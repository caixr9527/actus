#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 工具调用循环与仲裁逻辑。"""

import hashlib
import inspect
import json
import logging
import re
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from app.domain.external import LLM
from app.domain.models import (
    BrowserActionableElementsResult,
    BrowserCardExtractionResult,
    BrowserLinkMatchResult,
    BrowserMainContentResult,
    BrowserPageStructuredResult,
    BrowserPageType,
    SearchResults,
    Step,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
)
from app.domain.services.prompts import SYSTEM_PROMPT, REACT_SYSTEM_PROMPT
from app.domain.services.runtime.normalizers import truncate_text
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph_graphs.graph_parsers import (
    normalize_attachments,
    safe_parse_json,
)
from .runtime_logging import elapsed_ms, log_runtime, now_perf
from .settings import (
    ABSOLUTE_PATH_PATTERN,
    ACTION_PATTERN,
    ASK_USER_FUNCTION_NAME,
    BROWSER_ATOMIC_FUNCTION_NAMES,
    BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
    BROWSER_INTERACTION_PATTERN,
    BROWSER_NO_PROGRESS_LIMIT,
    BROWSER_PROGRESS_FUNCTIONS,
    CODING_PATTERN,
    CODE_BLOCK_PATTERN,
    FILE_FUNCTION_NAMES,
    FILE_PATTERN,
    NOTIFY_USER_FUNCTION_NAME,
    NUMBERED_LIST_PATTERN,
    PHATIC_PATTERN,
    REPEAT_TOOL_LIMIT,
    SEARCH_FUNCTION_NAMES,
    SEARCH_PATTERN,
    SEARCH_REPEAT_LIMIT,
    SEQUENCE_PATTERN,
    SHELL_COMMAND_PATTERN,
    TASK_MODE_ALLOWED_FUNCTIONS,
    TASK_MODE_ALLOWED_PREFIXES,
    TOOL_FAILURE_LIMIT,
    TOOL_REFERENCE_PATTERN,
    TOOL_RESULT_MAX_DICT_ITEMS,
    TOOL_RESULT_MAX_LIST_ITEMS,
    TOOL_RESULT_MAX_TEXT_CHARS,
    URL_PATTERN,
    WAIT_PATTERN,
    WEB_READING_PATTERN,
)

logger = logging.getLogger(__name__)


def _normalize_intent_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _analyze_text_intent(value: str) -> Dict[str, Any]:
    normalized_text = _normalize_intent_text(value)
    if not normalized_text:
        return {
            "text": "",
            "char_count": 0,
            "has_url": False,
            "has_absolute_path": False,
            "has_shell_command": False,
            "has_code_block": False,
            "has_numbered_list": False,
            "has_sequence_marker": False,
            "is_phatic": False,
            "needs_human_wait": False,
            "has_browser_interaction_signal": False,
            "has_web_reading_signal": False,
            "has_search_signal": False,
            "has_file_signal": False,
            "has_coding_signal": False,
            "has_action_signal": False,
            "has_tool_reference": False,
            "clause_count": 0,
        }

    clause_count = len(
        [
            segment
            for segment in re.split(r"[。！？!?；;\n]+", normalized_text)
            if str(segment).strip()
        ]
    )
    return {
        "text": normalized_text,
        "char_count": len(normalized_text),
        "has_url": bool(URL_PATTERN.search(normalized_text)),
        "has_absolute_path": bool(ABSOLUTE_PATH_PATTERN.search(normalized_text)),
        "has_shell_command": bool(SHELL_COMMAND_PATTERN.search(normalized_text)),
        "has_code_block": bool(CODE_BLOCK_PATTERN.search(normalized_text)),
        "has_numbered_list": bool(NUMBERED_LIST_PATTERN.search(normalized_text)),
        "has_sequence_marker": bool(SEQUENCE_PATTERN.search(normalized_text)),
        "is_phatic": bool(PHATIC_PATTERN.match(normalized_text)),
        "needs_human_wait": bool(WAIT_PATTERN.search(normalized_text)),
        "has_browser_interaction_signal": bool(BROWSER_INTERACTION_PATTERN.search(normalized_text)),
        "has_web_reading_signal": bool(WEB_READING_PATTERN.search(normalized_text)),
        "has_search_signal": bool(SEARCH_PATTERN.search(normalized_text)),
        "has_file_signal": bool(FILE_PATTERN.search(normalized_text)),
        "has_coding_signal": bool(CODING_PATTERN.search(normalized_text)),
        "has_action_signal": bool(ACTION_PATTERN.search(normalized_text)),
        "has_tool_reference": bool(TOOL_REFERENCE_PATTERN.search(normalized_text)),
        "clause_count": clause_count,
    }


def infer_entry_strategy(
        *,
        user_message: str,
        has_input_parts: bool,
        has_active_plan: bool,
) -> str:
    if has_active_plan:
        return "create_plan_or_reuse"
    if has_input_parts:
        return "recall_memory_context"

    signals = _analyze_text_intent(user_message)
    if signals["char_count"] == 0:
        return "recall_memory_context"
    if signals["needs_human_wait"]:
        return "direct_wait"
    if signals["is_phatic"] and not signals["has_action_signal"] and not signals["has_tool_reference"]:
        return "direct_answer"

    is_multi_step = (
            signals["has_numbered_list"]
            or signals["has_sequence_marker"]
            or signals["clause_count"] >= 3
            or signals["char_count"] >= 120
    )
    has_direct_execution_signal = any(
        (
            signals["has_tool_reference"],
            signals["has_url"],
            signals["has_absolute_path"],
            signals["has_shell_command"],
            signals["has_code_block"],
            signals["has_action_signal"],
        )
    )
    if has_direct_execution_signal and not is_multi_step:
        return "direct_execute"
    return "recall_memory_context"


def collect_available_tools(runtime_tools: Optional[List[BaseTool]]) -> List[Dict[str, Any]]:
    """收集当前步骤可用的工具 schema 列表。"""
    available_tools: List[Dict[str, Any]] = []
    for tool in list(runtime_tools or []):
        try:
            available_tools.extend(tool.get_tools())
        except Exception as e:
            log_runtime(
                logger,
                logging.WARNING,
                "工具Schema读取失败",
                tool_name=getattr(tool, "name", "unknown"),
                error=str(e),
            )

    def _tool_priority(tool_schema: Dict[str, Any]) -> Tuple[int, str]:
        function_name = str(
            (tool_schema.get("function") or {}).get("name")
            if isinstance(tool_schema, dict)
            else ""
        ).strip().lower()
        # 搜索类优先，浏览器类后置：多数检索任务先走 search 更快。
        if "search" in function_name:
            return 0, function_name
        if function_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES:
            return 10, function_name
        if function_name.startswith("browser_"):
            return 80, function_name
        return 20, function_name

    available_tools.sort(key=_tool_priority)
    return available_tools


def classify_step_task_mode(step: Step) -> str:
    signals = _analyze_text_intent(_build_step_candidate_text(step))

    if signals["needs_human_wait"]:
        return "human_wait"

    scores = {
        "web_reading": 0,
        "browser_interaction": 0,
        "coding": 0,
        "file_processing": 0,
        "research": 0,
    }
    if signals["has_browser_interaction_signal"]:
        scores["browser_interaction"] += 5
    if signals["has_web_reading_signal"]:
        scores["web_reading"] += 3
    if signals["has_url"]:
        scores["web_reading"] += 2
        scores["research"] += 2
    if signals["has_shell_command"] or signals["has_code_block"] or signals["has_coding_signal"]:
        scores["coding"] += 3
    if signals["has_absolute_path"] or signals["has_file_signal"]:
        scores["file_processing"] += 3
    if signals["has_search_signal"]:
        scores["research"] += 3
    if signals["has_tool_reference"]:
        if any(name in signals["text"] for name in
               ("browser_click", "browser_input", "browser_scroll", "browser_press_key", "browser_select_option")):
            scores["browser_interaction"] += 3
        if any(
                name in signals["text"]
                for name in (
                        "browser_view",
                        "browser_navigate",
                        "browser_restart",
                        "browser_read_current_page_structured",
                        "browser_extract_main_content",
                        "browser_extract_cards",
                        "browser_find_link_by_text",
                        "browser_find_actionable_elements",
                )
        ):
            scores["web_reading"] += 2
        if "shell_" in signals["text"]:
            scores["coding"] += 2
        if any(name in signals["text"] for name in FILE_FUNCTION_NAMES):
            scores["file_processing"] += 2
        if any(name in signals["text"] for name in SEARCH_FUNCTION_NAMES):
            scores["research"] += 2
    if scores["browser_interaction"] == 0 and scores["web_reading"] > 0:
        scores["research"] += 1

    ranked_modes = sorted(
        scores.items(),
        key=lambda item: (
            item[1],
            {
                "web_reading": 5,
                "browser_interaction": 4,
                "coding": 3,
                "file_processing": 2,
                "research": 1,
            }.get(item[0], 0),
        ),
        reverse=True,
    )
    if ranked_modes and ranked_modes[0][1] > 0:
        return ranked_modes[0][0]
    return "general"


def _extract_function_name(tool_schema: Dict[str, Any]) -> str:
    if not isinstance(tool_schema, dict):
        return ""
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip().lower()


def _tool_call_priority(
        function_name: str,
        *,
        preferred_function_names: Optional[Tuple[str, ...]] = None,
) -> int:
    normalized_name = function_name.strip().lower()
    preferred_names = tuple(
        str(item or "").strip().lower()
        for item in tuple(preferred_function_names or ())
        if str(item or "").strip()
    )
    if normalized_name in preferred_names:
        return -50 + preferred_names.index(normalized_name)
    if "search" in normalized_name:
        return 0
    if normalized_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES:
        return 10
    if normalized_name == NOTIFY_USER_FUNCTION_NAME:
        return 90
    if normalized_name == ASK_USER_FUNCTION_NAME:
        return 95
    if normalized_name.startswith("browser_"):
        return 80
    return 20


def pick_preferred_tool_call(
        tool_calls: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        *,
        preferred_function_names: Optional[Tuple[str, ...]] = None,
) -> Optional[Dict[str, Any]]:
    """从同轮多个 tool_call 中挑选本轮优先执行的候选。"""
    if len(tool_calls) == 0:
        return None
    if len(tool_calls) == 1:
        return tool_calls[0] if isinstance(tool_calls[0], dict) else None

    available_function_names: set[str] = set()
    for tool_schema in available_tools:
        function_name = _extract_function_name(tool_schema)
        if function_name:
            available_function_names.add(function_name)

    ranked_candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for index, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        function_name = str(function.get("name") or "").strip()
        if not function_name:
            continue

        normalized_name = function_name.lower()
        priority = _tool_call_priority(
            function_name,
            preferred_function_names=preferred_function_names,
        )
        if normalized_name not in available_function_names:
            priority += 1000
        ranked_candidates.append((priority, index, raw_call))

    if len(ranked_candidates) == 0:
        return None

    ranked_candidates.sort(key=lambda item: (item[0], item[1]))
    selected_call = ranked_candidates[0][2]
    selected_function = str((selected_call.get("function") or {}).get("name") or "")
    log_runtime(
        logger,
        logging.INFO,
        "工具候选仲裁完成",
        candidate_count=len(tool_calls),
        selected_function=selected_function,
    )
    return selected_call


def _resolve_tool_by_function_name(function_name: str, runtime_tools: Optional[List[BaseTool]]) -> Optional[BaseTool]:
    for tool in list(runtime_tools or []):
        try:
            if tool.has_tool(function_name):
                return tool
        except Exception:
            continue
    return None


def _parse_tool_call_args(raw_arguments: Any) -> Dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            log_runtime(
                logger,
                logging.WARNING,
                "工具参数解析失败",
                raw_length=len(raw_arguments),
            )
            return {}
    return {}


def _step_allows_user_wait(step: Step, function_args: Dict[str, Any]) -> bool:
    takeover = str(function_args.get("suggest_user_takeover") or "").strip().lower()
    if takeover == "browser":
        return True

    candidate_text = _build_step_candidate_text(step)
    if not candidate_text.strip():
        return False
    return bool(_analyze_text_intent(candidate_text)["needs_human_wait"])


def _build_step_candidate_text(step: Step) -> str:
    candidate_parts = [
        str(step.title or "").strip(),
        str(step.description or "").strip(),
        *[str(item or "").strip() for item in list(step.success_criteria or [])],
    ]
    return " ".join([part for part in candidate_parts if part])


def _filter_available_tools(
        available_tools: List[Dict[str, Any]],
        *,
        disallowed_function_names: Optional[set[str]] = None,
        allow_notify_user: bool,
        allow_ask_user: bool,
) -> List[Dict[str, Any]]:
    filtered_tools: List[Dict[str, Any]] = []
    blocked_names = set(disallowed_function_names or set())
    for tool_schema in available_tools:
        function_name = _extract_function_name(tool_schema)
        if function_name in blocked_names:
            continue
        if function_name == NOTIFY_USER_FUNCTION_NAME and not allow_notify_user:
            continue
        if function_name == ASK_USER_FUNCTION_NAME and not allow_ask_user:
            continue
        filtered_tools.append(tool_schema)
    return filtered_tools


def _truncate_tool_text(value: Any, *, max_chars: int = TOOL_RESULT_MAX_TEXT_CHARS) -> str:
    return truncate_text(value, max_chars=max_chars)


def _compact_tool_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, BaseModel):
        return _compact_tool_value(value.model_dump(mode="json"), depth=depth)
    if isinstance(value, str):
        return _truncate_tool_text(value)
    if depth >= 2:
        return _truncate_tool_text(value, max_chars=400)
    if isinstance(value, list):
        return [
            _compact_tool_value(item, depth=depth + 1)
            for item in value[:TOOL_RESULT_MAX_LIST_ITEMS]
        ]
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= TOOL_RESULT_MAX_DICT_ITEMS:
                break
            compacted[str(key)] = _compact_tool_value(item, depth=depth + 1)
        return compacted
    return _truncate_tool_text(value, max_chars=400)


def _hash_payload(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _build_tool_fingerprint(function_name: str, function_args: Dict[str, Any]) -> str:
    return _hash_payload(
        {
            "function_name": function_name.strip().lower(),
            "args": _compact_tool_value(function_args),
        }
    )


def _build_search_fingerprint(function_args: Dict[str, Any]) -> str:
    return _hash_payload(
        {
            "query": str(function_args.get("query") or "").strip().lower(),
            "engines": str(function_args.get("engines") or "").strip().lower(),
            "language": str(function_args.get("language") or "").strip().lower(),
            "time_range": str(function_args.get("time_range") or "").strip().lower(),
        }
    )


def _build_browser_observation_fingerprint(tool_result: ToolResult) -> str:
    result_data = tool_result.data if hasattr(tool_result, "data") else None
    return _hash_payload(
        {
            "message": _truncate_tool_text(getattr(tool_result, "message", ""), max_chars=200),
            "data": _compact_tool_value(result_data),
        }
    )


def _build_browser_route_state_key(
        *,
        browser_page_type: str,
        browser_url: str,
        browser_observation_fingerprint: str,
) -> str:
    # 这里把“页面类型 + 当前 URL + 最近一次浏览器观察结果”压成一个稳定 key。
    # 这个 key 的职责不是表达“信息已经完整”，而是表达“当前失败应该绑定在哪个页面状态上”。
    # 即使三项里有空字符串，它仍然有意义，因为“尚未读到页面结构/URL/观察结果”本身也是一个明确状态。
    # 这样首轮高阶工具失败时，会先被收敛到“初始空状态”；一旦后续读到了 URL 或页面观察变化，key 就会变化，失败封禁也会自然失效。
    return _hash_payload(
        {
            # page_type 用来区分正文页、列表页等不同路由状态，避免跨页面类型误复用失败记录。
            "page_type": browser_page_type.strip().lower(),
            # url 用来区分至少最常见的页面变化场景；只要跳到新页面，这里的 key 就会变化。
            "url": browser_url.strip(),
            # observation 记录最近一次浏览器高阶观察结果的摘要；URL 不变但页面内容变化时，也能触发 key 变化。
            "observation": browser_observation_fingerprint.strip(),
        }
    )


def _build_browser_high_level_failure_key(
        *,
        function_name: str,
        function_args: Dict[str, Any],
        browser_route_state_key: str,
) -> str:
    return _hash_payload(
        {
            "function_name": function_name.strip().lower(),
            "page": browser_route_state_key,
            "args": _compact_tool_value(function_args),
        }
    )


def _is_browser_high_level_temporarily_blocked(
        *,
        function_name: str,
        function_args: Dict[str, Any],
        browser_route_state_key: str,
        failed_high_level_keys: set[str],
) -> bool:
    normalized_function_name = function_name.strip().lower()
    if normalized_function_name not in BROWSER_HIGH_LEVEL_FUNCTION_NAMES:
        return False
    return _build_browser_high_level_failure_key(
        function_name=normalized_function_name,
        function_args=function_args,
        browser_route_state_key=browser_route_state_key,
    ) in failed_high_level_keys


def _collect_temporarily_blocked_browser_high_level_function_names(
        *,
        browser_route_state_key: str,
        failed_high_level_keys: set[str],
) -> set[str]:
    return {
        function_name
        for function_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES
        if _is_browser_high_level_temporarily_blocked(
            function_name=function_name,
            function_args={},
            browser_route_state_key=browser_route_state_key,
            failed_high_level_keys=failed_high_level_keys,
        )
    }


def _build_browser_preferred_function_names(
        *,
        task_mode: str,
        available_function_names: set[str],
        browser_page_type: str,
        browser_structured_ready: bool,
        browser_main_content_ready: bool,
        browser_cards_ready: bool,
        browser_link_match_ready: bool,
        browser_actionables_ready: bool,
        failed_high_level_functions: set[str],
) -> Tuple[str, ...]:
    is_listing_page = browser_page_type in {
        BrowserPageType.LISTING.value,
        BrowserPageType.SEARCH_RESULTS.value,
    }

    ordered_candidates: Tuple[str, ...] = ()
    if task_mode == "web_reading":
        if not browser_structured_ready:
            ordered_candidates = ("browser_read_current_page_structured",)
        elif is_listing_page:
            if not browser_cards_ready:
                ordered_candidates = ("browser_extract_cards",)
            elif not browser_link_match_ready:
                ordered_candidates = ("browser_find_link_by_text",)
            else:
                ordered_candidates = ("browser_click", "browser_navigate")
        elif not browser_main_content_ready:
            ordered_candidates = ("browser_extract_main_content",)
    elif task_mode == "browser_interaction":
        if not browser_structured_ready:
            ordered_candidates = ("browser_read_current_page_structured",)
        elif is_listing_page:
            if not browser_cards_ready:
                ordered_candidates = ("browser_extract_cards",)
            elif not browser_link_match_ready:
                ordered_candidates = ("browser_find_link_by_text",)
            else:
                ordered_candidates = ("browser_click", "browser_navigate")
        elif not browser_actionables_ready:
            ordered_candidates = ("browser_find_actionable_elements",)
    else:
        return ()

    return tuple(
        function_name
        for function_name in ordered_candidates
        if function_name in available_function_names and function_name not in failed_high_level_functions
    )


def _normalize_browser_page_type(value: Any) -> str:
    if isinstance(value, BrowserPageType):
        return value.value
    normalized_value = getattr(value, "value", value)
    return str(normalized_value or "").strip().lower()


def _extract_browser_tool_state(tool_result: ToolResult) -> Dict[str, Any]:
    data = tool_result.data
    if isinstance(
            data,
            (
                    BrowserPageStructuredResult,
                    BrowserMainContentResult,
                    BrowserCardExtractionResult,
                    BrowserActionableElementsResult,
            ),
    ):
        return {
            "url": str(data.url or "").strip(),
            "title": str(data.title or "").strip(),
            "page_type": _normalize_browser_page_type(data.page_type),
            "selector": "",
            "index": None,
        }
    if isinstance(data, BrowserLinkMatchResult):
        return {
            "url": str(data.url or "").strip(),
            "title": str(data.matched_text or "").strip(),
            "page_type": "",
            "selector": str(data.selector or "").strip(),
            "index": data.index,
        }
    if isinstance(data, dict):
        return {
            "url": str(data.get("url") or "").strip(),
            "title": str(data.get("title") or data.get("matched_text") or "").strip(),
            "page_type": _normalize_browser_page_type(data.get("page_type")),
            "selector": str(data.get("selector") or "").strip(),
            "index": data.get("index"),
        }
    return {"url": "", "title": "", "page_type": "", "selector": "", "index": None}


def _build_browser_atomic_allowlist(
        *,
        task_mode: str,
        browser_page_type: str,
        browser_structured_ready: bool,
        browser_link_match_ready: bool,
        browser_actionables_ready: bool,
        failed_high_level_functions: set[str],
) -> Tuple[str, ...]:
    # 这里不再保留 browser_cards_ready 这种未使用参数。
    # 原子工具是否放行，当前只依赖：
    # 1. 页面类型是否已经判明
    # 2. 链接是否已经匹配完成
    # 3. 可交互元素是否已经提取完成
    # 4. 哪些高阶能力在当前页面状态下失败过
    # 没有参与判断的状态就直接删除，避免函数签名和真实逻辑不一致。
    is_listing_page = browser_page_type in {
        BrowserPageType.LISTING.value,
        BrowserPageType.SEARCH_RESULTS.value,
    }

    if task_mode == "web_reading":
        if not browser_structured_ready and "browser_read_current_page_structured" in failed_high_level_functions:
            return ("browser_view", "browser_scroll_down", "browser_scroll_up")
        if is_listing_page:
            if browser_link_match_ready:
                return ("browser_click", "browser_navigate")
            if any(
                    function_name in failed_high_level_functions
                    for function_name in (
                            "browser_extract_cards",
                            "browser_read_current_page_structured",
                    )
            ):
                return ("browser_view", "browser_scroll_down", "browser_scroll_up", "browser_navigate")
            return ()
        if any(
                function_name in failed_high_level_functions
                for function_name in (
                        "browser_extract_main_content",
                        "browser_read_current_page_structured",
                )
        ):
            return ("browser_view", "browser_scroll_down", "browser_scroll_up")
        return ()

    if task_mode == "browser_interaction":
        if not browser_structured_ready and "browser_read_current_page_structured" in failed_high_level_functions:
            return BROWSER_ATOMIC_FUNCTION_NAMES
        if is_listing_page:
            if browser_link_match_ready:
                return ("browser_click", "browser_navigate")
            if any(
                    function_name in failed_high_level_functions
                    for function_name in (
                            "browser_extract_cards",
                            "browser_read_current_page_structured",
                    )
            ):
                return BROWSER_ATOMIC_FUNCTION_NAMES
            return ()
        if browser_actionables_ready or any(
                function_name in failed_high_level_functions
                for function_name in (
                        "browser_find_actionable_elements",
                        "browser_read_current_page_structured",
                )
        ):
            return BROWSER_ATOMIC_FUNCTION_NAMES
        return ()

    return ()


def _build_browser_capability_gap_allowlist(
        *,
        task_mode: str,
) -> Tuple[str, ...]:
    if task_mode == "web_reading":
        return ("browser_view", "browser_navigate", "browser_restart", "browser_scroll_down", "browser_scroll_up")
    if task_mode == "browser_interaction":
        return BROWSER_ATOMIC_FUNCTION_NAMES
    return ()


def _build_browser_route_block_message(
        *,
        task_mode: str,
        function_name: str,
        browser_page_type: str,
        browser_structured_ready: bool,
        browser_cards_ready: bool,
        browser_link_match_ready: bool,
        browser_actionables_ready: bool,
        last_browser_route_url: str,
        last_browser_route_selector: str,
        last_browser_route_index: Optional[int],
) -> str:
    is_listing_page = browser_page_type in {
        BrowserPageType.LISTING.value,
        BrowserPageType.SEARCH_RESULTS.value,
    }
    normalized_function_name = function_name.strip().lower()

    if not browser_structured_ready:
        return "当前步骤属于浏览器任务，请先调用 browser_read_current_page_structured 判断页面类型，再决定后续动作。"

    if is_listing_page:
        if not browser_cards_ready:
            return "当前页面是列表页，请先调用 browser_extract_cards 提取候选卡片，不要直接执行原子浏览器动作。"
        if not browser_link_match_ready:
            return "当前页面是列表页，已提取候选卡片，请先调用 browser_find_link_by_text 选定目标，不要直接执行原子浏览器动作。"
        if normalized_function_name not in {"browser_click", "browser_navigate"}:
            if last_browser_route_index is not None:
                url_hint = f"；如需直接跳转可使用 browser_navigate 或 fetch_page 读取 {last_browser_route_url}" if last_browser_route_url else ""
                selector_hint = f"；匹配定位: {last_browser_route_selector}" if last_browser_route_selector else ""
                return (
                    "当前页面已完成目标链接定位，请优先使用 "
                    f"browser_click(index={last_browser_route_index}){url_hint}{selector_hint}。"
                )
            if last_browser_route_url:
                return (
                    "当前页面已完成目标链接定位，请优先使用返回结果中的 URL 调用 browser_navigate"
                    f" 或 fetch_page，当前可用 URL: {last_browser_route_url}"
                )
            return "当前页面已完成目标链接定位，请优先使用 browser_click 或 browser_navigate。"

    if task_mode == "web_reading":
        return "当前步骤属于网页阅读任务，请优先使用高阶浏览器阅读能力，只有高阶提取失败后才能退回原子浏览器动作。"
    if task_mode == "browser_interaction" and not browser_actionables_ready:
        return "当前步骤属于浏览器交互任务，请先调用 browser_find_actionable_elements 确认可交互元素，再执行原子浏览器动作。"
    return "当前浏览器原子动作尚未开放，请先完成当前阶段要求的高阶浏览器能力。"


def _attach_browser_degrade_payload(
        tool_result: ToolResult,
        *,
        function_name: str,
        degrade_reason: str,
        browser_page_type: str,
        browser_url: str,
        browser_title: str,
) -> ToolResult:
    payload = tool_result.data if isinstance(tool_result.data, dict) else {}
    payload = {
        **payload,
        "degrade_reason": degrade_reason,
        "function_name": function_name,
        "page_type": browser_page_type,
        "url": browser_url,
        "title": browser_title,
    }
    tool_result.data = payload
    return tool_result


def _build_browser_high_level_retry_block_message(
        *,
        function_name: str,
        function_args: Dict[str, Any],
) -> str:
    normalized_function_name = function_name.strip().lower()
    if normalized_function_name == "browser_find_link_by_text":
        query = str(function_args.get("text") or "").strip()
        if query:
            return f"当前页面中“{query}”的链接匹配刚刚失败，请先更换关键词或改变页面状态后再试。"
        return "当前页面的链接匹配刚刚失败，请先更换关键词或改变页面状态后再试。"
    return f"{function_name} 在当前页面状态下刚刚失败，请先改变页面状态或尝试其他高阶路径后再试。"


def _coerce_optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped_value = value.strip()
        if stripped_value.isdigit() or (
                stripped_value.startswith("-") and stripped_value[1:].isdigit()
        ):
            return int(stripped_value)
    return None


def _build_listing_click_target_block_message(
        *,
        last_browser_route_index: Optional[int],
        last_browser_route_url: str,
        last_browser_route_selector: str,
) -> str:
    if last_browser_route_index is None:
        if last_browser_route_url:
            return (
                "当前列表页已完成目标链接定位，但匹配结果没有可点击 index，"
                f"请改用 browser_navigate 或 fetch_page 读取 {last_browser_route_url}。"
            )
        return "当前列表页已完成目标链接定位，但匹配结果没有可点击 index，请改用 browser_navigate。"
    selector_hint = f"；匹配定位: {last_browser_route_selector}" if last_browser_route_selector else ""
    url_hint = f"；目标 URL: {last_browser_route_url}" if last_browser_route_url else ""
    return (
        "当前列表页已完成目标链接定位，只允许点击刚刚匹配到的目标，"
        f"请调用 browser_click(index={last_browser_route_index}){selector_hint}{url_hint}。"
    )


def _extract_text_from_user_content(user_content: Optional[List[Dict[str, Any]]]) -> str:
    fragments: List[str] = []
    for item in list(user_content or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() != "text":
            continue
        text = str(item.get("text") or "").strip()
        if text:
            fragments.append(text)
    return "\n".join(fragments)


def _step_or_user_content_has_url(step: Step, user_content: Optional[List[Dict[str, Any]]]) -> bool:
    candidate_text = "\n".join(
        [
            _build_step_candidate_text(step),
            _extract_text_from_user_content(user_content),
        ]
    )
    return bool(URL_PATTERN.search(candidate_text))


def _extract_search_result_urls(tool_result: ToolResult) -> List[str]:
    data = tool_result.data
    urls: List[str] = []

    if isinstance(data, SearchResults):
        for item in list(data.results or []):
            url = str(getattr(item, "url", "") or "").strip()
            if url:
                urls.append(url)
    elif isinstance(data, dict):
        raw_results = data.get("results") or []
        if isinstance(raw_results, list):
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if url:
                    urls.append(url)

    deduped_urls: List[str] = []
    seen_urls: set[str] = set()
    for url in urls:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_urls.append(url)
    return deduped_urls[:5]


def _is_allowed_in_task_mode(function_name: str, task_mode: str) -> bool:
    normalized_name = function_name.strip().lower()
    allowed_functions = set(TASK_MODE_ALLOWED_FUNCTIONS.get(task_mode, TASK_MODE_ALLOWED_FUNCTIONS["general"]))
    allowed_prefixes = TASK_MODE_ALLOWED_PREFIXES.get(task_mode, TASK_MODE_ALLOWED_PREFIXES["general"])
    if normalized_name in allowed_functions:
        return True
    return any(normalized_name.startswith(prefix) for prefix in allowed_prefixes)


def _build_task_mode_disallowed_names(
        available_tools: List[Dict[str, Any]],
        *,
        task_mode: str,
) -> set[str]:
    blocked_names: set[str] = set()
    for tool_schema in available_tools:
        function_name = _extract_function_name(tool_schema)
        if not function_name:
            continue
        if _is_allowed_in_task_mode(function_name, task_mode):
            continue
        blocked_names.add(function_name)
    return blocked_names


def _build_loop_break_payload(
        *,
        step: Step,
        loop_break_reason: str,
        blocker: str,
        next_hint: str,
) -> Dict[str, Any]:
    log_runtime(
        logger,
        logging.INFO,
        "工具循环已收敛",
        step_id=str(step.id or ""),
        reason=loop_break_reason,
        blocker_count=1,
        error=blocker,
        next_hint=next_hint,
    )
    return {
        "success": False,
        "result": f"当前步骤暂时未能完成：{step.description}",
        "attachments": [],
    }


def _summarize_tool_result_data(function_name: str, tool_result: ToolResult) -> Dict[str, Any]:
    normalized_name = function_name.strip().lower()
    result_data = tool_result.data
    if normalized_name == NOTIFY_USER_FUNCTION_NAME:
        return {
            "notified": True,
            "message": "当前步骤进度通知已发送，请继续执行实际工具步骤，不要再次调用进度通知。",
        }
    if normalized_name == ASK_USER_FUNCTION_NAME:
        interrupt_payload = result_data.get("interrupt") if isinstance(result_data, dict) else None
        if isinstance(interrupt_payload, dict):
            return {
                "interrupt": {
                    "kind": str(interrupt_payload.get("kind") or "").strip(),
                    "prompt": _truncate_tool_text(interrupt_payload.get("prompt"), max_chars=200),
                    "title": _truncate_tool_text(interrupt_payload.get("title"), max_chars=120),
                }
            }
        return {"interrupt": True}
    if normalized_name == "write_file" and isinstance(result_data, dict):
        return {
            "filepath": str(
                result_data.get("filepath")
                or result_data.get("file_path")
                or result_data.get("path")
                or ""
            ).strip(),
            "message": _truncate_tool_text(tool_result.message, max_chars=200),
        }
    if normalized_name == "read_file" and isinstance(result_data, dict):
        return {
            "filepath": str(
                result_data.get("filepath")
                or result_data.get("file_path")
                or result_data.get("path")
                or ""
            ).strip(),
            "content": _truncate_tool_text(result_data.get("content"), max_chars=1800),
        }
    if normalized_name in {"list_files", "find_files"} and isinstance(result_data, dict):
        files = result_data.get("files") or result_data.get("results") or []
        if not isinstance(files, list):
            files = []
        return {
            "dir_path": str(result_data.get("dir_path") or "").strip(),
            "files": [_truncate_tool_text(item, max_chars=200) for item in files[:TOOL_RESULT_MAX_LIST_ITEMS]],
        }
    if normalized_name == "search_web" and isinstance(result_data, SearchResults):
        return {
            "query": str(result_data.query or "").strip(),
            "results": [
                {
                    "title": _truncate_tool_text(item.title, max_chars=120),
                    "url": _truncate_tool_text(item.url, max_chars=240),
                    "snippet": _truncate_tool_text(item.snippet, max_chars=200),
                }
                for item in list(result_data.results or [])[:5]
            ],
        }
    return {"data": _compact_tool_value(result_data)}


def _build_tool_feedback_content(function_name: str, tool_result: ToolResult) -> str:
    payload = {
        "success": bool(tool_result.success),
        "message": _truncate_tool_text(tool_result.message, max_chars=240),
        "data": _summarize_tool_result_data(function_name, tool_result),
    }
    return json.dumps(payload, ensure_ascii=False)


async def execute_step_with_prompt(
        *,
        llm: LLM,
        step: Step,
        runtime_tools: Optional[List[BaseTool]],
        max_tool_iterations: int = 5,
        task_mode: str = "general",
        on_tool_event: Optional[Callable[[ToolEvent], Optional[Awaitable[None]]]] = None,
        user_content: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[ToolEvent]]:
    """执行单步任务，支持“模型决策 -> 调工具 -> 回传模型”的最小循环。"""
    started_at = now_perf()

    emitted_tool_events: List[ToolEvent] = []

    async def _notify_tool_event(event: ToolEvent) -> None:
        try:
            emitted_tool_events.append(event)
            if on_tool_event is None:
                return
            maybe_awaitable = on_tool_event(event)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception as e:
            log_runtime(
                logger,
                logging.WARNING,
                "工具事件投递失败",
                tool_name=event.tool_name,
                function_name=event.function_name,
                status=event.status.value,
                error=str(e),
            )

    normalized_user_content = list(user_content or [])
    if len(normalized_user_content) == 0:
        prompt_text = str(step.description or "").strip()
        normalized_user_content = [{"type": "text", "text": prompt_text}]

    available_tools = collect_available_tools(runtime_tools)
    available_function_names = {
        _extract_function_name(tool_schema)
        for tool_schema in available_tools
        if _extract_function_name(tool_schema)
    }
    browser_route_enabled = (
            task_mode in {"web_reading", "browser_interaction"}
            and any(function_name in available_function_names for function_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES)
    )
    blocked_function_names = _build_task_mode_disallowed_names(
        available_tools,
        task_mode=task_mode,
    )
    if task_mode in {"web_reading", "browser_interaction"} and not browser_route_enabled:
        for function_name in _build_browser_capability_gap_allowlist(task_mode=task_mode):
            blocked_function_names.discard(function_name)
    log_runtime(
        logger,
        logging.INFO,
        "工具执行循环准备完成",
        step_id=str(step.id or ""),
        step_title=str(step.title or step.description or ""),
        task_mode=task_mode,
        available_tool_count=len(available_tools),
        blocked_tool_count=len(blocked_function_names),
        max_tool_iterations=max(1, int(max_tool_iterations)),
    )

    if len(available_tools) == 0:
        log_runtime(
            logger,
            logging.INFO,
            "当前步骤无可用工具",
            step_id=str(step.id or ""),
        )
        llm_started_at = now_perf()
        llm_message = await llm.invoke(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + REACT_SYSTEM_PROMPT},
                {"role": "user", "content": normalized_user_content}
            ],
            tools=[],
            response_format={"type": "json_object"},
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        parsed = safe_parse_json(llm_message.get("content"))
        log_runtime(
            logger,
            logging.INFO,
            "无工具模式执行完成",
            step_id=str(step.id or ""),
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
            attachment_count=len(normalize_attachments(parsed.get("attachments"))),
        )
        return {
            "success": bool(parsed.get("success", True)),
            "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
            "attachments": normalize_attachments(parsed.get("attachments")),
        }, []

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + REACT_SYSTEM_PROMPT},
        {"role": "user", "content": normalized_user_content}
    ]

    llm_message: Dict[str, Any] = {}
    notify_user_sent = False
    allow_ask_user = task_mode == "human_wait" or _step_allows_user_wait(step, {})
    research_route_enabled = (
            task_mode in {"research", "web_reading"}
            and {"search_web", "fetch_page"}.issubset(available_function_names)
    )
    research_has_explicit_url = research_route_enabled and _step_or_user_content_has_url(
        step,
        normalized_user_content,
    )
    research_search_ready = False
    research_fetch_completed = False
    research_candidate_urls: List[str] = []
    browser_page_type = ""
    browser_structured_ready = False
    browser_main_content_ready = False
    browser_cards_ready = False
    browser_link_match_ready = False
    browser_actionables_ready = False
    last_browser_route_url = ""
    last_browser_route_title = ""
    last_browser_route_selector = ""
    last_browser_route_index: Optional[int] = None
    last_tool_fingerprint = ""
    same_tool_repeat_count = 0
    search_repeat_counter: Dict[str, int] = {}
    last_browser_observation_fingerprint = ""
    browser_no_progress_count = 0
    failed_browser_high_level_keys: set[str] = set()
    consecutive_failure_count = 0

    for index in range(max(1, int(max_tool_iterations))):
        # 每轮都先根据“当前页面状态”生成一个 route key。
        # 这里即使 browser_page_type / last_browser_route_url / last_browser_observation_fingerprint 里有空值，key 仍然有意义：
        # 1. 空值组合本身就代表“还没读到页面结构/URL/观察结果”的初始状态。
        # 2. 首轮高阶工具如果在这个初始状态下失败，需要有地方把失败绑定起来，避免同页同参立刻空转重试。
        # 3. 只要后续任一维度从空值变成非空，或观察结果发生变化，这个 key 就会变化，旧失败记录不会再继续封禁新状态。
        # 4. 所以这个 key 的核心价值是“失败收敛按状态隔离”，不是“要求三项字段一开始就必须完整”。
        browser_route_state_key = _build_browser_route_state_key(
            # 当前已知页面类型；未读到结构化结果时这里会是空字符串。
            browser_page_type=browser_page_type,
            # 当前已知路由 URL；还没拿到浏览器结果时这里也可能为空。
            browser_url=last_browser_route_url,
            # 最近一次浏览器观察摘要；没有成功浏览器观察前同样允许为空。
            browser_observation_fingerprint=last_browser_observation_fingerprint,
        )
        # 再用这个页面状态 key 反推出：在“当前页面状态”下，哪些高阶浏览器函数应该被暂时封禁。
        # 这里拿到的是“函数名集合”，目的是给本轮工具白名单/黑名单直接使用。
        failed_browser_high_level_functions = _collect_temporarily_blocked_browser_high_level_function_names(
            # 当前轮次使用的页面状态 key。
            browser_route_state_key=browser_route_state_key,
            # 整个 step 生命周期里累积的“函数 + 参数 + 页面状态”失败键集合。
            failed_high_level_keys=failed_browser_high_level_keys,
        )
        # 先从任务模式级别的禁用工具开始构建本轮黑名单。
        iteration_blocked_function_names = set(blocked_function_names)
        # 再把“当前页面状态下暂时封禁的高阶浏览器函数”叠加进去。
        iteration_blocked_function_names.update(failed_browser_high_level_functions)
        if browser_route_enabled:
            # 浏览器强路由开启时，再单独算一遍原子浏览器工具的白名单。
            allowed_atomic_browser_functions = set(
                _build_browser_atomic_allowlist(
                    task_mode=task_mode,
                    browser_page_type=browser_page_type,
                    browser_structured_ready=browser_structured_ready,
                    browser_link_match_ready=browser_link_match_ready,
                    browser_actionables_ready=browser_actionables_ready,
                    failed_high_level_functions=failed_browser_high_level_functions,
                )
            )
            for function_name in BROWSER_ATOMIC_FUNCTION_NAMES:
                # 原子浏览器工具默认先禁用；只有进入 allowlist 的工具才会被放开。
                if function_name in available_function_names and function_name not in allowed_atomic_browser_functions:
                    iteration_blocked_function_names.add(function_name)
                # 已进入 allowlist 的原子工具，要从黑名单里移除，保证模型这一轮真的能调用到。
                elif function_name in allowed_atomic_browser_functions:
                    iteration_blocked_function_names.discard(function_name)
        iteration_tools = _filter_available_tools(
            available_tools,
            disallowed_function_names=iteration_blocked_function_names,
            allow_notify_user=not notify_user_sent,
            allow_ask_user=allow_ask_user,
        )
        log_runtime(
            logger,
            logging.INFO,
            "开始工具决策轮次",
            step_id=str(step.id or ""),
            iteration=index,
            task_mode=task_mode,
            available_tool_count=len(iteration_tools),
        )
        llm_started_at = now_perf()
        llm_message = await llm.invoke(
            messages=messages,
            tools=iteration_tools,
            tool_choice="auto",
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        tool_calls = llm_message.get("tool_calls") or []
        if len(tool_calls) == 0:
            parsed = safe_parse_json(llm_message.get("content"))
            # 如果模型返回结果为空，则尝试重新执行
            if parsed.get("success", True) and str(parsed.get("result", "")).strip() == '':
                log_runtime(
                    logger,
                    logging.WARNING,
                    "模型结果为空，准备重试",
                    step_id=str(step.id or ""),
                    iteration=index,
                    llm_elapsed_ms=llm_cost_ms,
                    elapsed_ms=elapsed_ms(started_at),
                )
                continue
            log_runtime(
                logger,
                logging.INFO,
                "未调用工具直接完成当前轮次",
                step_id=str(step.id or ""),
                iteration=index,
                success=bool(parsed.get("success", True)),
                attachment_count=len(normalize_attachments(parsed.get("attachments"))),
                llm_elapsed_ms=llm_cost_ms,
                elapsed_ms=elapsed_ms(started_at),
            )
            return {
                "success": bool(parsed.get("success", True)),
                "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
                "attachments": normalize_attachments(parsed.get("attachments")),
            }, emitted_tool_events

        selected_tool_call = pick_preferred_tool_call(
            tool_calls=[item for item in tool_calls if isinstance(item, dict)],
            available_tools=iteration_tools,
            preferred_function_names=(
                ("fetch_page",)
                if research_route_enabled and not research_fetch_completed and (
                            research_search_ready or research_has_explicit_url)
                else _build_browser_preferred_function_names(
                    task_mode=task_mode,
                    available_function_names=available_function_names,
                    browser_page_type=browser_page_type,
                    browser_structured_ready=browser_structured_ready,
                    browser_main_content_ready=browser_main_content_ready,
                    browser_cards_ready=browser_cards_ready,
                    browser_link_match_ready=browser_link_match_ready,
                    browser_actionables_ready=browser_actionables_ready,
                    failed_high_level_functions=failed_browser_high_level_functions,
                )
            ),
        )
        if selected_tool_call is None:
            continue

        messages.append(
            {
                "role": "assistant",
                "content": llm_message.get("content"),
                "tool_calls": [selected_tool_call],
            }
        )

        function = selected_tool_call.get("function")
        if not isinstance(function, dict):
            continue

        function_name = str(function.get("name") or "").strip()
        if not function_name:
            continue
        normalized_function_name = function_name.lower()
        tool_call_id = str(selected_tool_call.get("id") or uuid.uuid4())
        function_args = _parse_tool_call_args(function.get("arguments"))
        tool_fingerprint = _build_tool_fingerprint(normalized_function_name, function_args)
        if tool_fingerprint == last_tool_fingerprint:
            same_tool_repeat_count += 1
        else:
            same_tool_repeat_count = 1
            last_tool_fingerprint = tool_fingerprint
        log_runtime(
            logger,
            logging.INFO,
            "已选择工具调用",
            step_id=str(step.id or ""),
            iteration=index,
            tool_call_id=tool_call_id,
            function_name=function_name,
            task_mode=task_mode,
            same_tool_repeat_count=same_tool_repeat_count,
            arg_keys=sorted(function_args.keys()),
        )

        matched_tool = _resolve_tool_by_function_name(function_name=function_name, runtime_tools=runtime_tools)
        tool_name = matched_tool.name if matched_tool is not None else "unknown"
        tool_cost_ms = 0
        loop_break_reason = ""
        calling_event = ToolEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            function_name=function_name,
            function_args=function_args,
            status=ToolEventStatus.CALLING,
        )
        await _notify_tool_event(calling_event)

        if matched_tool is None:
            log_runtime(
                logger,
                logging.WARNING,
                "工具调用无效",
                step_id=str(step.id or ""),
                function_name=function_name,
            )
            tool_result = ToolResult(success=False, message=f"无效工具: {function_name}")
        elif normalized_function_name in iteration_blocked_function_names:
            if browser_route_enabled and normalized_function_name.startswith("browser_"):
                loop_break_reason = "browser_route_blocked"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "浏览器固定路径拦截工具调用",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    task_mode=task_mode,
                    browser_page_type=browser_page_type,
                    browser_structured_ready=browser_structured_ready,
                    browser_cards_ready=browser_cards_ready,
                    browser_link_match_ready=browser_link_match_ready,
                    browser_actionables_ready=browser_actionables_ready,
                )
                tool_result = ToolResult(
                    success=False,
                    message=_build_browser_route_block_message(
                        task_mode=task_mode,
                        function_name=function_name,
                        browser_page_type=browser_page_type,
                        browser_structured_ready=browser_structured_ready,
                        browser_cards_ready=browser_cards_ready,
                        browser_link_match_ready=browser_link_match_ready,
                        browser_actionables_ready=browser_actionables_ready,
                        last_browser_route_url=last_browser_route_url,
                        last_browser_route_selector=last_browser_route_selector,
                        last_browser_route_index=last_browser_route_index,
                    ),
                )
            else:
                loop_break_reason = "task_mode_tool_blocked"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "任务模式拦截工具调用",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    task_mode=task_mode,
                )
                tool_result = ToolResult(
                    success=False,
                    message=f"当前步骤的任务模式 {task_mode} 不允许调用工具: {function_name}",
                )
        elif research_route_enabled and normalized_function_name == "fetch_page" and not research_has_explicit_url and not research_search_ready:
            loop_break_reason = "research_route_search_required"
            tool_result = ToolResult(
                success=False,
                message="当前步骤属于检索/网页阅读任务，请先调用 search_web 获取候选链接，再使用 fetch_page 读取正文。",
            )
        elif research_route_enabled and normalized_function_name == "search_web" and research_has_explicit_url and not research_fetch_completed:
            loop_break_reason = "research_route_fetch_required"
            tool_result = ToolResult(
                success=False,
                message="当前步骤已提供明确 URL，请直接调用 fetch_page 读取页面正文，不要先重复搜索。",
            )
        elif research_route_enabled and normalized_function_name == "search_web" and research_search_ready and not research_fetch_completed:
            loop_break_reason = "research_route_fetch_required"
            candidate_hint = "；".join(research_candidate_urls[:3])
            tool_result = ToolResult(
                success=False,
                message=(
                        "已经拿到候选链接，请优先对搜索结果中的 URL 调用 fetch_page 读取正文。"
                        + (f" 可用链接示例: {candidate_hint}" if candidate_hint else "")
                ),
            )
        elif function_name == NOTIFY_USER_FUNCTION_NAME and notify_user_sent:
            log_runtime(
                logger,
                logging.INFO,
                "重复进度通知已收敛",
                step_id=str(step.id or ""),
                function_name=function_name,
            )
            tool_result = ToolResult(
                success=True,
                message="当前步骤已发送过进度通知，请继续调用实际工具或直接完成当前步骤。",
                data="Continue",
            )
        elif function_name == ASK_USER_FUNCTION_NAME and not _step_allows_user_wait(step, function_args):
            log_runtime(
                logger,
                logging.WARNING,
                "提前请求用户交互，已拦截",
                step_id=str(step.id or ""),
                function_name=function_name,
                step_description=str(step.description or ""),
            )
            tool_result = ToolResult(
                success=False,
                message="当前步骤不允许向用户提问。请先完成当前步骤，只能在明确需要用户确认/选择/输入的步骤中使用该工具。",
            )
        elif (
                browser_route_enabled
                and normalized_function_name == "browser_click"
                and browser_page_type in {
                    BrowserPageType.LISTING.value,
                    BrowserPageType.SEARCH_RESULTS.value,
                }
                and browser_link_match_ready
        ):
            requested_index = _coerce_optional_int(function_args.get("index"))
            coordinate_x = function_args.get("coordinate_x")
            coordinate_y = function_args.get("coordinate_y")
            if (
                    coordinate_x is not None
                    or coordinate_y is not None
                    or requested_index is None
                    or last_browser_route_index is None
                    or requested_index != last_browser_route_index
            ):
                loop_break_reason = "browser_click_target_blocked"
                log_runtime(
                    logger,
                    logging.INFO,
                    "列表页点击目标与已匹配结果不一致，已拦截",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    requested_index=requested_index,
                    matched_index=last_browser_route_index,
                    has_coordinate_x=coordinate_x is not None,
                    has_coordinate_y=coordinate_y is not None,
                )
                tool_result = ToolResult(
                    success=False,
                    message=_build_listing_click_target_block_message(
                        last_browser_route_index=last_browser_route_index,
                        last_browser_route_url=last_browser_route_url,
                        last_browser_route_selector=last_browser_route_selector,
                    ),
                )
            else:
                tool_started_at = now_perf()
                try:
                    tool_result = await matched_tool.invoke(function_name, **function_args)
                    tool_cost_ms = elapsed_ms(tool_started_at)
                    if not isinstance(tool_result, ToolResult):
                        tool_result = ToolResult(success=True, data=tool_result)
                except Exception as e:
                    tool_cost_ms = elapsed_ms(tool_started_at)
                    log_runtime(
                        logger,
                        logging.ERROR,
                        "工具调用失败",
                        step_id=str(step.id or ""),
                        function_name=function_name,
                        tool_name=tool_name,
                        error=str(e),
                        tool_elapsed_ms=tool_cost_ms,
                        elapsed_ms=elapsed_ms(started_at),
                        exc_info=True,
                    )
                    tool_result = ToolResult(success=False, message=f"调用工具失败: {function_name}")
        elif browser_route_enabled and _is_browser_high_level_temporarily_blocked(
                function_name=normalized_function_name,
                function_args=function_args,
                browser_route_state_key=browser_route_state_key,
                failed_high_level_keys=failed_browser_high_level_keys,
        ):
            loop_break_reason = "browser_high_level_retry_blocked"
            log_runtime(
                logger,
                logging.INFO,
                "浏览器高阶能力在当前页面状态下暂时封禁",
                step_id=str(step.id or ""),
                function_name=function_name,
                browser_route_state_key=browser_route_state_key,
            )
            tool_result = ToolResult(
                success=False,
                message=_build_browser_high_level_retry_block_message(
                    function_name=function_name,
                    function_args=function_args,
                ),
            )
        elif normalized_function_name == "search_web":
            search_fingerprint = _build_search_fingerprint(function_args)
            search_repeat_counter[search_fingerprint] = search_repeat_counter.get(search_fingerprint, 0) + 1
            if search_repeat_counter[search_fingerprint] > SEARCH_REPEAT_LIMIT:
                loop_break_reason = "search_repeat"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "重复搜索已收敛",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    search_repeat_count=search_repeat_counter[search_fingerprint],
                )
                tool_result = ToolResult(
                    success=False,
                    message="同一搜索查询已重复多次，请改写查询、缩小范围，或改用 fetch_page / 其他工具继续。",
                )
            else:
                tool_started_at = now_perf()
                try:
                    tool_result = await matched_tool.invoke(function_name, **function_args)
                    tool_cost_ms = elapsed_ms(tool_started_at)
                    if not isinstance(tool_result, ToolResult):
                        tool_result = ToolResult(success=True, data=tool_result)
                except Exception as e:
                    tool_cost_ms = elapsed_ms(tool_started_at)
                    log_runtime(
                        logger,
                        logging.ERROR,
                        "工具调用失败",
                        step_id=str(step.id or ""),
                        function_name=function_name,
                        tool_name=tool_name,
                        error=str(e),
                        tool_elapsed_ms=tool_cost_ms,
                        elapsed_ms=elapsed_ms(started_at),
                        exc_info=True,
                    )
                    tool_result = ToolResult(success=False, message=f"调用工具失败: {function_name}")
        elif same_tool_repeat_count > REPEAT_TOOL_LIMIT:
            loop_break_reason = "repeat_tool_call"
            log_runtime(
                logger,
                logging.WARNING,
                "重复工具调用已收敛",
                step_id=str(step.id or ""),
                function_name=function_name,
                same_tool_repeat_count=same_tool_repeat_count,
            )
            tool_result = ToolResult(
                success=False,
                message="检测到同一工具与相近参数被重复调用，请改用其他工具、调整参数，或结束当前步骤。",
            )
        else:
            try:
                tool_started_at = now_perf()
                tool_result = await matched_tool.invoke(function_name, **function_args)
                tool_cost_ms = elapsed_ms(tool_started_at)
                if not isinstance(tool_result, ToolResult):
                    # 兼容少数工具返回 dict/str 的历史实现，统一收敛为 ToolResult。
                    tool_result = ToolResult(success=True, data=tool_result)
            except Exception as e:
                tool_cost_ms = elapsed_ms(tool_started_at)
                log_runtime(
                    logger,
                    logging.ERROR,
                    "工具调用失败",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    tool_name=tool_name,
                    error=str(e),
                    tool_elapsed_ms=tool_cost_ms,
                    elapsed_ms=elapsed_ms(started_at),
                    exc_info=True,
                )
                tool_result = ToolResult(success=False, message=f"调用工具失败: {function_name}")

        if bool(tool_result.success):
            consecutive_failure_count = 0
        else:
            consecutive_failure_count += 1
            if (
                    normalized_function_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES
                    and loop_break_reason != "browser_high_level_retry_blocked"
            ):
                failed_browser_high_level_keys.add(
                    _build_browser_high_level_failure_key(
                        function_name=normalized_function_name,
                        function_args=function_args,
                        browser_route_state_key=browser_route_state_key,
                    )
                )
                degrade_reason = f"{normalized_function_name}_failed"
                tool_result = _attach_browser_degrade_payload(
                    tool_result,
                    function_name=function_name,
                    degrade_reason=degrade_reason,
                    browser_page_type=browser_page_type,
                    browser_url=last_browser_route_url,
                    browser_title=last_browser_route_title,
                )
                log_runtime(
                    logger,
                    logging.INFO,
                    "浏览器高阶能力失败，允许降级为其他浏览器能力",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    degrade_reason=degrade_reason,
                )

        if research_route_enabled and normalized_function_name == "search_web" and bool(tool_result.success):
            research_candidate_urls = _extract_search_result_urls(tool_result)
            research_search_ready = len(research_candidate_urls) > 0
        elif research_route_enabled and normalized_function_name == "fetch_page" and bool(tool_result.success):
            research_fetch_completed = True

        browser_tool_state = _extract_browser_tool_state(tool_result)
        if browser_tool_state["page_type"]:
            browser_page_type = browser_tool_state["page_type"]
        if browser_tool_state["url"]:
            last_browser_route_url = browser_tool_state["url"]
        if browser_tool_state["title"]:
            last_browser_route_title = browser_tool_state["title"]
        if browser_tool_state["selector"]:
            last_browser_route_selector = str(browser_tool_state["selector"])
        if browser_tool_state["index"] is not None:
            last_browser_route_index = int(browser_tool_state["index"])

        if bool(tool_result.success):
            if normalized_function_name == "browser_read_current_page_structured":
                browser_structured_ready = True
            elif normalized_function_name == "browser_extract_main_content":
                browser_main_content_ready = True
            elif normalized_function_name == "browser_extract_cards":
                browser_cards_ready = True
            elif normalized_function_name == "browser_find_link_by_text":
                browser_link_match_ready = True
            elif normalized_function_name == "browser_find_actionable_elements":
                browser_actionables_ready = True

        called_event = ToolEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            function_name=function_name,
            function_args=function_args,
            function_result=tool_result,
            status=ToolEventStatus.CALLED,
        )
        await _notify_tool_event(called_event)
        if normalized_function_name in BROWSER_PROGRESS_FUNCTIONS and bool(tool_result.success):
            browser_observation_fingerprint = _build_browser_observation_fingerprint(tool_result)
            if browser_observation_fingerprint == last_browser_observation_fingerprint:
                browser_no_progress_count += 1
            else:
                browser_no_progress_count = 0
                last_browser_observation_fingerprint = browser_observation_fingerprint
            if browser_no_progress_count >= BROWSER_NO_PROGRESS_LIMIT:
                loop_break_reason = "browser_no_progress"

        interrupt_request = (
            tool_result.data.get("interrupt")
            if isinstance(tool_result.data, dict)
            else None
        )
        log_runtime(
            logger,
            logging.INFO,
            "工具调用完成",
            step_id=str(step.id or ""),
            tool_call_id=tool_call_id,
            function_name=function_name,
            tool_name=tool_name,
            success=bool(tool_result.success),
            has_interrupt=bool(isinstance(interrupt_request, dict) and interrupt_request),
            loop_break_reason=loop_break_reason or "",
            browser_no_progress_count=browser_no_progress_count,
            consecutive_failure_count=consecutive_failure_count,
            llm_elapsed_ms=llm_cost_ms,
            tool_elapsed_ms=tool_cost_ms if matched_tool is not None else 0,
            elapsed_ms=elapsed_ms(started_at),
        )
        if isinstance(interrupt_request, dict) and interrupt_request:
            log_runtime(
                logger,
                logging.INFO,
                "工具请求进入等待",
                step_id=str(step.id or ""),
                tool_call_id=tool_call_id,
                function_name=function_name,
                interrupt_kind=str(interrupt_request.get("kind") or ""),
                elapsed_ms=elapsed_ms(started_at),
            )
            return {
                "success": True,
                "interrupt_request": interrupt_request,
                "result": "",
                "attachments": [],
            }, emitted_tool_events
        if function_name == NOTIFY_USER_FUNCTION_NAME and bool(tool_result.success):
            notify_user_sent = True
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "function_name": function_name,
                "content": _build_tool_feedback_content(function_name, tool_result),
            }
        )
        if loop_break_reason == "repeat_tool_call":
            return _build_loop_break_payload(
                step=step,
                loop_break_reason=loop_break_reason,
                blocker="同一工具及参数被重复调用过多次，当前步骤已被强制收敛。",
                next_hint="请改用其他工具、调整参数，或将当前步骤拆小后再执行。",
            ), emitted_tool_events
        if loop_break_reason == "search_repeat":
            return _build_loop_break_payload(
                step=step,
                loop_break_reason=loop_break_reason,
                blocker="同一搜索查询已重复触发多次，当前检索路径没有继续收获。",
                next_hint="请改写搜索关键词、缩小范围，或改用 fetch_page / 文件读取继续。",
            ), emitted_tool_events
        if loop_break_reason == "browser_no_progress":
            return _build_loop_break_payload(
                step=step,
                loop_break_reason=loop_break_reason,
                blocker="浏览器连续观察未发现新的有效信息，当前页面路径已无进展。",
                next_hint="请更换页面、改用搜索/正文读取，或重新规划当前步骤。",
            ), emitted_tool_events
        if consecutive_failure_count >= TOOL_FAILURE_LIMIT:
            return _build_loop_break_payload(
                step=step,
                loop_break_reason="tool_failure_limit",
                blocker="连续工具调用失败次数过多，当前步骤已停止继续重试。",
                next_hint="请检查参数、改换工具，或将当前步骤拆小后再执行。",
            ), emitted_tool_events

    parsed = safe_parse_json(llm_message.get("content"))
    log_runtime(
        logger,
        logging.INFO,
        "达到最大工具轮次，返回当前结果",
        step_id=str(step.id or ""),
        iteration_count=max(1, int(max_tool_iterations)),
        task_mode=task_mode,
        loop_break_reason="max_tool_iterations",
        success=bool(parsed.get("success", True)),
        attachment_count=len(normalize_attachments(parsed.get("attachments"))),
        elapsed_ms=elapsed_ms(started_at),
    )
    return {
        "success": bool(parsed.get("success", True)),
        "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
        "attachments": normalize_attachments(parsed.get("attachments")),
    }, emitted_tool_events
