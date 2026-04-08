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

from app.domain.external import LLM
from app.domain.models import Step, ToolEvent, ToolEventStatus, ToolResult
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
        if any(name in signals["text"] for name in ("browser_view", "browser_navigate", "browser_restart")):
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


def _tool_call_priority(function_name: str) -> int:
    normalized_name = function_name.strip().lower()
    if "search" in normalized_name:
        return 0
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
        priority = _tool_call_priority(function_name)
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
    blocked_function_names = _build_task_mode_disallowed_names(
        available_tools,
        task_mode=task_mode,
    )
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
    last_tool_fingerprint = ""
    same_tool_repeat_count = 0
    search_repeat_counter: Dict[str, int] = {}
    last_browser_observation_fingerprint = ""
    browser_no_progress_count = 0
    consecutive_failure_count = 0

    for index in range(max(1, int(max_tool_iterations))):
        iteration_tools = _filter_available_tools(
            available_tools,
            disallowed_function_names=blocked_function_names,
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
        elif normalized_function_name in blocked_function_names:
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
