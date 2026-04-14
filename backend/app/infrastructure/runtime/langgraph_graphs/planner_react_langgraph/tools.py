#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 工具调用循环与仲裁逻辑。"""

import inspect
import json
import logging
import uuid
from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.domain.external import LLM
from app.domain.models import (
    BrowserPageType,
    SearchResults,
    Step,
    StepArtifactPolicy,
    StepDeliveryContextState,
    StepDeliveryRole,
    StepOutputMode,
    StepTaskModeHint,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
)
from app.domain.services.prompts import SYSTEM_PROMPT, REACT_SYSTEM_PROMPT
from app.domain.services.runtime.normalizers import (
    normalize_controlled_value,
    normalize_execution_response,
    normalize_file_path_list,
)
from app.domain.services.workspace_runtime.policies import (
    build_human_wait_missing_interrupt_payload as _build_human_wait_missing_interrupt_payload,
    build_loop_break_payload as _build_loop_break_payload,
    build_recent_blocked_tool_call as _build_recent_blocked_tool_call,
    build_recent_failed_action as _build_recent_failed_action,
    build_search_fingerprint as _build_search_fingerprint,
    build_tool_feedback_content as _build_tool_feedback_content,
    build_tool_fingerprint as _build_tool_fingerprint,
    analyze_text_intent as _analyze_text_intent,
    attach_browser_degrade_payload as _attach_browser_degrade_payload,
    build_browser_atomic_allowlist as _build_browser_atomic_allowlist,
    build_browser_capability_gap_allowlist as _build_browser_capability_gap_allowlist,
    build_browser_high_level_failure_key as _build_browser_high_level_failure_key,
    build_browser_high_level_retry_block_message as _build_browser_high_level_retry_block_message,
    build_browser_observation_fingerprint as _build_browser_observation_fingerprint,
    build_browser_preferred_function_names as _build_browser_preferred_function_names,
    build_browser_route_block_message as _build_browser_route_block_message,
    build_browser_route_state_key as _build_browser_route_state_key,
    build_listing_click_target_block_message as _build_listing_click_target_block_message,
    build_step_candidate_text as _build_step_candidate_text,
    build_task_mode_disallowed_names as _build_task_mode_disallowed_names,
    classify_confirmed_user_task_mode,
    classify_step_task_mode,
    coerce_optional_int as _coerce_optional_int,
    collect_temporarily_blocked_browser_high_level_function_names as _collect_temporarily_blocked_browser_high_level_function_names,
    extract_browser_tool_state as _extract_browser_tool_state,
    infer_entry_strategy,
    is_browser_high_level_temporarily_blocked as _is_browser_high_level_temporarily_blocked,
    normalize_execution_payload as _normalize_execution_payload,
    requests_plan_only,
)
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph_graphs.graph_parsers import (
    normalize_attachments,
    safe_parse_json,
)
from .runtime_logging import elapsed_ms, log_runtime, now_perf
from .settings import (
    ASK_USER_FUNCTION_NAME,
    BROWSER_ATOMIC_FUNCTION_NAMES,
    BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
    BROWSER_NO_PROGRESS_LIMIT,
    FETCH_REPEAT_LIMIT,
    BROWSER_PROGRESS_FUNCTIONS,
    FILE_FUNCTION_NAMES,
    NOTIFY_USER_FUNCTION_NAME,
    READ_ONLY_FILE_FUNCTION_NAMES,
    REPEAT_TOOL_LIMIT,
    SEARCH_FUNCTION_NAMES,
    SEARCH_REPEAT_LIMIT,
    TASK_MODE_MAX_TOOL_ITERATIONS,
    TOOL_FAILURE_LIMIT,
)

logger = logging.getLogger(__name__)

def has_available_file_context(
        *,
        user_message: str,
        attachment_paths: Optional[List[str]] = None,
        artifact_paths: Optional[List[str]] = None,
) -> bool:
    """统一判断当前步骤是否已经具备真实可读的文件上下文。"""
    normalized_paths = normalize_file_path_list([*(attachment_paths or []), *(artifact_paths or [])])
    if len(normalized_paths) > 0:
        return True
    signals = _analyze_text_intent(user_message)
    return bool(signals["has_absolute_path"])


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


def _build_assistant_tool_call_message(
        llm_message: Dict[str, Any],
        *,
        selected_tool_call: Dict[str, Any],
) -> Dict[str, Any]:
    """回放 assistant tool-call 消息时保留所有未知字段，避免推理态信息被裁掉。"""
    assistant_message = deepcopy(llm_message if isinstance(llm_message, dict) else {})
    assistant_message["role"] = "assistant"
    assistant_message["tool_calls"] = [deepcopy(selected_tool_call)]
    if "content" not in assistant_message:
        assistant_message["content"] = llm_message.get("content")
    return assistant_message


def _step_allows_user_wait(step: Step, function_args: Dict[str, Any]) -> bool:
    takeover = str(function_args.get("suggest_user_takeover") or "").strip().lower()
    if takeover == "browser":
        return True

    # 等待能力先信 planner/replan 的结构化标记，文本正则只保留为旧计划兼容兜底。
    if normalize_controlled_value(getattr(step, "task_mode_hint", None), StepTaskModeHint) == "human_wait":
        return True

    candidate_text = _build_step_candidate_text(step)
    if not candidate_text.strip():
        return False
    return bool(_analyze_text_intent(candidate_text)["needs_human_wait"])


def _step_forbids_file_output(step: Step) -> bool:
    """结构化产物策略优先决定是否禁止当前步骤产出文件。"""
    return normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy) == "forbid_file_output"


def _step_outputs_inline_result(step: Step) -> bool:
    """读取步骤的结构化输出模式，避免展示类步骤继续绕回文件系统。"""
    return normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode) == "inline"


def _step_owns_final_delivery(step: Step) -> bool:
    """显式 final 交付步骤负责最终重正文，不应再漂移回检索链路。"""
    return normalize_controlled_value(getattr(step, "delivery_role", None), StepDeliveryRole) == "final"


def _resolve_step_delivery_context_state(step: Step, *, task_mode: str) -> str:
    """统一解析最终交付上下文状态；缺失时按当前步骤语义做结构化推断。"""
    delivery_context_state = normalize_controlled_value(
        getattr(step, "delivery_context_state", None),
        StepDeliveryContextState,
    )
    if delivery_context_state:
        return delivery_context_state
    if not _step_owns_final_delivery(step):
        return StepDeliveryContextState.NONE.value
    inferred_task_mode = (
        normalize_controlled_value(getattr(step, "task_mode_hint", None), StepTaskModeHint)
        or normalize_controlled_value(task_mode, StepTaskModeHint)
    )
    if inferred_task_mode == StepTaskModeHint.GENERAL.value:
        return StepDeliveryContextState.READY.value
    return StepDeliveryContextState.NEEDS_PREPARATION.value


def _step_final_delivery_context_ready(step: Step, *, task_mode: str) -> bool:
    """只有上下文已准备好的 final 步骤，才应该被禁止继续检索页面。"""
    return _resolve_step_delivery_context_state(step, task_mode=task_mode) == StepDeliveryContextState.READY.value


def _step_is_final_inline_delivery_ready(step: Step, *, task_mode: str) -> bool:
    # P3-1A 收敛修复：统一识别“最终交付正文步骤”，用于集中禁用漂移型工具。
    return (
        _step_outputs_inline_result(step)
        and _step_owns_final_delivery(step)
        and _step_final_delivery_context_ready(step, task_mode=task_mode)
    )


def _resolve_effective_max_tool_iterations(*, task_mode: str, requested_max_tool_iterations: int) -> int:
    # P3-2A 收敛修复：按任务模式收敛工具轮次，防止单步在默认 20 轮上限下长时间空转。
    mode_cap = TASK_MODE_MAX_TOOL_ITERATIONS.get(str(task_mode or "").strip().lower())
    if mode_cap is None:
        return requested_max_tool_iterations
    return min(requested_max_tool_iterations, max(1, int(mode_cap)))


def _build_fetch_fingerprint(function_args: Dict[str, Any]) -> str:
    # P3-2A 收敛修复：fetch_page 使用 URL 指纹去重，避免反复读取同一链接。
    return str(function_args.get("url") or "").strip().lower()


def _is_transient_research_transport_error(raw_error: Any) -> bool:
    # P3-3A 收敛修复：识别检索链路瞬时传输错误，避免在当前步骤内盲目重试。
    message = str(raw_error or "").strip().lower()
    if not message:
        return False
    transient_markers = (
        "remoteprotocolerror",
        "server disconnected",
        "connection reset",
        "connection aborted",
        "readtimeout",
        "connecttimeout",
        "connection error",
        "temporarily unavailable",
        "unexpected eof",
    )
    return any(marker in message for marker in transient_markers)


def _filter_available_tools(
        available_tools: List[Dict[str, Any]],
        *,
        disallowed_function_names: Optional[set[str]] = None,
        allow_ask_user: bool,
) -> List[Dict[str, Any]]:
    filtered_tools: List[Dict[str, Any]] = []
    blocked_names = set(disallowed_function_names or set())
    for tool_schema in available_tools:
        function_name = _extract_function_name(tool_schema)
        if function_name in blocked_names:
            continue
        if function_name == ASK_USER_FUNCTION_NAME and not allow_ask_user:
            continue
        filtered_tools.append(tool_schema)
    return filtered_tools


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


def _step_explicitly_requests_shell_execution(step: Step, user_content: Optional[List[Dict[str, Any]]]) -> bool:
    # P3-CASE3 修复：file_processing 默认禁用 shell_execute，只有显式命令意图才放开。
    candidate_text = "\n".join(
        [
            _build_step_candidate_text(step),
            _extract_text_from_user_content(user_content),
        ]
    )
    signals = _analyze_text_intent(candidate_text)
    if bool(signals.get("has_shell_command")):
        return True
    normalized_text = str(signals.get("text") or "").strip().lower()
    explicit_markers = (
        "shell_execute",
        "执行命令",
        "运行命令",
        "终端命令",
        "命令行执行",
        "run command",
        "execute command",
        "terminal command",
    )
    return any(marker in normalized_text for marker in explicit_markers)


def _step_or_user_content_has_url(step: Step, user_content: Optional[List[Dict[str, Any]]]) -> bool:
    candidate_text = "\n".join(
        [
            _build_step_candidate_text(step),
            _extract_text_from_user_content(user_content),
        ]
    )
    return bool(_analyze_text_intent(candidate_text)["has_url"])


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

async def execute_step_with_prompt(
        *,
        llm: LLM,
        step: Step,
        runtime_tools: Optional[List[BaseTool]],
        max_tool_iterations: int = 5,
        task_mode: str = "general",
        on_tool_event: Optional[Callable[[ToolEvent], Optional[Awaitable[None]]]] = None,
        user_content: Optional[List[Dict[str, Any]]] = None,
        has_available_file_context: bool = False,
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
        list(available_function_names),
        task_mode=task_mode,
    )
    research_file_context_blocked_function_names: set[str] = set()
    general_inline_blocked_function_names: set[str] = set()
    file_processing_shell_blocked_function_names: set[str] = set()
    artifact_policy_blocked_function_names: set[str] = set()
    final_delivery_search_blocked_function_names: set[str] = set()
    final_delivery_shell_blocked_function_names: set[str] = set()
    if task_mode == "research" and not has_available_file_context:
        research_file_context_blocked_function_names.update(READ_ONLY_FILE_FUNCTION_NAMES)
        blocked_function_names.update(research_file_context_blocked_function_names)
    if task_mode == "general" and _step_outputs_inline_result(step) and not has_available_file_context:
        # 展示型步骤应当直接把已有观察整理成文本，不要在无文件上下文时再试探读写文件。
        general_inline_blocked_function_names.update(FILE_FUNCTION_NAMES)
        blocked_function_names.update(general_inline_blocked_function_names)
    if task_mode == "file_processing" and not _step_explicitly_requests_shell_execution(step, normalized_user_content):
        # P3-CASE3 修复：文件处理默认只走文件工具，显式命令意图才允许 shell_execute。
        file_processing_shell_blocked_function_names.add("shell_execute")
        blocked_function_names.update(file_processing_shell_blocked_function_names)
    if _step_is_final_inline_delivery_ready(step, task_mode=task_mode):
        # 最终交付步骤只负责组织已准备好的上下文，不再重新发起搜索/页面读取。
        final_delivery_search_blocked_function_names.update(SEARCH_FUNCTION_NAMES)
        blocked_function_names.update(final_delivery_search_blocked_function_names)
        # P3-1A 收敛修复：最终交付正文阶段禁止 shell_execute，避免进入无意义命令循环。
        final_delivery_shell_blocked_function_names.add("shell_execute")
        blocked_function_names.update(final_delivery_shell_blocked_function_names)
    if _step_forbids_file_output(step):
        artifact_policy_blocked_function_names.add("write_file")
        blocked_function_names.update(artifact_policy_blocked_function_names)
    if task_mode in {"web_reading", "browser_interaction"} and not browser_route_enabled:
        for function_name in _build_browser_capability_gap_allowlist(task_mode=task_mode):
            blocked_function_names.discard(function_name)
    blocked_function_names.difference_update(
        {
            function_name
            for function_name in research_file_context_blocked_function_names
            if function_name not in available_function_names
        }
    )
    blocked_function_names.difference_update(
        {
            function_name
            for function_name in general_inline_blocked_function_names
            if function_name not in available_function_names
        }
    )
    blocked_function_names.difference_update(
        {
            function_name
            for function_name in file_processing_shell_blocked_function_names
            if function_name not in available_function_names
        }
    )
    blocked_function_names.difference_update(
        {
            function_name
            for function_name in artifact_policy_blocked_function_names
            if function_name not in available_function_names
        }
    )
    blocked_function_names.difference_update(
        {
            function_name
            for function_name in final_delivery_search_blocked_function_names
            if function_name not in available_function_names
        }
    )
    blocked_function_names.difference_update(
        {
            function_name
            for function_name in final_delivery_shell_blocked_function_names
            if function_name not in available_function_names
        }
    )
    # P3-2A 收敛修复：先记录外部请求上限，再计算 task_mode 分级后的真实轮次上限。
    requested_max_tool_iterations = max(1, int(max_tool_iterations))
    effective_max_tool_iterations = _resolve_effective_max_tool_iterations(
        task_mode=task_mode,
        requested_max_tool_iterations=requested_max_tool_iterations,
    )
    if task_mode == "human_wait" and ASK_USER_FUNCTION_NAME not in available_function_names:
        log_runtime(
            logger,
            logging.WARNING,
            "等待步骤缺少 ask_user 工具",
            step_id=str(step.id or ""),
        )
        return _build_loop_break_payload(
            step=step,
            blocker="当前等待步骤缺少可用的 message_ask_user 工具，无法向用户发起确认或选择。",
            next_hint="请先为当前运行时注入 message_ask_user 工具，再重新执行该等待步骤。",
        ), emitted_tool_events
    log_runtime(
        logger,
        logging.INFO,
        "工具执行循环准备完成",
        step_id=str(step.id or ""),
        step_title=str(step.title or step.description or ""),
        task_mode=task_mode,
        available_tool_count=len(available_tools),
        blocked_tool_count=len(blocked_function_names),
        requested_max_tool_iterations=requested_max_tool_iterations,
        max_tool_iterations=effective_max_tool_iterations,
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
        return _normalize_execution_payload(
            parsed,
            default_summary=f"已完成步骤：{step.description}",
        ), []

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + REACT_SYSTEM_PROMPT},
        {"role": "user", "content": normalized_user_content}
    ]

    llm_message: Dict[str, Any] = {}
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
    fetch_repeat_counter: Dict[str, int] = {}
    last_browser_observation_fingerprint = ""
    browser_no_progress_count = 0
    failed_browser_high_level_keys: set[str] = set()
    consecutive_failure_count = 0
    runtime_recent_action: Dict[str, Any] = {}

    for index in range(effective_max_tool_iterations):
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
            parsed_execution = normalize_execution_response(parsed)
            if task_mode == "human_wait":
                log_runtime(
                    logger,
                    logging.WARNING,
                    "等待步骤未发起 interrupt，已拒绝直接完成",
                    step_id=str(step.id or ""),
                    iteration=index,
                    llm_elapsed_ms=llm_cost_ms,
                    elapsed_ms=elapsed_ms(started_at),
                )
                return _build_human_wait_missing_interrupt_payload(
                    step,
                    runtime_recent_action=runtime_recent_action,
                ), emitted_tool_events
            has_summary = bool(str(parsed_execution.get("summary") or "").strip())
            has_delivery_text = bool(str(parsed_execution.get("delivery_text") or "").strip())
            if parsed_execution.get("success", True) and not has_summary and not has_delivery_text:
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
            return _normalize_execution_payload(
                {
                    **parsed,
                    "runtime_recent_action": runtime_recent_action,
                },
                default_summary=f"已完成步骤：{step.description}",
            ), emitted_tool_events

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
            _build_assistant_tool_call_message(
                llm_message,
                selected_tool_call=selected_tool_call,
            )
        )

        function = selected_tool_call.get("function")
        if not isinstance(function, dict):
            continue

        function_name = str(function.get("name") or "").strip()
        if not function_name:
            continue
        normalized_function_name = function_name.lower()
        tool_call_id = str(selected_tool_call.get("id") or selected_tool_call.get("call_id") or uuid.uuid4())
        tool_call_ref = str(selected_tool_call.get("call_id") or tool_call_id)
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
            if task_mode == "human_wait" and function_name != ASK_USER_FUNCTION_NAME:
                loop_break_reason = "human_wait_non_interrupt_tool_blocked"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "等待步骤已拦截非等待工具",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                )
                tool_result = ToolResult(
                    success=False,
                    message="当前步骤是等待用户确认/选择的步骤，只允许调用 message_ask_user 发起等待。",
                )
            elif normalized_function_name in research_file_context_blocked_function_names:
                loop_break_reason = "research_file_context_required"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "缺少明确文件上下文，已拦截文件工具调用",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    task_mode=task_mode,
                )
                tool_result = ToolResult(
                    success=False,
                    message="当前步骤属于检索任务，只有在用户消息或附件中出现明确文件路径/文件名时，才能调用文件工具。",
                )
            elif task_mode == "web_reading" and normalized_function_name in READ_ONLY_FILE_FUNCTION_NAMES:
                loop_break_reason = "web_reading_file_tool_blocked"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "网页读取步骤已拦截文件工具调用",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    task_mode=task_mode,
                )
                tool_result = ToolResult(
                    success=False,
                    message="当前步骤属于网页读取任务，请优先使用 search_web、fetch_page 或浏览器高阶读取工具，不要回退到文件工具。",
                )
            elif normalized_function_name in general_inline_blocked_function_names:
                loop_break_reason = "general_inline_file_context_required"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "内联展示步骤缺少文件上下文，已拦截文件工具调用",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    task_mode=task_mode,
                    output_mode=str(getattr(step, "output_mode", "") or ""),
                )
                tool_result = ToolResult(
                    success=False,
                    message="当前步骤是直接内联展示结果的步骤，且没有可用文件上下文，请直接返回文本结果，不要继续读写文件。",
                )
            elif normalized_function_name in file_processing_shell_blocked_function_names:
                loop_break_reason = "file_processing_shell_explicit_required"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "文件处理步骤缺少显式命令意图，已拦截 shell 工具调用",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    task_mode=task_mode,
                )
                tool_result = ToolResult(
                    success=False,
                    message="当前步骤属于文件处理，默认禁止调用 shell_execute。仅在用户明确要求执行命令时才允许。",
                )
            elif normalized_function_name in artifact_policy_blocked_function_names:
                loop_break_reason = "artifact_policy_file_output_blocked"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "步骤产物策略拦截文件产出工具调用",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    task_mode=task_mode,
                    artifact_policy=str(getattr(step, "artifact_policy", "") or ""),
                    output_mode=str(getattr(step, "output_mode", "") or ""),
                )
                tool_result = ToolResult(
                    success=False,
                    message="当前步骤的结构化产物策略禁止文件产出。请直接返回文本结果，或先通过重规划生成允许文件产出的步骤。",
                )
            elif normalized_function_name in final_delivery_search_blocked_function_names:
                loop_break_reason = "final_delivery_search_drift_blocked"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "最终交付步骤已拦截检索漂移",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    task_mode=task_mode,
                    delivery_role=str(getattr(step, "delivery_role", "") or ""),
                )
                tool_result = ToolResult(
                    success=False,
                    message="当前步骤负责最终交付正文，请直接基于已知上下文组织答案，不要重新调用 search_web 或 fetch_page。",
                )
            elif normalized_function_name in final_delivery_shell_blocked_function_names:
                # P3-1A 收敛修复：最终交付步骤禁止 shell_execute，防止总结阶段漂移成命令执行。
                loop_break_reason = "final_delivery_shell_drift_blocked"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "最终交付步骤已拦截 shell 漂移",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    task_mode=task_mode,
                    delivery_role=str(getattr(step, "delivery_role", "") or ""),
                )
                tool_result = ToolResult(
                    success=False,
                    message="当前步骤负责最终交付正文，请直接输出最终答案，不要调用 shell_execute。",
                )
            elif browser_route_enabled and normalized_function_name.startswith("browser_"):
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
                    # P3-3A 收敛修复：search_web 返回瞬时链路错误时立即降级，阻止同一步骤内继续空转重试。
                    if not bool(tool_result.success) and _is_transient_research_transport_error(tool_result.message):
                        loop_break_reason = "research_route_transport_error"
                        tool_result = ToolResult(
                            success=False,
                            message="检索链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或切换其他路径。",
                        )
                except Exception as e:
                    tool_cost_ms = elapsed_ms(tool_started_at)
                    # P3-3A 收敛修复：RemoteProtocolError 等瞬时异常直接收敛，不再继续长循环重试。
                    if _is_transient_research_transport_error(f"{e.__class__.__name__}: {e}"):
                        loop_break_reason = "research_route_transport_error"
                        log_runtime(
                            logger,
                            logging.WARNING,
                            "检索链路瞬时错误，已触发快速收敛",
                            step_id=str(step.id or ""),
                            function_name=function_name,
                            tool_name=tool_name,
                            error=f"{e.__class__.__name__}: {e}",
                            tool_elapsed_ms=tool_cost_ms,
                            elapsed_ms=elapsed_ms(started_at),
                        )
                        tool_result = ToolResult(
                            success=False,
                            message="检索链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或切换其他路径。",
                        )
                    else:
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
        elif normalized_function_name == "fetch_page":
            # P3-2A 收敛修复：fetch_page 也按 URL 指纹收敛，防止同一页面被反复读取。
            fetch_fingerprint = _build_fetch_fingerprint(function_args)
            if fetch_fingerprint:
                fetch_repeat_counter[fetch_fingerprint] = fetch_repeat_counter.get(fetch_fingerprint, 0) + 1
            if fetch_fingerprint and fetch_repeat_counter[fetch_fingerprint] > FETCH_REPEAT_LIMIT:
                loop_break_reason = "research_route_fingerprint_repeat"
                log_runtime(
                    logger,
                    logging.WARNING,
                    "重复抓取同一页面已收敛",
                    step_id=str(step.id or ""),
                    function_name=function_name,
                    fetch_repeat_count=fetch_repeat_counter[fetch_fingerprint],
                    fetch_url=fetch_fingerprint,
                )
                tool_result = ToolResult(
                    success=False,
                    message="同一页面 URL 已重复抓取多次，请切换其他候选链接或结束当前步骤。",
                )
            else:
                tool_started_at = now_perf()
                try:
                    tool_result = await matched_tool.invoke(function_name, **function_args)
                    tool_cost_ms = elapsed_ms(tool_started_at)
                    if not isinstance(tool_result, ToolResult):
                        tool_result = ToolResult(success=True, data=tool_result)
                    # P3-3A 收敛修复：fetch_page 返回瞬时链路错误时立即降级，避免同一步骤内长循环。
                    if not bool(tool_result.success) and _is_transient_research_transport_error(tool_result.message):
                        loop_break_reason = "research_route_transport_error"
                        tool_result = ToolResult(
                            success=False,
                            message="页面抓取链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或改用其他来源。",
                        )
                except Exception as e:
                    tool_cost_ms = elapsed_ms(tool_started_at)
                    # P3-3A 收敛修复：RemoteProtocolError 等瞬时异常快速收敛，不再进入长重试。
                    if _is_transient_research_transport_error(f"{e.__class__.__name__}: {e}"):
                        loop_break_reason = "research_route_transport_error"
                        log_runtime(
                            logger,
                            logging.WARNING,
                            "页面抓取链路瞬时错误，已触发快速收敛",
                            step_id=str(step.id or ""),
                            function_name=function_name,
                            tool_name=tool_name,
                            error=f"{e.__class__.__name__}: {e}",
                            tool_elapsed_ms=tool_cost_ms,
                            elapsed_ms=elapsed_ms(started_at),
                        )
                        tool_result = ToolResult(
                            success=False,
                            message="页面抓取链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或改用其他来源。",
                        )
                    else:
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
            runtime_recent_action["last_failed_action"] = _build_recent_failed_action(
                function_name=function_name,
                tool_result=tool_result,
            )
            blocked_tool_call = _build_recent_blocked_tool_call(
                function_name=function_name,
                tool_result=tool_result,
                loop_break_reason=loop_break_reason,
            )
            if blocked_tool_call:
                runtime_recent_action["last_blocked_tool_call"] = blocked_tool_call
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
                "summary": "",
                "result": "",
                "delivery_text": "",
                "attachments": [],
                "runtime_recent_action": runtime_recent_action,
            }, emitted_tool_events
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "call_id": tool_call_ref,
                "function_name": function_name,
                "content": _build_tool_feedback_content(function_name, tool_result),
            }
        )
        if loop_break_reason == "repeat_tool_call":
            return _build_loop_break_payload(
                step=step,
                blocker="同一工具及参数被重复调用过多次，当前步骤已被强制收敛。",
                next_hint="请改用其他工具、调整参数，或将当前步骤拆小后再执行。",
                runtime_recent_action=runtime_recent_action,
            ), emitted_tool_events
        if loop_break_reason == "search_repeat":
            return _build_loop_break_payload(
                step=step,
                blocker="同一搜索查询已重复触发多次，当前检索路径没有继续收获。",
                next_hint="请改写搜索关键词、缩小范围，或改用 fetch_page / 文件读取继续。",
                runtime_recent_action=runtime_recent_action,
            ), emitted_tool_events
        if loop_break_reason == "research_route_fingerprint_repeat":
            # P3-2A 收敛修复：研究链路（fetch_page）命中重复指纹后直接结束，避免继续空转。
            return _build_loop_break_payload(
                step=step,
                blocker="同一页面抓取请求已重复触发多次，当前检索路径没有新增信息。",
                next_hint="请切换其他候选 URL、改用其他工具，或结束当前步骤。",
                runtime_recent_action=runtime_recent_action,
            ), emitted_tool_events
        if loop_break_reason == "research_route_transport_error":
            # P3-3A 收敛修复：检索链路瞬时网络错误直接收敛，避免同一步骤内反复失败。
            return _build_loop_break_payload(
                step=step,
                blocker="检索/抓取链路出现瞬时网络错误，当前步骤已停止重试。",
                next_hint="请稍后重试，或先基于已有信息继续后续步骤。",
                runtime_recent_action=runtime_recent_action,
            ), emitted_tool_events
        if loop_break_reason == "browser_no_progress":
            return _build_loop_break_payload(
                step=step,
                blocker="浏览器连续观察未发现新的有效信息，当前页面路径已无进展。",
                next_hint="请更换页面、改用搜索/正文读取，或重新规划当前步骤。",
                runtime_recent_action=runtime_recent_action,
            ), emitted_tool_events
        if consecutive_failure_count >= TOOL_FAILURE_LIMIT:
            return _build_loop_break_payload(
                step=step,
                blocker="连续工具调用失败次数过多，当前步骤已停止继续重试。",
                next_hint="请检查参数、改换工具，或将当前步骤拆小后再执行。",
                runtime_recent_action=runtime_recent_action,
            ), emitted_tool_events

    parsed = safe_parse_json(llm_message.get("content"))
    if task_mode == "human_wait":
        log_runtime(
            logger,
            logging.WARNING,
            "等待步骤达到最大轮次仍未进入等待",
            step_id=str(step.id or ""),
            requested_max_tool_iterations=requested_max_tool_iterations,
            iteration_count=effective_max_tool_iterations,
            elapsed_ms=elapsed_ms(started_at),
        )
        return _build_human_wait_missing_interrupt_payload(
            step,
            runtime_recent_action=runtime_recent_action,
        ), emitted_tool_events
    log_runtime(
        logger,
        logging.INFO,
        "达到最大工具轮次，返回当前结果",
        step_id=str(step.id or ""),
        requested_max_tool_iterations=requested_max_tool_iterations,
        iteration_count=effective_max_tool_iterations,
        task_mode=task_mode,
        loop_break_reason="max_tool_iterations",
        success=bool(parsed.get("success", True)),
        attachment_count=len(normalize_attachments(parsed.get("attachments"))),
        elapsed_ms=elapsed_ms(started_at),
    )
    return _normalize_execution_payload(
        {
            **parsed,
            "runtime_recent_action": runtime_recent_action,
        },
        default_summary=f"已完成步骤：{step.description}",
    ), emitted_tool_events
