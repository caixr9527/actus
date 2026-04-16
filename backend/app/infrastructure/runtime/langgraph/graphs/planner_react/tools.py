#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 工具调用循环与仲裁逻辑。"""

import json
import logging
from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.domain.external import LLM
from app.domain.models import (
    Step,
    ToolEvent,
)
from app.domain.services.prompts import SYSTEM_PROMPT, REACT_SYSTEM_PROMPT
from app.domain.services.runtime.normalizers import (
    normalize_file_path_list,
    normalize_execution_response,
)
from app.domain.services.workspace_runtime.policies import (
    build_loop_break_payload as _build_loop_break_payload,
    build_tool_feedback_content as _build_tool_feedback_content,
    build_tool_fingerprint as _build_tool_fingerprint,
    analyze_text_intent as _analyze_text_intent,
    build_step_candidate_text as _build_step_candidate_text,
    build_browser_preferred_function_names as _build_browser_preferred_function_names,
    classify_confirmed_user_task_mode,
    classify_step_task_mode,
    infer_entry_strategy,
    normalize_execution_payload as _normalize_execution_payload,
    requests_plan_only,
)
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    normalize_attachments,
    safe_parse_json,
)
from .execution_context import build_execution_context
from .execution_state import ExecutionState
from .convergence.judge import ConvergenceJudge
from .iteration_context import build_iteration_context
from .policy_engine.engine import ToolPolicyEngine
from .tool_schema import extract_function_name
from app.domain.services.runtime.contracts.runtime_logging import elapsed_ms, log_runtime, now_perf
from app.domain.services.runtime.contracts.langgraph_settings import (
    ASK_USER_FUNCTION_NAME,
    BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
    NOTIFY_USER_FUNCTION_NAME,
)
from .tool_effects import build_interrupt_payload, extract_interrupt_request
from .tool_events import (
    ToolEventDispatcher,
    bind_tool_name,
    build_called_event,
    build_calling_event,
    build_tool_call_lifecycle,
    build_tool_feedback_message,
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


def _build_read_only_intent_text(
        *,
        step: Step,
        user_message: str,
        attachment_paths: Optional[List[str]],
        artifact_paths: Optional[List[str]],
) -> str:
    """只读治理只看业务语义输入，禁止把执行提示词正文混入判定。"""
    candidate_parts: List[str] = []
    normalized_user_message = str(user_message or "").strip()
    if normalized_user_message:
        candidate_parts.append(normalized_user_message)
    step_candidate_text = _build_step_candidate_text(step)
    if step_candidate_text:
        candidate_parts.append(step_candidate_text)
    file_context_paths = normalize_file_path_list([*(attachment_paths or []), *(artifact_paths or [])])
    if file_context_paths:
        candidate_parts.append("已知文件上下文: " + " ".join(file_context_paths[:8]))
    return "\n".join(candidate_parts).strip()


def _build_file_output_intent_text(
        *,
        step: Step,
        user_message: str,
) -> str:
    """文件产出判定同样只看业务语义输入，不读取系统提示词内容。"""
    candidate_parts: List[str] = []
    normalized_user_message = str(user_message or "").strip()
    if normalized_user_message:
        candidate_parts.append(normalized_user_message)
    step_candidate_text = _build_step_candidate_text(step)
    if step_candidate_text:
        candidate_parts.append(step_candidate_text)
    return "\n".join(candidate_parts).strip()


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
        function_name = extract_function_name(tool_schema)
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


async def execute_step_with_prompt(
        *,
        llm: LLM,
        step: Step,
        runtime_tools: Optional[List[BaseTool]],
        max_tool_iterations: int = 5,
        task_mode: str = "general",
        on_tool_event: Optional[Callable[[ToolEvent], Optional[Awaitable[None]]]] = None,
        user_content: Optional[List[Dict[str, Any]]] = None,
        user_message: str = "",
        attachment_paths: Optional[List[str]] = None,
        artifact_paths: Optional[List[str]] = None,
        has_available_file_context: bool = False,
) -> Tuple[Dict[str, Any], List[ToolEvent]]:
    """执行单步任务，支持“模型决策 -> 调工具 -> 回传模型”的最小循环。"""
    started_at = now_perf()
    # P3 重构：统一事件投递入口，tools 主流程不再内联事件分发细节。
    event_dispatcher = ToolEventDispatcher(
        logger=logger,
        on_tool_event=on_tool_event,
    )
    policy_engine = ToolPolicyEngine(logger=logger)
    convergence_judge = ConvergenceJudge()
    step_file_context: Dict[str, Any] = {
        "called_functions": set(),
    }

    available_tools = collect_available_tools(runtime_tools)
    available_function_names: set[str] = set()
    for tool_schema in available_tools:
        function_name = extract_function_name(tool_schema)
        if function_name:
            available_function_names.add(function_name)
    # P3 重构：上下文构建统一沉淀到独立模块，tools.py 仅保留编排职责。
    execution_context = build_execution_context(
        step=step,
        task_mode=task_mode,
        max_tool_iterations=max_tool_iterations,
        user_content=user_content,
        has_available_file_context=has_available_file_context,
        available_tools=available_tools,
        available_function_names=available_function_names,
        read_only_intent_text=_build_read_only_intent_text(
            step=step,
            user_message=user_message,
            attachment_paths=attachment_paths,
            artifact_paths=artifact_paths,
        ),
        file_output_intent_text=_build_file_output_intent_text(
            step=step,
            user_message=user_message,
        ),
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
        ), event_dispatcher.emitted_events
    log_runtime(
        logger,
        logging.INFO,
        "工具执行循环准备完成",
        step_id=str(step.id or ""),
        step_title=str(step.title or step.description or ""),
        task_mode=task_mode,
        available_tool_count=len(available_tools),
        blocked_tool_count=len(execution_context.blocked_function_names),
        requested_max_tool_iterations=execution_context.requested_max_tool_iterations,
        max_tool_iterations=execution_context.effective_max_tool_iterations,
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
                {"role": "user", "content": execution_context.normalized_user_content}
            ],
            tools=[],
            response_format={"type": "json_object"},
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        parsed = safe_parse_json(llm_message.get("content"))
        parsed_execution = normalize_execution_response(parsed)
        has_summary = bool(str(parsed_execution.get("summary") or "").strip())
        has_delivery_text = bool(str(parsed_execution.get("delivery_text") or "").strip())
        has_explicit_success = isinstance(parsed, dict) and "success" in parsed
        inferred_success = bool(parsed.get("success")) if has_explicit_success else bool(has_summary or has_delivery_text)
        log_runtime(
            logger,
            logging.INFO,
            "无工具模式执行完成",
            step_id=str(step.id or ""),
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
            attachment_count=len(normalize_attachments(parsed.get("attachments"))),
        )
        if not inferred_success and not has_summary and not has_delivery_text:
            return _build_loop_break_payload(
                step=step,
                blocker="当前步骤无可用工具，且模型未返回可交付结果。",
                next_hint="请补充更明确的执行指令，或启用对应工具后重试。",
            ), event_dispatcher.emitted_events
        return _normalize_execution_payload(
            {
                **parsed,
                "success": inferred_success,
            },
            default_summary=(
                f"步骤执行完成：{step.description}"
                if inferred_success
                else f"步骤暂未完成：{step.description}"
            ),
        ), event_dispatcher.emitted_events

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + REACT_SYSTEM_PROMPT},
        {"role": "user", "content": execution_context.normalized_user_content}
    ]

    execution_state = ExecutionState()

    for index in range(execution_context.effective_max_tool_iterations):
        # P3 重构：单轮工具白名单/黑名单计算下沉，主循环只保留调度职责。
        iteration_context = build_iteration_context(
            task_mode=task_mode,
            execution_context=execution_context,
            execution_state=execution_state,
        )
        log_runtime(
            logger,
            logging.INFO,
            "开始工具决策轮次",
            step_id=str(step.id or ""),
            iteration=index,
            task_mode=task_mode,
            available_tool_count=len(iteration_context.iteration_tools),
        )
        llm_started_at = now_perf()
        execution_state.llm_message = await llm.invoke(
            messages=messages,
            tools=iteration_context.iteration_tools,
            tool_choice="auto",
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        tool_calls = execution_state.llm_message.get("tool_calls") or []
        if len(tool_calls) == 0:
            no_tool_call_result = policy_engine.finalize_no_tool_call(
                step=step,
                task_mode=task_mode,
                llm_message=execution_state.llm_message,
                llm_cost_ms=llm_cost_ms,
                started_at=started_at,
                iteration=index,
                runtime_recent_action=execution_state.runtime_recent_action,
            )
            if no_tool_call_result.action == "retry":
                continue
            return no_tool_call_result.payload or {}, event_dispatcher.emitted_events

        selected_tool_call = pick_preferred_tool_call(
            tool_calls=[item for item in tool_calls if isinstance(item, dict)],
            available_tools=iteration_context.iteration_tools,
            preferred_function_names=(
                ("fetch_page",)
                if execution_context.research_route_enabled and not execution_state.research_fetch_completed and (
                            execution_state.research_search_ready or execution_context.research_has_explicit_url)
                else _build_browser_preferred_function_names(
                    task_mode=task_mode,
                    available_function_names=execution_context.available_function_names,
                    browser_page_type=execution_state.browser_page_type,
                    browser_structured_ready=execution_state.browser_structured_ready,
                    browser_main_content_ready=execution_state.browser_main_content_ready,
                    browser_cards_ready=execution_state.browser_cards_ready,
                    browser_link_match_ready=execution_state.browser_link_match_ready,
                    browser_actionables_ready=execution_state.browser_actionables_ready,
                    failed_high_level_functions=iteration_context.failed_browser_high_level_functions,
                )
            ),
        )
        if selected_tool_call is None:
            continue

        messages.append(
            _build_assistant_tool_call_message(
                execution_state.llm_message,
                selected_tool_call=selected_tool_call,
            )
        )

        lifecycle = build_tool_call_lifecycle(
            selected_tool_call=selected_tool_call,
            parse_tool_call_args=_parse_tool_call_args,
        )
        if lifecycle is None:
            continue

        tool_fingerprint = _build_tool_fingerprint(lifecycle.normalized_function_name, lifecycle.function_args)
        if tool_fingerprint == execution_state.last_tool_fingerprint:
            execution_state.same_tool_repeat_count += 1
        else:
            execution_state.same_tool_repeat_count = 1
            execution_state.last_tool_fingerprint = tool_fingerprint
        log_runtime(
            logger,
            logging.INFO,
            "已选择工具调用",
            step_id=str(step.id or ""),
            iteration=index,
            tool_call_id=lifecycle.tool_call_id,
            function_name=lifecycle.function_name,
            task_mode=task_mode,
            same_tool_repeat_count=execution_state.same_tool_repeat_count,
            arg_keys=sorted(lifecycle.function_args.keys()),
        )

        matched_tool = _resolve_tool_by_function_name(function_name=lifecycle.function_name, runtime_tools=runtime_tools)
        bind_tool_name(lifecycle, matched_tool)
        await event_dispatcher.emit(build_calling_event(lifecycle))
        policy_result = await policy_engine.evaluate_tool_call(
            step=step,
            task_mode=task_mode,
            function_name=lifecycle.function_name,
            normalized_function_name=lifecycle.normalized_function_name,
            function_args=lifecycle.function_args,
            matched_tool=matched_tool,
            tool_name=lifecycle.tool_name,
            browser_route_state_key=iteration_context.browser_route_state_key,
            iteration_blocked_function_names=iteration_context.iteration_blocked_function_names,
            execution_context=execution_context,
            execution_state=execution_state,
            started_at=started_at,
        )
        tool_result = policy_result.tool_result
        loop_break_reason = policy_result.loop_break_reason
        tool_cost_ms = policy_result.tool_cost_ms

        await event_dispatcher.emit(build_called_event(lifecycle, tool_result))
        interrupt_request = extract_interrupt_request(tool_result)
        log_runtime(
            logger,
            logging.INFO,
            "工具调用完成",
            step_id=str(step.id or ""),
            tool_call_id=lifecycle.tool_call_id,
            function_name=lifecycle.function_name,
            tool_name=lifecycle.tool_name,
            success=bool(tool_result.success),
            has_interrupt=bool(interrupt_request),
            loop_break_reason=loop_break_reason or "",
            browser_no_progress_count=execution_state.browser_no_progress_count,
            consecutive_failure_count=execution_state.consecutive_failure_count,
            llm_elapsed_ms=llm_cost_ms,
            tool_elapsed_ms=tool_cost_ms if matched_tool is not None else 0,
            elapsed_ms=elapsed_ms(started_at),
        )
        if interrupt_request is not None:
            log_runtime(
                logger,
                logging.INFO,
                "工具请求进入等待",
                step_id=str(step.id or ""),
                tool_call_id=lifecycle.tool_call_id,
                function_name=lifecycle.function_name,
                interrupt_kind=str(interrupt_request.get("kind") or ""),
                elapsed_ms=elapsed_ms(started_at),
            )
            return build_interrupt_payload(
                interrupt_request=interrupt_request,
                runtime_recent_action=execution_state.runtime_recent_action,
            ), event_dispatcher.emitted_events
        messages.append(
            build_tool_feedback_message(
                lifecycle=lifecycle,
                tool_result=tool_result,
                feedback_content_builder=_build_tool_feedback_content,
            )
        )
        convergence_result = policy_engine.evaluate_iteration_convergence(
            loop_break_reason=loop_break_reason or "",
            step=step,
            tool_result=tool_result,
            execution_state=execution_state,
        )
        if convergence_result.should_break and convergence_result.payload is not None:
            return convergence_result.payload, event_dispatcher.emitted_events
        progress_result = convergence_judge.evaluate_file_processing_progress(
            step=step,
            task_mode=task_mode,
            recent_function_name=lifecycle.normalized_function_name,
            tool_result_success=bool(tool_result.success),
            step_file_context=step_file_context,
            runtime_recent_action=execution_state.runtime_recent_action,
        )
        if progress_result.should_break and progress_result.payload is not None:
            log_runtime(
                logger,
                logging.INFO,
                "关键事实满足，提前收敛步骤",
                step_id=str(step.id or ""),
                task_mode=task_mode,
                reason_code=progress_result.reason_code,
                iteration=index,
            )
            return progress_result.payload, event_dispatcher.emitted_events

    return policy_engine.finalize_max_iterations(
        step=step,
        task_mode=task_mode,
        llm_message=execution_state.llm_message,
        started_at=started_at,
        requested_max_tool_iterations=execution_context.requested_max_tool_iterations,
        iteration_count=execution_context.effective_max_tool_iterations,
        runtime_recent_action=execution_state.runtime_recent_action,
        step_file_context=step_file_context,
    ), event_dispatcher.emitted_events
