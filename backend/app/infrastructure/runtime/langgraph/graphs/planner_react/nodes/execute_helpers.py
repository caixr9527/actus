#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""execute_step_node 专项 helper。

本模块承接执行节点的稳定职责：
1. 执行前输入装配：把 state/control/step 转成执行器可消费的输入；
2. 执行结果归一：把执行器返回的原始消息与 tool event 收口成统一结果；
3. interrupt / completed 分支的状态写回数据装配。

注意：
- 本模块不改变 wait / replan / summary 的业务语义；
- 本模块不直接决定 graph node 的流转，只给 execute_step_node 提供可复用的阶段 helper。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.domain.external import LLM
from app.domain.models import (
    ExecutionStatus,
    Plan,
    Step,
    StepDeliveryRole,
    StepEvent,
    StepOutcome,
    StepOutputMode,
    ToolEvent,
)
from app.domain.services.prompts import EXECUTION_PROMPT
from app.domain.services.runtime.contracts.langgraph_settings import MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS
from app.domain.services.runtime.langgraph_events import append_events
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import (
    normalize_controlled_value,
    normalize_execution_response,
    normalize_file_path_list,
    normalize_step_result_text,
    normalize_text_list,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    format_attachments_for_prompt,
    normalize_attachments,
)

from ..execution.tools import has_available_file_context
from .confirmed_facts import _build_step_execution_text, _normalize_confirmed_fact_map
from .control_state import replace_control_metadata
from .delivery_helpers import (
    _collect_available_file_context_refs,
    _infer_step_attachment_delivery_preference,
    _merge_step_outcome_into_working_memory,
    _sanitize_step_delivery_text,
)
from ..parsers import extract_write_file_paths_from_tool_events, merge_attachment_paths
from .prompt_context_helpers import (
    _append_prompt_context_to_prompt,
    _build_prompt_context_packet_async,
    _extract_prompt_context_state_updates,
)
from .wait_helpers import _build_step_label
from .working_memory import _ensure_working_memory


@dataclass(frozen=True)
class ExecuteStepPreparedInput:
    """执行阶段输入装配结果。

    业务含义：
    - 这是 execute_step_node 真正调用执行器前的完整输入快照；
    - 节点壳只负责决定是否执行，具体 prompt/context/附件装配由这里统一产出。
    """

    user_message: str
    task_mode: str
    working_memory: Dict[str, Any]
    confirmed_fact_keys: List[str]
    input_parts: List[Dict[str, Any]]
    attachments: List[str]
    available_file_context_refs: List[str]
    available_file_context: bool
    execute_context_updates: Dict[str, Any]
    user_content: List[Dict[str, Any]]


@dataclass(frozen=True)
class ExecuteStepNormalizedRuntimeResult:
    """执行器原始输出的统一归一结果。

    业务含义：
    - `raw_message` 保留执行器原始返回，供后续 interrupt/completed 分支继续消费；
    - `runtime_recent_action` 是 execute 节点后续写回上下文的依据；
    - `normalized_execution` 只在非 interrupt 完成分支使用，用来构造 `StepOutcome`。
    """

    raw_message: Dict[str, Any]
    runtime_recent_action: Dict[str, Any]
    normalized_execution: Dict[str, Any]


@dataclass(frozen=True)
class ExecuteStepInterruptTransition:
    """interrupt 分支的状态写回结果。

    业务含义：
    - 本结构只描述 execute 节点在进入 wait 前要写回的 state updates；
    - 节点壳继续负责事件归并与 `_reduce_state_with_events` 返回，不把 graph 流转语义下沉到 helper。
    """

    updates: Dict[str, Any]


@dataclass(frozen=True)
class ExecuteStepCompletedTransition:
    """completed 分支的结果落账与 state 写回结果。"""

    updates: Dict[str, Any]
    events: List[Any]
    next_step_id: Optional[str]
    step_success: bool
    step_summary: str
    artifact_count: int
    blocker_count: int
    open_question_count: int


async def _build_message(
        llm: LLM,
        user_message_prompt: str,
        input_parts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """按多模态能力统一构造执行节点 user content。"""
    if getattr(llm, "multimodal", False) and len(input_parts) > 0:
        multiplexed_message = await llm.format_multiplexed_message(input_parts)
        return [
            {"type": "text", "text": user_message_prompt},
            *multiplexed_message,
        ]
    return [{"type": "text", "text": user_message_prompt}]


def collect_recent_search_queries_from_tool_events(tool_events: List[ToolEvent]) -> List[str]:
    """从当前步骤 tool event 中提取搜索词，供 runtime_recent_action 回写。"""
    queries: List[str] = []
    for event in tool_events:
        if str(event.function_name or "").strip().lower() != "search_web":
            continue
        query = str((event.function_args or {}).get("query") or (event.function_args or {}).get("q") or "").strip()
        if not query or query in queries:
            continue
        queries.append(query)
    return queries


async def prepare_execute_step_input(
        *,
        state: PlannerReActLangGraphState,
        step: Step,
        llm: LLM,
        runtime_context_service: RuntimeContextService,
        task_mode: str,
        user_message: str,
) -> ExecuteStepPreparedInput:
    """装配执行阶段输入，不处理 graph state 写回。"""
    plan = state.get("plan")
    language = getattr(plan, "language", None) or "zh"
    working_memory = _ensure_working_memory(state)
    confirmed_fact_keys = sorted(
        list(
            _normalize_confirmed_fact_map(working_memory.get("confirmed_facts")).keys()
        )
    )
    step_execution_text = _build_step_execution_text(
        step,
        working_memory=working_memory,
    )
    input_parts = list(state.get("input_parts") or [])
    attachments = [str(part.get("sandbox_filepath") or "") for part in input_parts if part.get("sandbox_filepath")]
    available_file_context_refs = _collect_available_file_context_refs(state)
    available_file_context = has_available_file_context(
        user_message=user_message,
        attachment_paths=attachments,
        artifact_paths=available_file_context_refs,
    )
    user_message_prompt = EXECUTION_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
        language=language,
        step=step_execution_text,
        delivery_role=str(getattr(step, "delivery_role", "") or "none"),
        delivery_context_state=str(getattr(step, "delivery_context_state", "") or "none"),
    )
    execute_context_packet = await _build_prompt_context_packet_async(
        stage="execute",
        state=state,
        runtime_context_service=runtime_context_service,
        step=step,
        task_mode=task_mode,
    )
    execute_context_updates = _extract_prompt_context_state_updates(
        runtime_context_service=runtime_context_service,
        context_packet=execute_context_packet,
    )
    user_message_prompt = _append_prompt_context_to_prompt(user_message_prompt, execute_context_packet)
    user_content = await _build_message(llm, user_message_prompt, input_parts)
    return ExecuteStepPreparedInput(
        user_message=user_message,
        task_mode=task_mode,
        working_memory=working_memory,
        confirmed_fact_keys=confirmed_fact_keys,
        input_parts=input_parts,
        attachments=attachments,
        available_file_context_refs=available_file_context_refs,
        available_file_context=available_file_context,
        execute_context_updates=execute_context_updates,
        user_content=user_content,
    )


def build_timeout_execution_message(*, step: Step, timeout_seconds: int) -> Dict[str, Any]:
    """构造执行超时的标准失败消息。"""
    step_label = _build_step_label(step)
    return {
        "success": False,
        "summary": f"步骤执行超时：{step_label}",
        "delivery_text": "",
        "attachments": [],
        "blockers": [f"当前步骤超过 {timeout_seconds} 秒未完成"],
        "next_hint": "请缩小当前步骤范围后重试",
    }


def build_empty_execution_message(*, step: Step) -> Dict[str, Any]:
    """构造执行器未返回内容时的标准失败消息。"""
    step_label = _build_step_label(step)
    return {
        "success": False,
        "summary": f"步骤执行失败：{step_label}",
        "delivery_text": "",
        "attachments": [],
    }


def normalize_execute_runtime_result(
        *,
        llm_message: Optional[Dict[str, Any]],
        tool_events: List[ToolEvent],
        runtime_context_service: RuntimeContextService,
) -> ExecuteStepNormalizedRuntimeResult:
    """归一执行器输出，供 execute_step_node 后续分支消费。"""
    raw_message = dict(llm_message or {})
    runtime_recent_action = runtime_context_service.normalize_runtime_recent_action(
        raw_message.get("runtime_recent_action")
    )
    recent_search_queries = collect_recent_search_queries_from_tool_events(tool_events)
    if recent_search_queries:
        runtime_recent_action["recent_search_queries"] = recent_search_queries
    return ExecuteStepNormalizedRuntimeResult(
        raw_message=raw_message,
        runtime_recent_action=runtime_recent_action,
        normalized_execution=normalize_execution_response(raw_message),
    )


async def build_execute_interrupt_transition(
        *,
        state: PlannerReActLangGraphState,
        plan: Plan,
        step: Step,
        control: Dict[str, Any],
        runtime_context_service: RuntimeContextService,
        task_mode: str,
        execute_context_updates: Dict[str, Any],
        runtime_recent_action: Dict[str, Any],
        interrupt_request: Dict[str, Any],
        user_message: str,
) -> ExecuteStepInterruptTransition:
    """构造 execute 节点 interrupt 分支的 state updates。"""
    next_state = {
        **state,
        **runtime_context_service.merge_runtime_recent_action(
            state_updates=execute_context_updates,
            task_mode=task_mode,
            runtime_recent_action=runtime_recent_action,
        ),
        "plan": plan,
        "current_step_id": step.id,
        "pending_interrupt": interrupt_request,
    }
    interrupt_context_packet = await _build_prompt_context_packet_async(
        stage="execute",
        state=next_state,
        runtime_context_service=runtime_context_service,
        step=step,
        task_mode=task_mode,
    )
    interrupt_context_updates = runtime_context_service.merge_runtime_recent_action(
        state_updates=_extract_prompt_context_state_updates(
            runtime_context_service=runtime_context_service,
            context_packet=interrupt_context_packet,
        ),
        task_mode=task_mode,
        runtime_recent_action=runtime_recent_action,
    )
    updated_control = dict(control)
    updated_control["step_reuse_hit"] = False
    return ExecuteStepInterruptTransition(
        updates={
            "plan": plan,
            **interrupt_context_updates,
            "current_step_id": step.id,
            "user_message": user_message,
            "graph_metadata": replace_control_metadata(state, updated_control),
            "pending_interrupt": interrupt_request,
        }
    )


async def build_execute_completed_transition(
        *,
        state: PlannerReActLangGraphState,
        plan: Plan,
        step: Step,
        control: Dict[str, Any],
        runtime_context_service: RuntimeContextService,
        task_mode: str,
        execute_context_updates: Dict[str, Any],
        runtime_recent_action: Dict[str, Any],
        normalized_execution: Dict[str, Any],
        started_event: Any,
        tool_events: List[ToolEvent],
        user_message: str,
        working_memory: Dict[str, Any],
        is_direct_wait_execute_step: bool,
) -> ExecuteStepCompletedTransition:
    """构造 execute 节点 completed 分支的结果落账与状态写回。"""
    step_success = bool(normalized_execution.get("success", True))
    step_label = _build_step_label(step)
    step_summary = normalize_step_result_text(
        normalized_execution.get("summary"),
        fallback=f"步骤执行完成：{step_label}" if step_success else f"步骤执行失败：{step_label}",
    )
    step_delivery_text = normalize_step_result_text(
        _sanitize_step_delivery_text(step, normalized_execution.get("delivery_text")),
        fallback=(
            step_summary
            if normalize_controlled_value(getattr(step, "delivery_role", None), StepDeliveryRole)
               == StepDeliveryRole.FINAL.value
               and normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
               == StepOutputMode.INLINE.value
            else ""
        ),
    )
    step_deliver_result_as_attachment = _infer_step_attachment_delivery_preference(
        user_message=user_message,
        normalized_execution=normalized_execution,
    )
    model_attachment_paths = normalize_attachments(normalized_execution.get("attachments"))
    tool_attachment_paths = extract_write_file_paths_from_tool_events(tool_events)
    step_attachment_paths = normalize_file_path_list(
        merge_attachment_paths(model_attachment_paths, tool_attachment_paths),
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )
    step.outcome = StepOutcome(
        done=step_success,
        summary=step_summary,
        delivery_text=step_delivery_text,
        produced_artifacts=step_attachment_paths,
        blockers=normalize_text_list(normalized_execution.get("blockers")),
        facts_learned=normalize_text_list(normalized_execution.get("facts_learned")),
        open_questions=normalize_text_list(normalized_execution.get("open_questions")),
        deliver_result_as_attachment=step_deliver_result_as_attachment,
        next_hint=normalize_step_result_text(normalized_execution.get("next_hint")),
    )
    step.status = ExecutionStatus.COMPLETED if step_success else ExecutionStatus.FAILED
    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=step.status,
    )
    events: List[Any] = [started_event, *tool_events, completed_event]
    next_step = plan.get_next_step()
    updated_control = dict(control)
    updated_control["step_reuse_hit"] = False
    if is_direct_wait_execute_step:
        updated_control["direct_wait_original_task_executed"] = True
        updated_control.pop("direct_wait_execute_task_mode", None)
        updated_control.pop("direct_wait_original_message", None)
    updated_working_memory = _merge_step_outcome_into_working_memory(
        working_memory,
        step=step,
    )
    completed_context_packet = await _build_prompt_context_packet_async(
        stage="execute",
        state={
            **state,
            **runtime_context_service.merge_runtime_recent_action(
                state_updates=execute_context_updates,
                task_mode=task_mode,
                runtime_recent_action=runtime_recent_action,
            ),
            "plan": plan,
            "last_executed_step": step.model_copy(deep=True),
            "working_memory": updated_working_memory,
            "final_message": step_summary,
            "pending_interrupt": {},
            "emitted_events": append_events(state.get("emitted_events"), *events),
        },
        runtime_context_service=runtime_context_service,
        step=step,
        task_mode=task_mode,
    )
    completed_context_updates = runtime_context_service.merge_runtime_recent_action(
        state_updates=_extract_prompt_context_state_updates(
            runtime_context_service=runtime_context_service,
            context_packet=completed_context_packet,
        ),
        task_mode=task_mode,
        runtime_recent_action=runtime_recent_action,
    )
    next_step_id = str(next_step.id or "") if next_step is not None else ""
    return ExecuteStepCompletedTransition(
        updates={
            "plan": plan,
            **completed_context_updates,
            "last_executed_step": step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
            "user_message": user_message,
            "working_memory": updated_working_memory,
            "graph_metadata": replace_control_metadata(state, updated_control),
            "final_message": step_summary,
            "selected_artifacts": list(state.get("selected_artifacts") or []),
            "pending_interrupt": {},
        },
        events=events,
        next_step_id=next_step_id or None,
        step_success=step_success,
        step_summary=step_summary,
        artifact_count=len(step_attachment_paths),
        blocker_count=len(list(step.outcome.blockers or [])),
        open_question_count=len(list(step.outcome.open_questions or [])),
    )
