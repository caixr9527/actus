#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层步骤执行节点。"""

import asyncio
import logging
import sys
from typing import Dict, List, Optional

from app.domain.external import LLM
from app.domain.models import ExecutionStatus, StepEvent, StepEventStatus, ToolEvent
from app.domain.services.runtime import SkillGraphRuntime
from app.domain.services.runtime.contracts.langgraph_settings import STEP_EXECUTION_TIMEOUT_SECONDS
from app.domain.services.runtime.contracts.runtime_logging import elapsed_ms, log_runtime, now_perf
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.tools import BaseTool
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.policies import classify_step_task_mode

from ..execution.tools import execute_step_with_prompt
from ..live_events import emit_live_events
from .control_state import (
    get_control_metadata as _get_control_metadata,
    is_direct_wait_execute_step as _is_direct_wait_execute_step,
)
from .execute_helpers import (
    build_empty_execution_message,
    build_execute_completed_transition,
    build_execute_interrupt_transition,
    build_timeout_execution_message,
    normalize_execute_runtime_result,
    prepare_execute_step_input,
)
from .state_reducer import _reduce_state_with_events
from .wait_helpers import _build_step_label, _normalize_interrupt_request

logger = logging.getLogger(__name__)


def _resolve_emit_live_events():
    """统一从 nodes 包级入口解析事件发送函数，保持聚合入口的可替换性。"""
    package_module = sys.modules.get(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes"
    )
    if package_module is not None:
        package_emit_live_events = getattr(package_module, "emit_live_events", None)
        if callable(package_emit_live_events):
            return package_emit_live_events
    return emit_live_events


async def execute_step_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
        runtime_context_service: RuntimeContextService,
        skill_runtime: Optional[SkillGraphRuntime] = None,
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
) -> PlannerReActLangGraphState:
    """执行单个步骤；当前批次未完成时继续跑后续步骤，整批完成后再统一重规划。"""
    started_at = now_perf()
    plan = state.get("plan")
    if plan is None:
        return state

    step = plan.get_next_step()
    if step is None:
        return state

    control = _get_control_metadata(state)
    is_direct_wait_execute_step = _is_direct_wait_execute_step(step, control)
    explicit_direct_wait_task_mode = str(control.get("direct_wait_execute_task_mode") or "").strip()
    task_mode = explicit_direct_wait_task_mode if is_direct_wait_execute_step and explicit_direct_wait_task_mode else classify_step_task_mode(
        step)
    step.status = ExecutionStatus.RUNNING
    started_event = StepEvent(step=step.model_copy(deep=True), status=StepEventStatus.STARTED)
    emit_events = _resolve_emit_live_events()
    await emit_events(started_event)

    user_message = str(state.get("user_message", ""))
    if is_direct_wait_execute_step:
        # 确认后的执行阶段必须继续消费原始请求，而不是“继续/确认”这类恢复文本。
        user_message = str(control.get("direct_wait_original_message") or user_message).strip()
    prepared_execute_input = await prepare_execute_step_input(
        state=state,
        step=step,
        llm=llm,
        runtime_context_service=runtime_context_service,
        task_mode=task_mode,
        user_message=user_message,
    )
    working_memory = prepared_execute_input.working_memory
    execute_context_updates = prepared_execute_input.execute_context_updates

    log_runtime(
        logger,
        logging.INFO,
        "开始执行步骤",
        state=state,
        step_id=str(step.id or ""),
        step_title=str(step.title or step.description or ""),
        task_mode=task_mode,
        attachment_count=len(prepared_execute_input.input_parts),
        runtime_tool_count=len(list(runtime_tools or [])),
        has_skill_runtime=skill_runtime is not None,
        confirmed_fact_keys=prepared_execute_input.confirmed_fact_keys,
    )

    llm_message: Optional[Dict[str, object]] = None
    tool_events: List[ToolEvent] = []
    tool_cost_ms = 0
    skill_cost_ms = 0

    try:
        async with asyncio.timeout(STEP_EXECUTION_TIMEOUT_SECONDS):
            # 只要执行器能力已初始化（即便当前列表为空），都统一走 execute_step_with_prompt，
            # 这样“无工具纯文本完成”分支才能真正生效。
            if runtime_tools is not None:
                tool_started_at = now_perf()
                llm_message, tool_events = await execute_step_with_prompt(
                    llm=llm,
                    step=step,
                    runtime_tools=runtime_tools or [],
                    max_tool_iterations=max_tool_iterations,
                    task_mode=task_mode,
                    on_tool_event=emit_events,
                    user_content=prepared_execute_input.user_content,
                    user_message=user_message,
                    attachment_paths=prepared_execute_input.attachments,
                    artifact_paths=prepared_execute_input.available_file_context_refs,
                    has_available_file_context=prepared_execute_input.available_file_context,
                )
                tool_cost_ms = elapsed_ms(tool_started_at)

            # 暂时忽略, 因为技能能力目前尚未实现。
            # if llm_message is None and skill_runtime is not None:
            #     try:
            #         log_runtime(
            #             logger,
            #             logging.INFO,
            #             "开始执行技能",
            #             state=state,
            #             step_id=str(step.id or ""),
            #             skill_id=PLANNER_EXECUTE_STEP_SKILL_ID,
            #         )
            #         skill_started_at = now_perf()
            #         skill_result = await skill_runtime.execute_skill(
            #             skill_id=PLANNER_EXECUTE_STEP_SKILL_ID,
            #             payload={
            #                 "session_id": str(state.get("session_id") or ""),
            #                 "user_message": user_message,
            #                 "step_description": step.description,
            #                 "language": language,
            #                 "attachments": attachments,
            #                 "execution_context": execute_context_packet,
            #             },
            #         )
            #         skill_cost_ms = elapsed_ms(skill_started_at)
            #         llm_message = {
            #             "success": bool(getattr(skill_result, "success", True)),
            #             "result": str(getattr(skill_result, "result", "") or f"步骤执行完成：{step.description}"),
            #             "attachments": normalize_attachments(getattr(skill_result, "attachments", [])),
            #         }
            #     except Exception as e:
            #         log_runtime(
            #             logger,
            #             logging.WARNING,
            #             "技能执行失败，回退默认链路",
            #             state=state,
            #             step_id=str(step.id or ""),
            #             skill_id=PLANNER_EXECUTE_STEP_SKILL_ID,
            #             error=str(e),
            #             elapsed_ms=elapsed_ms(started_at),
            #         )
    except TimeoutError:
        step_label = _build_step_label(step)
        log_runtime(
            logger,
            logging.WARNING,
            "步骤执行超时",
            state=state,
            step_id=str(step.id or ""),
            step_description=step_label,
            timeout_seconds=STEP_EXECUTION_TIMEOUT_SECONDS,
            elapsed_ms=elapsed_ms(started_at),
        )
        llm_message = build_timeout_execution_message(
            step=step,
            timeout_seconds=STEP_EXECUTION_TIMEOUT_SECONDS,
        )

    if llm_message is None:
        llm_message = build_empty_execution_message(step=step)

    normalized_runtime_result = normalize_execute_runtime_result(
        llm_message=llm_message,
        tool_events=tool_events,
        runtime_context_service=runtime_context_service,
    )
    runtime_recent_action = normalized_runtime_result.runtime_recent_action
    interrupt_request = _normalize_interrupt_request(
        normalized_runtime_result.raw_message.get("interrupt_request")
    )
    if interrupt_request:
        log_runtime(
            logger,
            logging.INFO,
            "步骤请求进入等待",
            state=state,
            step_id=str(step.id or ""),
            interrupt_kind=str(interrupt_request.get("kind") or ""),
            tool_elapsed_ms=tool_cost_ms,
            skill_elapsed_ms=skill_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
        interrupt_transition = await build_execute_interrupt_transition(
            state=state,
            plan=plan,
            step=step,
            control=control,
            runtime_context_service=runtime_context_service,
            task_mode=task_mode,
            runtime_recent_action=runtime_recent_action,
            execute_context_updates=execute_context_updates,
            interrupt_request=interrupt_request,
            user_message=user_message,
        )
        return _reduce_state_with_events(
            state,
            updates=interrupt_transition.updates,
            events=[started_event, *tool_events],
        )

    completed_transition = await build_execute_completed_transition(
        state=state,
        plan=plan,
        step=step,
        control=control,
        runtime_context_service=runtime_context_service,
        task_mode=task_mode,
        execute_context_updates=execute_context_updates,
        runtime_recent_action=runtime_recent_action,
        normalized_execution=normalized_runtime_result.normalized_execution,
        started_event=started_event,
        tool_events=tool_events,
        user_message=user_message,
        working_memory=working_memory,
        is_direct_wait_execute_step=is_direct_wait_execute_step,
    )
    await emit_events(*completed_transition.events[len([started_event, *tool_events]):])
    log_runtime(
        logger,
        logging.INFO,
        "步骤执行完成",
        state=state,
        step_id=str(step.id or ""),
        status=step.status.value,
        success=completed_transition.step_success,
        artifact_count=completed_transition.artifact_count,
        blocker_count=completed_transition.blocker_count,
        open_question_count=completed_transition.open_question_count,
        next_step_id=completed_transition.next_step_id or "",
        tool_elapsed_ms=tool_cost_ms,
        skill_elapsed_ms=skill_cost_ms,
        elapsed_ms=elapsed_ms(started_at),
    )
    return _reduce_state_with_events(
        state,
        updates=completed_transition.updates,
        events=completed_transition.events,
    )
