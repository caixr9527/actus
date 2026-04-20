#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 节点实现。"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

from app.domain.external import LLM
from app.domain.models import (
    ExecutionStatus,
    LongTermMemory,
    StepEvent,
    StepEventStatus,
    ToolEvent,
)
from app.domain.repositories import LongTermMemoryRepository
from app.domain.services.runtime import SkillGraphRuntime
from app.domain.services.runtime.contracts.langgraph_settings import (
    STEP_EXECUTION_TIMEOUT_SECONDS,
)
from app.domain.services.runtime.contracts.runtime_logging import (
    elapsed_ms,
    log_runtime,
    now_perf
)
from app.domain.services.runtime.langgraph_state import (
    PlannerReActLangGraphState,
    normalize_retrieved_memories,
)
from app.domain.services.runtime.normalizers import (
    normalize_step_result_text,
    normalize_text_list,
    truncate_text,
)
from app.domain.services.tools import BaseTool
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.policies import (
    classify_step_task_mode,
)
from ..execution.tools import execute_step_with_prompt
from ..live_events import emit_live_events
from .confirmed_facts import (
    _extract_confirmed_facts_from_resume,
    _fact_value_is_missing,
    _fact_value_to_text,
    _merge_confirmed_facts,
)
from .control_state import (
    clear_direct_wait_control_state as _clear_direct_wait_control_state,
    clear_plan_only_control_state as _clear_plan_only_control_state,
    get_control_metadata as _get_control_metadata,
    is_direct_wait_execute_step as _is_direct_wait_execute_step,
    replace_control_metadata as _replace_control_metadata,
)
from .direct_nodes import (
    _build_direct_plan_title,
    direct_answer_node,
    direct_execute_node,
    direct_wait_node,
    entry_router_node,
)
from .memory_nodes import consolidate_memory_node
from .planner_nodes import create_or_reuse_plan_node
from .delivery_helpers import (
    _build_intermediate_round_summary_fallback,
    _build_intermediate_round_summary_prompt,
    _build_reused_step_outcome,
    _find_reusable_step_outcome,
    _is_intermediate_delivery_step,
    _merge_step_outcome_into_working_memory,
    _should_skip_summary_llm_for_final_delivery,
)
from .execute_helpers import (
    build_empty_execution_message,
    build_execute_completed_transition,
    build_execute_interrupt_transition,
    build_timeout_execution_message,
    normalize_execute_runtime_result,
    prepare_execute_step_input,
)
from .memory_helpers import (
    _build_memory_recall_queries,
    _dedupe_recalled_memories,
)
from .finalize_nodes import finalize_node
from .prompt_context_helpers import _build_prompt_context_packet_async, _extract_prompt_context_state_updates
from .replan_nodes import replan_node
from .state_reducer import _reduce_state_with_events
from .summary_nodes import summarize_node
from .wait_helpers import (
    _build_step_label,
    _normalize_interrupt_request,
)
from .wait_nodes import wait_for_human_node
from .working_memory import _ensure_working_memory

logger = logging.getLogger(__name__)


# 状态与交付 helper：负责工作记忆默认结构，以及“轻 summary / 重交付正文”分轨。


def _truncate_text(value: Any, *, max_chars: int) -> str:
    return truncate_text(value, max_chars=max_chars)


# 附件 helper：只负责当前 run 内附件引用的收集、合并与最终交付选择。


# 入口路由节点：决定 direct_answer / direct_wait / direct_execute / planner 主链入口。


# 规划前置节点：先召回记忆，再创建或复用当前计划。
async def recall_memory_context_node(
        state: PlannerReActLangGraphState,
        long_term_memory_repository: Optional[LongTermMemoryRepository] = None,
) -> PlannerReActLangGraphState:
    """统一整理线程级短期记忆，为后续 planner/react 节点提供稳定输入。"""
    started_at = now_perf()
    log_runtime(
        logger,
        logging.INFO,
        "开始召回记忆上下文",
        state=state,
        existing_memory_count=len(list(state.get("retrieved_memories") or [])),
        has_repository=long_term_memory_repository is not None,
    )
    plan = state.get("plan")
    working_memory = _ensure_working_memory(state)
    if not str(working_memory.get("goal") or "").strip():
        working_memory["goal"] = str(getattr(plan, "goal", "") or state.get("user_message") or "")
    if not list(working_memory.get("open_questions") or []):
        working_memory["open_questions"] = normalize_text_list(state.get("session_open_questions"))

    retrieved_memories = list(state.get("retrieved_memories") or [])
    recall_cost_ms = 0
    if long_term_memory_repository is not None:
        try:
            recall_started_at = now_perf()
            recalled_memories: List[LongTermMemory] = []
            for query in _build_memory_recall_queries(state):
                recalled_memories.extend(await long_term_memory_repository.search(query))
            recalled_memories = _dedupe_recalled_memories(recalled_memories)
            retrieved_memories = normalize_retrieved_memories(
                [memory.model_dump(mode="json") for memory in recalled_memories]
            )
            recall_cost_ms = elapsed_ms(recall_started_at)
        except Exception as e:
            log_runtime(
                logger,
                logging.WARNING,
                "记忆召回失败，回退线程快照",
                state=state,
                error=str(e),
                recall_elapsed_ms=recall_cost_ms,
                elapsed_ms=elapsed_ms(started_at),
            )
    # P3-一次性收口：planner 前禁止把 profile 记忆回写到 working_memory，避免跨任务偏好污染计划。

    log_runtime(
        logger,
        logging.INFO,
        "记忆召回完成",
        state=state,
        recalled_memory_count=len(retrieved_memories),
        open_question_count=len(list(working_memory.get("open_questions") or [])),
        preference_count=len(dict(working_memory.get("user_preferences") or {})),
        recall_elapsed_ms=recall_cost_ms,
        elapsed_ms=elapsed_ms(started_at),
    )

    return {
        **state,
        "working_memory": working_memory,
        "retrieved_memories": retrieved_memories,
    }


# 步骤复用节点：在真实执行前先做当前 run 内的等价目标复用。
async def guard_step_reuse_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """在真实执行前做当前 run 内复用，命中时直接跳过执行节点。"""
    plan = state.get("plan")
    if plan is None:
        return state

    step = plan.get_next_step()
    if step is None:
        return state

    log_runtime(
        logger,
        logging.INFO,
        "开始检查步骤复用",
        state=state,
        step_id=str(step.id or ""),
        objective_key=str(step.objective_key or ""),
    )

    reusable_step = _find_reusable_step_outcome(state=state)
    control = _get_control_metadata(state)
    control["step_reuse_hit"] = False

    if reusable_step is None:
        log_runtime(
            logger,
            logging.INFO,
            "步骤复用未命中",
            state=state,
            step_id=str(step.id or ""),
        )
        return {
            **state,
            "graph_metadata": _replace_control_metadata(state, control),
        }

    source_outcome, reused_from_run_id, reused_from_step_id = reusable_step
    step.outcome = _build_reused_step_outcome(
        source_outcome,
        reused_from_run_id=reused_from_run_id,
        reused_from_step_id=reused_from_step_id,
    )
    step.status = ExecutionStatus.COMPLETED

    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=StepEventStatus.COMPLETED,
    )

    await emit_live_events(completed_event)

    next_step = plan.get_next_step()
    working_memory = _merge_step_outcome_into_working_memory(
        _ensure_working_memory(state),
        step=step,
    )
    control["step_reuse_hit"] = True
    log_runtime(
        logger,
        logging.INFO,
        "步骤复用命中",
        state=state,
        step_id=str(step.id or ""),
        source_run_id=reused_from_run_id,
        source_step_id=reused_from_step_id,
        artifact_count=len(list(step.outcome.produced_artifacts or [])),
    )

    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "last_executed_step": step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
            "working_memory": working_memory,
            "graph_metadata": _replace_control_metadata(state, control),
            "final_message": normalize_step_result_text(step.outcome.summary),
            "selected_artifacts": list(state.get("selected_artifacts") or []),
            "pending_interrupt": {},
        },
        events=[completed_event],
    )


# 执行节点：负责单步工具执行与步骤结果落账。
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
    await emit_live_events(started_event)

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

    llm_message: Optional[Dict[str, Any]] = None
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
                    on_tool_event=emit_live_events,
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
    await emit_live_events(*completed_transition.events[len([started_event, *tool_events]):])
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
