#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层等待恢复节点。

本模块承载 human_wait 与入口 wait 的恢复节点实现。
"""

import logging
import sys
from typing import Optional

from langgraph.types import interrupt

from app.domain.models import (
    ExecutionStatus,
    Step,
    StepEvent,
    StepEventStatus,
    StepOutcome,
)
from app.domain.services.runtime.contracts.runtime_logging import (
    elapsed_ms,
    log_runtime,
    now_perf,
)
from app.domain.services.runtime.contracts.final_output_contract import (
    RuntimeOutputStage,
    assert_state_update_allowed,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import normalize_step_result_text
from ..live_events import emit_live_events
from .confirmed_facts import (
    _extract_confirmed_facts_from_resume,
    _merge_confirmed_facts,
    _normalize_confirmed_fact_map,
)
from .control_state import (
    clear_wait_entry_runtime_state as _clear_wait_entry_runtime_state,
    get_entry_contract_payload as _get_entry_contract_payload,
    get_control_metadata as _get_control_metadata,
    replace_control_metadata as _replace_control_metadata,
)
from .delivery_helpers import _merge_step_outcome_into_working_memory
from .memory_helpers import _append_message_window_entry
from .state_reducer import _reduce_state_with_events
from .wait_helpers import (
    _build_post_wait_execute_step,
    _build_wait_cancel_step_summary,
    _build_wait_resume_outcome,
    _normalize_interrupt_request,
    _resolve_wait_resume_branch,
    _resume_value_to_message,
)
from .working_memory import _ensure_working_memory

logger = logging.getLogger(__name__)


def _resolve_interrupt_callable():
    """统一从 nodes 包级入口解析 interrupt，保持节点聚合入口的可替换性。"""
    package_module = sys.modules.get(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes"
    )
    if package_module is not None:
        package_interrupt = getattr(package_module, "interrupt", None)
        if callable(package_interrupt):
            return package_interrupt
    return interrupt


async def wait_for_human_node(
        state: PlannerReActLangGraphState,
) -> PlannerReActLangGraphState:
    """在等待节点中恢复用户输入，并回到当前批次继续执行剩余步骤。"""
    started_at = now_perf()
    interrupt_request = _normalize_interrupt_request(state.get("pending_interrupt"))
    if not interrupt_request:
        log_runtime(
            logger,
            logging.INFO,
            "跳过恢复处理",
            state=state,
            reason="没有待恢复的中断",
            elapsed_ms=elapsed_ms(started_at),
        )
        return {
            **state,
            "pending_interrupt": {},
        }

    log_runtime(
        logger,
        logging.INFO,
        "开始处理等待恢复",
        state=state,
        interrupt_kind=str(interrupt_request.get("kind") or ""),
    )

    resume_value = _resolve_interrupt_callable()(interrupt_request)
    resumed_message = _resume_value_to_message(interrupt_request, resume_value)
    message_window = list(state.get("message_window") or [])
    if resumed_message:
        message_window = _append_message_window_entry(
            message_window,
            role="user",
            message=resumed_message,
            attachments=[],
        )

    control = _get_control_metadata(state)
    control.pop("wait_resume_action", None)

    waiting_step_id = str(state.get("current_step_id") or "").strip()

    plan = state.get("plan")
    if plan is None or not waiting_step_id:
        log_runtime(
            logger,
            logging.INFO,
            "恢复完成，但未绑定步骤",
            state=state,
            resumed_message_length=len(resumed_message),
            elapsed_ms=elapsed_ms(started_at),
        )
        return {
            **state,
            "user_message": resumed_message,
            "input_parts": [],
            "message_window": message_window,
            "graph_metadata": _replace_control_metadata(state, control),
            "pending_interrupt": {},
        }

    waiting_step: Optional[Step] = None
    for candidate in list(plan.steps or []):
        if str(candidate.id or "").strip() == waiting_step_id:
            waiting_step = candidate
            break

    if waiting_step is None:
        log_runtime(
            logger,
            logging.WARNING,
            "恢复时未找到等待步骤",
            state=state,
            waiting_step_id=waiting_step_id,
            elapsed_ms=elapsed_ms(started_at),
        )
        return {
            **state,
            "user_message": resumed_message,
            "input_parts": [],
            "message_window": message_window,
            "graph_metadata": _replace_control_metadata(state, control),
            "pending_interrupt": {},
        }

    resume_branch = _resolve_wait_resume_branch(interrupt_request, resume_value)
    confirmed_facts = _extract_confirmed_facts_from_resume(
        waiting_step=waiting_step,
        payload=interrupt_request,
        resume_value=resume_value,
        resumed_message=resumed_message,
    )

    if resume_branch == "confirm_cancel":
        waiting_step.outcome = StepOutcome(
            done=False,
            summary=_build_wait_cancel_step_summary(waiting_step, resumed_message),
        )
        waiting_step.status = ExecutionStatus.CANCELLED
        cancelled_event = StepEvent(
            step=waiting_step.model_copy(deep=True),
            status=StepEventStatus.CANCELLED,
        )
        await emit_live_events(cancelled_event)
        working_memory = _merge_step_outcome_into_working_memory(
            _ensure_working_memory(state),
            step=waiting_step,
        )
        # 入口等待被取消后，本轮后续由 Planner 重新判断；入口合同保留用于审计。
        _clear_wait_entry_runtime_state(control)
        control["wait_resume_action"] = "replan"
        log_runtime(
            logger,
            logging.INFO,
            "等待恢复收到取消确认，转入重规划",
            state=state,
            step_id=str(waiting_step.id or ""),
            resumed_message_length=len(resumed_message),
            elapsed_ms=elapsed_ms(started_at),
        )
        updates = {
            "plan": plan,
            "user_message": resumed_message,
            "input_parts": [],
            "message_window": message_window,
            "graph_metadata": _replace_control_metadata(state, control),
            "pending_interrupt": {},
            "last_executed_step": waiting_step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": None,
            "working_memory": working_memory,
            # 等待恢复结果只进入步骤摘要，不应覆盖最终正文；但状态合同要求 final_message 键稳定存在。
            "final_message": str(state.get("final_message") or ""),
        }
        assert_state_update_allowed(
            stage=RuntimeOutputStage.WAIT,
            before_state=state,
            updates=updates,
        )
        return _reduce_state_with_events(
            state,
            updates=updates,
            events=[cancelled_event],
        )

    waiting_step.outcome = _build_wait_resume_outcome(
        waiting_step,
        branch=resume_branch,
        resumed_message=resumed_message,
    )
    waiting_step.status = ExecutionStatus.COMPLETED
    completed_event = StepEvent(
        step=waiting_step.model_copy(deep=True),
        status=StepEventStatus.COMPLETED,
    )
    await emit_live_events(completed_event)
    next_step = plan.get_next_step()
    original_user_message = str(state.get("user_message") or "").strip()
    next_user_message = original_user_message or resumed_message
    working_memory = _ensure_working_memory(state)
    merged_confirmed_facts = _merge_confirmed_facts(
        memory_facts=_normalize_confirmed_fact_map(working_memory.get("confirmed_facts")),
        new_facts=confirmed_facts,
    )
    if merged_confirmed_facts:
        working_memory["confirmed_facts"] = merged_confirmed_facts
    if str(waiting_step.id or "").strip() == "direct-wait-confirm":
        # synthetic confirm 只负责授权继续执行，后续节点仍应保留入口合同中的原始请求。
        entry_contract_payload = _get_entry_contract_payload(control)
        source_payload = entry_contract_payload.get("source") if isinstance(entry_contract_payload, dict) else {}
        next_user_message = str(
            source_payload.get("user_message") if isinstance(source_payload, dict) else ""
        ).strip() or resumed_message
    elif next_step is None and merged_confirmed_facts:
        generated_next_step = _build_post_wait_execute_step(
            original_user_message=original_user_message,
            resumed_message=resumed_message,
            waiting_step=waiting_step,
        )
        plan.steps.append(generated_next_step)
        next_step = generated_next_step
    working_memory = _merge_step_outcome_into_working_memory(
        working_memory,
        step=waiting_step,
    )
    log_runtime(
        logger,
        logging.INFO,
        "等待恢复完成",
        state=state,
        step_id=str(waiting_step.id or ""),
        resumed_message_length=len(resumed_message),
        confirmed_fact_keys=sorted(list(merged_confirmed_facts.keys())),
        next_step_id=str(next_step.id or "") if next_step is not None else "",
        elapsed_ms=elapsed_ms(started_at),
    )

    updates = {
        "plan": plan,
        "user_message": next_user_message,
        "input_parts": [],
        "message_window": message_window,
        "graph_metadata": _replace_control_metadata(state, control),
        "pending_interrupt": {},
        "last_executed_step": waiting_step.model_copy(deep=True),
        "execution_count": int(state.get("execution_count", 0)) + 1,
        "current_step_id": next_step.id if next_step is not None else None,
        "working_memory": working_memory,
        # 等待恢复完成后继续后续步骤，不把当前等待步骤摘要提升为最终正文。
        "final_message": str(state.get("final_message") or ""),
    }
    assert_state_update_allowed(
        stage=RuntimeOutputStage.WAIT,
        before_state=state,
        updates=updates,
    )
    return _reduce_state_with_events(
        state,
        updates=updates,
        events=[completed_event],
    )
