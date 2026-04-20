#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层步骤复用节点。"""

import logging
import sys

from app.domain.models import ExecutionStatus, StepEvent, StepEventStatus
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import normalize_step_result_text

from ..live_events import emit_live_events
from .control_state import (
    get_control_metadata as _get_control_metadata,
    replace_control_metadata as _replace_control_metadata,
)
from .delivery_helpers import (
    _build_reused_step_outcome,
    _find_reusable_step_outcome,
    _merge_step_outcome_into_working_memory,
)
from .state_reducer import _reduce_state_with_events
from .working_memory import _ensure_working_memory

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

    await _resolve_emit_live_events()(completed_event)

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
