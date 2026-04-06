#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行取消收敛辅助函数。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from app.domain.models import (
    BaseEvent,
    ExecutionStatus,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Step,
    StepEvent,
    StepEventStatus,
    StepOutcome,
)


def extract_plan_from_runtime_metadata(runtime_metadata: Dict[str, Any]) -> Optional[Plan]:
    """从运行时元数据中提取当前计划快照。"""
    graph_contract = runtime_metadata.get("graph_state_contract")
    if not isinstance(graph_contract, dict):
        return None

    graph_state = graph_contract.get("graph_state")
    if not isinstance(graph_state, dict):
        return None

    plan_payload = graph_state.get("plan")
    if not isinstance(plan_payload, dict) or len(plan_payload) == 0:
        return None

    try:
        return Plan.model_validate(plan_payload)
    except Exception:
        return None


def _extract_plan_from_run_events(run_events: Iterable[BaseEvent]) -> tuple[Optional[Plan], int]:
    """从当前 run 的事件流中取出最近一次计划快照及其索引。"""
    event_list = list(run_events)
    for index in range(len(event_list) - 1, -1, -1):
        event = event_list[index]
        if not isinstance(event, PlanEvent):
            continue
        return event.plan.model_copy(deep=True), index
    return None, -1


def _reduce_plan_with_step_events(
        plan: Plan,
        *,
        run_events: Iterable[BaseEvent],
        start_index: int,
) -> Plan:
    """用计划之后的 StepEvent 收敛最终步骤快照。"""
    reduced_plan = plan.model_copy(deep=True)
    ordered_steps = list(reduced_plan.steps)
    step_index_by_id = {
        str(step.id): index
        for index, step in enumerate(ordered_steps)
        if str(step.id).strip()
    }

    event_list = list(run_events)
    for event in event_list[start_index + 1:]:
        if not isinstance(event, StepEvent):
            continue

        next_step = event.step.model_copy(deep=True)
        existing_index = step_index_by_id.get(str(next_step.id))
        if existing_index is None:
            step_index_by_id[str(next_step.id)] = len(ordered_steps)
            ordered_steps.append(next_step)
            continue
        ordered_steps[existing_index] = next_step

    reduced_plan.steps = ordered_steps
    return reduced_plan


def _build_plan_from_step_events(run_events: Iterable[BaseEvent]) -> Optional[Plan]:
    """在缺少计划事件时，用当前 run 的步骤事件合成最小计划快照。"""
    ordered_steps: list[Step] = []
    step_index_by_id: dict[str, int] = {}

    for event in run_events:
        if not isinstance(event, StepEvent):
            continue

        next_step = event.step.model_copy(deep=True)
        existing_index = step_index_by_id.get(str(next_step.id))
        if existing_index is None:
            step_index_by_id[str(next_step.id)] = len(ordered_steps)
            ordered_steps.append(next_step)
            continue
        ordered_steps[existing_index] = next_step

    if len(ordered_steps) == 0:
        return None

    return Plan(steps=ordered_steps)


def resolve_plan_for_cancellation(
        runtime_metadata: Dict[str, Any],
        *,
        run_events: Iterable[BaseEvent],
) -> Optional[Plan]:
    """为 stop/cancel 收敛恢复计划快照。

    优先级：
    1. 当前 run 的事件流真相源；
    2. runtime_metadata 中的 graph_state_contract；
    3. 仅由 StepEvent 合成的最小计划。
    """
    event_list = list(run_events)

    plan_from_events, plan_event_index = _extract_plan_from_run_events(event_list)
    if plan_from_events is not None:
        return _reduce_plan_with_step_events(
            plan_from_events,
            run_events=event_list,
            start_index=plan_event_index,
        )

    plan_from_runtime = extract_plan_from_runtime_metadata(runtime_metadata)
    if plan_from_runtime is not None:
        return _reduce_plan_with_step_events(
            plan_from_runtime,
            run_events=event_list,
            start_index=-1,
        )

    return _build_plan_from_step_events(event_list)


def build_cancelled_plan_snapshot(
        plan: Optional[Plan],
        *,
        current_step_id: Optional[str],
        cancellation_summary: str = "任务已取消",
) -> Tuple[Optional[Plan], Optional[Step]]:
    """将计划中的未完成步骤统一收敛为 cancelled，并返回当前步骤快照。"""
    if plan is None:
        return None, None

    cancelled_plan = plan.model_copy(deep=True)
    cancelled_step: Optional[Step] = None
    normalized_current_step_id = str(current_step_id or "").strip()

    for step in cancelled_plan.steps:
        if step.done:
            continue

        step.status = ExecutionStatus.CANCELLED
        if step.outcome is None:
            step.outcome = StepOutcome(done=False)

        if not str(step.outcome.summary or "").strip():
            step.outcome.summary = cancellation_summary

        if normalized_current_step_id and str(step.id or "").strip() == normalized_current_step_id:
            cancelled_step = step.model_copy(deep=True)

    if cancelled_step is None:
        next_cancelled_step = next(
            (
                step.model_copy(deep=True)
                for step in cancelled_plan.steps
                if step.status == ExecutionStatus.CANCELLED
            ),
            None,
        )
        cancelled_step = next_cancelled_step

    cancelled_plan.status = ExecutionStatus.CANCELLED
    return cancelled_plan, cancelled_step


def build_cancelled_runtime_events(
        runtime_metadata: Dict[str, Any],
        *,
        run_events: Iterable[BaseEvent],
        current_step_id: Optional[str],
        cancellation_summary: str = "任务已取消",
) -> Tuple[Optional[PlanEvent], Optional[StepEvent]]:
    """为取消分支构造 cancelled 计划/步骤事件。"""
    cancelled_plan, cancelled_step = build_cancelled_plan_snapshot(
        resolve_plan_for_cancellation(
            runtime_metadata,
            run_events=run_events,
        ),
        current_step_id=current_step_id,
        cancellation_summary=cancellation_summary,
    )

    plan_event = (
        PlanEvent(
            plan=cancelled_plan,
            status=PlanEventStatus.CANCELLED,
        )
        if cancelled_plan is not None
        else None
    )
    step_event = (
        StepEvent(
            step=cancelled_step,
            status=StepEventStatus.CANCELLED,
        )
        if cancelled_step is not None
        else None
    )
    return plan_event, step_event
