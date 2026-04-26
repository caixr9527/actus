#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 路由判断。"""

import logging
from typing import Literal

from app.domain.models import ExecutionStatus
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState, get_graph_control
from app.domain.services.runtime.normalizers import normalize_controlled_value
from app.domain.services.workspace_runtime.entry import EntryContract, EntryRoute

logger = logging.getLogger(__name__)


def route_after_entry(
        state: PlannerReActLangGraphState,
) -> Literal["direct_answer", "direct_wait", "atomic_action", "create_plan_or_reuse", "recall_memory_context"]:
    route = _get_entry_route(state)
    if route == EntryRoute.ANSWER.value:
        return "direct_answer"
    if route == EntryRoute.WAIT.value:
        return "direct_wait"
    if route == EntryRoute.ATOMIC_ACTION.value:
        return "atomic_action"
    if route == EntryRoute.RESUME_PLAN.value:
        return "create_plan_or_reuse"
    return "recall_memory_context"


def _get_entry_contract_payload(state: PlannerReActLangGraphState) -> dict:
    payload = get_graph_control(state.get("graph_metadata")).get("entry_contract")
    return dict(payload) if isinstance(payload, dict) else {}


def _get_entry_route(state: PlannerReActLangGraphState) -> str:
    return str(_get_entry_contract_payload(state).get("route") or "").strip()


def _get_entry_contract(state: PlannerReActLangGraphState) -> EntryContract | None:
    payload = _get_entry_contract_payload(state)
    if not payload:
        return None
    return EntryContract.model_validate(payload)


def _should_summarize_after_plan_finished(state: PlannerReActLangGraphState) -> bool:
    contract = _get_entry_contract(state)
    if contract is None:
        return False
    return contract.needs_summary and contract.route in {EntryRoute.WAIT, EntryRoute.ATOMIC_ACTION}


def _has_reached_execution_limit(state: PlannerReActLangGraphState) -> bool:
    execution_count = int(state.get("execution_count", 0))
    max_execution_steps = int(state.get("max_execution_steps", 20))
    if execution_count < max_execution_steps:
        return False

    log_runtime(
        logger,
        logging.WARNING,
        "达到执行上限，转入总结",
        state=state,
        execution_count=execution_count,
        max_execution_steps=max_execution_steps,
    )
    return True


def _route_after_completed_step(
        state: PlannerReActLangGraphState,
) -> Literal["guard_step_reuse", "replan", "summarize"]:
    """步骤已完成后，优先继续当前批次；仅在批次跑完后进入重规划。"""
    if _has_reached_execution_limit(state):
        return "summarize"

    plan = state.get("plan")
    next_step = plan.get_next_step() if plan is not None else None
    if next_step is not None:
        log_runtime(
            logger,
            logging.INFO,
            "当前批次仍有待执行步骤，继续执行",
            state=state,
            next_step_id=str(next_step.id or ""),
        )
        return "guard_step_reuse"

    if _should_summarize_after_plan_finished(state):
        log_runtime(
            logger,
            logging.INFO,
            "计划已完成，按入口合同收尾",
            state=state,
        )
        return "summarize"

    log_runtime(
        logger,
        logging.INFO,
        "当前批次步骤已全部完成，进入重规划",
        state=state,
    )
    return "replan"


def route_after_plan(state: PlannerReActLangGraphState) -> Literal[
    "guard_step_reuse", "summarize", "consolidate_memory"]:
    """规划阶段后的分支路由。"""
    plan = state.get("plan")
    contract = _get_entry_contract(state)

    if plan is None or plan.status == ExecutionStatus.COMPLETED:
        return "consolidate_memory"
    if contract is not None and contract.plan_only:
        log_runtime(
            logger,
            logging.INFO,
            "命中仅规划模式，跳过执行与总结",
            state=state,
        )
        return "consolidate_memory"

    if _has_reached_execution_limit(state):
        return "summarize"

    return "guard_step_reuse" if plan.get_next_step() is not None else "summarize"


def route_after_guard(
        state: PlannerReActLangGraphState,
) -> Literal["guard_step_reuse", "execute_step", "replan", "summarize"]:
    """复用防护后的分支路由。"""
    plan = state.get("plan")
    if plan is None:
        return "summarize"

    if bool(get_graph_control(state.get("graph_metadata")).get("step_reuse_hit")):
        return _route_after_completed_step(state)

    return "execute_step" if plan.get_next_step() is not None else "summarize"


def route_after_execute(
        state: PlannerReActLangGraphState,
) -> Literal["wait_for_human", "guard_step_reuse", "replan", "summarize"]:
    """步骤执行后，优先处理等待；否则继续当前批次或进入重规划。"""
    pending_interrupt = state.get("pending_interrupt")
    if isinstance(pending_interrupt, dict) and len(pending_interrupt) > 0:
        return "wait_for_human"

    control = get_graph_control(state.get("graph_metadata"))
    if isinstance(control.get("entry_upgrade"), dict) and control.get("entry_upgrade"):
        log_runtime(
            logger,
            logging.INFO,
            "原子动作触发入口升级，进入重规划",
            state=state,
        )
        return "replan"

    last_step = state.get("last_executed_step")
    # P3-一次性收口：统一用受控枚举归一化，避免 Enum 字符串化导致 fail-fast 失效。
    last_step_status = normalize_controlled_value(
        getattr(last_step, "status", ""),
        ExecutionStatus,
    )
    if last_step_status == ExecutionStatus.FAILED.value:
        if not _should_summarize_after_plan_finished(state):
            log_runtime(
                logger,
                logging.INFO,
                "当前步骤失败，停止后续步骤并进入重规划",
                state=state,
                last_step_id=str(getattr(last_step, "id", "") or ""),
            )
            return "replan"
        log_runtime(
            logger,
            logging.INFO,
            "当前步骤失败，按当前路径收尾总结",
            state=state,
            last_step_id=str(getattr(last_step, "id", "") or ""),
        )
        return "summarize"
    return _route_after_completed_step(state)


def route_after_wait(
        state: PlannerReActLangGraphState,
) -> Literal["guard_step_reuse", "replan", "summarize"]:
    """等待恢复后，继续当前批次剩余步骤；批次结束后再重规划。"""
    if get_graph_control(state.get("graph_metadata")).get("wait_resume_action") == "replan":
        return "replan"
    return _route_after_completed_step(state)


def route_after_replan(state: PlannerReActLangGraphState) -> Literal[
    "guard_step_reuse", "summarize", "consolidate_memory"]:
    """重规划阶段后的分支路由，与规划后逻辑保持一致。"""
    return route_after_plan(state)
