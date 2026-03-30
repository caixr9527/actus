#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 路由判断。"""

import logging
from typing import Literal

from app.domain.models import ExecutionStatus
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState

logger = logging.getLogger(__name__)


def route_after_plan(state: PlannerReActLangGraphState) -> Literal["execute_step", "summarize", "consolidate_memory"]:
    """规划阶段后的分支路由。"""
    plan = state.get("plan")

    if plan is None or plan.status == ExecutionStatus.COMPLETED:
        return "consolidate_memory"

    if state.get("execution_count", 0) >= state.get("max_execution_steps", 20):
        logger.warning("执行次数达到上限，提前进入总结阶段")
        return "summarize"

    return "execute_step" if plan.get_next_step() is not None else "summarize"


def route_after_replan(state: PlannerReActLangGraphState) -> Literal["execute_step", "summarize", "consolidate_memory"]:
    """重规划阶段后的分支路由，与规划后逻辑保持一致。"""
    return route_after_plan(state)
