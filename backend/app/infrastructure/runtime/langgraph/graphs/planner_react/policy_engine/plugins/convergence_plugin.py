#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：循环收敛插件。"""

from typing import Dict, Optional

from app.domain.models import Step, ToolResult
from app.domain.services.workspace_runtime.policies import build_loop_break_payload as _build_loop_break_payload
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import (
    ExecutionState,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.loop_breaks import build_loop_break_result
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_effects import (
    reached_tool_failure_limit,
)


def run_convergence_plugin(
        *,
        loop_break_reason: str,
        step: Step,
        tool_result: ToolResult,
        execution_state: ExecutionState,
) -> Optional[Dict[str, object]]:
    """判断是否应当结束当前步骤循环。"""
    loop_break_payload = build_loop_break_result(
        loop_break_reason=loop_break_reason,
        step=step,
        tool_result=tool_result,
        runtime_recent_action=execution_state.runtime_recent_action,
    )
    if loop_break_payload is not None:
        return loop_break_payload
    if reached_tool_failure_limit(execution_state):
        return _build_loop_break_payload(
            step=step,
            blocker="连续工具调用失败次数过多，当前步骤已停止继续重试。",
            next_hint="请检查参数、改换工具，或将当前步骤拆小后再执行。",
            runtime_recent_action=execution_state.runtime_recent_action,
        )
    return None
