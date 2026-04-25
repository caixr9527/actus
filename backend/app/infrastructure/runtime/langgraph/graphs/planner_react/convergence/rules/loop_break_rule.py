#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行停止类收敛规则。"""

from app.domain.services.workspace_runtime.policies import build_loop_break_payload as _build_loop_break_payload
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.contracts import (
    ConvergenceDecision,
    IterationConvergenceContext,
    IterationConvergenceRule,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.loop_breaks import build_loop_break_result
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_effects import (
    reached_tool_failure_limit,
)


class LoopBreakConvergenceRule(IterationConvergenceRule):
    """负责 loop_break 与连续失败上限，不代表证据成功收敛。"""

    name = "loop_break"

    def evaluate_after_iteration(
            self,
            *,
            context: IterationConvergenceContext,
    ) -> ConvergenceDecision:
        loop_break_payload = build_loop_break_result(
            loop_break_reason=context.loop_break_reason,
            step=context.step,
            tool_result=context.tool_result,
            runtime_recent_action=context.execution_state.runtime_recent_action,
        )
        if loop_break_payload is not None:
            return ConvergenceDecision(
                should_break=True,
                payload=loop_break_payload,
                reason_code=context.loop_break_reason,
                rule_name=self.name,
            )
        if reached_tool_failure_limit(context.execution_state):
            return ConvergenceDecision(
                should_break=True,
                payload=_build_loop_break_payload(
                    step=context.step,
                    blocker="连续工具调用失败次数过多，当前步骤已停止继续重试。",
                    next_hint="请检查参数、改换工具，或将当前步骤拆小后再执行。",
                    runtime_recent_action=context.execution_state.runtime_recent_action,
                ),
                reason_code="tool_failure_limit_reached",
                rule_name=self.name,
            )
        return ConvergenceDecision(should_break=False, rule_name=self.name)
