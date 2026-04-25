#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文件事实收敛规则。"""

from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.contracts import (
    ConvergenceDecision,
    IterationConvergenceContext,
    IterationConvergenceRule,
    MaxIterationConvergenceContext,
    MaxIterationConvergenceRule,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.judge import ConvergenceJudge


class FileFactConvergenceRule(IterationConvergenceRule, MaxIterationConvergenceRule):
    name = "file_fact"
    log_message = "关键事实满足，提前收敛步骤"
    max_iteration_log_message = "达到最大工具轮次但关键事实已满足，按成功收敛"

    def __init__(self) -> None:
        self._judge = ConvergenceJudge()

    def evaluate_after_iteration(
            self,
            *,
            context: IterationConvergenceContext,
    ) -> ConvergenceDecision:
        decision = self._judge.evaluate_file_processing_progress(
            step=context.step,
            task_mode=context.task_mode,
            recent_function_name=context.recent_function_name,
            function_args=context.function_args,
            tool_result_data=context.tool_result.data,
            tool_result_success=bool(context.tool_result.success),
            step_file_context=context.step_file_context,
            runtime_recent_action=context.execution_state.runtime_recent_action,
        )
        decision.rule_name = self.name
        decision.log_message = self.log_message if decision.should_break else ""
        return decision

    def evaluate_max_iteration(
            self,
            *,
            context: MaxIterationConvergenceContext,
    ) -> ConvergenceDecision:
        payload = ConvergenceJudge.build_max_iteration_convergence_payload(
            step=context.step,
            task_mode=context.task_mode,
            runtime_recent_action=context.runtime_recent_action,
            step_file_context=dict(context.step_file_context or {}),
        )
        if payload is None:
            return ConvergenceDecision(should_break=False, rule_name=self.name)
        return ConvergenceDecision(
            should_break=True,
            payload=payload,
            reason_code="file_processing_facts_ready",
            rule_name=self.name,
            log_message=self.max_iteration_log_message,
        )
