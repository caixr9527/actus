#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""general 文件观察收敛规则。"""

from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.contracts import (
    ConvergenceDecision,
    IterationConvergenceContext,
    IterationConvergenceRule,
    MaxIterationConvergenceContext,
    MaxIterationConvergenceRule,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.judge.general_convergence import (
    GeneralConvergenceJudge,
)


class GeneralFileObservationConvergenceRule(IterationConvergenceRule, MaxIterationConvergenceRule):
    name = "general_file_observation"
    log_message = "general 文件观察事实满足，提前收敛步骤"
    max_iteration_log_message = "达到最大工具轮次但 general 文件观察事实可用，按成功收敛"

    def __init__(self) -> None:
        self._judge = GeneralConvergenceJudge()

    def evaluate_after_iteration(
            self,
            *,
            context: IterationConvergenceContext,
    ) -> ConvergenceDecision:
        decision = self._judge.evaluate_after_iteration(
            step=context.step,
            task_mode=context.task_mode,
            runtime_recent_action=context.execution_state.runtime_recent_action,
            iteration=context.iteration,
        )
        decision.rule_name = self.name
        decision.log_message = self.log_message if decision.should_break else ""
        return decision

    def evaluate_max_iteration(
            self,
            *,
            context: MaxIterationConvergenceContext,
    ) -> ConvergenceDecision:
        payload = GeneralConvergenceJudge.build_max_iteration_convergence_payload(
            step=context.step,
            task_mode=context.task_mode,
            runtime_recent_action=context.runtime_recent_action,
        )
        if payload is None:
            return ConvergenceDecision(should_break=False, rule_name=self.name)
        return ConvergenceDecision(
            should_break=True,
            payload=payload,
            reason_code="general_file_observation_ready",
            rule_name=self.name,
            log_message=self.max_iteration_log_message,
        )
