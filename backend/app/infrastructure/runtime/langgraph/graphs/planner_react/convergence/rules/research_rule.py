#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究证据收敛规则。"""

from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.contracts import (
    ConvergenceDecision,
    IterationConvergenceContext,
    IterationConvergenceRule,
    MaxIterationConvergenceContext,
    MaxIterationConvergenceRule,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.judge.research_convergence import (
    ResearchConvergenceJudge,
)


class ResearchConvergenceRule(IterationConvergenceRule, MaxIterationConvergenceRule):
    name = "research"
    log_message = "研究证据满足，提前收敛步骤"
    max_iteration_log_message = "达到最大工具轮次但研究摘要证据可用，按阶段性结果收敛"

    def __init__(self) -> None:
        self._judge = ResearchConvergenceJudge()

    def evaluate_after_iteration(
            self,
            *,
            context: IterationConvergenceContext,
    ) -> ConvergenceDecision:
        decision = self._judge.evaluate_after_iteration(
            step=context.step,
            task_mode=context.task_mode,
            recent_function_name=context.recent_function_name,
            execution_state=context.execution_state,
        )
        decision.rule_name = self.name
        decision.log_message = self.log_message if decision.should_break else ""
        return decision

    def evaluate_max_iteration(
            self,
            *,
            context: MaxIterationConvergenceContext,
    ) -> ConvergenceDecision:
        payload = ResearchConvergenceJudge.build_max_iteration_convergence_payload(
            step=context.step,
            task_mode=context.task_mode,
            runtime_recent_action=context.runtime_recent_action,
        )
        if payload is None:
            return ConvergenceDecision(should_break=False, rule_name=self.name)
        return ConvergenceDecision(
            should_break=True,
            payload=payload,
            reason_code="research_max_iteration_snippet_fallback",
            rule_name=self.name,
            log_message=self.max_iteration_log_message,
        )
