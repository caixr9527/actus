#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""收敛规则编排引擎。"""

import logging
from typing import List

from app.domain.services.runtime.contracts.runtime_logging import elapsed_ms, log_runtime
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.contracts import (
    ConvergenceDecision,
    IterationConvergenceContext,
    IterationConvergenceRule,
    MaxIterationConvergenceContext,
    MaxIterationConvergenceRule,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.rules import (
    FileFactConvergenceRule,
    GeneralFileObservationConvergenceRule,
    LoopBreakConvergenceRule,
    ResearchConvergenceRule,
    WebReadingConvergenceRule,
)


class ConvergenceEngine:
    """统一维护收敛规则顺序和日志输出。"""

    def __init__(self, *, logger: logging.Logger) -> None:
        self._logger = logger
        self._iteration_rules: List[IterationConvergenceRule] = [
            LoopBreakConvergenceRule(),
            ResearchConvergenceRule(),
            WebReadingConvergenceRule(),
            FileFactConvergenceRule(),
            GeneralFileObservationConvergenceRule(),
        ]
        self._max_iteration_rules: List[MaxIterationConvergenceRule] = [
            ResearchConvergenceRule(),
            WebReadingConvergenceRule(),
            GeneralFileObservationConvergenceRule(),
            FileFactConvergenceRule(),
        ]

    def evaluate_after_iteration(
            self,
            *,
            context: IterationConvergenceContext,
    ) -> ConvergenceDecision:
        for rule in self._iteration_rules:
            decision = rule.evaluate_after_iteration(context=context)
            if decision.should_break and decision.payload is not None:
                self._log_iteration_decision(decision=decision, context=context)
                return decision
        return ConvergenceDecision(should_break=False)

    def evaluate_max_iteration(
            self,
            *,
            context: MaxIterationConvergenceContext,
            started_at: float,
    ) -> ConvergenceDecision:
        for rule in self._max_iteration_rules:
            decision = rule.evaluate_max_iteration(context=context)
            if decision.should_break and decision.payload is not None:
                self._log_max_iteration_decision(
                    decision=decision,
                    context=context,
                    started_at=started_at,
                )
                return decision
        return ConvergenceDecision(should_break=False)

    def _log_iteration_decision(
            self,
            *,
            decision: ConvergenceDecision,
            context: IterationConvergenceContext,
    ) -> None:
        if not decision.log_message:
            return
        log_runtime(
            self._logger,
            logging.INFO,
            decision.log_message,
            step_id=str(context.step.id or ""),
            task_mode=context.task_mode,
            reason_code=decision.reason_code,
            iteration=context.iteration,
        )

    def _log_max_iteration_decision(
            self,
            *,
            decision: ConvergenceDecision,
            context: MaxIterationConvergenceContext,
            started_at: float,
    ) -> None:
        if not decision.log_message:
            return
        log_runtime(
            self._logger,
            logging.INFO,
            decision.log_message,
            step_id=str(context.step.id or ""),
            task_mode=context.task_mode,
            requested_max_tool_iterations=context.requested_max_tool_iterations,
            iteration_count=context.iteration_count,
            elapsed_ms=elapsed_ms(started_at),
        )
