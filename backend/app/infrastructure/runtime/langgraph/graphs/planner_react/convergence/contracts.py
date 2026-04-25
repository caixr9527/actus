#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""收敛层统一合同。

执行层只负责提供上下文并消费最终决策；具体有哪些收敛规则、规则顺序和日志文案
都集中在 convergence 层维护，避免工具循环重新长出业务分支。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.domain.models import Step, ToolResult
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import (
    ExecutionState,
)


@dataclass(slots=True)
class ConvergenceDecision:
    """统一的收敛决策结果。"""

    should_break: bool
    payload: Optional[Dict[str, Any]] = None
    reason_code: str = ""
    rule_name: str = ""
    log_message: str = ""


@dataclass(slots=True)
class IterationConvergenceContext:
    """单轮工具调用后的收敛输入。"""

    step: Step
    task_mode: str
    iteration: int
    recent_function_name: str
    function_args: Dict[str, Any]
    tool_result: ToolResult
    loop_break_reason: str
    execution_state: ExecutionState
    step_file_context: Dict[str, Any]


@dataclass(slots=True)
class MaxIterationConvergenceContext:
    """达到最大工具轮次后的证据收敛输入。"""

    step: Step
    task_mode: str
    requested_max_tool_iterations: int
    iteration_count: int
    runtime_recent_action: Optional[Dict[str, Any]] = None
    step_file_context: Dict[str, Any] = field(default_factory=dict)


class IterationConvergenceRule(ABC):
    """每轮工具调用后的收敛规则接口。"""

    name: str

    @abstractmethod
    def evaluate_after_iteration(
            self,
            *,
            context: IterationConvergenceContext,
    ) -> ConvergenceDecision:
        raise NotImplementedError


class MaxIterationConvergenceRule(ABC):
    """最大工具轮次后的证据收敛规则接口。"""

    name: str

    @abstractmethod
    def evaluate_max_iteration(
            self,
            *,
            context: MaxIterationConvergenceContext,
    ) -> ConvergenceDecision:
        raise NotImplementedError
