#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：工具执行结果归并插件。"""

import logging
from typing import Dict

from app.domain.models import Step, ToolResult
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution_context import ExecutionContext
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution_state import ExecutionState
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_effects import ToolEffectsResult
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_effects import apply_tool_preinvoke_effects
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_effects import apply_rewrite_effects
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_effects import apply_tool_result_effects


def run_effects_plugin(
        *,
        logger: logging.Logger,
        step: Step,
        function_name: str,
        normalized_function_name: str,
        function_args: Dict[str, object],
        tool_result: ToolResult,
        loop_break_reason: str,
        guard_reason_code: str = "",
        browser_route_state_key: str,
        execution_context: ExecutionContext,
        execution_state: ExecutionState,
        tool_executed: bool = True,
) -> ToolEffectsResult:
    """effects 插件统一入口。

    业务含义：
    - 保持 `ToolPolicyEngine` 不直接感知执行态细节；
    - 由这里把 executor/guard 产出的结果统一交给 `tool_effects` 写回执行状态。
    """
    return apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name=function_name,
        normalized_function_name=normalized_function_name,
        function_args=function_args,
        tool_result=tool_result,
        loop_break_reason=loop_break_reason,
        guard_reason_code=guard_reason_code,
        browser_route_state_key=browser_route_state_key,
        execution_context=execution_context,
        execution_state=execution_state,
        tool_executed=tool_executed,
    )


def run_preinvoke_effects_plugin(
        *,
        normalized_function_name: str,
        function_args: Dict[str, object],
        execution_state: ExecutionState,
) -> None:
    """调用前计数入账统一入口。

    注意：
    - 只对最终实际执行目标调用一次；
    - 不允许在 tools 主流程或 constraint engine 中散落写重复计数。
    """
    apply_tool_preinvoke_effects(
        normalized_function_name=normalized_function_name,
        function_args=function_args,
        execution_state=execution_state,
    )


def run_rewrite_effects_plugin(
        *,
        rewrite_reason: str,
        rewrite_metadata: Dict[str, object],
        execution_state: ExecutionState,
) -> None:
    """rewrite 状态写入统一入口。

    业务含义：
    - 约束层只返回 rewrite 决策，不写状态；
    - rewrite 对执行态的影响统一在 effects 域落账。
    """
    apply_rewrite_effects(
        rewrite_reason=rewrite_reason,
        rewrite_metadata=rewrite_metadata,
        execution_state=execution_state,
    )
