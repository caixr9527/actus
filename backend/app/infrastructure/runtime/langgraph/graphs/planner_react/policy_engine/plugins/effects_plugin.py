#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：工具执行结果归并插件。"""

import logging
from typing import Dict

from app.domain.models import Step, ToolResult
from ...execution_context import ExecutionContext
from ...execution_state import ExecutionState
from ...tool_effects import ToolEffectsResult, apply_tool_result_effects


def run_effects_plugin(
    *,
    logger: logging.Logger,
    step: Step,
    function_name: str,
    normalized_function_name: str,
    function_args: Dict[str, object],
    tool_result: ToolResult,
    loop_break_reason: str,
    browser_route_state_key: str,
    execution_context: ExecutionContext,
    execution_state: ExecutionState,
    tool_executed: bool = True,
) -> ToolEffectsResult:
    """归并工具执行结果，更新执行态。"""
    return apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name=function_name,
        normalized_function_name=normalized_function_name,
        function_args=function_args,
        tool_result=tool_result,
        loop_break_reason=loop_break_reason,
        browser_route_state_key=browser_route_state_key,
        execution_context=execution_context,
        execution_state=execution_state,
        tool_executed=tool_executed,
    )
