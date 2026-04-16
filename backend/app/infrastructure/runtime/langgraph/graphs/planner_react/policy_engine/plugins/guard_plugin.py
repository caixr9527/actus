#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：工具调用守卫插件。"""

import logging
from typing import Dict, Optional, Set

from app.domain.models import Step
from app.domain.services.tools import BaseTool
from ...execution_context import ExecutionContext
from ...execution_state import ExecutionState
from ...tool_guards import GuardDecision, evaluate_tool_guard


def run_guard_plugin(
    *,
    logger: logging.Logger,
    step: Step,
    task_mode: str,
    function_name: str,
    normalized_function_name: str,
    function_args: Dict[str, object],
    matched_tool: Optional[BaseTool],
    iteration_blocked_function_names: Set[str],
    execution_context: ExecutionContext,
    execution_state: ExecutionState,
) -> GuardDecision:
    """执行守卫插件，判断是否允许进入工具执行。"""
    return evaluate_tool_guard(
        logger=logger,
        step=step,
        task_mode=task_mode,
        function_name=function_name,
        normalized_function_name=normalized_function_name,
        function_args=function_args,
        matched_tool=matched_tool,
        iteration_blocked_function_names=iteration_blocked_function_names,
        ctx=execution_context,
        state=execution_state,
    )

