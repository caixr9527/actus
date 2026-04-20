#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：工具执行插件。"""

import logging
from typing import Dict

from app.domain.models import Step
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_handlers import (
    ToolExecutionDecision,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_handlers import (
    execute_tool_with_policy,
)


async def run_executor_plugin(
        *,
        logger: logging.Logger,
        step: Step,
        function_name: str,
        normalized_function_name: str,
        function_args: Dict[str, object],
        matched_tool: BaseTool,
        tool_name: str,
        started_at: float,
) -> ToolExecutionDecision:
    """执行工具并返回执行决策。"""
    return await execute_tool_with_policy(
        logger=logger,
        step=step,
        function_name=function_name,
        normalized_function_name=normalized_function_name,
        function_args=function_args,
        matched_tool=matched_tool,
        tool_name=tool_name,
        started_at=started_at,
    )
