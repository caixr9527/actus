#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：统一工具执行策略引擎。"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from app.domain.models import Step, ToolResult
from app.domain.services.tools import BaseTool
from ..execution_context import ExecutionContext
from ..execution_state import ExecutionState
from ..finalizer import finalize_max_iterations, finalize_no_tool_call
from .plugins import (
    RewriteDecision,
    run_convergence_plugin,
    run_effects_plugin,
    run_executor_plugin,
    run_guard_plugin,
    run_rewrite_plugin,
)


@dataclass(slots=True)
class PolicyEvaluationResult:
    """单次工具调用策略评估结果。"""

    tool_result: ToolResult
    loop_break_reason: str
    tool_cost_ms: int


@dataclass(slots=True)
class PolicyConvergenceResult:
    """循环收敛判定结果。"""

    should_break: bool
    payload: Optional[Dict[str, Any]] = None


class ToolPolicyEngine:
    """封装 guard/handler/effects/loop_break 的统一策略入口。"""

    def __init__(self, *, logger: logging.Logger) -> None:
        self._logger = logger

    def evaluate_rewrite(
        self,
        *,
        lifecycle: Any,
        execution_context: ExecutionContext,
        execution_state: ExecutionState,
        step: Step,
    ) -> RewriteDecision:
        """统一处理调用前工具改写策略。"""
        return run_rewrite_plugin(
            lifecycle=lifecycle,
            execution_context=execution_context,
            execution_state=execution_state,
            step=step,
        )

    @staticmethod
    def _require_matched_tool(matched_tool: Optional[BaseTool]) -> BaseTool:
        if matched_tool is None:
            # guard 插件在 should_skip=False 时已经保证 matched_tool 存在。
            raise RuntimeError("matched_tool is required before executor plugin")
        return matched_tool

    async def evaluate_tool_call(
        self,
        *,
        step: Step,
        task_mode: str,
        function_name: str,
        normalized_function_name: str,
        function_args: Dict[str, Any],
        matched_tool: Optional[BaseTool],
        tool_name: str,
        browser_route_state_key: str,
        iteration_blocked_function_names: Set[str],
        execution_context: ExecutionContext,
        execution_state: ExecutionState,
        started_at: float,
    ) -> PolicyEvaluationResult:
        """统一执行 guard -> handler -> effects。"""
        guard_decision = run_guard_plugin(
            logger=self._logger,
            step=step,
            task_mode=task_mode,
            function_name=function_name,
            normalized_function_name=normalized_function_name,
            function_args=function_args,
            matched_tool=matched_tool,
            iteration_blocked_function_names=iteration_blocked_function_names,
            execution_context=execution_context,
            execution_state=execution_state,
        )

        if guard_decision.should_skip:
            loop_break_reason = guard_decision.loop_break_reason
            tool_result = guard_decision.tool_result or ToolResult(
                success=False,
                message=f"调用工具失败: {function_name}",
            )
            tool_cost_ms = 0
            tool_executed = False
        else:
            execution_decision = await run_executor_plugin(
                logger=self._logger,
                step=step,
                function_name=function_name,
                normalized_function_name=normalized_function_name,
                function_args=function_args,
                matched_tool=self._require_matched_tool(matched_tool),
                tool_name=tool_name,
                browser_route_state_key=browser_route_state_key,
                execution_context=execution_context,
                execution_state=execution_state,
                started_at=started_at,
            )
            loop_break_reason = execution_decision.loop_break_reason
            tool_cost_ms = execution_decision.tool_cost_ms
            tool_result = execution_decision.tool_result
            tool_executed = True

        effects_result = run_effects_plugin(
            logger=self._logger,
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
        return PolicyEvaluationResult(
            tool_result=effects_result.tool_result,
            loop_break_reason=effects_result.loop_break_reason,
            tool_cost_ms=tool_cost_ms,
        )

    def evaluate_iteration_convergence(
        self,
        *,
        loop_break_reason: str,
        step: Step,
        tool_result: ToolResult,
        execution_state: ExecutionState,
    ) -> PolicyConvergenceResult:
        """统一处理 loop_break 与失败上限收敛。"""
        loop_break_payload = run_convergence_plugin(
            loop_break_reason=loop_break_reason,
            step=step,
            tool_result=tool_result,
            execution_state=execution_state,
        )
        if loop_break_payload is None:
            return PolicyConvergenceResult(should_break=False, payload=None)
        return PolicyConvergenceResult(should_break=True, payload=loop_break_payload)

    def finalize_no_tool_call(
        self,
        *,
        step: Step,
        task_mode: str,
        llm_message: Dict[str, Any],
        llm_cost_ms: int,
        started_at: float,
        iteration: int,
        runtime_recent_action: Optional[Dict[str, Any]] = None,
    ):
        """无工具调用分支统一出口。"""
        return finalize_no_tool_call(
            logger=self._logger,
            step=step,
            task_mode=task_mode,
            llm_message=llm_message,
            llm_cost_ms=llm_cost_ms,
            started_at=started_at,
            iteration=iteration,
            runtime_recent_action=runtime_recent_action,
        )

    def finalize_max_iterations(
        self,
        *,
        step: Step,
        task_mode: str,
        llm_message: Dict[str, Any],
        started_at: float,
        requested_max_tool_iterations: int,
        iteration_count: int,
        runtime_recent_action: Optional[Dict[str, Any]] = None,
        step_file_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """达到最大轮次统一出口。"""
        return finalize_max_iterations(
            logger=self._logger,
            step=step,
            task_mode=task_mode,
            llm_message=llm_message,
            started_at=started_at,
            requested_max_tool_iterations=requested_max_tool_iterations,
            iteration_count=iteration_count,
            runtime_recent_action=runtime_recent_action,
            step_file_context=step_file_context,
        )
