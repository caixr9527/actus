#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：统一工具执行策略引擎。"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.domain.models import Step, ToolResult
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_context import (
    ExecutionContext,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import (
    ExecutionState,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintInput
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.engine import ConstraintEngine
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_REPEAT_TOOL_CALL
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_REPEAT_TOOL_CALL_SUCCESS_FALLBACK
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.finalizer import (
    finalize_max_iterations,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.finalizer import (
    finalize_no_tool_call,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.policy_engine.plugins.convergence_plugin import run_convergence_plugin
from app.infrastructure.runtime.langgraph.graphs.planner_react.policy_engine.plugins.effects_plugin import run_effects_plugin
from app.infrastructure.runtime.langgraph.graphs.planner_react.policy_engine.plugins.effects_plugin import run_preinvoke_effects_plugin
from app.infrastructure.runtime.langgraph.graphs.planner_react.policy_engine.plugins.effects_plugin import run_rewrite_effects_plugin
from app.infrastructure.runtime.langgraph.graphs.planner_react.policy_engine.plugins.executor_plugin import run_executor_plugin
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_argument_normalizers import (
    normalize_tool_execution_args,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_handlers import (
    build_repeat_success_fallback_result,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_resolution import (
    resolve_matched_tool,
)


@dataclass(slots=True)
class PolicyEvaluationResult:
    """单次工具调用策略评估结果。"""

    tool_result: ToolResult
    loop_break_reason: str
    tool_cost_ms: int
    rewrite_reason: str = ""
    rewrite_metadata: Dict[str, Any] = field(default_factory=dict)
    final_function_name: str = ""
    final_normalized_function_name: str = ""
    final_function_args: Dict[str, Any] = field(default_factory=dict)
    executed_function_args: Dict[str, Any] = field(default_factory=dict)
    final_matched_tool: Optional[BaseTool] = None
    final_tool_name: str = ""


@dataclass(slots=True)
class PolicyConvergenceResult:
    """循环收敛判定结果。"""

    should_break: bool
    payload: Optional[Dict[str, Any]] = None


class ToolPolicyEngine:
    """封装 constraint_engine/executor/effects/loop_break 的统一策略入口。

    实现语义：
    - 先通过 `ConstraintEngine` 决定当前轮最终应该执行哪个函数；
    - 再由 executor 真正调用工具；
    - 最后统一进入 effects / convergence，更新执行态并判断是否收敛。
    """

    def __init__(self, *, logger: logging.Logger) -> None:
        self._logger = logger
        self._constraint_engine = ConstraintEngine(logger=logger)

    @staticmethod
    def _require_matched_tool(matched_tool: Optional[BaseTool]) -> BaseTool:
        if matched_tool is None:
            # allow 分支下约束引擎已验证工具可执行。
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
            runtime_tools: Optional[List[BaseTool]],
            browser_route_state_key: str,
            iteration_blocked_function_names: Set[str],
            execution_context: ExecutionContext,
            execution_state: ExecutionState,
            started_at: float,
    ) -> PolicyEvaluationResult:
        """统一执行 `constraint guard -> executor -> effects`。

        业务语义：
        - 这是 tools 主循环进入真实工具调用前的唯一策略入口；
        - 返回值中的 `final_function_*` 是已经过 rewrite 收口后的最终执行目标；
        - `tool_result` 与 `loop_break_reason` 已经过 effects 域统一归并，可直接供 tools 主循环消费。
        """
        engine_result = self._constraint_engine.evaluate_guard(
            constraint_input=ConstraintInput(
                step=step,
                task_mode=task_mode,
                function_name=function_name,
                normalized_function_name=normalized_function_name,
                function_args=dict(function_args or {}),
                matched_tool=matched_tool,
                iteration_blocked_function_names=iteration_blocked_function_names,
                execution_context=execution_context,
                execution_state=execution_state,
                runtime_tools=list(runtime_tools or []),
            ),
            logger=self._logger,
            allow_rewrite=True,
        )
        final_function_name = str(engine_result.final_function_name or function_name or "")
        final_normalized_function_name = str(
            engine_result.final_normalized_function_name or normalized_function_name or ""
        ).strip().lower()
        final_function_args = dict(engine_result.final_function_args or function_args or {})
        normalized_final_function_args = normalize_tool_execution_args(
            normalized_function_name=final_normalized_function_name,
            function_args=final_function_args,
        )
        if engine_result.rewrite_applied:
            run_rewrite_effects_plugin(
                rewrite_reason=str(engine_result.rewrite_reason or ""),
                rewrite_metadata=dict(engine_result.rewrite_metadata or {}),
                execution_state=execution_state,
            )
        log_runtime(
            self._logger,
            logging.INFO,
            "工具策略评估完成",
            step_id=str(getattr(step, "id", "") or ""),
            function_name=str(function_name or ""),
            final_function_name=final_function_name,
            task_mode=task_mode,
            final_action=str(engine_result.constraint_decision.action or ""),
            winning_policy=str(engine_result.winning_policy or ""),
            reason_code=str(engine_result.constraint_decision.reason_code or ""),
            rewrite_applied=bool(engine_result.rewrite_applied),
            tool_call_fingerprint=str(engine_result.tool_call_fingerprint or ""),
        )
        guard_decision = engine_result.constraint_decision
        guard_reason_code = str(guard_decision.reason_code or "")

        if guard_decision.action == "allow":
            # P3-一次性收口：调用前计数只对最终真实执行目标入账一次。
            run_preinvoke_effects_plugin(
                normalized_function_name=final_normalized_function_name,
                function_args=normalized_final_function_args,
                execution_state=execution_state,
            )
            resolved_matched_tool = resolve_matched_tool(
                function_name=final_function_name,
                fallback_tool=matched_tool,
                runtime_tools=runtime_tools,
            )
            execution_decision = await run_executor_plugin(
                logger=self._logger,
                step=step,
                function_name=final_function_name,
                normalized_function_name=final_normalized_function_name,
                function_args=normalized_final_function_args,
                matched_tool=self._require_matched_tool(resolved_matched_tool),
                tool_name=str(getattr(resolved_matched_tool, "name", "") or ""),
                started_at=started_at,
            )
            loop_break_reason = execution_decision.loop_break_reason
            tool_cost_ms = execution_decision.tool_cost_ms
            tool_result = execution_decision.tool_result
            final_function_args = dict(execution_decision.executed_function_args or normalized_final_function_args)
            tool_executed = True
        else:
            resolved_matched_tool = resolve_matched_tool(
                function_name=final_function_name,
                fallback_tool=matched_tool,
                runtime_tools=runtime_tools,
            )
            payload = guard_decision.tool_result_payload
            loop_break_reason = str(guard_decision.loop_break_reason or "")
            tool_result = ToolResult(
                success=False,
                message=str(payload.message if payload is not None else f"调用工具失败: {final_function_name}"),
                data=dict(payload.data or {}) if payload is not None else {},
            )
            # 重复调用阻断时，若最近成功结果可复用，则直接转换为成功收敛，避免有效信息丢失。
            if str(guard_decision.reason_code or "") == REASON_REPEAT_TOOL_CALL:
                repeated_success_result = build_repeat_success_fallback_result(
                    function_name=final_normalized_function_name,
                    function_args=normalized_final_function_args,
                    current_tool_fingerprint=execution_state.last_tool_fingerprint,
                    last_successful_tool_call=execution_state.last_successful_tool_call,
                    last_successful_tool_fingerprint=execution_state.last_successful_tool_fingerprint,
                )
                if repeated_success_result is not None:
                    loop_break_reason = REASON_REPEAT_TOOL_CALL_SUCCESS_FALLBACK
                    tool_result = repeated_success_result
            tool_cost_ms = 0
            final_function_args = dict(normalized_final_function_args)
            tool_executed = False
            log_runtime(
                self._logger,
                logging.INFO,
                "工具策略阻断执行，改写为虚拟失败结果",
                step_id=str(getattr(step, "id", "") or ""),
                function_name=final_function_name,
                task_mode=task_mode,
                reason_code=str(guard_decision.reason_code or ""),
                block_mode=str(guard_decision.block_mode or ""),
                loop_break_reason=loop_break_reason,
            )

        effects_result = run_effects_plugin(
            logger=self._logger,
            step=step,
            function_name=final_function_name,
            normalized_function_name=final_normalized_function_name,
            function_args=final_function_args,
            tool_result=tool_result,
            loop_break_reason=loop_break_reason,
            guard_reason_code=guard_reason_code,
            browser_route_state_key=browser_route_state_key,
            execution_context=execution_context,
            execution_state=execution_state,
            tool_executed=tool_executed,
        )
        return PolicyEvaluationResult(
            tool_result=effects_result.tool_result,
            loop_break_reason=effects_result.loop_break_reason,
            tool_cost_ms=tool_cost_ms,
            rewrite_reason=(
                str(engine_result.rewrite_reason or "")
                if engine_result.rewrite_applied
                else ""
            ),
            rewrite_metadata=(
                dict(engine_result.rewrite_metadata or {})
                if engine_result.rewrite_applied
                else {}
            ),
            final_function_name=final_function_name,
            final_normalized_function_name=final_normalized_function_name,
            final_function_args=final_function_args,
            executed_function_args=final_function_args,
            final_matched_tool=resolved_matched_tool,
            final_tool_name=str(getattr(resolved_matched_tool, "name", "") or ""),
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
