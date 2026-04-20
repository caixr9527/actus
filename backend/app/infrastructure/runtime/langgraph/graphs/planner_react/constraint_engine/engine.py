#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束引擎。"""

import json
import logging
from dataclasses import dataclass

from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.workspace_runtime.policies import build_tool_fingerprint as _build_tool_fingerprint
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintDecision
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintEngineResult
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintInput
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintPolicyTraceEntry
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintToolResultPayload
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import build_default_external_signals_snapshot
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.artifact_policy import evaluate_artifact_policy
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.final_delivery_policy import evaluate_final_delivery_policy
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.human_wait_policy import evaluate_human_wait_policy
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.repeat_loop_policy import evaluate_repeat_loop_policy
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.research_route_policy import build_research_route_rewrite_decision
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.research_route_policy import evaluate_research_route_policy
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.task_mode_policy import evaluate_task_mode_policy
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_ALLOW
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_CONSTRAINT_ENGINE_ERROR
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_RESEARCH_ROUTE_FETCH_REQUIRED
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_REWRITE_CHAIN_BLOCKED
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_resolution import (
    resolve_matched_tool,
)


@dataclass(slots=True)
class PolicyEvaluationError(Exception):
    """策略评估异常，携带策略名以便 fail-closed 日志追踪。"""

    policy_name: str
    cause: Exception

    def __str__(self) -> str:
        return str(self.cause)


class ConstraintEngine:
    """执行前约束评估与改写唯一入口。

    实现语义：
    - 顺序执行固定 policy 列表，按 `block > rewrite > allow` 聚合；
    - 命中 rewrite 时，会用 rewrite 后的函数名/参数再做一次完整约束评估；
    - 引擎只返回决策与可追溯元数据，不直接执行业务工具，也不写执行状态。
    """

    def __init__(self, *, logger: logging.Logger) -> None:
        self._logger = logger

    def evaluate_guard(
            self,
            *,
            constraint_input: ConstraintInput,
            logger: logging.Logger | None = None,
            allow_rewrite: bool = True,
    ) -> ConstraintEngineResult:
        """评估当前轮工具调用是否允许执行，并返回最终执行目标。

        调用方约定：
        - `ToolPolicyEngine` 在真正执行工具前调用这里；
        - 若返回 `rewrite`，本方法内部已经完成 rewrite 后二次评估，并将最终函数名/参数写入结果；
        - 任意异常都 fail-closed，统一转换为 `constraint_engine_error`。
        """
        active_logger = logger or self._logger
        normalized_input = ConstraintInput(
            step=constraint_input.step,
            task_mode=constraint_input.task_mode,
            function_name=constraint_input.function_name,
            normalized_function_name=constraint_input.normalized_function_name,
            function_args=dict(constraint_input.function_args or {}),
            matched_tool=constraint_input.matched_tool,
            iteration_blocked_function_names=set(constraint_input.iteration_blocked_function_names or set()),
            execution_context=constraint_input.execution_context,
            execution_state=constraint_input.execution_state,
            # 统一补齐默认快照字段，避免各 policy 里重复判空。
            external_signals_snapshot=build_default_external_signals_snapshot(
                constraint_input.external_signals_snapshot
            ),
            runtime_tools=list(constraint_input.runtime_tools or []),
        )
        policy_name_for_error = "constraint_engine"
        try:
            engine_result = self._evaluate_guard_with_optional_rewrite(
                constraint_input=normalized_input,
                logger=active_logger,
                allow_rewrite=allow_rewrite,
            )
        except Exception as exc:
            actual_error = exc
            if isinstance(exc, PolicyEvaluationError):
                policy_name_for_error = str(exc.policy_name or "constraint_engine")
                actual_error = exc.cause
            # 约束层异常必须 fail-closed，避免约束绕过。
            fallback_message = "约束引擎异常，已停止当前工具调用，请调整步骤或重试"
            tool_call_fingerprint = _build_tool_fingerprint(
                str(normalized_input.normalized_function_name or "").strip().lower(),
                dict(normalized_input.function_args or {}),
            )
            log_runtime(
                active_logger,
                logging.ERROR,
                "约束引擎评估异常，已执行失败安全拦截",
                step_id=str(getattr(normalized_input.step, "id", "") or ""),
                function_name=str(normalized_input.function_name or ""),
                task_mode=str(normalized_input.task_mode or ""),
                policy_name=policy_name_for_error,
                tool_call_fingerprint=tool_call_fingerprint,
                error_type=actual_error.__class__.__name__,
                error_message=str(actual_error),
                exc_info=True,
            )
            fallback_decision = ConstraintDecision(
                action="block",
                reason_code=REASON_CONSTRAINT_ENGINE_ERROR,
                block_mode="hard_block_break",
                loop_break_reason=REASON_CONSTRAINT_ENGINE_ERROR,
                tool_result_payload=ConstraintToolResultPayload(
                    success=False,
                    message=fallback_message,
                ),
                message_for_model=fallback_message,
            )
            engine_result = ConstraintEngineResult(
                constraint_decision=fallback_decision,
                final_function_name=str(normalized_input.function_name or ""),
                final_normalized_function_name=str(normalized_input.normalized_function_name or "").strip().lower(),
                final_function_args=dict(normalized_input.function_args or {}),
                policy_trace=[
                    ConstraintPolicyTraceEntry(
                        policy_name=policy_name_for_error,
                        action="block",
                        reason_code=REASON_CONSTRAINT_ENGINE_ERROR,
                    )
                ],
                winning_policy=policy_name_for_error,
                tool_call_fingerprint=tool_call_fingerprint,
            )
        self._log_final_decision(active_logger, normalized_input, engine_result)
        return engine_result

    def _evaluate_guard_with_optional_rewrite(
            self,
            *,
            constraint_input: ConstraintInput,
            logger: logging.Logger,
            allow_rewrite: bool,
    ) -> ConstraintEngineResult:
        """执行一次主评估，并在需要时对 rewrite 目标做二次 guard 校验。"""
        first_pass = self._evaluate_policies_once(
            constraint_input=constraint_input,
            allow_rewrite=allow_rewrite,
        )
        first_decision = first_pass["decision"]
        first_trace = list(first_pass["trace"])
        first_winning_policy = str(first_pass.get("winning_policy") or "")
        if (not allow_rewrite) or first_decision.action != "rewrite":
            return ConstraintEngineResult(
                constraint_decision=first_decision,
                final_function_name=str(constraint_input.function_name or ""),
                final_normalized_function_name=str(constraint_input.normalized_function_name or "").strip().lower(),
                final_function_args=dict(constraint_input.function_args or {}),
                policy_trace=first_trace,
                winning_policy=first_winning_policy,
                tool_call_fingerprint=_build_tool_fingerprint(
                    str(constraint_input.normalized_function_name or "").strip().lower(),
                    dict(constraint_input.function_args or {}),
                ),
            )

        rewrite_target = dict(first_decision.rewrite_target or {})
        rewrite_function_name = str(rewrite_target.get("function_name") or "").strip() or str(
            constraint_input.function_name or ""
        ).strip()
        rewrite_normalized_name = str(rewrite_target.get("normalized_function_name") or "").strip().lower() or str(
            rewrite_function_name or ""
        ).strip().lower()
        rewrite_function_args = dict(rewrite_target.get("function_args") or {})
        rewritten_matched_tool = resolve_matched_tool(
            function_name=rewrite_function_name,
            fallback_tool=constraint_input.matched_tool,
            runtime_tools=list(constraint_input.runtime_tools or []),
        )
        rewritten_input = ConstraintInput(
            step=constraint_input.step,
            task_mode=constraint_input.task_mode,
            function_name=rewrite_function_name,
            normalized_function_name=rewrite_normalized_name,
            function_args=rewrite_function_args,
            matched_tool=rewritten_matched_tool,
            iteration_blocked_function_names=set(constraint_input.iteration_blocked_function_names or set()),
            execution_context=constraint_input.execution_context,
            execution_state=constraint_input.execution_state,
            external_signals_snapshot=dict(constraint_input.external_signals_snapshot or {}),
            runtime_tools=list(constraint_input.runtime_tools or []),
        )
        second_pass = self._evaluate_policies_once(
            constraint_input=rewritten_input,
            allow_rewrite=False,
        )
        log_runtime(
            logger,
            logging.INFO,
            "执行约束改写二次评估完成",
            step_id=str(getattr(constraint_input.step, "id", "") or ""),
            function_name=str(constraint_input.function_name or ""),
            final_function_name=rewrite_function_name,
            rewrite_reason=str(first_decision.reason_code or ""),
            winning_policy=str(second_pass.get("winning_policy") or ""),
        )
        second_decision = second_pass["decision"]
        second_trace = list(second_pass["trace"])
        if second_decision.action == "rewrite":
            second_decision = ConstraintDecision(
                action="block",
                reason_code=REASON_REWRITE_CHAIN_BLOCKED,
                block_mode="hard_block_break",
                loop_break_reason=REASON_REWRITE_CHAIN_BLOCKED,
                tool_result_payload=ConstraintToolResultPayload(
                    success=False,
                    message="同一轮约束改写仅允许一次，已阻断重复改写。",
                ),
                message_for_model="同一轮约束改写仅允许一次，已阻断重复改写。",
            )
            second_trace.append(
                ConstraintPolicyTraceEntry(
                    policy_name="constraint_engine",
                    action="block",
                    reason_code=REASON_REWRITE_CHAIN_BLOCKED,
                )
            )
            winning_policy = "constraint_engine"
        else:
            winning_policy = str(second_pass.get("winning_policy") or "")
        return ConstraintEngineResult(
            constraint_decision=second_decision,
            final_function_name=rewrite_function_name,
            final_normalized_function_name=rewrite_normalized_name,
            final_function_args=rewrite_function_args,
            policy_trace=[*first_trace, *second_trace],
            winning_policy=winning_policy or first_winning_policy,
            tool_call_fingerprint=_build_tool_fingerprint(
                rewrite_normalized_name,
                rewrite_function_args,
            ),
            rewrite_applied=True,
            rewrite_reason=str(first_decision.reason_code or ""),
            rewrite_metadata=dict(first_decision.metadata or {}),
        )

    def _evaluate_policies_once(
            self,
            *,
            constraint_input: ConstraintInput,
            allow_rewrite: bool,
    ) -> dict:
        """按固定顺序执行单轮 policy，返回该轮聚合结果。

        注意：
        - 这里不处理 executor/effects/convergence；
        - 若 `allow_rewrite=False`，research rewrite 候选会被转换成显式 block，避免绕过约束。
        """
        trace: list[ConstraintPolicyTraceEntry] = []
        winning_policy = ""
        for policy_name, evaluator in (
                ("task_mode_policy", evaluate_task_mode_policy),
                ("artifact_policy", evaluate_artifact_policy),
                ("final_delivery_policy", evaluate_final_delivery_policy),
                ("human_wait_policy", evaluate_human_wait_policy),
                ("research_route_policy", evaluate_research_route_policy),
                ("repeat_loop_policy", evaluate_repeat_loop_policy),
        ):
            try:
                decision = evaluator(constraint_input)
            except Exception as exc:
                raise PolicyEvaluationError(policy_name=policy_name, cause=exc) from exc
            if policy_name == "research_route_policy" and decision is None:
                rewrite_decision = build_research_route_rewrite_decision(constraint_input=constraint_input)
                if rewrite_decision is not None:
                    if allow_rewrite:
                        log_runtime(
                            self._logger,
                            logging.INFO,
                            "执行约束命中改写候选",
                            step_id=str(getattr(constraint_input.step, "id", "") or ""),
                            function_name=str(constraint_input.function_name or ""),
                            final_function_name=str(
                                (rewrite_decision.rewrite_target or {}).get("function_name") or ""
                            ),
                            reason_code=str(rewrite_decision.reason_code or ""),
                            policy_name=policy_name,
                        )
                        decision = rewrite_decision
                    else:
                        log_runtime(
                            self._logger,
                            logging.INFO,
                            "执行约束禁止改写并转为阻断",
                            step_id=str(getattr(constraint_input.step, "id", "") or ""),
                            function_name=str(constraint_input.function_name or ""),
                            reason_code=REASON_RESEARCH_ROUTE_FETCH_REQUIRED,
                            policy_name=policy_name,
                        )
                        decision = ConstraintDecision(
                            action="block",
                            reason_code=REASON_RESEARCH_ROUTE_FETCH_REQUIRED,
                            block_mode="hard_block_break",
                            loop_break_reason=REASON_RESEARCH_ROUTE_FETCH_REQUIRED,
                            tool_result_payload=ConstraintToolResultPayload(
                                success=False,
                                message="当前步骤需要先执行 fetch_page 读取目标页面，已阻断原始 search_web 调用。",
                            ),
                            message_for_model="当前步骤需要先执行 fetch_page 读取目标页面，已阻断原始 search_web 调用。",
                            metadata=dict(rewrite_decision.metadata or {}),
                        )
            if decision is None:
                trace.append(
                    ConstraintPolicyTraceEntry(
                        policy_name=policy_name,
                        action="allow",
                        reason_code=REASON_ALLOW,
                    )
                )
                continue
            trace.append(
                ConstraintPolicyTraceEntry(
                    policy_name=policy_name,
                    action=str(decision.action or "allow"),
                    reason_code=str(decision.reason_code or REASON_ALLOW),
                )
            )
            winning_policy = policy_name
            if decision.action == "block":
                return {"decision": decision, "trace": trace, "winning_policy": winning_policy}
            if decision.action == "rewrite":
                if allow_rewrite:
                    return {"decision": decision, "trace": trace, "winning_policy": winning_policy}
                blocked = ConstraintDecision(
                    action="block",
                    reason_code=REASON_REWRITE_CHAIN_BLOCKED,
                    block_mode="hard_block_break",
                    loop_break_reason=REASON_REWRITE_CHAIN_BLOCKED,
                    tool_result_payload=ConstraintToolResultPayload(
                        success=False,
                        message="同一轮约束改写仅允许一次，已阻断重复改写。",
                    ),
                    message_for_model="同一轮约束改写仅允许一次，已阻断重复改写。",
                )
                return {"decision": blocked, "trace": trace, "winning_policy": "constraint_engine"}
        return {
            "decision": ConstraintDecision(
                action="allow",
                reason_code=REASON_ALLOW,
            ),
            "trace": trace,
            "winning_policy": winning_policy,
        }

    def _log_final_decision(
            self,
            logger: logging.Logger,
            constraint_input: ConstraintInput,
            engine_result: ConstraintEngineResult,
    ) -> None:
        decision = engine_result.constraint_decision
        policy_trace = [
            {
                "policy_name": entry.policy_name,
                "action": entry.action,
                "reason_code": entry.reason_code,
            }
            for entry in list(engine_result.policy_trace or [])
        ]
        metadata: dict[str, object] = {
            "step_id": str(getattr(constraint_input.step, "id", "") or ""),
            "function_name": str(constraint_input.function_name or ""),
            "final_function_name": str(engine_result.final_function_name or ""),
            "final_action": str(decision.action or ""),
            "winning_policy": str(engine_result.winning_policy or ""),
            "policy_trace": json.dumps(policy_trace, ensure_ascii=False),
            "reason_code": str(decision.reason_code or ""),
            "tool_call_fingerprint": str(engine_result.tool_call_fingerprint or ""),
            "rewrite_applied": bool(engine_result.rewrite_applied),
        }
        if decision.loop_break_reason:
            metadata["loop_break_reason"] = str(decision.loop_break_reason)
        if decision.action == "block":
            metadata["block_mode"] = str(decision.block_mode or "")
        if engine_result.rewrite_reason:
            metadata["rewrite_reason"] = str(engine_result.rewrite_reason)
        if engine_result.rewrite_metadata:
            metadata["rewrite_metadata"] = json.dumps(dict(engine_result.rewrite_metadata), ensure_ascii=False)
        log_runtime(
            logger,
            logging.INFO,
            "执行约束决策完成",
            **metadata,
        )
