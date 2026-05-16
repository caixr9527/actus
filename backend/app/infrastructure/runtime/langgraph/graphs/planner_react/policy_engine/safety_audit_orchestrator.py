#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safety Audit 工具调用薄协调层。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from app.domain.models import Step
from app.domain.models.safety_audit import (
    SafetyAuditDecision,
    SafetyAuditPolicyTraceEntry,
    SafetyAuditRecorderPort,
    SafetyAuditRecordCommand,
    SafetyAuditRecordResult,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.tools import BaseTool

from ..constraint_engine.contracts import ConstraintEngineResult, ConstraintPolicyTraceEntry
from ..execution.execution_context import ExecutionContext
from ..execution.execution_state import ExecutionState
from .engine import PolicyEvaluationResult, ToolPolicyEngine


class SafetyAuditToolCallError(RuntimeError):
    """工具调用审计无法写入时必须 fail closed。"""


@dataclass(slots=True)
class SafetyAuditToolCallContext:
    scope: AccessScopeResult
    step: Step
    task_mode: str
    tool_call_id: str
    function_name: str
    normalized_function_name: str
    function_args: Dict[str, Any]
    matched_tool: Optional[BaseTool]
    runtime_tools: Optional[list[BaseTool]]
    browser_route_state_key: str
    iteration_blocked_function_names: Set[str]
    execution_context: ExecutionContext
    execution_state: ExecutionState
    started_at: float
    evidence_reuse_snapshot: Any = None
    has_previous_completed_steps: bool = False
    previous_completed_step_task_modes: Optional[Dict[str, str]] = None


class ConstraintEngineResultSafetyAuditAdapter:
    """将 constraint guard 结果转换为 Safety Audit 领域 command。"""

    def build_command(
            self,
            *,
            scope: AccessScopeResult,
            step: Step,
            tool_call_id: str,
            requested_function_name: str,
            requested_normalized_function_name: str,
            requested_args: Dict[str, Any],
            matched_tool: Optional[BaseTool],
            engine_result: ConstraintEngineResult,
    ) -> SafetyAuditRecordCommand:
        self._validate_scope(scope=scope, step=step)
        decision = (
            SafetyAuditDecision.REWRITE
            if bool(engine_result.rewrite_applied)
            else _map_decision(str(engine_result.constraint_decision.action or ""))
        )
        final_function_name = str(engine_result.final_function_name or requested_function_name or "").strip()
        final_normalized_function_name = str(
            engine_result.final_normalized_function_name
            or requested_normalized_function_name
            or final_function_name
        ).strip().lower()
        reason_code = (
            str(engine_result.rewrite_reason or "").strip()
            if bool(engine_result.rewrite_applied)
            else str(engine_result.constraint_decision.reason_code or "").strip()
        ) or "unknown"
        return SafetyAuditRecordCommand(
            scope=scope,
            user_id=scope.user_id,
            session_id=str(scope.session_id or ""),
            workspace_id=str(scope.workspace_id or ""),
            run_id=str(scope.run_id or ""),
            step_id=str(step.id or "").strip(),
            tool_call_id=str(tool_call_id or "").strip() or None,
            capability_id=_resolve_capability_id(matched_tool=matched_tool, function_name=final_normalized_function_name),
            tool_family=_resolve_tool_family(matched_tool=matched_tool, function_name=final_normalized_function_name),
            function_name=str(requested_function_name or "").strip(),
            normalized_function_name=str(requested_normalized_function_name or requested_function_name or "").strip().lower(),
            requested_args=dict(requested_args or {}),
            final_function_name=final_function_name,
            final_normalized_function_name=final_normalized_function_name,
            final_args=dict(engine_result.final_function_args or {}),
            decision=decision,
            reason_code=reason_code,
            policy_trace=[
                _policy_trace_entry(entry)
                for entry in list(engine_result.policy_trace or [])
            ],
            winning_policy=str(engine_result.winning_policy or "").strip() or "constraint_engine",
            tool_call_fingerprint=str(engine_result.tool_call_fingerprint or "").strip(),
            rewrite_applied=bool(engine_result.rewrite_applied),
            rewrite_reason=str(engine_result.rewrite_reason or "").strip() or None,
            rewrite_metadata=dict(engine_result.rewrite_metadata or {}),
        )

    @staticmethod
    def _validate_scope(*, scope: AccessScopeResult, step: Step) -> None:
        if scope is None:
            raise SafetyAuditToolCallError("Safety Audit 缺少 AccessScopeResult")
        required = [scope.user_id, scope.session_id, scope.workspace_id, scope.run_id]
        if any(not str(value or "").strip() for value in required):
            raise SafetyAuditToolCallError("Safety Audit scope 缺少 user/session/workspace/run")
        step_id = str(step.id or "").strip()
        scope_step_id = str(scope.current_step_id or "").strip()
        if not step_id or scope_step_id != step_id:
            raise SafetyAuditToolCallError("Safety Audit scope.current_step_id 与当前 step 不一致")


class SafetyAuditToolCallOrchestrator:
    """包裹 policy engine，在真实 executor 前写 pending audit。"""

    def __init__(
            self,
            *,
            policy_engine: ToolPolicyEngine,
            recorder: SafetyAuditRecorderPort,
            adapter: ConstraintEngineResultSafetyAuditAdapter | None = None,
            logger: logging.Logger,
    ) -> None:
        self._policy_engine = policy_engine
        self._recorder = recorder
        self._adapter = adapter or ConstraintEngineResultSafetyAuditAdapter()
        self._logger = logger

    async def evaluate_tool_call(self, context: SafetyAuditToolCallContext) -> PolicyEvaluationResult:
        if self._recorder is None:
            raise SafetyAuditToolCallError("SafetyAuditRecorderPort 未装配")
        audit_result_holder: dict[str, SafetyAuditRecordResult | None] = {"record": None}

        async def _record_audit(engine_result: ConstraintEngineResult) -> SafetyAuditRecordResult:
            command = self._adapter.build_command(
                scope=context.scope,
                step=context.step,
                tool_call_id=context.tool_call_id,
                requested_function_name=context.function_name,
                requested_normalized_function_name=context.normalized_function_name,
                requested_args=context.function_args,
                matched_tool=context.matched_tool,
                engine_result=engine_result,
            )
            try:
                write_result = await self._recorder.record_constraint_decision(command)
            except Exception as exc:
                log_runtime(
                    self._logger,
                    logging.ERROR,
                    "safety_audit_record_failed",
                    step_id=str(context.step.id or ""),
                    tool_call_id=str(context.tool_call_id or ""),
                    function_name=str(context.function_name or ""),
                    error_type=exc.__class__.__name__,
                    reason_code="safety_audit_record_failed",
                )
                raise SafetyAuditToolCallError("safety_audit_record_failed") from exc
            audit_result_holder["record"] = write_result.record
            return write_result.record

        policy_result = await self._policy_engine.evaluate_tool_call(
            step=context.step,
            task_mode=context.task_mode,
            function_name=context.function_name,
            normalized_function_name=context.normalized_function_name,
            function_args=context.function_args,
            matched_tool=context.matched_tool,
            runtime_tools=context.runtime_tools,
            browser_route_state_key=context.browser_route_state_key,
            iteration_blocked_function_names=context.iteration_blocked_function_names,
            execution_context=context.execution_context,
            execution_state=context.execution_state,
            started_at=context.started_at,
            evidence_reuse_snapshot=context.evidence_reuse_snapshot,
            has_previous_completed_steps=context.has_previous_completed_steps,
            previous_completed_step_task_modes=context.previous_completed_step_task_modes,
            after_guard_decision=_record_audit,
        )
        audit_record = audit_result_holder["record"]
        if audit_record is None or not str(audit_record.audit_id or "").strip():
            raise SafetyAuditToolCallError("safety_audit_record_failed")
        policy_result.audit_id = audit_record.audit_id
        policy_result.safety_audit_metadata = _build_safety_audit_metadata(audit_record)
        return policy_result


def _map_decision(action: str) -> SafetyAuditDecision:
    normalized = str(action or "").strip().lower()
    if normalized == "allow":
        return SafetyAuditDecision.ALLOW
    if normalized == "rewrite":
        return SafetyAuditDecision.REWRITE
    if normalized == "require_confirmation":
        return SafetyAuditDecision.REQUIRE_CONFIRMATION
    return SafetyAuditDecision.BLOCK


def _policy_trace_entry(entry: ConstraintPolicyTraceEntry) -> SafetyAuditPolicyTraceEntry:
    return SafetyAuditPolicyTraceEntry(
        policy_name=str(entry.policy_name or "").strip() or "unknown_policy",
        action=str(entry.action or "").strip() or "unknown",
        reason_code=str(entry.reason_code or "").strip() or "unknown",
    )


def _resolve_tool_family(*, matched_tool: Optional[BaseTool], function_name: str) -> str:
    tool_name = str(getattr(matched_tool, "name", "") or "").strip()
    if tool_name:
        return tool_name
    normalized = str(function_name or "").strip().lower()
    if normalized:
        return normalized.split("_", 1)[0]
    return "unknown"


def _resolve_capability_id(*, matched_tool: Optional[BaseTool], function_name: str) -> str:
    tool_family = _resolve_tool_family(matched_tool=matched_tool, function_name=function_name)
    normalized = str(function_name or "").strip().lower() or "unknown"
    return f"{tool_family}.{normalized}"


def _build_safety_audit_metadata(record: SafetyAuditRecordResult) -> dict[str, Any]:
    return {
        "audit_id": record.audit_id,
        "action_id": record.action_id,
        "decision": record.decision.value,
        "risk_level": record.risk_level.value,
        "reason_code": record.reason_code,
    }
