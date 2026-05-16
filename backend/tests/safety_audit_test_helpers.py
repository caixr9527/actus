from __future__ import annotations

from typing import Any

from app.domain.models.safety_audit import (
    SafetyAuditDecision,
    SafetyAuditRecordCommand,
    SafetyAuditRecordResult,
    SafetyAuditRiskClassificationInput,
    SafetyAuditRiskLevel,
    SafetyAuditWriteResult,
    SafetyAuditWriteStatus,
    classify_safety_audit_risk,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.sandbox_fact_ports import ToolEventFactProjectionResult


def fake_safety_audit_scope(step_id: str = "step-1") -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        current_step_id=step_id,
    )


class FakeSafetyAuditRecorder:
    def __init__(self) -> None:
        self.commands: list[SafetyAuditRecordCommand] = []
        self.attach_calls: list[dict[str, Any]] = []

    async def record_constraint_decision(self, command: SafetyAuditRecordCommand) -> SafetyAuditWriteResult:
        self.commands.append(command)
        audit_id = f"audit-{len(self.commands)}"
        risk_level = (
            classify_safety_audit_risk(command.risk_input).risk_level
            if command.risk_input is not None
            else classify_safety_audit_risk(
                SafetyAuditRiskClassificationInput(
                    decision=command.decision,
                    normalized_function_name=command.final_normalized_function_name,
                    tool_family=command.tool_family,
                    capability_id=command.capability_id,
                    action_kind=command.final_normalized_function_name,
                    reason_code=command.reason_code,
                )
            ).risk_level
        )
        record = SafetyAuditRecordResult(
            audit_id=audit_id,
            action_id=f"action-{audit_id}",
            decision=command.decision,
            risk_level=risk_level,
            reason_code=command.reason_code,
            run_id=command.run_id,
            step_id=command.step_id,
            tool_call_id=command.tool_call_id,
        )
        return SafetyAuditWriteResult(
            audit_id=audit_id,
            record=record,
            status=SafetyAuditWriteStatus.CREATED,
            reason_code=command.reason_code,
        )

    async def attach_tool_event_source(
            self,
            audit_id: str,
            tool_event_source_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        self.attach_calls.append(
            {
                "audit_id": audit_id,
                "tool_event_source_event_id": tool_event_source_event_id,
                "scope": scope,
            }
        )
        record = SafetyAuditRecordResult(
            audit_id=audit_id,
            action_id=f"action-{audit_id}",
            decision=self.commands[-1].decision if self.commands else SafetyAuditDecision.ALLOW,
            risk_level=SafetyAuditRiskLevel.MEDIUM,
            reason_code=self.commands[-1].reason_code if self.commands else "allow",
            run_id=str(scope.run_id or ""),
            step_id=scope.current_step_id,
            tool_call_id=self.commands[-1].tool_call_id if self.commands else None,
        )
        return SafetyAuditWriteResult(
            audit_id=audit_id,
            record=record,
            status=SafetyAuditWriteStatus.REUSED,
            reason_code=record.reason_code,
        )


class FakeAccessControlService:
    async def assert_session_access(self, *, user_id: str, session_id: str, action) -> AccessScopeResult:
        return AccessScopeResult(
            tenant_id=user_id,
            user_id=user_id,
            session_id=session_id,
            workspace_id="workspace-1",
            run_id="run-1",
            current_step_id=None,
        )


class FakeRuntimeToolEventPersistence:
    def __init__(self, *, artifact_revision_count: int = 1) -> None:
        self.artifact_revision_count = artifact_revision_count
        self.calls: list[dict[str, Any]] = []

    async def persist_tool_event_and_record_facts(
            self,
            *,
            event,
            run_id: str,
            session_id: str,
            current_step_id: str,
    ) -> ToolEventFactProjectionResult:
        source_event_id = str(getattr(event, "id", "") or f"evt-{len(self.calls) + 1}")
        projection = {
            "source_event_id": source_event_id,
            "fact_count": 0,
            "artifact_revision_count": self.artifact_revision_count,
            "sandbox_fact_event_persisted": False,
            "event_inserted": True,
        }
        event.id = source_event_id
        event.runtime_fact_projection = dict(projection)
        self.calls.append(
            {
                "event": event,
                "run_id": run_id,
                "session_id": session_id,
                "current_step_id": current_step_id,
            }
        )
        return ToolEventFactProjectionResult(**projection)


async def execute_step_with_fake_safety_audit(execute_step_with_prompt, **kwargs):
    step = kwargs.get("step")
    if "access_scope" not in kwargs and step is not None:
        kwargs["access_scope"] = fake_safety_audit_scope(str(getattr(step, "id", "") or "step-1"))
    if "safety_audit_recorder" not in kwargs:
        kwargs["safety_audit_recorder"] = FakeSafetyAuditRecorder()
    return await execute_step_with_prompt(**kwargs)
