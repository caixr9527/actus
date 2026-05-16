from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.application.service.safety_audit_ledger_service import (
    SafetyAuditLedgerService,
    SafetyAuditLinkedEventError,
    SafetyAuditScopeError,
)
from app.domain.models import MessageEvent, ToolEvent, WorkflowRunEventRecord
from app.domain.models.safety_audit import (
    NonToolSafetyAuditCommand,
    SafetyAuditDecision,
    SafetyAuditNonToolActionKind,
    SafetyAuditPolicyTraceEntry,
    SafetyAuditRecord,
    SafetyAuditRecordCommand,
    SafetyAuditRiskLevel,
    SafetyAuditWriteStatus,
)
from app.infrastructure.repositories.db_safety_audit_repository import SafetyAuditLinkageConflictError
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult


def _scope(
        *,
        user_id: str = "user-1",
        session_id: str = "session-1",
        workspace_id: str = "workspace-1",
        run_id: str = "run-1",
        current_step_id: str | None = "step-1",
) -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id=user_id,
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        run_id=run_id,
        current_step_id=current_step_id,
    )


def _record_command(
        *,
        scope: AccessScopeResult | None = None,
        decision: SafetyAuditDecision = SafetyAuditDecision.ALLOW,
        reason_code: str = "allow",
        function_name: str = "read_file",
        normalized_function_name: str = "read_file",
        capability_id: str = "filesystem.read",
        tool_family: str = "filesystem",
) -> SafetyAuditRecordCommand:
    actual_scope = scope or _scope()
    return SafetyAuditRecordCommand(
        scope=actual_scope,
        user_id=actual_scope.user_id,
        session_id=str(actual_scope.session_id),
        workspace_id=str(actual_scope.workspace_id),
        run_id=str(actual_scope.run_id),
        step_id=actual_scope.current_step_id,
        tool_call_id="call-1",
        capability_id=capability_id,
        tool_family=tool_family,
        function_name=function_name,
        normalized_function_name=normalized_function_name,
        requested_args={"path": "/workspace/a.txt"},
        final_function_name=function_name,
        final_normalized_function_name=normalized_function_name,
        final_args={"path": "/workspace/a.txt"},
        decision=decision,
        reason_code=reason_code,
        policy_trace=[
            SafetyAuditPolicyTraceEntry(
                policy_name="task_mode_policy",
                action=decision.value,
                reason_code=reason_code,
            )
        ],
        winning_policy="task_mode_policy",
        tool_call_fingerprint="sha256:fingerprint",
    )


def _non_tool_command(
        *,
        action_kind: SafetyAuditNonToolActionKind = SafetyAuditNonToolActionKind.ARTIFACT_DOWNLOAD,
        decision: SafetyAuditDecision = SafetyAuditDecision.BLOCK,
        reason_code: str = "artifact_revision_scope_mismatch",
) -> NonToolSafetyAuditCommand:
    scope = _scope()
    return NonToolSafetyAuditCommand(
        scope=scope,
        action_kind=action_kind,
        user_id=scope.user_id,
        session_id=str(scope.session_id),
        workspace_id=str(scope.workspace_id),
        run_id=str(scope.run_id),
        step_id=scope.current_step_id,
        action_id_hint="download-artifact-1-rev-1",
        capability_id="artifact.delivery",
        function_name=action_kind.value,
        requested_args={"artifact_id": "artifact-1", "revision_id": "rev-1", "content_hash": "sha256:abc"},
        final_args={"artifact_id": "artifact-1", "revision_id": "rev-1", "content_hash": "sha256:abc"},
        decision=decision,
        reason_code=reason_code,
        artifact_delivery_state="blocked",
    )


class _FakeSafetyAuditRepository:
    def __init__(self) -> None:
        self.records: list[SafetyAuditRecord] = []
        self.events: dict[str, WorkflowRunEventRecord] = {}
        self.event_lookups: list[dict] = []
        self.attached: list[dict] = []

    async def save_once(self, record: SafetyAuditRecord) -> SafetyAuditRecord:
        for existing in self.records:
            if (
                existing.user_id == record.user_id
                and existing.session_id == record.session_id
                and existing.run_id == record.run_id
                and existing.action_id == record.action_id
            ):
                return existing
        self.records.append(record)
        return record

    async def get_by_scope(self, *, user_id: str, session_id: str, audit_id: str) -> SafetyAuditRecord | None:
        if not user_id or not session_id:
            raise ValueError("scope required")
        for record in self.records:
            if record.user_id == user_id and record.session_id == session_id and record.id == audit_id:
                return record
        return None

    async def list_by_run(self, *, user_id: str, session_id: str, run_id: str, limit: int = 100) -> list[SafetyAuditRecord]:
        return [
            record for record in self.records
            if record.user_id == user_id and record.session_id == session_id and record.run_id == run_id
        ][:limit]

    async def list_by_step(self, *, user_id: str, session_id: str, run_id: str, step_id: str, limit: int = 100) -> list[SafetyAuditRecord]:
        return [
            record for record in self.records
            if record.user_id == user_id
            and record.session_id == session_id
            and record.run_id == run_id
            and record.step_id == step_id
        ][:limit]

    async def list_by_tool_event_source(
            self,
            *,
            user_id: str,
            session_id: str,
            tool_event_source_event_id: str,
    ) -> list[SafetyAuditRecord]:
        return [
            record for record in self.records
            if record.user_id == user_id
            and record.session_id == session_id
            and record.tool_event_source_event_id == tool_event_source_event_id
        ]

    async def list_by_decision_event(
            self,
            *,
            user_id: str,
            session_id: str,
            decision_event_id: str,
    ) -> list[SafetyAuditRecord]:
        return [
            record for record in self.records
            if record.user_id == user_id
            and record.session_id == session_id
            and record.decision_event_id == decision_event_id
        ]

    async def list_by_confirmation_event(
            self,
            *,
            user_id: str,
            session_id: str,
            confirmation_event_id: str,
    ) -> list[SafetyAuditRecord]:
        return [
            record for record in self.records
            if record.user_id == user_id
            and record.session_id == session_id
            and record.confirmation_event_id == confirmation_event_id
        ]

    async def attach_linkage(self, **kwargs) -> SafetyAuditRecord:
        record = await self.get_by_scope(
            user_id=kwargs["user_id"],
            session_id=kwargs["session_id"],
            audit_id=kwargs["audit_id"],
        )
        if record is None:
            raise ValueError("missing")
        updates = {}
        for key in {
            "decision_event_id",
            "tool_event_source_event_id",
            "confirmation_event_id",
            "source_event_type",
        }:
            new_value = str(kwargs.get(key) or "").strip()
            if not new_value:
                continue
            current_value = getattr(record, key)
            if current_value is not None and current_value != new_value:
                raise SafetyAuditLinkageConflictError(f"{key} conflict")
            if current_value is None:
                updates[key] = new_value
        source_linked_at = kwargs.get("source_linked_at")
        if source_linked_at is not None:
            if record.source_linked_at is not None and record.source_linked_at != source_linked_at:
                raise SafetyAuditLinkageConflictError("source_linked_at conflict")
            if record.source_linked_at is None:
                updates["source_linked_at"] = source_linked_at
        if not updates:
            return record
        self.attached.append({**kwargs, "_updates": updates})
        updated = record.model_copy(update=updates)
        self.records[self.records.index(record)] = updated
        return updated


class _FakeWorkflowRunRepository:
    def __init__(self, events: dict[str, WorkflowRunEventRecord], lookups: list[dict]) -> None:
        self.events = events
        self.lookups = lookups

    async def get_event_record_by_event_id(self, *, user_id: str, session_id: str, run_id: str, event_id: str):
        self.lookups.append(
            {"user_id": user_id, "session_id": session_id, "run_id": run_id, "event_id": event_id}
        )
        return self.events.get(event_id)


class _FakeUow:
    def __init__(self, safety_audit: _FakeSafetyAuditRepository) -> None:
        self.safety_audit = safety_audit
        self.workflow_run = _FakeWorkflowRunRepository(safety_audit.events, safety_audit.event_lookups)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


def _service(repo: _FakeSafetyAuditRepository) -> SafetyAuditLedgerService:
    return SafetyAuditLedgerService(uow_factory=lambda: _FakeUow(repo))


def _event(
        event_id: str,
        event_type: str,
        *,
        role: str = "assistant",
        step_id: str = "step-1",
        tool_call_id: str = "call-1",
) -> WorkflowRunEventRecord:
    if event_type == "message":
        payload = MessageEvent(id=event_id, role=role, message="ok")
    elif event_type == "tool":
        payload = ToolEvent(
            id=event_id,
            step_id=step_id,
            tool_call_id=tool_call_id,
            tool_name="file",
            function_name="read_file",
            function_args={"path": "/workspace/a.txt"},
        )
    else:
        payload = MessageEvent(id=event_id)
    payload.type = event_type if event_type in {"message"} else payload.type
    return WorkflowRunEventRecord(
        run_id="run-1",
        session_id="session-1",
        user_id="user-1",
        event_id=event_id,
        event_type=event_type,
        event_payload=payload,
    )


def test_pr2_command_result_snapshot_contracts_are_strict() -> None:
    command = _record_command()
    payload = command.model_dump(mode="json")
    payload["extra"] = "forbidden"
    with pytest.raises(ValidationError):
        SafetyAuditRecordCommand.model_validate(payload)

    non_tool = _non_tool_command()
    non_tool_payload = non_tool.model_dump(mode="json")
    non_tool_payload["raw_args"] = {"secret": "token=secret-value-12345"}
    with pytest.raises(ValidationError):
        NonToolSafetyAuditCommand.model_validate(non_tool_payload)


def test_record_constraint_decision_writes_audit_and_reuses_idempotent_record() -> None:
    repo = _FakeSafetyAuditRepository()
    service = _service(repo)

    first = asyncio.run(service.record_constraint_decision(_record_command()))
    second = asyncio.run(service.record_constraint_decision(_record_command()))

    assert first.status == SafetyAuditWriteStatus.CREATED
    assert second.status == SafetyAuditWriteStatus.REUSED
    assert first.audit_id == second.audit_id
    assert len(repo.records) == 1
    record = repo.records[0]
    assert record.risk_level == SafetyAuditRiskLevel.MEDIUM
    assert record.requested_args_digest.fields["path"].path.basename == "a.txt"


def test_record_fails_closed_without_scope_or_scope_mismatch() -> None:
    repo = _FakeSafetyAuditRepository()
    service = _service(repo)
    command = _record_command().model_copy(update={"scope": None})
    with pytest.raises(SafetyAuditScopeError):
        asyncio.run(service.record_constraint_decision(command))

    mismatch_scope = _scope(user_id="user-2")
    mismatch = _record_command(scope=mismatch_scope).model_copy(update={"user_id": "user-1"})
    with pytest.raises(SafetyAuditScopeError):
        asyncio.run(service.record_constraint_decision(mismatch))

    missing_step = _record_command().model_copy(update={"step_id": None})
    with pytest.raises(SafetyAuditScopeError):
        asyncio.run(service.record_constraint_decision(missing_step))


def test_linkage_event_type_rules_are_fail_closed() -> None:
    repo = _FakeSafetyAuditRepository()
    service = _service(repo)
    saved = asyncio.run(service.record_constraint_decision(_record_command()))
    scope = _scope()
    repo.events["tool-1"] = _event("tool-1", "tool")
    repo.events["tool-event-1"] = _event("tool-event-1", "tool_event")
    repo.events["audit-evt-1"] = _event("audit-evt-1", "safety_audit")
    repo.events["audit-tool-evt-1"] = _event("audit-tool-evt-1", "tool")
    repo.events["msg-user-1"] = _event("msg-user-1", "message", role="user")
    repo.events["msg-assistant-1"] = _event("msg-assistant-1", "message", role="assistant")
    repo.events["wait-1"] = _event("wait-1", "wait")
    repo.events["resume-1"] = _event("resume-1", "resume")

    tool_result = asyncio.run(service.attach_tool_event_source(saved.audit_id, "tool-1", scope=scope))
    assert tool_result.audit_id == saved.audit_id
    assert repo.attached[-1]["tool_event_source_event_id"] == "tool-1"
    assert repo.event_lookups[-1] == {
        "user_id": "user-1",
        "session_id": "session-1",
        "run_id": "run-1",
        "event_id": "tool-1",
    }

    with pytest.raises(SafetyAuditLinkedEventError):
        asyncio.run(service.attach_tool_event_source(saved.audit_id, "tool-event-1", scope=scope))

    decision_result = asyncio.run(service.attach_decision_event(saved.audit_id, "audit-evt-1", scope=scope))
    assert decision_result.audit_id == saved.audit_id
    with pytest.raises(SafetyAuditLinkedEventError):
        asyncio.run(service.attach_decision_event(saved.audit_id, "audit-tool-evt-1", scope=scope))

    confirmation_result = asyncio.run(service.attach_confirmation_event(saved.audit_id, "msg-user-1", scope=scope))
    assert confirmation_result.audit_id == saved.audit_id

    with pytest.raises(SafetyAuditLinkedEventError):
        asyncio.run(service.attach_confirmation_event(saved.audit_id, "msg-assistant-1", scope=scope))
    with pytest.raises(SafetyAuditLinkedEventError):
        asyncio.run(service.attach_confirmation_event(saved.audit_id, "wait-1", scope=scope))
    with pytest.raises(SafetyAuditLinkedEventError):
        asyncio.run(service.attach_confirmation_event(saved.audit_id, "resume-1", scope=scope))
    with pytest.raises(SafetyAuditScopeError):
        asyncio.run(service.attach_tool_event_source(saved.audit_id, "tool-1", scope=None))


def test_tool_event_attach_requires_matching_step_and_tool_call() -> None:
    repo = _FakeSafetyAuditRepository()
    service = _service(repo)
    saved = asyncio.run(service.record_constraint_decision(_record_command()))
    scope = _scope()
    repo.events["tool-ok"] = _event("tool-ok", "tool", step_id="step-1", tool_call_id="call-1")
    repo.events["tool-wrong-step"] = _event("tool-wrong-step", "tool", step_id="step-2", tool_call_id="call-1")
    repo.events["tool-wrong-call"] = _event("tool-wrong-call", "tool", step_id="step-1", tool_call_id="call-2")

    result = asyncio.run(service.attach_tool_event_source(saved.audit_id, "tool-ok", scope=scope))
    assert result.audit_id == saved.audit_id

    second_repo = _FakeSafetyAuditRepository()
    second_service = _service(second_repo)
    second_saved = asyncio.run(second_service.record_constraint_decision(_record_command()))
    second_repo.events = repo.events
    with pytest.raises(SafetyAuditLinkedEventError):
        asyncio.run(second_service.attach_tool_event_source(second_saved.audit_id, "tool-wrong-step", scope=scope))
    with pytest.raises(SafetyAuditLinkedEventError):
        asyncio.run(second_service.attach_tool_event_source(second_saved.audit_id, "tool-wrong-call", scope=scope))


def test_linkage_attach_keeps_source_fields_once_and_rejects_conflicting_same_field() -> None:
    repo = _FakeSafetyAuditRepository()
    service = _service(repo)
    saved = asyncio.run(service.record_constraint_decision(_record_command()))
    scope = _scope()
    repo.events["tool-1"] = _event("tool-1", "tool")
    repo.events["tool-2"] = _event("tool-2", "tool")
    repo.events["audit-evt-1"] = _event("audit-evt-1", "safety_audit")
    repo.events["msg-user-1"] = _event("msg-user-1", "message", role="user")

    asyncio.run(service.attach_tool_event_source(saved.audit_id, "tool-1", scope=scope))
    after_tool = repo.records[0]
    assert after_tool.source_event_type == "tool"
    assert after_tool.source_linked_at is not None

    asyncio.run(service.attach_tool_event_source(saved.audit_id, "tool-1", scope=scope))
    assert repo.records[0].tool_event_source_event_id == "tool-1"

    asyncio.run(service.attach_decision_event(saved.audit_id, "audit-evt-1", scope=scope))
    asyncio.run(service.attach_confirmation_event(saved.audit_id, "msg-user-1", scope=scope))
    after_all = repo.records[0]
    assert after_all.source_event_type == "tool"
    assert after_all.source_linked_at == after_tool.source_linked_at
    assert after_all.decision_event_id == "audit-evt-1"
    assert after_all.confirmation_event_id == "msg-user-1"

    with pytest.raises(SafetyAuditLinkageConflictError):
        asyncio.run(service.attach_tool_event_source(saved.audit_id, "tool-2", scope=scope))


def test_non_tool_artifact_download_and_preview_commands_write_audit() -> None:
    repo = _FakeSafetyAuditRepository()
    service = _service(repo)

    download = asyncio.run(service.record_artifact_download_decision(_non_tool_command()))
    preview = asyncio.run(
        service.record_artifact_preview_decision(
            _non_tool_command(action_kind=SafetyAuditNonToolActionKind.ARTIFACT_PREVIEW)
        )
    )

    assert download.record.decision == SafetyAuditDecision.BLOCK
    assert download.record.risk_level == SafetyAuditRiskLevel.HIGH
    assert preview.record.risk_level == SafetyAuditRiskLevel.HIGH
    assert {record.function_name for record in repo.records} == {"artifact_download", "artifact_preview"}

    with pytest.raises(SafetyAuditScopeError):
        asyncio.run(
            service.record_artifact_download_decision(
                _non_tool_command(action_kind=SafetyAuditNonToolActionKind.ARTIFACT_PREVIEW)
            )
        )


def test_snapshot_is_built_from_audit_records_without_raw_payloads() -> None:
    repo = _FakeSafetyAuditRepository()
    service = _service(repo)

    asyncio.run(service.record_constraint_decision(_record_command()))
    asyncio.run(
        service.record_constraint_decision(
            _record_command(
                decision=SafetyAuditDecision.BLOCK,
                reason_code="constraint_engine_error",
                function_name="run_shell",
                normalized_function_name="run_shell",
                capability_id="shell.execute",
                tool_family="shell",
            )
        )
    )
    snapshot = asyncio.run(service.get_snapshot(scope=_scope()))

    assert snapshot.decision_counts.allow == 1
    assert snapshot.decision_counts.block == 1
    assert snapshot.risk_counts.critical == 1
    assert len(snapshot.critical_findings) == 1
    serialized = snapshot.model_dump(mode="json")
    assert "requested_args_digest" not in str(serialized)
    assert "raw_args" not in str(serialized)


def test_linkage_queries_use_scope_and_do_not_expose_raw_payloads() -> None:
    repo = _FakeSafetyAuditRepository()
    service = _service(repo)
    saved = asyncio.run(service.record_constraint_decision(_record_command()))
    scope = _scope()
    repo.events["tool-1"] = _event("tool-1", "tool")
    repo.events["audit-evt-1"] = _event("audit-evt-1", "safety_audit")
    repo.events["msg-user-1"] = _event("msg-user-1", "message", role="user")
    asyncio.run(service.attach_tool_event_source(saved.audit_id, "tool-1", scope=scope))
    asyncio.run(service.attach_decision_event(saved.audit_id, "audit-evt-1", scope=scope))
    asyncio.run(service.attach_confirmation_event(saved.audit_id, "msg-user-1", scope=scope))

    by_tool = asyncio.run(service.list_by_tool_event_source(scope=scope, tool_event_source_event_id="tool-1"))
    by_decision = asyncio.run(service.list_by_decision_event(scope=scope, decision_event_id="audit-evt-1"))
    by_confirmation = asyncio.run(service.list_by_confirmation_event(scope=scope, confirmation_event_id="msg-user-1"))

    assert [record.id for record in by_tool] == [saved.audit_id]
    assert [record.id for record in by_decision] == [saved.audit_id]
    assert [record.id for record in by_confirmation] == [saved.audit_id]
    assert by_tool[0].requested_args_digest.fields["path"].path.basename == "a.txt"
    assert "raw_args" not in str([record.model_dump(mode="json") for record in by_tool])

    other_scope = _scope(user_id="user-2")
    assert asyncio.run(service.list_by_tool_event_source(scope=other_scope, tool_event_source_event_id="tool-1")) == []
    with pytest.raises(SafetyAuditScopeError):
        asyncio.run(service.list_by_tool_event_source(scope=None, tool_event_source_event_id="tool-1"))
    with pytest.raises(SafetyAuditScopeError):
        asyncio.run(service.list_by_decision_event(scope=scope, decision_event_id=" "))
