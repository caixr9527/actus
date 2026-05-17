from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.application.service.runtime_observation_service import (
    RuntimeObservationContextResult,
    RuntimeObservationService,
)
from app.application.service.projecting_safety_audit_recorder import ProjectingSafetyAuditRecorder
from app.application.service.safety_audit_event_projector import SafetyAuditEventProjector
from app.application.service.safety_audit_ledger_service import SafetyAuditLedgerService
from app.domain.models import SafetyAuditEvent, SessionStatus, ToolEvent, WorkflowRunEventRecord
from app.domain.models.safety_audit import (
    SafetyAuditNonToolActionKind,
    SafetyAuditDecision,
    SafetyAuditEventPayload,
    SafetyAuditEventRef,
    SafetyAuditRecord,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.interfaces.schemas.event import EventMapper

from tests.test_safety_audit_pr2 import _FakeSafetyAuditRepository, _non_tool_command, _record_command, _scope


def _record(
        *,
        decision: SafetyAuditDecision = SafetyAuditDecision.BLOCK,
        reason_code: str = "artifact_revision_scope_mismatch",
        tool_call_id: str = "call-1",
        function_name: str = "read_file",
) -> SafetyAuditRecord:
    command = _record_command(
        decision=decision,
        reason_code=reason_code,
        function_name=function_name,
        normalized_function_name=function_name,
    )
    command = command.model_copy(update={"tool_call_id": tool_call_id})
    from app.application.service.safety_audit_ledger_service import SafetyAuditLedgerService

    return SafetyAuditLedgerService(uow_factory=lambda: None)._build_record(command)


class _WorkflowRunRepository:
    def __init__(self, repo: _FakeSafetyAuditRepository) -> None:
        self.repo = repo

    async def add_event_record_if_absent(self, *, session_id: str, run_id: str, event) -> bool:
        if event.id in self.repo.events:
            return False
        self.repo.events[event.id] = WorkflowRunEventRecord(
            run_id=run_id,
            session_id=session_id,
            user_id="user-1",
            event_id=event.id,
            event_type=event.type,
            event_payload=event,
            created_at=event.created_at,
        )
        return True

    async def get_event_record_by_event_id(self, *, user_id: str, session_id: str, run_id: str, event_id: str):
        return self.repo.events.get(event_id)


class _Uow:
    def __init__(self, repo: _FakeSafetyAuditRepository) -> None:
        self.safety_audit = repo
        self.workflow_run = _WorkflowRunRepository(repo)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class _Coordinator:
    def __init__(self, repo: _FakeSafetyAuditRepository, *, force_duplicate: bool = False) -> None:
        self.repo = repo
        self.force_duplicate = force_duplicate
        self.persisted_event_ids: list[str] = []
        self.persist_results: list[object] = []

    async def persist_runtime_event(self, *, session_id: str, event, **_kwargs):
        self.persisted_event_ids.append(event.id)
        if self.force_duplicate:
            await _WorkflowRunRepository(self.repo).add_event_record_if_absent(
                session_id=session_id,
                run_id="run-1",
                event=event,
            )
            result = type("PersistResult", (), {"event_inserted": False, "event_id": event.id})()
            self.persist_results.append(result)
            return result
        inserted = await _WorkflowRunRepository(self.repo).add_event_record_if_absent(
            session_id=session_id,
            run_id="run-1",
            event=event,
        )
        result = type("PersistResult", (), {"event_inserted": inserted, "event_id": event.id})()
        self.persist_results.append(result)
        return result


class _Recorder:
    def __init__(self, repo: _FakeSafetyAuditRepository, *, fail_attach: bool = False) -> None:
        self.repo = repo
        self.fail_attach = fail_attach
        self.decision_attach_calls: list[tuple[str, str]] = []

    async def attach_decision_event(
            self,
            audit_id: str,
            decision_event_id: str,
            *,
            scope: AccessScopeResult,
    ):
        self.decision_attach_calls.append((audit_id, decision_event_id))
        if self.fail_attach:
            raise RuntimeError("decision attach failed")
        service = __import__(
            "app.application.service.safety_audit_ledger_service",
            fromlist=["SafetyAuditLedgerService"],
        ).SafetyAuditLedgerService(uow_factory=lambda: _Uow(self.repo))
        return await service.attach_decision_event(audit_id, decision_event_id, scope=scope)


def _projector(
        repo: _FakeSafetyAuditRepository,
        *,
        recorder: _Recorder | None = None,
        coordinator: _Coordinator | None = None,
) -> SafetyAuditEventProjector:
    return SafetyAuditEventProjector(
        uow_factory=lambda: _Uow(repo),
        runtime_state_coordinator=coordinator or _Coordinator(repo),
        recorder=recorder or _Recorder(repo),
    )


def _projecting_recorder(
        repo: _FakeSafetyAuditRepository,
        *,
        projector: SafetyAuditEventProjector | None = None,
) -> ProjectingSafetyAuditRecorder:
    ledger = SafetyAuditLedgerService(uow_factory=lambda: _Uow(repo))
    return ProjectingSafetyAuditRecorder(
        recorder=ledger,
        projector=projector or _projector(repo, recorder=_Recorder(repo)),
    )


def test_safety_audit_event_payload_rejects_raw_or_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SafetyAuditEventPayload.model_validate(
            {
                "audit_refs": [
                    {
                        "audit_id": "audit-1",
                        "decision": "block",
                        "risk_level": "high",
                        "reason_code": "secret_detected",
                        "step_id": "step-1",
                        "tool_call_id": "call-1",
                        "function_name": "read_file",
                        "raw_args": {"secret": "value"},
                    }
                ],
                "source_event_ids": [],
                "decision_counts": {"block": 1},
                "risk_counts": {"high": 1},
                "blocked_count": 1,
                "rewrite_count": 0,
                "confirmation_count": 0,
                "summary": "blocked",
                "runtime_metadata": {
                    "visibility": "hidden",
                    "projection_key": "safety_audit:v1:user:session:run:hash",
                    "schema_version": "safety_audit_event.v1",
                },
            }
        )

    with pytest.raises(ValidationError):
        SafetyAuditEventRef.model_validate(
            {
                "audit_id": "audit-1",
                "decision": "block",
                "risk_level": "high",
                "reason_code": "secret_detected",
                "step_id": "step-1",
                "tool_call_id": "call-1",
                "function_name": "read_file",
                "action_id": "must-not-leak",
            }
        )


def test_projector_projects_tool_source_once_and_attaches_decision_event() -> None:
    repo = _FakeSafetyAuditRepository()
    record = _record()
    record = record.model_copy(update={"tool_event_source_event_id": "tool-event-1"})
    repo.records.append(record)
    repo.events["tool-event-1"] = WorkflowRunEventRecord(
        run_id="run-1",
        session_id="session-1",
        user_id="user-1",
        event_id="tool-event-1",
        event_type="tool",
        event_payload=ToolEvent(
            step_id="step-1",
            tool_call_id="call-1",
            tool_name="filesystem",
            function_name="read_file",
            function_args={},
        ),
        created_at=datetime.now(),
    )

    recorder = _Recorder(repo)
    result = asyncio.run(_projector(repo, recorder=recorder).project_tool_event_source(
        scope=_scope(),
        tool_event_source_event_id="tool-event-1",
    ))
    second = asyncio.run(_projector(repo, recorder=recorder).project_tool_event_source(
        scope=_scope(),
        tool_event_source_event_id="tool-event-1",
    ))

    assert result.projected is True
    assert result.event_id is not None
    assert second.projected is False
    assert len([event for event in repo.events.values() if event.event_type == "safety_audit"]) == 1
    updated = repo.records[0]
    assert updated.decision_event_id == result.event_id
    event = repo.events[str(result.event_id)].event_payload
    assert isinstance(event, SafetyAuditEvent)
    payload = event.payload.model_dump(mode="json")
    assert payload["source_event_ids"] == ["tool-event-1"]
    assert payload["audit_refs"][0] == {
        "audit_id": record.id,
        "decision": "block",
        "risk_level": "high",
        "reason_code": "artifact_revision_scope_mismatch",
        "step_id": "step-1",
        "tool_call_id": "call-1",
        "function_name": "read_file",
    }
    assert "raw_args" not in str(payload)
    assert "policy_trace" not in str(payload)


def test_projector_attaches_decision_when_event_write_is_duplicate() -> None:
    repo = _FakeSafetyAuditRepository()
    record = _record()
    record = record.model_copy(update={"tool_event_source_event_id": "tool-event-1"})
    repo.records.append(record)
    coordinator = _Coordinator(repo, force_duplicate=True)
    recorder = _Recorder(repo)

    result = asyncio.run(_projector(
        repo,
        recorder=recorder,
        coordinator=coordinator,
    ).project_tool_event_source(
        scope=_scope(),
        tool_event_source_event_id="tool-event-1",
    ))

    assert result.projected is True
    assert result.event_id is not None
    assert coordinator.persisted_event_ids == [result.event_id]
    assert coordinator.persist_results[0].event_inserted is False
    assert coordinator.persist_results[0].event_id == result.event_id
    assert repo.records[0].decision_event_id == result.event_id
    assert repo.events[str(result.event_id)].event_payload.payload.runtime_metadata.projection_key == result.event_id


def test_projector_keeps_event_when_decision_attach_fails(caplog) -> None:
    repo = _FakeSafetyAuditRepository()
    repo.records.append(_record())
    recorder = _Recorder(repo, fail_attach=True)

    with caplog.at_level("ERROR"):
        result = asyncio.run(_projector(repo, recorder=recorder).project_single_audit(
            scope=_scope(),
            audit_id=repo.records[0].id,
        ))

    assert result.projected is True
    assert result.event_id in repo.events
    assert repo.records[0].decision_event_id is None
    assert "safety_audit_decision_event_attach_failed" in caplog.text


@pytest.mark.parametrize(
    ("action_kind", "method_name"),
    [
        (SafetyAuditNonToolActionKind.ARTIFACT_DOWNLOAD, "record_artifact_download_decision"),
        (SafetyAuditNonToolActionKind.ARTIFACT_PREVIEW, "record_artifact_preview_decision"),
        (SafetyAuditNonToolActionKind.DOCUMENT_PREFLIGHT, "record_document_preflight_decision"),
    ],
)
def test_projecting_recorder_projects_non_tool_actions_and_attaches_decision_event(
        action_kind: SafetyAuditNonToolActionKind,
        method_name: str,
) -> None:
    repo = _FakeSafetyAuditRepository()
    service = _projecting_recorder(repo)
    command = _non_tool_command(action_kind=action_kind)

    result = asyncio.run(getattr(service, method_name)(command))
    second = asyncio.run(getattr(service, method_name)(command))

    safety_events = [event for event in repo.events.values() if event.event_type == "safety_audit"]
    assert result.audit_id == second.audit_id
    assert len(repo.records) == 1
    assert len(safety_events) == 1
    assert repo.records[0].decision_event_id == safety_events[0].event_id
    assert isinstance(safety_events[0].event_payload, SafetyAuditEvent)
    assert safety_events[0].event_payload.payload.runtime_metadata.visibility == "hidden"
    assert safety_events[0].event_payload.payload.audit_refs[0].audit_id == result.audit_id


def test_projecting_recorder_projects_generic_non_tool_action() -> None:
    repo = _FakeSafetyAuditRepository()
    service = _projecting_recorder(repo)
    command = _non_tool_command(action_kind=SafetyAuditNonToolActionKind.ARTIFACT_DOWNLOAD)

    result = asyncio.run(service.record_non_tool_action(command))

    safety_events = [event for event in repo.events.values() if event.event_type == "safety_audit"]
    assert len(safety_events) == 1
    assert repo.records[0].id == result.audit_id
    assert repo.records[0].decision_event_id == safety_events[0].event_id


class _FailingProjector:
    async def project_single_audit(self, **_kwargs):
        raise RuntimeError("projection failed")

    async def project_tool_event_source(self, **_kwargs):
        raise RuntimeError("unused")


def test_projecting_recorder_keeps_ledger_payload_when_projection_fails(caplog) -> None:
    repo = _FakeSafetyAuditRepository()
    service = _projecting_recorder(repo, projector=_FailingProjector())
    command = _non_tool_command(action_kind=SafetyAuditNonToolActionKind.ARTIFACT_DOWNLOAD)

    with caplog.at_level("ERROR"):
        result = asyncio.run(service.record_artifact_download_decision(command))

    record = repo.records[0]
    assert result.audit_id == repo.records[0].id
    assert record.decision_event_id is None
    assert repo.events == {}
    assert record.decision == SafetyAuditDecision.BLOCK
    assert record.risk_level.value == "high"
    assert record.reason_code == "artifact_revision_scope_mismatch"
    assert str(record.requested_args_digest.fields["artifact_id"].hash).startswith("sha256:")
    assert str(record.final_args_digest.fields["revision_id"].hash).startswith("sha256:")
    assert record.policy_trace == []
    assert "safety_audit_event_projection_failed" in caplog.text


def test_safety_audit_event_mapper_outputs_hidden_event_without_raw_payload() -> None:
    repo = _FakeSafetyAuditRepository()
    repo.records.append(_record())
    result = asyncio.run(_projector(repo).project_single_audit(
        scope=_scope(),
        audit_id=repo.records[0].id,
    ))
    event = repo.events[str(result.event_id)].event_payload
    service = RuntimeObservationService(uow_factory=lambda: _Uow(repo))
    envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=event,
        run_id="run-1",
        source_event_id=str(result.event_id),
        cursor_event_id=str(result.event_id),
        source="snapshot",
        context=RuntimeObservationContextResult(
            session_id="session-1",
            run_id="run-1",
            status=SessionStatus.RUNNING,
            current_step_id="step-1",
        ),
    ))

    mapped = EventMapper.observable_event_to_sse_event(envelope).model_dump(mode="json")

    assert mapped["event"] == "safety_audit"
    assert mapped["data"]["runtime"]["visibility"] == "hidden"
    assert mapped["data"]["payload"]["runtime_metadata"]["visibility"] == "hidden"
    assert mapped["data"]["event_id"] == result.event_id
    serialized = str(mapped)
    assert "raw_args" not in serialized
    assert "policy_trace" not in serialized
    assert "secret" not in serialized
