from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.application.service.feedback_ledger_service import FeedbackLedgerService
from app.application.service.runtime_feedback_adapter import RuntimeFeedbackAdapter
from app.application.service.runtime_feedback_gap_buffer import RuntimeFeedbackGapBuffer
from app.domain.models import (
    EvidenceEvent,
    EvidenceEventRef,
    ExecutionStatus,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
    WorkflowRunEventRecord,
)
from app.domain.models.evidence import (
    EvidenceKind,
    EvidenceQualityStatus,
    EvidenceRecord,
    EvidenceScope,
    EvidenceSourceRef,
    EvidenceSourceType,
    EvidenceSubjectRef,
    EvidenceSupportLevel,
    build_evidence_idempotency_key,
    build_evidence_payload_hash,
    build_evidence_result_refs_hash,
    validate_evidence_payload,
)
from app.domain.models.feedback import (
    FeedbackCategory,
    FeedbackGapKind,
    FeedbackKind,
    FeedbackReasonCode,
    FeedbackScopeKind,
    FeedbackSnapshotCursorResult,
    FeedbackSnapshotItemResult,
    FeedbackSnapshotResult,
    FeedbackSnapshotScopeResult,
    FeedbackSeverity,
    FeedbackSnapshotStage,
    FeedbackSourceKind,
    FeedbackStatus,
    FeedbackTargetType,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.models.safety_audit import (
    SafetyAuditArgsDigest,
    SafetyAuditDecision,
    SafetyAuditRecord,
    SafetyAuditRiskLevel,
    build_safety_audit_action_id,
)
from app.domain.models.sandbox_fact import (
    SandboxFactKind,
    SandboxFactProfileRef,
    SandboxFactRecord,
    SandboxFactScope,
    SandboxFactSourceRef,
    SandboxFactSourceType,
    SandboxFactSubjectRef,
    build_sandbox_fact_idempotency_key,
    build_sandbox_fact_payload_hash,
    classify_sandbox_fact_data,
    validate_sandbox_fact_payload,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.data_access_contract import DataOrigin


def _scope() -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        current_step_id="step-1",
    )


class _WorkflowRunRepo:
    def __init__(self, records: dict[str, WorkflowRunEventRecord] | None = None) -> None:
        self.records = dict(records or {})

    async def get_event_record_by_event_id(self, *, user_id: str, session_id: str, run_id: str, event_id: str):
        record = self.records.get(event_id)
        if record is None:
            return None
        if record.user_id != user_id or record.session_id != session_id or record.run_id != run_id:
            return None
        return record

    async def get_event_record_by_event_id_in_session(self, *, user_id: str, session_id: str, event_id: str):
        record = self.records.get(event_id)
        if record is None:
            return None
        if record.user_id != user_id or record.session_id != session_id:
            return None
        return record

    async def get_by_id_for_user_session(self, *, run_id: str, user_id: str, session_id: str):
        if run_id != "run-1":
            return None
        return SimpleNamespace(id=run_id, user_id=user_id, session_id=session_id)


class _FeedbackRepo:
    def __init__(self) -> None:
        self.saved = []

    async def save_once(self, record):
        self.saved.append(record)
        return record

    async def list_by_scope(self, **kwargs):
        return []


class _SandboxFactRepo:
    def __init__(self, records: dict[str, SandboxFactRecord] | None = None) -> None:
        self.records = dict(records or {})

    async def list_by_ids(self, *, user_id: str, session_id: str, fact_ids: list[str], limit: int = 100):
        return [
            fact
            for fact_id in fact_ids[:limit]
            if (fact := self.records.get(fact_id)) is not None
            and fact.user_id == user_id
            and fact.session_id == session_id
        ]


class _EvidenceRepo:
    def __init__(self, records: dict[str, EvidenceRecord] | None = None) -> None:
        self.records = dict(records or {})

    async def list_by_ids(self, *, user_id: str, session_id: str, evidence_ids: list[str], limit: int = 100):
        return [
            evidence
            for evidence_id in evidence_ids[:limit]
            if (evidence := self.records.get(evidence_id)) is not None
            and evidence.user_id == user_id
            and evidence.session_id == session_id
        ]


class _SafetyAuditRepo:
    def __init__(self, records: dict[str, SafetyAuditRecord] | None = None) -> None:
        self.records = dict(records or {})

    async def get_by_scope(self, *, user_id: str, session_id: str, audit_id: str):
        audit = self.records.get(audit_id)
        if audit is None or audit.user_id != user_id or audit.session_id != session_id:
            return None
        return audit


class _UoW:
    def __init__(
            self,
            *,
            feedback_repo: _FeedbackRepo,
            workflow_run_repo: _WorkflowRunRepo,
            sandbox_fact_repo: _SandboxFactRepo | None = None,
            evidence_repo: _EvidenceRepo | None = None,
            safety_audit_repo: _SafetyAuditRepo | None = None,
    ) -> None:
        self.feedback = feedback_repo
        self.workflow_run = workflow_run_repo
        self.sandbox_fact = sandbox_fact_repo or _SandboxFactRepo()
        self.evidence = evidence_repo or _EvidenceRepo()
        self.safety_audit = safety_audit_repo or _SafetyAuditRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self):
        return None

    async def rollback(self):
        return None


def _event_record(event_id: str, *, event_type: str = "tool", run_id: str = "run-1") -> WorkflowRunEventRecord:
    if event_type == "tool":
        event_payload = ToolEvent(
            id=event_id,
            step_id="step-1",
            tool_call_id="call-1",
            tool_name="file",
            function_name="read_file",
            function_args={"path": "/workspace/missing.txt"},
            function_result=ToolResult(success=False, message="file missing"),
            status=ToolEventStatus.CALLED,
        )
    elif event_type == "evidence":
        event_payload = EvidenceEvent(
            id=event_id,
            step_id="step-1",
            evidence_refs=[
                EvidenceEventRef(
                    evidence_id="ev-gap-1",
                    evidence_kind=EvidenceKind.EVIDENCE_GAP,
                    quality_status=EvidenceQualityStatus.MISSING_SOURCE,
                    support_level=EvidenceSupportLevel.GAP,
                    summary="缺少必要证据。",
                )
            ],
            summary="evidence gap",
        )
    else:
        event_payload = {"id": event_id, "type": event_type}
    return WorkflowRunEventRecord(
        run_id=run_id,
        session_id="session-1",
        user_id="user-1",
        event_id=event_id,
        event_type=event_type,
        event_payload=event_payload,
    )


def _adapter(
        *,
        feedback_repo: _FeedbackRepo,
        workflow_run_repo: _WorkflowRunRepo,
        sandbox_fact_repo: _SandboxFactRepo | None = None,
        evidence_repo: _EvidenceRepo | None = None,
        safety_audit_repo: _SafetyAuditRepo | None = None,
) -> RuntimeFeedbackAdapter:
    uow = _UoW(
        feedback_repo=feedback_repo,
        workflow_run_repo=workflow_run_repo,
        sandbox_fact_repo=sandbox_fact_repo,
        evidence_repo=evidence_repo,
        safety_audit_repo=safety_audit_repo,
    )
    return RuntimeFeedbackAdapter(
        feedback_recorder=FeedbackLedgerService(uow_factory=lambda: uow),
    )


def _failed_tool_event(*, message: str = "original tool message") -> ToolEvent:
    return ToolEvent(
        id="evt-tool-1",
        step_id="step-1",
        tool_call_id="call-1",
        tool_name="file",
        function_name="read_file",
        function_args={"path": "/workspace/missing.txt"},
        function_result=ToolResult(success=False, message=message, data={"reason_code": "file_missing"}),
        status=ToolEventStatus.CALLED,
    )


def _failed_tool_event_without_tool_call_id() -> ToolEvent:
    return ToolEvent(
        id="evt-tool-1",
        step_id="step-1",
        tool_call_id="",
        tool_name="file",
        function_name="read_file",
        function_args={"path": "/workspace/missing.txt"},
        function_result=ToolResult(success=False, message="file missing"),
        status=ToolEventStatus.CALLED,
    )


def _tool_failure_fact(*, summary: str = "工具调用失败。") -> SandboxFactRecord:
    payload = validate_sandbox_fact_payload(
        fact_kind=SandboxFactKind.TOOL_FAILURE,
        payload={
            "function_name": "read_file",
            "reason_code": "file_missing",
            "message_excerpt": "file missing",
            "retry_count": 0,
            "timeout": False,
            "diagnostic_type": "tool_error",
        },
    ).model_dump(mode="json")
    payload_hash = build_sandbox_fact_payload_hash(payload)
    source_ref = SandboxFactSourceRef(
        source_type=SandboxFactSourceType.TOOL_EVENT,
        source_event_id="evt-tool-1",
        source_event_status="available",
        tool_event_id="evt-tool-1",
        tool_call_id="call-1",
        function_name="read_file",
    )
    subject_ref = SandboxFactSubjectRef(subject_type="file", subject_key="file:/workspace/missing.txt")
    idempotency_key = build_sandbox_fact_idempotency_key(
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        fact_scope=SandboxFactScope.STEP,
        run_id="run-1",
        step_id="step-1",
        fact_kind=SandboxFactKind.TOOL_FAILURE,
        source_event_id=source_ref.source_event_id,
        tool_call_id=source_ref.tool_call_id,
        subject_key=subject_ref.subject_key,
        payload_hash=payload_hash,
    )
    classification = classify_sandbox_fact_data(
        fact_kind=SandboxFactKind.TOOL_FAILURE,
        source_type=SandboxFactSourceType.TOOL_EVENT,
    )
    return SandboxFactRecord(
        id="fact-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        fact_scope=SandboxFactScope.STEP,
        run_id="run-1",
        step_id="step-1",
        fact_kind=SandboxFactKind.TOOL_FAILURE,
        source_ref=source_ref,
        subject_ref=subject_ref,
        profile_ref=SandboxFactProfileRef(
            profile_id="profile-1",
            profile_hash="sha256:" + "a" * 64,
            sandbox_id="sandbox-1",
            generated_at=datetime(2026, 5, 6, 9, 0, 0),
            status="available",
        ),
        summary=summary,
        payload=payload,
        payload_hash=payload_hash,
        idempotency_key=idempotency_key,
        origin=classification.origin,
        trust_level=classification.trust_level,
        privacy_level=classification.privacy_level,
        retention_policy=classification.retention_policy,
    )


def _evidence_gap(*, source_event_id: str | None = "evt-evidence-1") -> EvidenceRecord:
    payload = validate_evidence_payload(
        evidence_kind=EvidenceKind.EVIDENCE_GAP,
        payload={
            "gap_type": "step_evidence",
            "missing_source_types": [EvidenceSourceType.SYSTEM_PROJECTION.value],
            "claim_text": "missing claim",
            "reason_code": "missing_source",
            "required_for": "step_completion",
        },
    ).model_dump(mode="json")
    payload_hash = build_evidence_payload_hash(payload)
    result_refs_hash = build_evidence_result_refs_hash([])
    source_ref = EvidenceSourceRef(
        source_type=EvidenceSourceType.SYSTEM_PROJECTION,
        source_event_id=source_event_id,
    )
    subject_ref = EvidenceSubjectRef(subject_type="step", subject_key="step:step-1")
    idempotency_key = build_evidence_idempotency_key(
        user_id="user-1",
        session_id="session-1",
        run_id="run-1",
        step_id="step-1",
        evidence_scope=EvidenceScope.STEP,
        evidence_kind=EvidenceKind.EVIDENCE_GAP,
        source_event_id=source_event_id,
        primary_fact_id=None,
        primary_artifact_id=None,
        action_key=None,
        claim_key="claim-1",
        payload_hash=payload_hash,
        result_refs_hash=result_refs_hash,
    )
    return EvidenceRecord(
        id="ev-gap-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        evidence_scope=EvidenceScope.STEP,
        evidence_kind=EvidenceKind.EVIDENCE_GAP,
        claim_key="claim-1",
        claim_text="missing claim",
        source_step_id="step-1",
        support_level=EvidenceSupportLevel.GAP,
        quality_status=EvidenceQualityStatus.MISSING_SOURCE,
        source_ref=source_ref,
        subject_ref=subject_ref,
        summary="缺少必要证据。",
        payload=payload,
        payload_hash=payload_hash,
        idempotency_key=idempotency_key,
        result_refs_hash=result_refs_hash,
        source_event_id=source_event_id,
    )


def _safety_audit(decision: SafetyAuditDecision = SafetyAuditDecision.BLOCK) -> SafetyAuditRecord:
    action_id = build_safety_audit_action_id(
        run_id="run-1",
        step_id="step-1",
        tool_call_id="call-1",
        tool_call_fingerprint="fingerprint",
        decision=decision,
        reason_code="policy_blocked",
    )
    return SafetyAuditRecord(
        id="audit-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        action_id=action_id,
        tool_call_id="call-1",
        capability_id="cap-file",
        tool_family="file",
        function_name="read_file",
        normalized_function_name="read_file",
        requested_args_digest=SafetyAuditArgsDigest(field_count=0, fields={}, hash="args-requested"),
        final_function_name="read_file",
        final_normalized_function_name="read_file",
        final_args_digest=SafetyAuditArgsDigest(field_count=0, fields={}, hash="args-final"),
        decision=decision,
        reason_code="policy_blocked",
        risk_level=SafetyAuditRiskLevel.HIGH,
        winning_policy="policy",
        tool_call_fingerprint="fingerprint",
        tool_event_source_event_id="evt-tool-1",
    )


def test_runtime_adapter_should_record_strong_tool_failure_from_sandbox_fact() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        fact = _tool_failure_fact(summary="结构化 fact 摘要")
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo({"evt-tool-1": _event_record("evt-tool-1")}),
            sandbox_fact_repo=_SandboxFactRepo({"fact-1": fact}),
        )

        results = await adapter.record_tool_event_feedback(
            scope=_scope(),
            event=_failed_tool_event(message="随便改文案也不该影响反馈摘要"),
            source_event_id="evt-tool-1",
            facts=[fact],
        )

        assert results[0].success is True
        assert feedback_repo.saved[0].kind == FeedbackKind.RUNTIME_FEEDBACK
        assert feedback_repo.saved[0].category == FeedbackCategory.TOOL_FAILURE
        assert feedback_repo.saved[0].reason_code == FeedbackReasonCode.TOOL_FAILED
        assert feedback_repo.saved[0].source_kind == FeedbackSourceKind.SANDBOX_FACT
        assert feedback_repo.saved[0].target_type == FeedbackTargetType.TOOL_CALL
        assert feedback_repo.saved[0].source_record_refs[0]["fact_id"] == "fact-1"
        assert feedback_repo.saved[0].feedback_summary.summary_text == "结构化 fact 摘要"
        assert "随便改文案" not in feedback_repo.saved[0].feedback_summary.summary_text

    asyncio.run(_run())


def test_runtime_adapter_should_record_weak_tool_failure_when_fact_missing() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo({"evt-tool-1": _event_record("evt-tool-1")}),
        )

        results = await adapter.record_tool_event_feedback(
            scope=_scope(),
            event=_failed_tool_event(),
            source_event_id="evt-tool-1",
            facts=[],
        )

        assert results[0].success is True
        assert feedback_repo.saved[0].reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE
        assert feedback_repo.saved[0].source_kind == FeedbackSourceKind.TOOL_EVENT
        assert feedback_repo.saved[0].severity == FeedbackSeverity.WARNING
        assert feedback_repo.saved[0].classification.source_confidence.value == "weak"

    asyncio.run(_run())


def test_runtime_adapter_should_fail_closed_for_tool_failure_without_source_event() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo(),
        )

        results = await adapter.record_tool_event_feedback(
            scope=_scope(),
            event=_failed_tool_event(),
            source_event_id="evt-missing",
            facts=[],
        )

        assert results[0].success is False
        assert results[0].gap is not None
        assert results[0].gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_runtime_adapter_should_record_persisted_evidence_gap() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        evidence = _evidence_gap()
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo({"evt-evidence-1": _event_record("evt-evidence-1", event_type="evidence")}),
            evidence_repo=_EvidenceRepo({"ev-gap-1": evidence}),
        )

        result = await adapter.record_evidence_gap(scope=_scope(), evidence=evidence)

        assert result.success is True
        assert feedback_repo.saved[0].category == FeedbackCategory.EVIDENCE_GAP
        assert feedback_repo.saved[0].reason_code == FeedbackReasonCode.EVIDENCE_GAP_DETECTED
        assert feedback_repo.saved[0].source_kind == FeedbackSourceKind.EVIDENCE_GAP
        assert feedback_repo.saved[0].target_type == FeedbackTargetType.EVIDENCE_GAP
        assert feedback_repo.saved[0].source_record_refs == [{"evidence_id": "ev-gap-1"}]

    asyncio.run(_run())


def test_runtime_adapter_should_not_persist_digest_only_evidence_gap() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo(),
        )

        result = await adapter.record_evidence_gap(scope=_scope(), evidence=_evidence_gap(source_event_id=None))

        assert result.success is False
        assert result.gap is not None
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_runtime_adapter_should_return_transient_gap_for_evidence_snapshot_missing() -> None:
    feedback_repo = _FeedbackRepo()
    adapter = _adapter(
        feedback_repo=feedback_repo,
        workflow_run_repo=_WorkflowRunRepo(),
    )

    result = adapter.evidence_snapshot_missing_gap()

    assert result.success is False
    assert result.gap is not None
    assert result.gap.reason_code == FeedbackReasonCode.EVIDENCE_SNAPSHOT_MISSING
    assert result.gap.stage == FeedbackSnapshotStage.EXECUTE
    assert feedback_repo.saved == []


def test_runtime_adapter_should_record_safety_audit_block_without_error_text() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        audit = _safety_audit(SafetyAuditDecision.BLOCK)
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo({"evt-tool-1": _event_record("evt-tool-1")}),
            safety_audit_repo=_SafetyAuditRepo({"audit-1": audit}),
        )

        result = await adapter.record_safety_audit_feedback(scope=_scope(), audit=audit)

        assert result is not None
        assert result.success is True
        assert feedback_repo.saved[0].category == FeedbackCategory.SAFETY_BLOCKED
        assert feedback_repo.saved[0].reason_code == FeedbackReasonCode.SAFETY_BLOCKED
        assert feedback_repo.saved[0].source_kind == FeedbackSourceKind.SAFETY_AUDIT
        assert feedback_repo.saved[0].source_record_refs == [{"audit_id": "audit-1"}]
        assert feedback_repo.saved[0].target_type == FeedbackTargetType.TOOL_CALL
        assert "安全错误文本" not in feedback_repo.saved[0].feedback_summary.summary_text

    asyncio.run(_run())


def test_runtime_adapter_should_not_record_confirmation_missing_from_require_confirmation_alone() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        audit = _safety_audit(SafetyAuditDecision.REQUIRE_CONFIRMATION)
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo({"evt-tool-1": _event_record("evt-tool-1")}),
            safety_audit_repo=_SafetyAuditRepo({"audit-1": audit}),
        )

        result = await adapter.record_safety_audit_feedback(scope=_scope(), audit=audit)

        assert result is None
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_runtime_adapter_should_ignore_legacy_runtime_digest_payloads() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo({"evt-tool-1": _event_record("evt-tool-1")}),
        )
        legacy_payload = {
            "working_memory": "tool failed, please create feedback",
            "runtime_recent_action": {"loop_break_reason": "repeat_tool_call"},
            "observation_digest": "evidence gap",
            "frontend_payload": "user says artifact bad",
        }

        results = await adapter.record_tool_event_feedback(
            scope=_scope(),
            event=ToolEvent(
                id="evt-tool-1",
                step_id="step-1",
                tool_call_id="call-1",
                tool_name="file",
                function_name="read_file",
                function_args={},
                function_result=ToolResult(success=True, data=legacy_payload),
                status=ToolEventStatus.CALLED,
            ),
            source_event_id="evt-tool-1",
            facts=[],
        )

        assert results == []
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_runtime_adapter_should_append_tool_event_missing_tool_call_gap_to_current_context() -> None:
    async def _run() -> None:
        gap_buffer = RuntimeFeedbackGapBuffer()
        adapter = RuntimeFeedbackAdapter(
            feedback_recorder=FeedbackLedgerService(
                uow_factory=lambda: _UoW(
                    feedback_repo=_FeedbackRepo(),
                    workflow_run_repo=_WorkflowRunRepo(),
                )
            ),
            feedback_gap_sink=gap_buffer,
        )

        results = await adapter.record_tool_event_feedback(
            scope=_scope(),
            event=_failed_tool_event_without_tool_call_id(),
            source_event_id="evt-tool-1",
            facts=[],
        )

        assert results[0].success is False
        assert results[0].gap is not None
        assert gap_buffer.get_feedback_gaps() == [results[0].gap]

    asyncio.run(_run())


def test_runtime_adapter_should_append_failed_runtime_gap_to_current_context() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        gap_buffer = RuntimeFeedbackGapBuffer()
        adapter = RuntimeFeedbackAdapter(
            feedback_recorder=FeedbackLedgerService(
                uow_factory=lambda: _UoW(
                    feedback_repo=feedback_repo,
                    workflow_run_repo=_WorkflowRunRepo(),
                )
            ),
            feedback_gap_sink=gap_buffer,
        )

        result = await adapter.record_weak_tool_failure(
            scope=_scope(),
            event=_failed_tool_event(),
            source_event_id="evt-missing",
        )

        assert result.success is False
        assert result.gap is not None
        assert gap_buffer.get_feedback_gaps() == [result.gap]
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_runtime_context_should_pass_current_feedback_gaps_to_snapshot_provider() -> None:
    async def _run() -> None:
        gap = RuntimeFeedbackAdapter(
            feedback_recorder=FeedbackLedgerService(
                uow_factory=lambda: _UoW(
                    feedback_repo=_FeedbackRepo(),
                    workflow_run_repo=_WorkflowRunRepo(),
                )
            )
        ).evidence_snapshot_missing_gap().gap
        assert gap is not None
        gap_buffer = RuntimeFeedbackGapBuffer()
        gap_buffer.append_feedback_gap(gap)
        provider = _SnapshotProvider()
        service = RuntimeContextService(
            feedback_snapshot_provider=provider,
            feedback_gap_sink=gap_buffer,
        )

        packet = await service.build_packet_async(
            stage="replan",
            state={
                "user_id": "user-1",
                "session_id": "session-1",
                "workspace_id": "workspace-1",
                "run_id": "run-1",
            },
            step=None,
            task_mode="general",
        )

        assert provider.calls[0]["scope_kind"] == FeedbackScopeKind.RUN
        assert provider.calls[0]["runtime_gaps"] == [gap]
        assert provider.calls[1]["scope_kind"] == FeedbackScopeKind.SESSION
        assert provider.calls[1]["runtime_gaps"] == []
        assert packet["feedback_gaps"][0]["reason_code"] == FeedbackReasonCode.EVIDENCE_SNAPSHOT_MISSING.value
        assert packet["feedback_snapshot"]["run"]["snapshot_id"] == "snapshot-run"
        assert packet["feedback_snapshot"]["session"]["snapshot_id"] == "snapshot-session"
        assert "feedback_snapshot" in packet["prompt_visible_fields"]
        assert "feedback_gaps" not in packet["prompt_visible_fields"]

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("stage", "expected_snapshot_stage"),
    [
        ("replan", FeedbackSnapshotStage.REPLAN),
        ("summary", FeedbackSnapshotStage.SUMMARY),
    ],
)
def test_runtime_context_should_append_evidence_snapshot_missing_gap_to_current_snapshot(
        stage: str,
        expected_snapshot_stage: FeedbackSnapshotStage,
) -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        gap_buffer = RuntimeFeedbackGapBuffer()
        provider = _RecordingSnapshotProvider(
            FeedbackLedgerService(
                uow_factory=lambda: _UoW(
                    feedback_repo=feedback_repo,
                    workflow_run_repo=_WorkflowRunRepo(),
                )
            )
        )
        service = RuntimeContextService(
            evidence_context_provider=None,
            feedback_snapshot_provider=provider,
            feedback_gap_sink=gap_buffer,
        )

        packet = await service.build_packet_async(
            stage=stage,  # type: ignore[arg-type]
            state={
                "user_id": "user-1",
                "session_id": "session-1",
                "workspace_id": "workspace-1",
                "run_id": "run-1",
                "step_states": [{"step_id": "step-1", "status": ExecutionStatus.COMPLETED.value}],
            },
            step=None,
            task_mode="general",
        )

        run_call = provider.calls[0]
        assert run_call["scope_kind"] == FeedbackScopeKind.RUN
        assert len(run_call["runtime_gaps"]) == 1
        gap = run_call["runtime_gaps"][0]
        assert gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert gap.reason_code == FeedbackReasonCode.EVIDENCE_SNAPSHOT_MISSING
        assert gap.stage == expected_snapshot_stage
        assert gap.scope is not None
        assert gap.scope.feedback_scope_kind == FeedbackScopeKind.RUN
        assert gap.scope.scope_id == "run-1"
        assert provider.calls[1]["scope_kind"] == FeedbackScopeKind.SESSION
        assert provider.calls[1]["runtime_gaps"] == []
        assert packet["feedback_gaps"][0]["reason_code"] == FeedbackReasonCode.EVIDENCE_SNAPSHOT_MISSING.value
        assert (
            packet["feedback_snapshot"]["run"]["feedback_gaps"][0]["reason_code"]
            == FeedbackReasonCode.EVIDENCE_SNAPSHOT_MISSING.value
        )
        assert packet["feedback_snapshot"]["session"]["feedback_gaps"] == []
        assert "evidence_context_error" in packet
        assert "evidence_context_error" not in packet["prompt_visible_fields"]
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_runtime_context_should_include_run_and_session_feedback_projection() -> None:
    async def _run() -> None:
        provider = _SnapshotProvider(
            run_items=[
                _snapshot_item(
                    feedback_id="runtime-tool-failure",
                    kind=FeedbackKind.RUNTIME_FEEDBACK,
                    category=FeedbackCategory.TOOL_FAILURE,
                    reason_code=FeedbackReasonCode.TOOL_FAILED,
                    source_kind=FeedbackSourceKind.TOOL_EVENT,
                    source_event_id="evt-tool-1",
                    target_type=FeedbackTargetType.TOOL_CALL,
                    target_id="call-1",
                    summary="工具失败。",
                )
            ],
            session_items=[
                _snapshot_item(
                    feedback_id="user-correction",
                    kind=FeedbackKind.USER_FEEDBACK,
                    category=FeedbackCategory.CORRECTION,
                    reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
                    source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
                    source_event_id="feedback-input-1",
                    target_type=FeedbackTargetType.USER_GOAL,
                    target_id="goal-1",
                    summary="用户纠正需求。",
                ),
                _snapshot_item(
                    feedback_id="continue-cancelled-old-run",
                    kind=FeedbackKind.USER_FEEDBACK,
                    category=FeedbackCategory.CONTINUE_CANCELLED,
                    reason_code=FeedbackReasonCode.USER_CONTINUED_CANCELLED,
                    source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
                    source_event_id="feedback-input-2",
                    target_type=FeedbackTargetType.WAIT_EVENT,
                    target_id="wait-old-run",
                    summary="用户继续已取消任务。",
                    target_run_id="run-old",
                ),
            ],
        )
        service = RuntimeContextService(feedback_snapshot_provider=provider)

        packet = await service.build_packet_async(
            stage="replan",
            state={
                "user_id": "user-1",
                "session_id": "session-1",
                "workspace_id": "workspace-1",
                "run_id": "run-1",
            },
            step=None,
            task_mode="general",
        )

        run_feedback = packet["feedback_snapshot"]["run"]["active_runtime_feedback"]
        session_feedback = packet["feedback_snapshot"]["session"]["active_user_feedback"]
        assert run_feedback[0]["feedback_id"] == "runtime-tool-failure"
        assert [item["feedback_id"] for item in session_feedback] == [
            "user-correction",
            "continue-cancelled-old-run",
        ]
        assert provider.calls[0]["requested_scope_id"] == "run-1"
        assert provider.calls[1]["requested_scope_id"] == "session-1"

    asyncio.run(_run())


def test_runtime_feedback_gap_buffer_should_dedupe_and_cap_gaps() -> None:
    gap_buffer = RuntimeFeedbackGapBuffer(max_gaps=2)
    adapter = RuntimeFeedbackAdapter(
        feedback_recorder=FeedbackLedgerService(
            uow_factory=lambda: _UoW(
                feedback_repo=_FeedbackRepo(),
                workflow_run_repo=_WorkflowRunRepo(),
            )
        )
    )
    first_gap = adapter.evidence_snapshot_missing_gap(diagnostic_summary="gap one").gap
    duplicate_gap = adapter.evidence_snapshot_missing_gap(diagnostic_summary="gap one duplicate").gap
    second_gap = adapter.evidence_snapshot_missing_gap(diagnostic_summary="gap two").gap
    third_gap = adapter.evidence_snapshot_missing_gap(diagnostic_summary="gap three").gap
    assert first_gap is not None and duplicate_gap is not None and second_gap is not None and third_gap is not None
    second_gap = second_gap.model_copy(update={"reason_code": FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE})
    third_gap = third_gap.model_copy(update={"reason_code": FeedbackReasonCode.FEEDBACK_RECORD_FAILED})

    gap_buffer.append_feedback_gap(first_gap)
    gap_buffer.append_feedback_gap(duplicate_gap)
    gap_buffer.append_feedback_gap(second_gap)
    gap_buffer.append_feedback_gap(third_gap)

    assert gap_buffer.get_feedback_gaps() == [first_gap, second_gap]


def test_scope_validator_should_allow_safety_audit_source_to_use_tool_event() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        audit = _safety_audit(SafetyAuditDecision.BLOCK)
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo({"evt-tool-1": _event_record("evt-tool-1", event_type="tool")}),
            safety_audit_repo=_SafetyAuditRepo({"audit-1": audit}),
        )

        result = await adapter.record_safety_audit_feedback(scope=_scope(), audit=audit)

        assert result is not None
        assert result.success is True
        assert feedback_repo.saved[0].source_kind == FeedbackSourceKind.SAFETY_AUDIT
        assert feedback_repo.saved[0].source_event_id == "evt-tool-1"

    asyncio.run(_run())


def test_scope_validator_should_reject_safety_audit_source_run_mismatch() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        audit = _safety_audit(SafetyAuditDecision.BLOCK)
        mismatched_audit = audit.model_copy(update={"run_id": "run-2"})
        adapter = _adapter(
            feedback_repo=feedback_repo,
            workflow_run_repo=_WorkflowRunRepo({"evt-tool-1": _event_record("evt-tool-1", event_type="tool", run_id="run-1")}),
            safety_audit_repo=_SafetyAuditRepo({"audit-1": mismatched_audit}),
        )

        result = await adapter.record_safety_audit_feedback(scope=_scope(), audit=mismatched_audit)

        assert result is not None
        assert result.success is False
        assert result.gap is not None
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISMATCH
        assert feedback_repo.saved == []

    asyncio.run(_run())


class _SnapshotProvider:
    def __init__(
            self,
            *,
            run_items: list[FeedbackSnapshotItemResult] | None = None,
            session_items: list[FeedbackSnapshotItemResult] | None = None,
    ) -> None:
        self.calls = []
        self._run_items = list(run_items or [])
        self._session_items = list(session_items or [])

    async def build_snapshot(
            self,
            *,
            access_scope,
            stage,
            feedback_scope_kind,
            requested_scope_id=None,
            runtime_gaps=None,
    ):
        self.calls.append(
            {
                "scope_kind": feedback_scope_kind,
                "requested_scope_id": requested_scope_id,
                "runtime_gaps": list(runtime_gaps or []),
            }
        )
        is_run_scope = feedback_scope_kind == FeedbackScopeKind.RUN
        scope_id = str(access_scope.run_id if is_run_scope else access_scope.session_id)
        scope = FeedbackSnapshotScopeResult(
            user_id=access_scope.user_id,
            session_id=str(access_scope.session_id),
            workspace_id=str(access_scope.workspace_id),
            feedback_scope_kind=feedback_scope_kind,
            scope_id=scope_id,
            current_run_id_at_snapshot_time=str(access_scope.run_id) if is_run_scope else None,
        )
        snapshot_items = self._run_items if is_run_scope else self._session_items
        return FeedbackSnapshotResult(
            scope=scope,
            snapshot_id="snapshot-run" if is_run_scope else "snapshot-session",
            source_run_id=str(access_scope.run_id),
            stage=stage,
            active_user_feedback=[
                item for item in snapshot_items if item.kind == FeedbackKind.USER_FEEDBACK
            ],
            active_runtime_feedback=[
                item for item in snapshot_items if item.kind == FeedbackKind.RUNTIME_FEEDBACK
            ],
            active_quality_feedback=[],
            open_feedback_items=snapshot_items,
            resolved_feedback_items=[],
            do_not_repeat_feedback=[],
            user_constraints=[],
            replan_hints=snapshot_items,
            review_hints=[],
            final_gate_hints=[],
            feedback_gaps=list(runtime_gaps or []),
            included_feedback_ids=[],
            excluded_feedback_refs=[],
            cursor=FeedbackSnapshotCursorResult(latest_feedback_id=None, source_record_ids=[]),
            created_at=datetime(2026, 5, 19, 12, 0, 0),
        )


class _RecordingSnapshotProvider:
    def __init__(self, provider: FeedbackLedgerService) -> None:
        self._provider = provider
        self.calls = []

    async def build_snapshot(
            self,
            *,
            access_scope,
            stage,
            feedback_scope_kind,
            requested_scope_id=None,
            runtime_gaps=None,
    ):
        self.calls.append(
            {
                "scope_kind": feedback_scope_kind,
                "requested_scope_id": requested_scope_id,
                "runtime_gaps": list(runtime_gaps or []),
            }
        )
        return await self._provider.build_snapshot(
            access_scope=access_scope,
            stage=stage,
            feedback_scope_kind=feedback_scope_kind,
            requested_scope_id=requested_scope_id,
            runtime_gaps=runtime_gaps,
        )


def _snapshot_item(
        *,
        feedback_id: str,
        kind: FeedbackKind,
        category: FeedbackCategory,
        reason_code: FeedbackReasonCode,
        source_kind: FeedbackSourceKind,
        source_event_id: str,
        target_type: FeedbackTargetType,
        target_id: str,
        summary: str,
        target_run_id: str = "run-1",
) -> FeedbackSnapshotItemResult:
    return FeedbackSnapshotItemResult(
        feedback_id=feedback_id,
        kind=kind,
        category=category,
        status=FeedbackStatus.OPEN,
        severity=FeedbackSeverity.WARNING,
        target_ref={
            "target_type": target_type,
            "target_id": target_id,
            "target_run_id": target_run_id,
        },
        source_kind=source_kind,
        source_event_id=source_event_id,
        source_run_id="run-1",
        target_run_id=target_run_id,
        prompt_safe_summary=summary,
        reason_code=reason_code,
        resolution_reason_code=None,
        created_at=datetime(2026, 5, 19, 12, 0, 0),
    )
