import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from app.application.service.evidence_ledger_service import (
    EvidenceLedgerService,
    EvidenceScopeMismatchError,
    EvidenceSourceMissingError,
)
from app.application.service.evidence_fact_assembler import EvidenceFactAssembler
from app.application.service.evidence_ledger_inputs import EvidenceRecordInput
from app.application.service.evidence_result_handle_resolver import EvidenceResultHandleResolver
from app.domain.models import MessageEvent, WorkflowRunEventRecord, WorkspaceArtifact
from app.domain.models.evidence import (
    EvidenceKind,
    EvidenceQualityStatus,
    EvidenceReadStrategy,
    EvidenceResolvedStatus,
    EvidenceResultRef,
    EvidenceResultRefType,
    EvidenceReusePolicy,
    EvidenceScope,
    EvidenceSourceRef,
    EvidenceSourceType,
    EvidenceStalenessPolicy,
    EvidenceSubjectRef,
    EvidenceSupportLevel,
    build_evidence_result_handle,
)
from app.domain.models.sandbox_fact import (
    SandboxFactKind,
    SandboxFactRecord,
    SandboxFactScope,
    SandboxFactSourceRef,
    SandboxFactSourceType,
    SandboxFactSubjectRef,
    build_sandbox_fact_idempotency_key,
    build_sandbox_fact_payload_hash,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.document_input_contract import DocumentParseStatus
from app.infrastructure.repositories.db_workflow_run_repository import DBWorkflowRunRepository
from app.infrastructure.repositories.db_workspace_artifact_repository import DBWorkspaceArtifactRepository


class _EvidenceRepo:
    def __init__(self, records=None) -> None:
        self.saved = []
        self.records = {record.id: record for record in list(records or [])}
        self.list_reusable_calls = []
        self.list_action_subject_calls = []

    async def save_once(self, evidence):
        self.saved.append(evidence)
        self.records[evidence.id] = evidence
        return evidence

    async def list_by_ids(self, *, evidence_ids, **kwargs):
        return [self.records[evidence_id] for evidence_id in evidence_ids if evidence_id in self.records]

    async def list_reusable_by_run(self, **kwargs):
        self.list_reusable_calls.append(kwargs)
        return []

    async def list_by_action_subject(self, **kwargs):
        self.list_action_subject_calls.append(kwargs)
        return []


class _FactRepo:
    def __init__(self, facts=None) -> None:
        self.facts = {fact.id: fact for fact in list(facts or [])}

    async def list_by_ids(self, *, fact_ids, user_id, session_id, **kwargs):
        return [
            self.facts[fact_id]
            for fact_id in fact_ids
            if fact_id in self.facts
            and self.facts[fact_id].user_id == user_id
            and self.facts[fact_id].session_id == session_id
        ]


class _WorkflowRunRepo:
    def __init__(self, event_exists: bool = True) -> None:
        self.event_exists = event_exists
        self.event_calls = []

    async def get_event_record_by_event_id(self, **kwargs):
        self.event_calls.append(kwargs)
        if not self.event_exists:
            return None
        return WorkflowRunEventRecord(
            run_id=kwargs["run_id"],
            session_id=kwargs["session_id"],
            user_id=kwargs["user_id"],
            event_id=kwargs["event_id"],
            event_type="message",
            event_payload=MessageEvent(id=kwargs["event_id"], role="user", message="ok"),
        )


class _ArtifactRepo:
    def __init__(self, artifacts=None) -> None:
        self.artifacts = {artifact.id: artifact for artifact in list(artifacts or [])}
        self.calls = []

    async def get_by_user_workspace_id_and_id(self, *, user_id, workspace_id, artifact_id):
        self.calls.append({"user_id": user_id, "workspace_id": workspace_id, "artifact_id": artifact_id})
        artifact = self.artifacts.get(artifact_id)
        if artifact is None or artifact.user_id != user_id or artifact.workspace_id != workspace_id:
            return None
        return artifact.model_copy(deep=True)

    async def list_by_user_workspace_id(self, *args, **kwargs):
        raise AssertionError("PR2 read_artifact 禁止 list 全量 artifact 后过滤")


class _UoW:
    def __init__(
            self,
            *,
            evidence_repo=None,
            fact_repo=None,
            workflow_run_repo=None,
            artifact_repo=None,
    ) -> None:
        self.evidence = evidence_repo or _EvidenceRepo()
        self.sandbox_fact = fact_repo or _FactRepo()
        self.workflow_run = workflow_run_repo or _WorkflowRunRepo()
        self.workspace_artifact = artifact_repo or _ArtifactRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self):
        return None

    async def rollback(self):
        return None


def _scope(**overrides) -> AccessScopeResult:
    values = {
        "tenant_id": "user-1",
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "current_step_id": "step-1",
    }
    values.update(overrides)
    return AccessScopeResult(**values)


def _ledger_service(*, uow_factory) -> EvidenceLedgerService:
    return EvidenceLedgerService(uow_factory=uow_factory, assembler=EvidenceFactAssembler())


def _fact(
        *,
        fact_id: str = "fact-1",
        user_id: str = "user-1",
        session_id: str = "session-1",
        workspace_id: str = "workspace-1",
        run_id: str = "run-1",
        step_id: str = "step-1",
        payload: dict | None = None,
        fact_kind: SandboxFactKind = SandboxFactKind.FILE_READ,
) -> SandboxFactRecord:
    actual_payload = payload or {
        "path": "/workspace/a.txt",
        "exists": True,
        "size": 12,
        "content_sha256": "sha256:file",
        "content_sha256_kind": "read_content_sha256",
        "mime_type": "text/plain",
        "line_range": None,
        "excerpt": "safe excerpt",
        "is_truncated": False,
        "mtime": None,
        "missing_fields": None,
        "reason_code": None,
    }
    payload_hash = build_sandbox_fact_payload_hash(actual_payload)
    source_ref = SandboxFactSourceRef(
        source_type=SandboxFactSourceType.TOOL_EVENT,
        source_event_id="event-1",
        source_event_status="available",
        tool_call_id="tool-call-1",
        function_name="read_file",
    )
    subject_ref = SandboxFactSubjectRef(subject_type="file", subject_key="/workspace/a.txt", path="/workspace/a.txt")
    idempotency_key = build_sandbox_fact_idempotency_key(
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        fact_scope=SandboxFactScope.STEP,
        run_id=run_id,
        step_id=step_id,
        fact_kind=fact_kind,
        source_event_id=source_ref.source_event_id,
        tool_call_id=source_ref.tool_call_id,
        subject_key=subject_ref.subject_key,
        payload_hash=payload_hash,
    )
    return SandboxFactRecord(
        id=fact_id,
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        fact_scope=SandboxFactScope.STEP,
        run_id=run_id,
        step_id=step_id,
        fact_kind=fact_kind,
        source_ref=source_ref,
        subject_ref=subject_ref,
        summary="fact summary",
        payload=actual_payload,
        payload_hash=payload_hash,
        idempotency_key=idempotency_key,
    )


def _record_input(**overrides) -> EvidenceRecordInput:
    values = {
        "evidence_kind": EvidenceKind.FILE_EVIDENCE,
        "source_ref": EvidenceSourceRef(
            source_type=EvidenceSourceType.SANDBOX_FACT,
            source_event_id="event-1",
            fact_ids=["fact-1"],
            tool_call_id="tool-call-1",
        ),
        "subject_ref": EvidenceSubjectRef(subject_type="file", subject_key="/workspace/a.txt", path="/workspace/a.txt"),
        "support_level": EvidenceSupportLevel.STRONG,
        "quality_status": EvidenceQualityStatus.VALID,
        "action_key": "read_file:/workspace/a.txt",
        "summary": "api_key=abcdefghijklmnop " + "x" * 700,
        "payload": {
            "path": "/workspace/a.txt",
            "operation": "read",
            "exists": True,
            "content_sha256": "sha256:file",
            "content_sha256_kind": "read_content_sha256",
            "source_fact_id": "fact-1",
            "excerpt": "password=secret12345 " + "z" * 5000,
            "is_truncated": False,
        },
        "reusable": True,
        "reuse_policy": EvidenceReusePolicy.REUSE_ALLOWED,
        "staleness_policy": EvidenceStalenessPolicy.RUN_SCOPED,
        "result_refs": [_result_ref()],
    }
    values.update(overrides)
    return EvidenceRecordInput(**values)


def _result_ref(**overrides) -> EvidenceResultRef:
    values = {
        "result_ref_type": EvidenceResultRefType.FACT_REF,
        "ref_id": "fact-1",
        "source_step_id": "step-1",
        "source_evidence_id": "evidence-1",
        "source_fact_id": "fact-1",
        "source_event_id": "event-1",
        "subject_key": "/workspace/a.txt",
        "payload_hash": "sha256:payload",
        "quality_status": EvidenceQualityStatus.VALID,
        "support_level": EvidenceSupportLevel.STRONG,
        "reuse_policy": EvidenceReusePolicy.REUSE_ALLOWED,
        "staleness_policy": EvidenceStalenessPolicy.RUN_SCOPED,
        "read_strategy": EvidenceReadStrategy.READ_FACT_PAYLOAD,
        "summary": "safe summary",
    }
    values.update(overrides)
    return EvidenceResultRef(**values)


def _artifact(
        *,
        artifact_id: str = "artifact-1",
        session_id: str | None = "session-1",
        run_id: str | None = "run-1",
        source_step_id: str | None = "step-1",
        content_hash: str = "sha256:artifact",
) -> WorkspaceArtifact:
    return WorkspaceArtifact(
        id=artifact_id,
        user_id="user-1",
        session_id=session_id,
        workspace_id="workspace-1",
        run_id=run_id,
        path="/workspace/report.md",
        artifact_type="file",
        summary="report",
        source_step_id=source_step_id,
        metadata={"content_hash": content_hash} if content_hash else {},
    )


def test_record_evidence_should_generate_hashes_sanitize_and_save_once() -> None:
    evidence_repo = _EvidenceRepo()
    fact_repo = _FactRepo([_fact()])
    workflow_repo = _WorkflowRunRepo(event_exists=True)
    service = _ledger_service(
        uow_factory=lambda: _UoW(evidence_repo=evidence_repo, fact_repo=fact_repo, workflow_run_repo=workflow_repo)
    )

    evidence = asyncio.run(service.record_evidence(scope=_scope(), evidence_input=_record_input()))

    assert evidence_repo.saved == [evidence]
    assert evidence.user_id == "user-1"
    assert evidence.session_id == "session-1"
    assert evidence.workspace_id == "workspace-1"
    assert evidence.run_id == "run-1"
    assert evidence.step_id == "step-1"
    assert evidence.payload_hash.startswith("sha256:")
    assert evidence.idempotency_key.startswith("sha256:")
    assert evidence.result_refs_hash.startswith("sha256:")
    assert "[REDACTED]" in evidence.summary
    assert "[REDACTED]" in evidence.payload["excerpt"]
    assert "secret12345" not in str(evidence.payload)
    assert len(evidence.payload["excerpt"]) == 4000
    assert workflow_repo.event_calls[0]["event_id"] == "event-1"


def test_record_evidence_should_fail_closed_for_cross_scope_and_missing_sources() -> None:
    service_missing_event = _ledger_service(
        uow_factory=lambda: _UoW(fact_repo=_FactRepo([_fact()]), workflow_run_repo=_WorkflowRunRepo(event_exists=False))
    )
    with pytest.raises(EvidenceSourceMissingError):
        asyncio.run(service_missing_event.record_evidence(scope=_scope(), evidence_input=_record_input()))

    service_missing_fact = _ledger_service(
        uow_factory=lambda: _UoW(fact_repo=_FactRepo([]), workflow_run_repo=_WorkflowRunRepo(event_exists=True))
    )
    with pytest.raises(EvidenceSourceMissingError):
        asyncio.run(service_missing_fact.record_evidence(scope=_scope(), evidence_input=_record_input()))

    service_cross_run = _ledger_service(
        uow_factory=lambda: _UoW(fact_repo=_FactRepo([_fact(run_id="run-2")]), workflow_run_repo=_WorkflowRunRepo())
    )
    with pytest.raises(EvidenceScopeMismatchError):
        asyncio.run(service_cross_run.record_evidence(scope=_scope(), evidence_input=_record_input()))


def test_record_evidence_should_validate_payload_fact_refs() -> None:
    service_missing_payload_fact = _ledger_service(
        uow_factory=lambda: _UoW(fact_repo=_FactRepo([_fact()]), workflow_run_repo=_WorkflowRunRepo())
    )
    with pytest.raises(EvidenceScopeMismatchError):
        asyncio.run(
            service_missing_payload_fact.record_evidence(
                scope=_scope(),
                evidence_input=_record_input(
                    source_ref=EvidenceSourceRef(
                        source_type=EvidenceSourceType.SANDBOX_FACT,
                        source_event_id="event-1",
                        fact_ids=["fact-1"],
                    ),
                    payload={
                        "path": "/workspace/a.txt",
                        "operation": "read",
                        "exists": True,
                        "content_sha256": "sha256:file",
                        "content_sha256_kind": "read_content_sha256",
                        "source_fact_id": "fact-missing",
                        "excerpt": "safe",
                        "is_truncated": False,
                    },
                ),
            )
        )

    service_cross_session = _ledger_service(
        uow_factory=lambda: _UoW(fact_repo=_FactRepo([_fact(session_id="session-2")]), workflow_run_repo=_WorkflowRunRepo())
    )
    with pytest.raises(EvidenceSourceMissingError):
        asyncio.run(
            service_cross_session.record_evidence(
                scope=_scope(),
                evidence_input=_record_input(
                    source_ref=EvidenceSourceRef(
                        source_type=EvidenceSourceType.SANDBOX_FACT,
                        source_event_id="event-1",
                        fact_ids=["fact-1"],
                    ),
                ),
            )
        )

    service_cross_workspace = _ledger_service(
        uow_factory=lambda: _UoW(fact_repo=_FactRepo([_fact(workspace_id="workspace-2")]), workflow_run_repo=_WorkflowRunRepo())
    )
    with pytest.raises(EvidenceScopeMismatchError):
        asyncio.run(service_cross_workspace.record_evidence(scope=_scope(), evidence_input=_record_input()))


def test_record_evidence_should_reject_payload_refs_inconsistent_with_source_ref() -> None:
    service = _ledger_service(
        uow_factory=lambda: _UoW(
            fact_repo=_FactRepo([_fact(), _fact(fact_id="fact-2")]),
            workflow_run_repo=_WorkflowRunRepo(),
            artifact_repo=_ArtifactRepo([_artifact(), _artifact(artifact_id="artifact-2")]),
        )
    )

    with pytest.raises(EvidenceScopeMismatchError):
        asyncio.run(
            service.record_evidence(
                scope=_scope(),
                evidence_input=_record_input(
                    source_ref=EvidenceSourceRef(
                        source_type=EvidenceSourceType.SANDBOX_FACT,
                        source_event_id="event-1",
                        fact_ids=["fact-1"],
                    ),
                    result_refs=[_result_ref(source_fact_id="fact-1", payload_hash=_fact().payload_hash)],
                    payload={
                        "path": "/workspace/a.txt",
                        "operation": "read",
                        "exists": True,
                        "content_sha256": "sha256:file",
                        "content_sha256_kind": "read_content_sha256",
                        "source_fact_id": "fact-2",
                        "excerpt": "safe",
                        "is_truncated": False,
                    },
                ),
            )
        )

    with pytest.raises(EvidenceScopeMismatchError):
        asyncio.run(
            service.record_evidence(
                scope=_scope(),
                evidence_input=_record_input(
                    evidence_kind=EvidenceKind.ARTIFACT_EVIDENCE,
                    source_ref=EvidenceSourceRef(
                        source_type=EvidenceSourceType.ARTIFACT,
                        source_event_id="event-1",
                        artifact_ids=["artifact-1"],
                    ),
                    result_refs=[
                        _result_ref(
                            result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
                            ref_id="artifact-1",
                            source_fact_id=None,
                            artifact_id="artifact-1",
                            content_hash="sha256:artifact",
                            payload_hash=None,
                            read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
                        )
                    ],
                    payload={
                        "artifact_id": "artifact-2",
                        "artifact_path": "/workspace/report.md",
                        "artifact_type": "file",
                        "source_fact_ids": [],
                        "current_hash": "sha256:artifact",
                        "hash_kind": "content_hash",
                        "delivery_candidate": True,
                    },
                ),
            )
        )


def test_record_evidence_should_fail_closed_for_missing_or_cross_scope_artifact() -> None:
    artifact_ref = _result_ref(
        result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
        ref_id="artifact-1",
        source_fact_id=None,
        artifact_id="artifact-1",
        content_hash="sha256:artifact",
        read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
    )
    evidence_input = _record_input(
        source_ref=EvidenceSourceRef(
            source_type=EvidenceSourceType.ARTIFACT,
            source_event_id="event-1",
            artifact_ids=["artifact-1"],
        ),
        result_refs=[artifact_ref],
        payload={
            "artifact_id": "artifact-1",
            "artifact_path": "/workspace/report.md",
            "artifact_type": "file",
            "source_fact_ids": [],
            "current_hash": "sha256:artifact",
            "hash_kind": "content_hash",
            "delivery_candidate": True,
        },
        evidence_kind=EvidenceKind.ARTIFACT_EVIDENCE,
    )
    service = _ledger_service(
        uow_factory=lambda: _UoW(
            fact_repo=_FactRepo([]),
            workflow_run_repo=_WorkflowRunRepo(),
            artifact_repo=_ArtifactRepo([]),
        )
    )

    with pytest.raises(EvidenceSourceMissingError):
        asyncio.run(service.record_evidence(scope=_scope(), evidence_input=evidence_input))

    mismatch_input = _record_input(
        source_ref=EvidenceSourceRef(
            source_type=EvidenceSourceType.ARTIFACT,
            source_event_id="event-1",
            artifact_ids=["artifact-1"],
        ),
        result_refs=[artifact_ref],
        payload={
            "artifact_id": "artifact-2",
            "artifact_path": "/workspace/report.md",
            "artifact_type": "file",
            "source_fact_ids": [],
            "current_hash": "sha256:artifact",
            "hash_kind": "content_hash",
            "delivery_candidate": True,
        },
        evidence_kind=EvidenceKind.ARTIFACT_EVIDENCE,
    )
    service_with_artifact = _ledger_service(
        uow_factory=lambda: _UoW(
            workflow_run_repo=_WorkflowRunRepo(),
            artifact_repo=_ArtifactRepo([_artifact()]),
        )
    )
    with pytest.raises(EvidenceScopeMismatchError):
        asyncio.run(service_with_artifact.record_evidence(scope=_scope(), evidence_input=mismatch_input))


@pytest.mark.parametrize(
    "artifact",
    [
        _artifact(session_id="session-2"),
        _artifact(run_id="run-2"),
        _artifact(run_id=None),
        _artifact(source_step_id=None),
    ],
)
def test_record_evidence_should_fail_closed_for_artifact_scope_mismatch(artifact: WorkspaceArtifact) -> None:
    artifact_ref = _result_ref(
        result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
        ref_id=artifact.id,
        source_fact_id=None,
        artifact_id=artifact.id,
        content_hash="sha256:artifact",
        read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
    )
    service = _ledger_service(
        uow_factory=lambda: _UoW(
            workflow_run_repo=_WorkflowRunRepo(),
            artifact_repo=_ArtifactRepo([artifact]),
        )
    )

    with pytest.raises(EvidenceScopeMismatchError):
        asyncio.run(
            service.record_evidence(
                scope=_scope(),
                evidence_input=_record_input(
                    evidence_kind=EvidenceKind.ARTIFACT_EVIDENCE,
                    source_ref=EvidenceSourceRef(
                        source_type=EvidenceSourceType.ARTIFACT,
                        source_event_id="event-1",
                        artifact_ids=[artifact.id],
                    ),
                    result_refs=[artifact_ref],
                    payload={
                        "artifact_id": artifact.id,
                        "artifact_path": artifact.path,
                        "artifact_type": artifact.artifact_type,
                        "source_fact_ids": [],
                        "current_hash": "sha256:artifact",
                        "hash_kind": "content_hash",
                        "delivery_candidate": True,
                    },
                ),
            )
        )


def test_evidence_service_queries_should_use_scope_filters() -> None:
    repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(evidence_repo=repo))

    asyncio.run(service.list_reusable_by_run(scope=_scope(), run_id="run-1"))
    asyncio.run(
        service.list_by_action_subject(
            scope=_scope(),
            query=SimpleNamespace(run_id="run-1", action_key="read", subject_key="file", limit=10),
        )
    )

    assert repo.list_reusable_calls[0]["user_id"] == "user-1"
    assert repo.list_reusable_calls[0]["session_id"] == "session-1"
    assert repo.list_reusable_calls[0]["run_id"] == "run-1"
    assert repo.list_action_subject_calls[0]["action_key"] == "read"


def test_resolver_should_cover_digest_verify_and_not_readable() -> None:
    resolver = EvidenceResultHandleResolver(uow_factory=lambda: _UoW())

    digest_handle = build_evidence_result_handle(
        _result_ref(read_strategy=EvidenceReadStrategy.USE_DIGEST_SUMMARY, payload_hash="sha256:payload")
    )
    resolved = asyncio.run(resolver.resolve(scope=_scope(), handle=digest_handle))
    assert resolved.status == EvidenceResolvedStatus.RESOLVED
    assert resolved.resolved_payload["summary"] == "safe summary"

    verify_handle = build_evidence_result_handle(
        _result_ref(
            result_ref_type=EvidenceResultRefType.VERIFICATION_REF,
            ref_id="verify-1",
            source_fact_id=None,
            read_strategy=EvidenceReadStrategy.VERIFY_BEFORE_USE,
            reason_code="query_external_may_change",
            allowed_verification_actions=["verification_search"],
            payload_hash=None,
        )
    )
    verify = asyncio.run(resolver.resolve(scope=_scope(), handle=verify_handle))
    assert verify.status == EvidenceResolvedStatus.REQUIRES_VERIFICATION
    assert verify.reason_code == "query_external_may_change"

    not_readable_handle = build_evidence_result_handle(
        _result_ref(
            read_strategy=EvidenceReadStrategy.NOT_READABLE,
            reason_code="source_missing",
            payload_hash="sha256:payload",
        )
    )
    not_readable = asyncio.run(resolver.resolve(scope=_scope(), handle=not_readable_handle))
    assert not_readable.status == EvidenceResolvedStatus.NOT_READABLE
    assert not_readable.reason_code == "source_missing"


def test_resolver_should_read_fact_payload_and_detect_hash_mismatch() -> None:
    fact = _fact()
    resolver = EvidenceResultHandleResolver(uow_factory=lambda: _UoW(fact_repo=_FactRepo([fact])))

    handle = build_evidence_result_handle(_result_ref(payload_hash=fact.payload_hash))
    resolved = asyncio.run(resolver.resolve(scope=_scope(), handle=handle))

    assert resolved.status == EvidenceResolvedStatus.RESOLVED
    assert resolved.payload_hash == fact.payload_hash
    assert "stdout" not in str(resolved.resolved_payload)

    stale_handle = build_evidence_result_handle(_result_ref(payload_hash="sha256:old"))
    stale = asyncio.run(resolver.resolve(scope=_scope(), handle=stale_handle))
    assert stale.status == EvidenceResolvedStatus.STALE
    assert stale.reason_code == "fact_payload_hash_mismatch"

    missing_hash_ref = _result_ref(
        ref_id="event-1",
        result_ref_type=EvidenceResultRefType.SOURCE_EVENT_REF,
        source_fact_id="fact-1",
        source_event_id="event-1",
        read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
        payload_hash=None,
    )
    missing_hash_handle = build_evidence_result_handle(missing_hash_ref)
    missing_hash = asyncio.run(resolver.resolve(scope=_scope(), handle=missing_hash_handle))
    assert missing_hash.status == EvidenceResolvedStatus.NOT_READABLE
    assert missing_hash.reason_code == "fact_payload_hash_missing"


def test_resolver_should_read_artifact_by_id_and_detect_hash_states() -> None:
    artifact = WorkspaceArtifact(
        id="artifact-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        path="/workspace/report.md",
        artifact_type="file",
        summary="report",
        source_step_id="step-1",
        metadata={"content_hash": "sha256:artifact", "secret": "hidden"},
    )
    artifact_repo = _ArtifactRepo([artifact])
    resolver = EvidenceResultHandleResolver(uow_factory=lambda: _UoW(artifact_repo=artifact_repo))
    handle = build_evidence_result_handle(
        _result_ref(
            result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
            ref_id="artifact-1",
            source_fact_id=None,
            artifact_id="artifact-1",
            artifact_path="/workspace/report.md",
            content_hash="sha256:artifact",
            payload_hash=None,
            read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
        )
    )

    resolved = asyncio.run(resolver.resolve(scope=_scope(), handle=handle))

    assert resolved.status == EvidenceResolvedStatus.RESOLVED
    assert artifact_repo.calls[0]["artifact_id"] == "artifact-1"
    assert resolved.resolved_payload["metadata"] == {"content_hash": "sha256:artifact"}

    changed_handle = build_evidence_result_handle(
        _result_ref(
            result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
            ref_id="artifact-1",
            source_fact_id=None,
            artifact_id="artifact-1",
            content_hash="sha256:old",
            payload_hash=None,
            read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
        )
    )
    changed = asyncio.run(resolver.resolve(scope=_scope(), handle=changed_handle))
    assert changed.status == EvidenceResolvedStatus.STALE
    assert changed.reason_code == "artifact_hash_changed"

    missing_hash_artifact = artifact.model_copy(update={"metadata": {}})
    missing_hash_resolver = EvidenceResultHandleResolver(
        uow_factory=lambda: _UoW(artifact_repo=_ArtifactRepo([missing_hash_artifact]))
    )
    missing_hash = asyncio.run(missing_hash_resolver.resolve(scope=_scope(), handle=handle))
    assert missing_hash.status == EvidenceResolvedStatus.NOT_READABLE
    assert missing_hash.reason_code == "artifact_hash_missing"


@pytest.mark.parametrize(
    "artifact",
    [
        _artifact(session_id="session-2"),
        _artifact(run_id="run-2"),
        _artifact(run_id=None),
        _artifact(source_step_id=None),
    ],
)
def test_resolver_should_fail_closed_for_artifact_scope_mismatch(artifact: WorkspaceArtifact) -> None:
    handle = build_evidence_result_handle(
        _result_ref(
            result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
            ref_id=artifact.id,
            source_fact_id=None,
            artifact_id=artifact.id,
            content_hash="sha256:artifact",
            payload_hash=None,
            read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
        )
    )
    resolver = EvidenceResultHandleResolver(uow_factory=lambda: _UoW(artifact_repo=_ArtifactRepo([artifact])))

    result = asyncio.run(resolver.resolve(scope=_scope(), handle=handle))

    assert result.status == EvidenceResolvedStatus.SCOPE_MISMATCH


def test_resolver_should_read_document_source_without_full_file_download() -> None:
    document_payload = {
        "file_id": "file-1",
        "filename_extension": ".pdf",
        "mime_type": "application/pdf",
        "parse_status": DocumentParseStatus.PARSED.value,
        "reason_code": None,
        "full_file_sha256": "sha256:full",
        "read_content_sha256": "sha256:read",
        "is_truncated": False,
        "excerpt_char_count": 120,
        "missing_fields": None,
    }
    fact = _fact(
        fact_id="fact-doc",
        fact_kind=SandboxFactKind.DOCUMENT_CONTEXT,
        payload=document_payload,
    )
    handle = build_evidence_result_handle(
        _result_ref(
            result_ref_type=EvidenceResultRefType.DOCUMENT_SOURCE_REF,
            ref_id="file-1",
            source_fact_id="fact-doc",
            document_file_id="file-1",
            content_hash="sha256:read",
            payload_hash=None,
            read_strategy=EvidenceReadStrategy.READ_DOCUMENT_SOURCE,
        )
    )
    resolver = EvidenceResultHandleResolver(uow_factory=lambda: _UoW(fact_repo=_FactRepo([fact])))

    resolved = asyncio.run(resolver.resolve(scope=_scope(), handle=handle))

    assert resolved.status == EvidenceResolvedStatus.RESOLVED
    assert resolved.resolved_payload == {
        "file_id": "file-1",
        "parse_status": "parsed",
        "reason_code": None,
        "full_file_sha256": "sha256:full",
        "read_content_sha256": "sha256:read",
        "is_truncated": False,
        "excerpt_char_count": 120,
    }
    assert "document_text" not in str(resolved.resolved_payload)

    missing_hash_handle = build_evidence_result_handle(
        _result_ref(
            result_ref_type=EvidenceResultRefType.DOCUMENT_SOURCE_REF,
            ref_id="file-1",
            source_fact_id="fact-doc",
            document_file_id="file-1",
            payload_hash=None,
            content_hash=None,
            read_strategy=EvidenceReadStrategy.READ_DOCUMENT_SOURCE,
        )
    )
    missing_hash = asyncio.run(resolver.resolve(scope=_scope(), handle=missing_hash_handle))
    assert missing_hash.status == EvidenceResolvedStatus.NOT_READABLE
    assert missing_hash.reason_code == "document_source_hash_missing"

    stale_handle = build_evidence_result_handle(
        _result_ref(
            result_ref_type=EvidenceResultRefType.DOCUMENT_SOURCE_REF,
            ref_id="file-1",
            source_fact_id="fact-doc",
            document_file_id="file-1",
            content_hash="sha256:old",
            payload_hash=None,
            read_strategy=EvidenceReadStrategy.READ_DOCUMENT_SOURCE,
        )
    )
    stale = asyncio.run(resolver.resolve(scope=_scope(), handle=stale_handle))
    assert stale.status == EvidenceResolvedStatus.STALE
    assert stale.reason_code == "document_source_hash_mismatch"


def test_workflow_run_event_query_should_use_user_session_run_event_filters() -> None:
    db_session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None)))
    repository = DBWorkflowRunRepository(db_session=db_session)

    result = asyncio.run(
        repository.get_event_record_by_event_id(
            user_id="user-1",
            session_id="session-1",
            run_id="run-1",
            event_id="event-1",
        )
    )

    assert result is None
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "workflow_run_events.user_id" in compiled_sql
    assert "workflow_run_events.session_id" in compiled_sql
    assert "workflow_run_events.run_id" in compiled_sql
    assert "workflow_run_events.event_id" in compiled_sql


def test_workspace_artifact_query_by_id_should_use_strong_filters() -> None:
    db_session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None)))
    repository = DBWorkspaceArtifactRepository(db_session=db_session)

    result = asyncio.run(
        repository.get_by_user_workspace_id_and_id(
            user_id="user-1",
            workspace_id="workspace-1",
            artifact_id="artifact-1",
        )
    )

    assert result is None
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "workspace_artifacts.user_id" in compiled_sql
    assert "workspace_artifacts.workspace_id" in compiled_sql
    assert "workspace_artifacts.id" in compiled_sql
