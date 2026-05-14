import asyncio
import hashlib
import io

import pytest
from pydantic import ValidationError

from app.application.service.artifact_revision_projector import ArtifactRevisionProjector
from app.application.service.artifact_revision_resolver import ArtifactRevisionResolver
from app.application.service.evidence_digest_projector import EvidenceDigestProjector
from app.application.service.evidence_result_handle_resolver import EvidenceResultHandleResolver
from app.application.service.runtime_tool_event_persistence_service import RuntimeToolEventPersistenceService
from app.application.service.sandbox_fact_ledger_service import SandboxFactLedgerService
from app.domain.models import File, ToolEvent, ToolEventStatus, WorkspaceArtifactRevision
from app.domain.models.evidence import (
    EvidenceKind,
    EvidenceQualityStatus,
    EvidenceReadStrategy,
    EvidenceResolvedStatus,
    EvidenceRecord,
    EvidenceResultRef,
    EvidenceResultRefType,
    EvidenceReusePolicy,
    EvidenceScope,
    EvidenceSourceRef,
    EvidenceSourceType,
    EvidenceStalenessPolicy,
    EvidenceSubjectRef,
    EvidenceSupportLevel,
    build_evidence_idempotency_key,
    build_evidence_payload_hash,
    build_evidence_result_handle,
    build_evidence_result_refs_hash,
)
from app.domain.models.tool_result import ToolResult
from app.domain.services.runtime.contracts.sandbox_fact_ports import SandboxFactProjectionContext
from app.domain.services.runtime.contracts.sandbox_fact_contract import SandboxFactProfileRef
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionRegistrationCommand,
    ArtifactRevisionResolveCommand,
    ArtifactRevisionResolveStatus,
    ArtifactRevisionSourceKind,
    ArtifactStorageBackend,
    ArtifactStorageRef,
    ArtifactType,
)
from app.domain.services.workspace_runtime.projectors.sandbox_fact_tool_event_projector import SandboxFactToolEventProjector
from app.domain.services.workspace_runtime.projectors.tool_event_projector import ToolEventProjector
from app.domain.services.tools import CapabilityRegistry, ToolRuntimeAdapter


def _scope() -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        current_step_id="step-1",
    )


def _storage_ref(*, file_id: str = "file-1", content_hash: str = "sha256:" + "a" * 64) -> ArtifactStorageRef:
    return ArtifactStorageRef(
        storage_backend=ArtifactStorageBackend.FILE_STORAGE,
        object_key=f"objects/{file_id}.md",
        file_id=file_id,
        storage_hash=content_hash,
        size_bytes=128,
        mime_type="text/markdown",
    )


def _revision(
        *,
        revision_id: str = "revision-1",
        content_hash: str = "sha256:" + "a" * 64,
        path: str = "/workspace/report.md",
        revision_no: int = 1,
        delivery_state: ArtifactDeliveryState = ArtifactDeliveryState.CANDIDATE,
        run_id: str = "run-1",
        session_id: str = "session-1",
) -> WorkspaceArtifactRevision:
    return WorkspaceArtifactRevision(
        revision_id=revision_id,
        artifact_id="artifact-1",
        revision_no=revision_no,
        user_id="user-1",
        session_id=session_id,
        workspace_id="workspace-1",
        run_id=run_id,
        step_id="step-1",
        path=path,
        storage_ref=_storage_ref(content_hash=content_hash),
        content_hash=content_hash,
        storage_hash=content_hash,
        size_bytes=128,
        mime_type="text/markdown",
        artifact_type=ArtifactType.FILE,
        delivery_state=delivery_state,
        source_kind=ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
        source_event_id="event-1",
        source_fact_ids=["fact-1"],
        tool_call_id="tool-call-1",
        function_name="write_file",
    )


class _RevisionRepo:
    def __init__(self, revisions: list[WorkspaceArtifactRevision]) -> None:
        self.revisions = list(revisions)

    async def get_by_identity(self, **kwargs):
        for revision in self.revisions:
            if (
                    revision.user_id == kwargs["user_id"]
                    and revision.workspace_id == kwargs["workspace_id"]
                    and revision.session_id == kwargs["session_id"]
                    and revision.artifact_id == kwargs["artifact_id"]
                    and revision.revision_id == kwargs["revision_id"]
                    and revision.content_hash == kwargs["content_hash"]
            ):
                return revision
        return None


class _UoW:
    def __init__(self, revisions: list[WorkspaceArtifactRevision]) -> None:
        self.workspace_artifact_revision = _RevisionRepo(revisions)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _uow_factory(revisions: list[WorkspaceArtifactRevision]):
    return lambda: _UoW(revisions)


class _OutputStream:
    def __init__(self) -> None:
        self.sequence = 0
        self.messages: list[str] = []
        self.deleted: list[str] = []

    async def put(self, message: str) -> str:
        self.sequence += 1
        self.messages.append(message)
        return f"event-{self.sequence}"

    async def delete_message(self, message_id: str) -> bool:
        self.deleted.append(message_id)
        return True


class _Task:
    def __init__(self) -> None:
        self.output_stream = _OutputStream()


class _PersistingCoordinator:
    async def persist_runtime_event(self, *, session_id, event, projection=None, allow_status_transition=True):
        return type("PersistResult", (), {"event_inserted": True})()


class _ContextBuilder:
    async def build_for_tool_event(self, *, source_event_id: str, current_step_id: str | None = None):
        return SandboxFactProjectionContext(
            scope=_scope().model_copy(update={"current_step_id": current_step_id or "step-1"}),
            profile_ref=SandboxFactProfileRef(profile_hash="profile-hash", status="available"),
            sandbox_id="sandbox-1",
            source_event_id=source_event_id,
            current_step_id=current_step_id,
        )


class _Sandbox:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = dict(files)
        self.downloaded_paths: list[str] = []

    async def download_file(self, file_path: str):
        self.downloaded_paths.append(file_path)
        return io.BytesIO(self.files[file_path])


class _FileStorage:
    def __init__(self) -> None:
        self.uploaded: list[dict[str, object]] = []

    async def upload_file(self, upload_file, user_id=None):
        content = upload_file.file.read()
        upload_file.file.seek(0)
        self.uploaded.append(
            {
                "filename": upload_file.filename,
                "size": upload_file.size,
                "content": content,
                "user_id": user_id,
            }
        )
        return File(
            id=f"file-{len(self.uploaded)}",
            filename=upload_file.filename,
            key=f"objects/{upload_file.filename}",
            mime_type="text/markdown",
            size=len(content),
        )

    def get_file_url(self, file: File) -> str:
        return f"https://cdn.example.test/{file.key}"


class _Browser:
    async def screenshot(self) -> bytes:
        return b""


class _WorkspaceRuntime:
    async def get_latest_shell_tool_result(self):
        return ToolResult(success=False, data={"console_records": []})


class _SandboxFactRepo:
    def __init__(self) -> None:
        self.saved = []

    async def save_once(self, fact):
        self.saved.append(fact)
        return fact


class _SandboxFactUoW:
    def __init__(self, repo: _SandboxFactRepo) -> None:
        self.sandbox_fact = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Ledger:
    def __init__(self) -> None:
        self.commands: list[ArtifactRevisionRegistrationCommand] = []

    async def register_revision(self, *, command: ArtifactRevisionRegistrationCommand):
        self.commands.append(command)
        return object()


def _tool_event_persistence_service(
        *,
        sandbox_files: dict[str, bytes],
        ledger: _Ledger,
) -> RuntimeToolEventPersistenceService:
    sandbox = _Sandbox(sandbox_files)
    file_storage = _FileStorage()
    display_projector = ToolEventProjector(
        adapter=ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1()),
        browser=_Browser(),
        file_storage=file_storage,
        sandbox=sandbox,
        workspace_runtime_service=_WorkspaceRuntime(),
        user_id="user-1",
    )
    sandbox_fact_repo = _SandboxFactRepo()
    sandbox_fact_service = SandboxFactLedgerService(uow_factory=lambda: _SandboxFactUoW(sandbox_fact_repo))
    return RuntimeToolEventPersistenceService(
        session_id="session-1",
        task=_Task(),
        uow_factory=lambda: _UoW([]),
        runtime_state_coordinator=_PersistingCoordinator(),
        sandbox_fact_recorder=SandboxFactToolEventProjector(ledger_service=sandbox_fact_service),
        sandbox_fact_context_builder=_ContextBuilder(),
        tool_event_display_projector=display_projector,
        artifact_revision_projector=ArtifactRevisionProjector(ledger_service=ledger),
    )


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _artifact_ref(revision: WorkspaceArtifactRevision, *, content_hash: str | None = None) -> EvidenceResultRef:
    return EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
        ref_id=revision.revision_id,
        source_step_id=revision.step_id,
        source_fact_id=revision.source_fact_ids[0],
        source_event_id=revision.source_event_id,
        artifact_id=revision.artifact_id,
        revision_id=revision.revision_id,
        artifact_path=revision.path,
        artifact_version_locked=True,
        storage_ref=revision.storage_ref,
        content_hash=content_hash or revision.content_hash,
        quality_status=EvidenceQualityStatus.VALID,
        support_level=EvidenceSupportLevel.STRONG,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
        summary="artifact",
    )


def _artifact_record(revision: WorkspaceArtifactRevision, *, result_ref: EvidenceResultRef) -> EvidenceRecord:
    payload = {
        "artifact_id": revision.artifact_id,
        "revision_id": result_ref.revision_id,
        "content_hash": result_ref.content_hash,
        "storage_ref": revision.storage_ref.model_dump(mode="json"),
        "artifact_path": revision.path,
        "artifact_type": revision.artifact_type.value,
        "source_fact_ids": list(revision.source_fact_ids),
        "source_event_id": revision.source_event_id,
        "delivery_candidate": True,
        "version_locked": True,
    }
    payload_hash = build_evidence_payload_hash(payload)
    result_refs_hash = build_evidence_result_refs_hash([result_ref])
    return EvidenceRecord(
        id="evidence-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        evidence_scope=EvidenceScope.STEP,
        evidence_kind=EvidenceKind.ARTIFACT_EVIDENCE,
        subject_key=revision.path,
        source_step_id="step-1",
        support_level=EvidenceSupportLevel.STRONG,
        quality_status=EvidenceQualityStatus.VALID,
        source_ref=EvidenceSourceRef(
            source_type=EvidenceSourceType.SANDBOX_FACT,
            source_event_id=revision.source_event_id,
            fact_ids=list(revision.source_fact_ids),
            tool_call_id=revision.tool_call_id,
            artifact_ids=[revision.artifact_id],
        ),
        subject_ref=EvidenceSubjectRef(subject_type="artifact", subject_key=revision.path, artifact_path=revision.path),
        summary="artifact evidence",
        payload=payload,
        payload_hash=payload_hash,
        idempotency_key=build_evidence_idempotency_key(
            user_id="user-1",
            session_id="session-1",
            run_id="run-1",
            step_id="step-1",
            evidence_scope=EvidenceScope.STEP,
            evidence_kind=EvidenceKind.ARTIFACT_EVIDENCE,
            source_event_id=revision.source_event_id,
            primary_fact_id=revision.source_fact_ids[0],
            primary_artifact_id=revision.artifact_id,
            action_key=None,
            claim_key=None,
            payload_hash=payload_hash,
            result_refs_hash=result_refs_hash,
        ),
        reusable=True,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        result_refs=[result_ref],
        result_refs_hash=result_refs_hash,
        source_event_id=revision.source_event_id,
        tool_call_id=revision.tool_call_id,
        primary_fact_id=revision.source_fact_ids[0],
        primary_artifact_id=revision.artifact_id,
    )


class _EvidenceRepo:
    def __init__(self, records: list[EvidenceRecord]) -> None:
        self.records = records

    async def list_by_run(self, **_kwargs):
        return list(self.records)


class _DigestUoW(_UoW):
    def __init__(self, *, records: list[EvidenceRecord], revisions: list[WorkspaceArtifactRevision]) -> None:
        super().__init__(revisions)
        self.evidence = _EvidenceRepo(records)


def test_artifact_result_ref_should_require_revision_id() -> None:
    with pytest.raises(ValidationError):
        EvidenceResultRef(
            result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
            ref_id="artifact-1",
            artifact_id="artifact-1",
            content_hash="sha256:" + "a" * 64,
            quality_status=EvidenceQualityStatus.VALID,
            support_level=EvidenceSupportLevel.STRONG,
            reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
            staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
            read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
        )


def test_artifact_revision_resolver_should_resolve_by_strong_identity() -> None:
    revision = _revision()
    result = asyncio.run(
        ArtifactRevisionResolver(uow_factory=_uow_factory([revision])).resolve(
            ArtifactRevisionResolveCommand(
                user_id="user-1",
                workspace_id="workspace-1",
                session_id="session-1",
                artifact_id="artifact-1",
                revision_id="revision-1",
                content_hash=revision.content_hash,
                run_id="run-1",
            )
        )
    )

    assert result.status == ArtifactRevisionResolveStatus.RESOLVED
    assert result.revision == revision


def test_artifact_revision_resolver_should_not_read_current_when_hash_differs() -> None:
    revision = _revision(revision_id="revision-old", content_hash="sha256:" + "a" * 64)
    result = asyncio.run(
        ArtifactRevisionResolver(uow_factory=_uow_factory([revision])).resolve(
            ArtifactRevisionResolveCommand(
                user_id="user-1",
                workspace_id="workspace-1",
                session_id="session-1",
                artifact_id="artifact-1",
                revision_id="revision-old",
                content_hash="sha256:" + "b" * 64,
                run_id="run-1",
            )
        )
    )

    assert result.status == ArtifactRevisionResolveStatus.NOT_FOUND
    assert result.reason_code == "artifact_revision_not_found"


def test_evidence_result_handle_resolver_should_use_artifact_revision_resolver() -> None:
    revision = _revision()
    result_ref = EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
        ref_id=revision.revision_id,
        source_step_id="step-1",
        source_fact_id="fact-1",
        source_event_id="event-1",
        artifact_id=revision.artifact_id,
        revision_id=revision.revision_id,
        artifact_path=revision.path,
        storage_ref=revision.storage_ref,
        content_hash=revision.content_hash,
        quality_status=EvidenceQualityStatus.VALID,
        support_level=EvidenceSupportLevel.STRONG,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
        summary="report",
    )
    resolved = asyncio.run(
        EvidenceResultHandleResolver(uow_factory=_uow_factory([revision])).resolve(
            scope=_scope(),
            handle=build_evidence_result_handle(result_ref),
        )
    )

    assert resolved.status == EvidenceResolvedStatus.RESOLVED
    assert resolved.resolved_payload["revision_id"] == revision.revision_id
    assert resolved.content_hash == revision.content_hash


@pytest.mark.parametrize(
    ("function_name", "function_args", "result_data", "final_bytes"),
    [
        (
            "write_file",
            {"filepath": "/workspace/report.md", "content": "draft"},
            {"filepath": "/workspace/report.md", "content": "draft"},
            b"draft",
        ),
        (
            "write_file",
            {"filepath": "/workspace/report.md", "content": "ignored", "append": True},
            {"filepath": "/workspace/report.md", "content": "ignored"},
            b"existing\nappended",
        ),
        (
            "replace_in_file",
            {"filepath": "/workspace/report.md", "old_str": "old", "new_str": "new"},
            {"filepath": "/workspace/report.md", "after_content": "guessed"},
            b"new final bytes",
        ),
    ],
)
def test_tool_event_main_chain_should_register_revision_from_storage_hook_final_bytes(
        function_name: str,
        function_args: dict,
        result_data: dict,
        final_bytes: bytes,
) -> None:
    ledger = _Ledger()
    service = _tool_event_persistence_service(
        sandbox_files={"/workspace/report.md": final_bytes},
        ledger=ledger,
    )
    event = ToolEvent(
        id="tool-event-1",
        step_id="step-1",
        tool_call_id="tool-call-1",
        tool_name="file",
        function_name=function_name,
        function_args=function_args,
        function_result=ToolResult(success=True, data=result_data),
        status=ToolEventStatus.CALLED,
    )

    result = asyncio.run(
        service.persist_tool_event_and_record_facts(
            event=event,
            run_id="run-1",
            session_id="session-1",
            current_step_id="step-1",
        )
    )

    expected_hash = _sha256(final_bytes)
    assert result.artifact_revision_count == 1
    assert len(ledger.commands) == 1
    command = ledger.commands[0]
    assert command.path == "/workspace/report.md"
    assert command.storage_ref.storage_backend == ArtifactStorageBackend.FILE_STORAGE
    assert command.storage_ref.file_id == "file-1"
    assert command.storage_ref.object_key == "objects/report.md"
    assert command.size_bytes == len(final_bytes)
    assert command.content_hash == expected_hash
    assert command.storage_hash == expected_hash
    assert command.source_event_id == "event-1"
    assert command.source_fact_ids
    assert command.tool_call_id == "tool-call-1"
    assert command.function_name == function_name
    assert event.runtime_fact_projection["artifact_revision_count"] == 1


def test_evidence_digest_available_artifacts_should_require_resolved_revision() -> None:
    revision = _revision()
    record = _artifact_record(revision, result_ref=_artifact_ref(revision))
    digest = asyncio.run(
        EvidenceDigestProjector(
            uow_factory=lambda: _DigestUoW(records=[record], revisions=[revision])
        ).build_digest(
            scope=_scope(),
            current_step_id="step-1",
            completed_step_ids=["step-1"],
        )
    )

    assert digest is not None
    assert len(digest.available_artifacts) == 1
    available = digest.available_artifacts[0]
    assert available.version_locked is True
    assert available.revision_id == revision.revision_id
    assert available.content_hash == revision.content_hash


@pytest.mark.parametrize(
    ("record_revision", "stored_revision"),
    [
        (
            _revision(content_hash="sha256:" + "b" * 64),
            _revision(content_hash="sha256:" + "a" * 64),
        ),
        (
            _revision(),
            _revision(delivery_state=ArtifactDeliveryState.EXPIRED),
        ),
        (
            _revision(),
            _revision(delivery_state=ArtifactDeliveryState.QUARANTINED),
        ),
        (
            _revision(),
            _revision(run_id="run-other"),
        ),
        (
            _revision(),
            _revision(session_id="session-other"),
        ),
    ],
)
def test_evidence_digest_available_artifacts_should_skip_unresolved_revisions(
        record_revision: WorkspaceArtifactRevision,
        stored_revision: WorkspaceArtifactRevision,
) -> None:
    record = _artifact_record(record_revision, result_ref=_artifact_ref(record_revision))
    digest = asyncio.run(
        EvidenceDigestProjector(
            uow_factory=lambda: _DigestUoW(records=[record], revisions=[stored_revision])
        ).build_digest(
            scope=_scope(),
            current_step_id="step-1",
            completed_step_ids=["step-1"],
        )
    )

    assert digest is not None
    assert digest.available_artifacts == []
