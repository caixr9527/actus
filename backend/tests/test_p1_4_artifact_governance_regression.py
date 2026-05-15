import asyncio
import io

import pytest

from app.application.errors import NotFoundError, ValidationError
from app.application.service.artifact_delivery_service import ArtifactDeliveryConflictError, ArtifactDeliveryService
from app.application.service.artifact_ledger_service import ArtifactLedgerService
from app.domain.models import File, Workspace, WorkspaceArtifact, WorkspaceArtifactRevision
from app.domain.services.runtime.artifact_file_hash import calculate_sha256_stream, verify_file_storage_stream
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactEventPayload,
    ArtifactRevisionRegistrationCommand,
    ArtifactRevisionSourceKind,
    ArtifactStorageBackend,
    ArtifactStorageRef,
    ArtifactType,
)


def _hash(content: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(content).hexdigest()


def _scope() -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        current_step_id="step-1",
    )


def _revision(
        *,
        revision_id: str = "revision-1",
        artifact_id: str = "artifact-1",
        content: bytes = b"hello",
        file_id: str = "file-1",
        backend: ArtifactStorageBackend = ArtifactStorageBackend.FILE_STORAGE,
        session_id: str = "session-1",
) -> WorkspaceArtifactRevision:
    content_hash = _hash(content)
    if backend == ArtifactStorageBackend.FILE_STORAGE:
        storage_ref = ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.FILE_STORAGE,
            file_id=file_id,
            object_key=f"objects/{file_id}.md",
            storage_hash=content_hash,
            size_bytes=len(content),
            mime_type="text/markdown",
        )
    else:
        storage_ref = ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.SANDBOX,
            sandbox_path="/workspace/report.md",
            size_bytes=len(content),
            mime_type="text/markdown",
        )
    return WorkspaceArtifactRevision(
        revision_id=revision_id,
        artifact_id=artifact_id,
        revision_no=1,
        user_id="user-1",
        session_id=session_id,
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        path="/workspace/report.md",
        storage_ref=storage_ref,
        content_hash=content_hash,
        storage_hash=content_hash if backend == ArtifactStorageBackend.FILE_STORAGE else None,
        size_bytes=len(content),
        mime_type="text/markdown",
        artifact_type=ArtifactType.FILE,
        delivery_state=ArtifactDeliveryState.CANDIDATE,
        source_kind=ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
        source_event_id="event-1",
        source_run_id="run-1",
        source_fact_ids=["fact-1"],
        tool_call_id="tool-call-1",
        function_name="write_file",
        profile_hash="sha256:" + "a" * 64,
        profile_status="available",
    )


class _AccessControl:
    async def assert_session_access(self, *, user_id, session_id, action):
        if user_id != "user-1" or session_id != "session-1":
            raise NotFoundError()
        return _scope()


class _RevisionRepo:
    def __init__(self, revisions: list[WorkspaceArtifactRevision]) -> None:
        self.revisions = revisions

    async def get_by_identity(self, *, user_id, workspace_id, session_id, artifact_id, revision_id, content_hash):
        for revision in self.revisions:
            if (
                    revision.user_id == user_id
                    and revision.workspace_id == workspace_id
                    and revision.session_id == session_id
                    and revision.artifact_id == artifact_id
                    and revision.revision_id == revision_id
                    and revision.content_hash == content_hash
            ):
                return revision.model_copy(deep=True)
        return None


class _DeliveryUoW:
    def __init__(self, revisions: list[WorkspaceArtifactRevision]) -> None:
        self.workspace_artifact_revision = _RevisionRepo(revisions)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FileStorage:
    def __init__(self, files: dict[str, tuple[bytes, File]]) -> None:
        self.files = files
        self.download_calls: list[tuple[str, str | None]] = []

    async def download_file(self, file_id: str, user_id: str | None = None):
        self.download_calls.append((file_id, user_id))
        content, file = self.files[file_id]
        return io.BytesIO(content), file.model_copy(deep=True)


def test_p1_4_artifact_governance_hash_helper_should_stream_and_verify_large_file() -> None:
    content = b"a" * (1024 * 1024 + 17)
    content_hash, stream = calculate_sha256_stream(io.BytesIO(content))

    assert content_hash == _hash(content)
    assert stream.read() == content

    verified = verify_file_storage_stream(
        stream=io.BytesIO(content),
        file=File(id="file-1", filename="large.bin", size=len(content)),
        expected_content_hash=content_hash,
    )

    assert verified.content_hash == content_hash
    assert verified.stream.read() == content


def test_p1_4_artifact_governance_download_should_read_locked_revision_not_current_path() -> None:
    old_content = b"old report"
    new_content = b"new report"
    old_revision = _revision(revision_id="revision-old", content=old_content, file_id="file-old")
    new_revision = _revision(revision_id="revision-new", content=new_content, file_id="file-new")
    file_storage = _FileStorage({
        "file-old": (old_content, File(id="file-old", filename="report.md", size=len(old_content), mime_type="text/markdown")),
        "file-new": (new_content, File(id="file-new", filename="report.md", size=len(new_content), mime_type="text/markdown")),
    })
    service = ArtifactDeliveryService(
        uow_factory=lambda: _DeliveryUoW([old_revision, new_revision]),
        file_storage=file_storage,
        access_control_service=_AccessControl(),
    )

    result = asyncio.run(service.download_revision(
        user_id="user-1",
        session_id="session-1",
        artifact_id="artifact-1",
        revision_id="revision-old",
        content_hash=old_revision.content_hash,
    ))

    assert result.stream.read() == old_content
    assert file_storage.download_calls == [("file-old", "user-1")]


def test_p1_4_artifact_governance_download_should_reject_cross_session_and_hash_mismatch() -> None:
    revision = _revision()
    service = ArtifactDeliveryService(
        uow_factory=lambda: _DeliveryUoW([revision]),
        file_storage=_FileStorage({
            "file-1": (b"hello", File(id="file-1", filename="report.md", size=5, mime_type="text/markdown")),
        }),
        access_control_service=_AccessControl(),
    )

    with pytest.raises(NotFoundError):
        asyncio.run(service.download_revision(
            user_id="user-1",
            session_id="session-2",
            artifact_id="artifact-1",
            revision_id="revision-1",
            content_hash=revision.content_hash,
        ))
    with pytest.raises(ValidationError):
        asyncio.run(service.download_revision(
            user_id="user-1",
            session_id="session-1",
            artifact_id="artifact-1",
            revision_id="revision-1",
            content_hash="not-a-sha",
        ))
    with pytest.raises(NotFoundError):
        asyncio.run(service.download_revision(
            user_id="user-1",
            session_id="session-1",
            artifact_id="artifact-1",
            revision_id="revision-1",
            content_hash="sha256:" + "b" * 64,
        ))


def test_p1_4_artifact_governance_download_should_reject_sandbox_revision() -> None:
    revision = _revision(backend=ArtifactStorageBackend.SANDBOX)
    service = ArtifactDeliveryService(
        uow_factory=lambda: _DeliveryUoW([revision]),
        file_storage=_FileStorage({}),
        access_control_service=_AccessControl(),
    )

    with pytest.raises(ArtifactDeliveryConflictError) as exc:
        asyncio.run(service.preview_revision(
            user_id="user-1",
            session_id="session-1",
            artifact_id="artifact-1",
            revision_id="revision-1",
            content_hash=revision.content_hash,
        ))

    assert exc.value.error_params == {"reason_code": "artifact_storage_not_deliverable"}


def test_p1_4_artifact_governance_download_should_reject_file_storage_hash_changed() -> None:
    revision = _revision(content=b"locked content")
    service = ArtifactDeliveryService(
        uow_factory=lambda: _DeliveryUoW([revision]),
        file_storage=_FileStorage({
            "file-1": (
                b"tampered content",
                File(id="file-1", filename="report.md", size=16, mime_type="text/markdown"),
            ),
        }),
        access_control_service=_AccessControl(),
    )

    with pytest.raises(ArtifactDeliveryConflictError) as exc:
        asyncio.run(service.download_revision(
            user_id="user-1",
            session_id="session-1",
            artifact_id="artifact-1",
            revision_id="revision-1",
            content_hash=revision.content_hash,
        ))

    assert exc.value.status_code == 409
    assert exc.value.error_params == {"reason_code": "artifact_hash_changed"}


class _ArtifactRepo:
    def __init__(self) -> None:
        self.artifact = WorkspaceArtifact(
            id="artifact-1",
            workspace_id="workspace-1",
            user_id="user-1",
            session_id="session-1",
            run_id="run-1",
            path="/workspace/report.md",
            artifact_type="file",
            delivery_state="candidate",
        )

    async def get_by_user_workspace_id_and_path(self, user_id, workspace_id, path):
        if user_id == self.artifact.user_id and workspace_id == self.artifact.workspace_id and path == self.artifact.path:
            return self.artifact.model_copy(deep=True)
        return None

    async def insert_current_index_if_absent(self, artifact):
        return None

    async def list_by_user_workspace_id_and_paths(self, user_id, workspace_id, paths):
        return [self.artifact.model_copy(deep=True)]


class _LedgerRevisionRepo:
    def __init__(self, revision: WorkspaceArtifactRevision) -> None:
        self.revision = revision

    async def append_revision_for_artifact(self, revision):
        return self.revision.model_copy(deep=True)


class _WorkflowRunRepo:
    def __init__(self) -> None:
        self.events = []

    async def add_event_record_if_absent(self, session_id, run_id, event):
        for existing_session_id, existing_run_id, existing_event in self.events:
            if existing_session_id == session_id and existing_run_id == run_id and existing_event.id == event.id:
                return False
        self.events.append((session_id, run_id, event))
        return True


class _LedgerUoW:
    def __init__(self, workflow_run: _WorkflowRunRepo) -> None:
        self.workspace_artifact = _ArtifactRepo()
        self.workspace_artifact_revision = _LedgerRevisionRepo(_revision())
        self.workflow_run = workflow_run

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_p1_4_artifact_governance_revision_registration_should_emit_artifact_event_with_revision_refs() -> None:
    workflow_run = _WorkflowRunRepo()
    service = ArtifactLedgerService(uow_factory=lambda: _LedgerUoW(workflow_run))
    revision = _revision()

    asyncio.run(service.register_revision(
        command=ArtifactRevisionRegistrationCommand(
            scope=_scope(),
            path=revision.path,
            storage_ref=revision.storage_ref,
            content_hash=revision.content_hash,
            storage_hash=revision.storage_hash,
            size_bytes=revision.size_bytes,
            mime_type=revision.mime_type,
            artifact_type=revision.artifact_type,
            delivery_state=revision.delivery_state,
            source_kind=revision.source_kind,
            source_event_id=revision.source_event_id or "",
            source_run_id=revision.source_run_id,
            source_fact_ids=list(revision.source_fact_ids),
            tool_call_id=revision.tool_call_id,
            function_name=revision.function_name,
            profile_hash=revision.profile_hash,
            profile_status=revision.profile_status,
        )
    ))

    assert len(workflow_run.events) == 1
    session_id, run_id, event = workflow_run.events[0]
    assert session_id == "session-1"
    assert run_id == "run-1"
    assert event.type == "artifact"
    assert event.id == f"artifact-revision:{revision.revision_id}"
    assert event.payload.revision_refs[0].revision_id == revision.revision_id
    assert event.payload.revision_refs[0].content_hash == revision.content_hash


def test_p1_4_artifact_governance_revision_registration_should_not_duplicate_artifact_event_for_existing_revision() -> None:
    workflow_run = _WorkflowRunRepo()
    service = ArtifactLedgerService(uow_factory=lambda: _LedgerUoW(workflow_run))
    revision = _revision()
    command = ArtifactRevisionRegistrationCommand(
        scope=_scope(),
        path=revision.path,
        storage_ref=revision.storage_ref,
        content_hash=revision.content_hash,
        storage_hash=revision.storage_hash,
        size_bytes=revision.size_bytes,
        mime_type=revision.mime_type,
        artifact_type=revision.artifact_type,
        delivery_state=revision.delivery_state,
        source_kind=revision.source_kind,
        source_event_id=revision.source_event_id or "",
        source_run_id=revision.source_run_id,
        source_fact_ids=list(revision.source_fact_ids),
        tool_call_id=revision.tool_call_id,
        function_name=revision.function_name,
        profile_hash=revision.profile_hash,
        profile_status=revision.profile_status,
    )

    first = asyncio.run(service.register_revision(command=command))
    second = asyncio.run(service.register_revision(command=command))

    assert first.revision_id == revision.revision_id
    assert second.revision_id == revision.revision_id
    assert [(run_id, event.id) for _, run_id, event in workflow_run.events] == [
        ("run-1", f"artifact-revision:{revision.revision_id}")
    ]


def test_p1_4_artifact_governance_artifact_event_payload_should_require_revision_identity() -> None:
    with pytest.raises(Exception):
        ArtifactEventPayload(
            revision_refs=[
                {
                    "artifact_id": "artifact-1",
                    "path": "/workspace/report.md",
                    "artifact_type": "file",
                    "delivery_state": "candidate",
                    "source_event_id": "event-1",
                }
            ],
        )
