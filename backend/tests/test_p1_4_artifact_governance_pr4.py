import asyncio
import hashlib
import io

import pytest

from app.application.service.artifact_ledger_service import ArtifactLedgerService
from app.application.service.derived_export_projector import DerivedExportError, DerivedExportProjector
from app.application.service.final_message_artifact_projector import FinalMessageArtifactProjector
from app.domain.models import (
    File,
    MessageEvent,
    Session,
    Workspace,
    WorkspaceArtifact,
    WorkspaceArtifactRevision,
    WorkflowRunEventRecord,
)
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionSourceKind,
    ArtifactStorageBackend,
    ArtifactStorageRef,
    ArtifactType,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.delivery_helpers import (
    _resolve_summary_selected_artifact_revisions,
)


def _storage_ref(
        *,
        backend: ArtifactStorageBackend = ArtifactStorageBackend.FILE_STORAGE,
        file_id: str = "file-1",
) -> dict:
    if backend == ArtifactStorageBackend.FILE_STORAGE:
        return ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.FILE_STORAGE,
            file_id=file_id,
            object_key=f"objects/{file_id}.md",
            storage_hash="sha256:" + "a" * 64,
            size_bytes=128,
            mime_type="text/markdown",
        ).model_dump(mode="json")
    return ArtifactStorageRef(
        storage_backend=ArtifactStorageBackend.INLINE_SNAPSHOT,
        missing_fields=["storage_hash", "size_bytes", "mime_type"],
        reason_code="inline_snapshot_no_materialized_storage",
    ).model_dump(mode="json")


def _available_artifact(
        *,
        path: str = "/workspace/report.md",
        revision_id: str = "revision-1",
        source_kind: str = "tool_write_file",
        backend: ArtifactStorageBackend = ArtifactStorageBackend.FILE_STORAGE,
        version_locked: bool = True,
) -> dict:
    return {
        "artifact_id": "artifact-1",
        "revision_id": revision_id,
        "content_hash": "sha256:" + "a" * 64,
        "storage_ref": _storage_ref(backend=backend),
        "path": path,
        "artifact_type": "file",
        "delivery_state": "candidate",
        "session_id": "session-1",
        "run_id": "run-1",
        "source_run_id": "run-1",
        "source_step_id": "step-1",
        "source_event_id": "event-1",
        "source_fact_ids": ["fact-1"],
        "source_kind": source_kind,
        "source_evidence_ids": ["evidence-1"],
        "delivery_candidate": True,
        "version_locked": version_locked,
        "reuse_policy": "reuse_allowed",
    }


def test_summary_selection_should_only_select_version_locked_file_storage_revisions() -> None:
    selected = _resolve_summary_selected_artifact_revisions(
        summary_context_packet={
            "summary_evidence_context": {
                "available_artifacts": [
                    _available_artifact(path="/workspace/report.md", revision_id="revision-file"),
                    _available_artifact(
                        path="/workspace/snapshot.md",
                        revision_id="revision-snapshot",
                        source_kind="final_answer_snapshot",
                        backend=ArtifactStorageBackend.INLINE_SNAPSHOT,
                    ),
                    _available_artifact(
                        path="/workspace/unlocked.md",
                        revision_id="revision-unlocked",
                        version_locked=False,
                    ),
                ],
                "cursor": "cursor-1",
            }
        },
        parsed_attachments=[
            "/workspace/report.md",
            "/workspace/snapshot.md",
            "/workspace/unlocked.md",
            "/workspace/missing.md",
        ],
    )

    assert [item["path"] for item in selected] == ["/workspace/report.md"]
    assert selected[0]["revision_id"] == "revision-file"
    assert selected[0]["session_id"] == "session-1"
    assert selected[0]["source_run_id"] == "run-1"
    assert selected[0]["source_kind"] == "tool_write_file"


class _WorkspaceArtifactRepo:
    def __init__(self) -> None:
        self.artifacts: dict[str, WorkspaceArtifact] = {}

    async def get_by_user_workspace_id_and_path(self, user_id: str, workspace_id: str, path: str):
        for artifact in self.artifacts.values():
            if artifact.user_id == user_id and artifact.workspace_id == workspace_id and artifact.path == path:
                return artifact.model_copy(deep=True)
        return None

    async def insert_current_index_if_absent(self, artifact: WorkspaceArtifact) -> None:
        existing = await self.get_by_user_workspace_id_and_path(
            user_id=artifact.user_id,
            workspace_id=artifact.workspace_id,
            path=artifact.path,
        )
        if existing is None:
            self.artifacts[artifact.id] = artifact.model_copy(deep=True)

    async def save(self, artifact: WorkspaceArtifact) -> None:
        self.artifacts[artifact.id] = artifact.model_copy(deep=True)

    async def get_by_user_workspace_id_and_id(self, user_id: str, workspace_id: str, artifact_id: str):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None or artifact.user_id != user_id or artifact.workspace_id != workspace_id:
            return None
        return artifact.model_copy(deep=True)


class _WorkspaceArtifactRevisionRepo:
    def __init__(self, artifact_repo: _WorkspaceArtifactRepo) -> None:
        self._artifact_repo = artifact_repo
        self.revisions: list[WorkspaceArtifactRevision] = []

    async def append_revision_for_artifact(self, revision: WorkspaceArtifactRevision) -> WorkspaceArtifactRevision:
        saved = revision.model_copy(deep=True)
        saved.revision_no = len(self.revisions) + 1
        self.revisions.append(saved)
        artifact = await self._artifact_repo.get_by_user_workspace_id_and_id(
            user_id=saved.user_id,
            workspace_id=saved.workspace_id,
            artifact_id=saved.artifact_id,
        )
        assert artifact is not None
        artifact.current_revision_id = saved.revision_id
        artifact.latest_content_hash = saved.content_hash
        await self._artifact_repo.save(artifact)
        return saved.model_copy(deep=True)

    async def get_latest_final_answer_snapshot(self, *, user_id: str, workspace_id: str, session_id: str, source_run_id: str):
        snapshots = [
            revision
            for revision in self.revisions
            if revision.user_id == user_id
            and revision.workspace_id == workspace_id
            and revision.session_id == session_id
            and revision.source_run_id == source_run_id
            and revision.source_kind == ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT
        ]
        return snapshots[-1].model_copy(deep=True) if snapshots else None


class _SessionRepo:
    async def get_by_id_without_events(self, session_id: str, user_id: str):
        if session_id == "session-1" and user_id == "user-1":
            return Session(id="session-1", user_id="user-1", workspace_id="workspace-1", current_run_id="run-1")
        return None


class _WorkspaceRepo:
    async def get_by_session_id_for_user(self, session_id: str, user_id: str):
        if session_id == "session-1" and user_id == "user-1":
            return Workspace(id="workspace-1", session_id="session-1", user_id="user-1", current_run_id="run-1")
        return None


class _WorkflowRunRepo:
    def __init__(self) -> None:
        self.event_records: dict[tuple[str, str], WorkflowRunEventRecord] = {}

    async def get_event_record_by_event_id(self, *, user_id: str, session_id: str, run_id: str, event_id: str):
        record = self.event_records.get((run_id, event_id))
        if record is None:
            return None
        if record.user_id != user_id or record.session_id != session_id:
            return None
        return record


class _UoW:
    def __init__(self) -> None:
        self.workspace_artifact = _WorkspaceArtifactRepo()
        self.workspace_artifact_revision = _WorkspaceArtifactRevisionRepo(self.workspace_artifact)
        self.session = _SessionRepo()
        self.workspace = _WorkspaceRepo()
        self.workflow_run = _WorkflowRunRepo()
        self.workflow_run.event_records[("run-1", "stream-event-1")] = WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            user_id="user-1",
            event_id="stream-event-1",
            event_type="message",
            event_payload=MessageEvent(id="stream-event-1", role="assistant", message="最终正文", stage="final"),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_final_message_projector_should_create_snapshot_bound_to_persisted_event_id() -> None:
    uow = _UoW()
    projector = FinalMessageArtifactProjector(
        uow_factory=lambda: uow,
        ledger_service=ArtifactLedgerService(uow_factory=lambda: uow),
        user_id="user-1",
        session_id="session-1",
    )

    asyncio.run(
        projector.project_final_message(
            event=MessageEvent(role="assistant", message="最终正文", stage="final"),
            persisted_event_id="stream-event-1",
            run_id="run-1",
        )
    )

    assert len(uow.workspace_artifact_revision.revisions) == 1
    revision = uow.workspace_artifact_revision.revisions[0]
    expected_hash = "sha256:" + hashlib.sha256("最终正文".encode("utf-8")).hexdigest()
    assert revision.source_kind == ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT
    assert revision.source_event_id == "stream-event-1"
    assert revision.source_message_event_id == "stream-event-1"
    assert revision.source_run_id == "run-1"
    assert revision.source_final_answer_hash == expected_hash
    assert revision.content_hash == expected_hash
    assert revision.storage_ref.storage_backend == ArtifactStorageBackend.INLINE_SNAPSHOT
    assert revision.artifact_type == ArtifactType.FINAL_ANSWER_SNAPSHOT
    assert revision.delivery_state == ArtifactDeliveryState.CANDIDATE


def test_final_message_projector_should_use_persisted_run_id_when_current_run_changed() -> None:
    uow = _UoW()
    uow.workflow_run.event_records[("old-run", "stream-event-1")] = WorkflowRunEventRecord(
        run_id="old-run",
        session_id="session-1",
        user_id="user-1",
        event_id="stream-event-1",
        event_type="message",
        event_payload=MessageEvent(id="stream-event-1", role="assistant", message="旧正文", stage="final"),
    )
    projector = FinalMessageArtifactProjector(
        uow_factory=lambda: uow,
        ledger_service=ArtifactLedgerService(uow_factory=lambda: uow),
        user_id="user-1",
        session_id="session-1",
    )

    asyncio.run(
        projector.project_final_message(
            event=MessageEvent(role="assistant", message="旧正文", stage="final"),
            persisted_event_id="stream-event-1",
            run_id="old-run",
        )
    )

    assert uow.workspace_artifact_revision.revisions[0].source_run_id == "old-run"


class _DerivedExportFileStorage:
    def __init__(self) -> None:
        self.uploaded_content: bytes | None = None
        self.uploaded_file: File | None = None

    async def upload_file(self, upload_file, user_id=None):
        self.uploaded_content = upload_file.file.read()
        self.uploaded_file = File(
            id="file-export-1",
            user_id=user_id,
            filename=upload_file.filename,
            key="objects/export.md",
            mime_type=upload_file.content_type,
            size=upload_file.size,
        )
        return self.uploaded_file.model_copy(deep=True)

    async def download_file(self, file_id, user_id=None):
        assert file_id == "file-export-1"
        assert self.uploaded_content is not None
        assert self.uploaded_file is not None
        return io.BytesIO(self.uploaded_content), self.uploaded_file.model_copy(deep=True)


def _scope(run_id: str = "run-2") -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id=run_id,
        current_step_id=None,
    )


def test_derived_export_projector_should_create_file_revision_from_final_answer_snapshot() -> None:
    uow = _UoW()
    ledger = ArtifactLedgerService(uow_factory=lambda: uow)
    snapshot_projector = FinalMessageArtifactProjector(
        uow_factory=lambda: uow,
        ledger_service=ledger,
        user_id="user-1",
        session_id="session-1",
    )
    asyncio.run(
        snapshot_projector.project_final_message(
            event=MessageEvent(role="assistant", message="最终正文", stage="final"),
            persisted_event_id="stream-event-1",
            run_id="run-1",
        )
    )
    file_storage = _DerivedExportFileStorage()
    export_projector = DerivedExportProjector(
        uow_factory=lambda: uow,
        ledger_service=ledger,
        file_storage=file_storage,
    )

    resolved = asyncio.run(
        export_projector.export_latest_final_answer_as_markdown(
            scope=_scope(run_id="run-2"),
            source_run_id="run-1",
        )
    )

    assert file_storage.uploaded_content == "最终正文\n".encode("utf-8")
    export_revision = uow.workspace_artifact_revision.revisions[-1]
    assert resolved.revision_id == export_revision.revision_id
    assert export_revision.source_kind == ArtifactRevisionSourceKind.DERIVED_EXPORT
    assert export_revision.run_id == "run-2"
    assert export_revision.source_run_id == "run-1"
    assert export_revision.source_revision_id == uow.workspace_artifact_revision.revisions[0].revision_id
    assert export_revision.source_final_answer_hash == uow.workspace_artifact_revision.revisions[0].source_final_answer_hash
    assert export_revision.derived_content_hash == export_revision.content_hash
    assert export_revision.storage_ref.storage_backend == ArtifactStorageBackend.FILE_STORAGE
    assert export_revision.storage_ref.file_id == "file-export-1"


def test_derived_export_projector_should_reject_regenerated_body_hash_mismatch() -> None:
    uow = _UoW()
    ledger = ArtifactLedgerService(uow_factory=lambda: uow)
    snapshot = WorkspaceArtifactRevision(
        artifact_id="artifact-snapshot",
        revision_no=1,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        path="inline://final-answer/stream-event-1",
        storage_ref=ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.INLINE_SNAPSHOT,
            missing_fields=["storage_hash", "size_bytes", "mime_type"],
            reason_code="inline_snapshot_no_materialized_storage",
        ),
        content_hash="sha256:" + "0" * 64,
        artifact_type=ArtifactType.FINAL_ANSWER_SNAPSHOT,
        source_kind=ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT,
        source_event_id="stream-event-1",
        source_run_id="run-1",
        source_message_event_id="stream-event-1",
        source_final_answer_hash="sha256:" + "0" * 64,
    )
    uow.workspace_artifact_revision.revisions.append(snapshot)
    export_projector = DerivedExportProjector(
        uow_factory=lambda: uow,
        ledger_service=ledger,
        file_storage=_DerivedExportFileStorage(),
    )

    with pytest.raises(DerivedExportError, match="derived_export_hash_mismatch"):
        asyncio.run(
            export_projector.export_latest_final_answer_as_markdown(
                scope=_scope(run_id="run-2"),
                source_run_id="run-1",
            )
        )
