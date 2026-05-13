import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from app.application.service.artifact_ledger_service import ArtifactLedgerService
from app.domain.models import Workspace, WorkspaceArtifact, WorkspaceArtifactRevision
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionIdentity,
    ArtifactRevisionRegistrationCommand,
    ArtifactRevisionSourceKind,
    ArtifactStorageBackend,
    ArtifactStorageRef,
    ArtifactType,
    ResolvedArtifactRevisionResult,
)
from app.domain.services.workspace_runtime import WorkspaceRuntimeService
from app.infrastructure.repositories.db_workspace_artifact_revision_repository import (
    DBWorkspaceArtifactRevisionRepository,
)


def _scope(*, user_id: str = "user-1", workspace_id: str = "workspace-1") -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id=user_id,
        user_id=user_id,
        session_id="session-1",
        workspace_id=workspace_id,
        run_id="run-1",
        current_step_id="step-1",
    )


def _storage_ref(*, file_id: str = "file-1", content_hash: str = "sha256:" + "a" * 64) -> ArtifactStorageRef:
    return ArtifactStorageRef(
        storage_backend=ArtifactStorageBackend.FILE_STORAGE,
        object_key=f"objects/{file_id}.md",
        file_id=file_id,
        sandbox_path=None,
        storage_hash=content_hash,
        size_bytes=128,
        mime_type="text/markdown",
        missing_fields=[],
        reason_code=None,
    )


def _command(
        *,
        path: str = "/workspace/report.md",
        content_hash: str = "sha256:" + "a" * 64,
        file_id: str = "file-1",
        source_event_id: str = "event-1",
        tool_call_id: str = "tool-call-1",
        scope: AccessScopeResult | None = None,
) -> ArtifactRevisionRegistrationCommand:
    return ArtifactRevisionRegistrationCommand(
        scope=scope or _scope(),
        path=path,
        storage_ref=_storage_ref(file_id=file_id, content_hash=content_hash),
        content_hash=content_hash,
        storage_hash=content_hash,
        hash_algorithm="sha256",
        size_bytes=128,
        mime_type="text/markdown",
        artifact_type=ArtifactType.FILE,
        delivery_state=ArtifactDeliveryState.CANDIDATE,
        source_kind=ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
        source_event_id=source_event_id,
        source_run_id="run-1",
        source_message_event_id=None,
        source_revision_id=None,
        source_fact_ids=["fact-1"],
        source_evidence_ids=[],
        source_final_answer_hash=None,
        derived_content_hash=None,
        tool_call_id=tool_call_id,
        function_name="write_file",
        profile_hash="sha256:" + "b" * 64,
        profile_status="available",
        metadata={"summary": "report"},
    )


def _revision(*, revision_no: int = 1, revision_id: str = "revision-new") -> WorkspaceArtifactRevision:
    content_hash = "sha256:" + "a" * 64
    return WorkspaceArtifactRevision(
        revision_id=revision_id,
        artifact_id="artifact-1",
        revision_no=revision_no,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        path="/workspace/report.md",
        storage_ref=_storage_ref(content_hash=content_hash),
        content_hash=content_hash,
        storage_hash=content_hash,
        hash_algorithm="sha256",
        size_bytes=128,
        mime_type="text/markdown",
        artifact_type=ArtifactType.FILE,
        delivery_state=ArtifactDeliveryState.CANDIDATE,
        source_kind=ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
        source_event_id="event-1",
        source_run_id="run-1",
        source_fact_ids=["fact-1"],
        tool_call_id="tool-call-1",
        function_name="write_file",
        profile_hash="sha256:" + "b" * 64,
        profile_status="available",
        metadata={"summary": "report"},
    )


class _WorkspaceArtifactRepo:
    def __init__(self) -> None:
        self.artifacts_by_id: dict[str, WorkspaceArtifact] = {}
        self.persisted_artifact_after_save: WorkspaceArtifact | None = None

    async def save(self, artifact: WorkspaceArtifact) -> None:
        if self.persisted_artifact_after_save is not None and artifact.id == self.persisted_artifact_after_save.id:
            self.persisted_artifact_after_save = artifact.model_copy(deep=True)
        self.artifacts_by_id[artifact.id] = artifact.model_copy(deep=True)

    async def insert_current_index_if_absent(self, artifact: WorkspaceArtifact) -> None:
        if self.persisted_artifact_after_save is not None:
            return
        if any(
                existing.user_id == artifact.user_id
                and existing.workspace_id == artifact.workspace_id
                and existing.path == artifact.path
                for existing in self.artifacts_by_id.values()
        ):
            return
        self.artifacts_by_id[artifact.id] = artifact.model_copy(deep=True)

    async def list_by_user_workspace_id(self, user_id: str, workspace_id: str):
        return [
            artifact.model_copy(deep=True)
            for artifact in self.artifacts_by_id.values()
            if artifact.user_id == user_id and artifact.workspace_id == workspace_id
        ]

    async def list_by_user_workspace_id_and_paths(self, user_id: str, workspace_id: str, paths: list[str]):
        path_set = {str(path or "").strip() for path in paths}
        return [
            artifact.model_copy(deep=True)
            for artifact in self.artifacts_by_id.values()
            if artifact.user_id == user_id and artifact.workspace_id == workspace_id and artifact.path in path_set
        ]

    async def get_by_user_workspace_id_and_path(self, user_id: str, workspace_id: str, path: str):
        if self.persisted_artifact_after_save is not None:
            artifact = self.persisted_artifact_after_save
            if artifact.user_id == user_id and artifact.workspace_id == workspace_id and artifact.path == path:
                return artifact.model_copy(deep=True)
        for artifact in self.artifacts_by_id.values():
            if artifact.user_id == user_id and artifact.workspace_id == workspace_id and artifact.path == path:
                return artifact.model_copy(deep=True)
        return None

    async def get_by_user_workspace_id_and_id(self, user_id: str, workspace_id: str, artifact_id: str):
        if self.persisted_artifact_after_save is not None:
            artifact = self.persisted_artifact_after_save
            if artifact.id == artifact_id and artifact.user_id == user_id and artifact.workspace_id == workspace_id:
                return artifact.model_copy(deep=True)
        artifact = self.artifacts_by_id.get(artifact_id)
        if artifact is None or artifact.user_id != user_id or artifact.workspace_id != workspace_id:
            return None
        return artifact.model_copy(deep=True)


class _WorkspaceArtifactRevisionRepo:
    def __init__(self, artifact_repo: _WorkspaceArtifactRepo) -> None:
        self._artifact_repo = artifact_repo
        self.revisions: list[WorkspaceArtifactRevision] = []
        self.assigned_revision_numbers: list[int] = []

    async def insert_or_get_existing(self, revision: WorkspaceArtifactRevision) -> WorkspaceArtifactRevision:
        for existing in self.revisions:
            if (
                    existing.user_id == revision.user_id
                    and existing.workspace_id == revision.workspace_id
                    and existing.source_kind == revision.source_kind
                    and existing.source_event_id == revision.source_event_id
                    and existing.tool_call_id == revision.tool_call_id
                    and existing.content_hash == revision.content_hash
            ):
                return existing.model_copy(deep=True)
        self.revisions.append(revision.model_copy(deep=True))
        await self._update_current_projection(revision)
        return revision.model_copy(deep=True)

    async def append_revision_for_artifact(self, revision: WorkspaceArtifactRevision) -> WorkspaceArtifactRevision:
        next_revision = revision.model_copy(deep=True)
        next_revision.revision_no = (
            max(
                [
                    item.revision_no
                    for item in self.revisions
                    if item.artifact_id == revision.artifact_id
                    and item.user_id == revision.user_id
                    and item.workspace_id == revision.workspace_id
                ],
                default=0,
            )
            + 1
        )
        self.assigned_revision_numbers.append(next_revision.revision_no)
        return await self.insert_or_get_existing(next_revision)

    async def get_by_user_workspace_revision_id(self, *, user_id: str, workspace_id: str, revision_id: str):
        for revision in self.revisions:
            if revision.user_id == user_id and revision.workspace_id == workspace_id and revision.revision_id == revision_id:
                return revision.model_copy(deep=True)
        return None

    async def list_by_user_workspace_artifact_id(self, *, user_id: str, workspace_id: str, artifact_id: str):
        return [
            revision.model_copy(deep=True)
            for revision in self.revisions
            if revision.user_id == user_id and revision.workspace_id == workspace_id and revision.artifact_id == artifact_id
        ]

    async def update_delivery_state_by_identities(
            self,
            *,
            user_id: str,
            workspace_id: str,
            session_id: str,
            identities: list[ArtifactRevisionIdentity],
            delivery_state: ArtifactDeliveryState,
    ):
        updated: list[WorkspaceArtifactRevision] = []
        identity_keys = {
            (identity.artifact_id, identity.revision_id, identity.content_hash)
            for identity in identities
        }
        next_revisions: list[WorkspaceArtifactRevision] = []
        for revision in self.revisions:
            key = (revision.artifact_id, revision.revision_id, revision.content_hash)
            if (
                    revision.user_id == user_id
                    and revision.workspace_id == workspace_id
                    and revision.session_id == session_id
                    and key in identity_keys
            ):
                next_revision = revision.model_copy(deep=True)
                next_revision.delivery_state = delivery_state
                next_revisions.append(next_revision)
                updated.append(next_revision.model_copy(deep=True))
                artifact = await self._artifact_repo.get_by_user_workspace_id_and_id(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    artifact_id=revision.artifact_id,
                )
                if artifact is not None and artifact.current_revision_id == revision.revision_id:
                    artifact.delivery_state = delivery_state.value
                    await self._artifact_repo.save(artifact)
            else:
                next_revisions.append(revision.model_copy(deep=True))
        self.revisions = next_revisions
        return updated

    async def _update_current_projection(self, revision: WorkspaceArtifactRevision) -> None:
        artifact = await self._artifact_repo.get_by_user_workspace_id_and_id(
            user_id=revision.user_id,
            workspace_id=revision.workspace_id,
            artifact_id=revision.artifact_id,
        )
        assert artifact is not None
        artifact.current_revision_id = revision.revision_id
        artifact.latest_content_hash = revision.content_hash
        artifact.latest_size = revision.size_bytes
        artifact.latest_mime_type = revision.mime_type
        artifact.artifact_type = revision.artifact_type.value
        artifact.delivery_state = revision.delivery_state.value
        await self._artifact_repo.save(artifact)


class _WorkspaceRepo:
    def __init__(self) -> None:
        self.workspace = Workspace(
            id="workspace-1",
            user_id="user-1",
            session_id="session-1",
            current_run_id="run-1",
        )

    async def get_by_session_id_for_user(self, session_id: str, user_id: str):
        if self.workspace.session_id == session_id and self.workspace.user_id == user_id:
            return self.workspace.model_copy(deep=True)
        return None

    async def save(self, workspace: Workspace) -> None:
        self.workspace = workspace.model_copy(deep=True)


class _UoW:
    def __init__(self) -> None:
        self.workspace_artifact = _WorkspaceArtifactRepo()
        self.workspace_artifact_revision = _WorkspaceArtifactRevisionRepo(self.workspace_artifact)
        self.workspace = _WorkspaceRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_workspace_runtime_import_should_not_trigger_application_barrel_cycle() -> None:
    from app.domain.services.workspace_runtime import WorkspaceRuntimeService as ImportedService

    assert ImportedService is WorkspaceRuntimeService


def test_registration_command_should_reject_artifact_id() -> None:
    payload = _command().model_dump(mode="python")
    payload["artifact_id"] = "caller-controlled"
    with pytest.raises(ValidationError):
        ArtifactRevisionRegistrationCommand.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool_call_id", None),
        ("source_fact_ids", []),
        ("function_name", "replace_in_file"),
    ],
)
def test_registration_command_should_require_tool_source_fields(field: str, value) -> None:
    payload = _command().model_dump(mode="python")
    payload[field] = value
    with pytest.raises(ValidationError):
        ArtifactRevisionRegistrationCommand.model_validate(payload)


def test_workspace_artifact_revision_should_require_tool_source_fields() -> None:
    payload = _revision().model_dump(mode="python")
    payload["source_fact_ids"] = []
    with pytest.raises(ValidationError):
        WorkspaceArtifactRevision.model_validate(payload)


def test_resolved_artifact_revision_result_should_not_include_selection_fields() -> None:
    result = ResolvedArtifactRevisionResult(
        artifact_id="artifact-1",
        revision_id="revision-1",
        content_hash="sha256:" + "a" * 64,
        path="/workspace/report.md",
        artifact_type=ArtifactType.FILE,
        delivery_state=ArtifactDeliveryState.CANDIDATE,
        session_id="session-1",
        run_id="run-1",
        source_run_id="run-1",
        source_step_id="step-1",
        source_event_id="event-1",
        source_kind=ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
    )
    payload = result.model_dump(mode="json")
    assert "selected_reason" not in payload
    assert "selected_at" not in payload


def test_artifact_ledger_register_revision_should_reuse_artifact_and_append_revisions() -> None:
    uow = _UoW()
    service = ArtifactLedgerService(uow_factory=lambda: uow)

    first = asyncio.run(service.register_revision(command=_command()))
    second = asyncio.run(
        service.register_revision(
            command=_command(
                content_hash="sha256:" + "c" * 64,
                file_id="file-2",
                source_event_id="event-2",
                tool_call_id="tool-call-2",
            )
        )
    )

    assert first.artifact_id == second.artifact_id
    assert [revision.revision_no for revision in uow.workspace_artifact_revision.revisions] == [1, 2]
    artifact = next(iter(uow.workspace_artifact.artifacts_by_id.values()))
    assert artifact.current_revision_id == second.revision_id
    assert artifact.latest_content_hash == second.content_hash


def test_artifact_ledger_register_revision_should_use_persisted_artifact_after_create_conflict() -> None:
    uow = _UoW()
    persisted = WorkspaceArtifact(
        id="artifact-db",
        workspace_id="workspace-1",
        user_id="user-1",
        session_id="session-1",
        run_id="run-1",
        path="/workspace/report.md",
        artifact_type="file",
    )
    uow.workspace_artifact.persisted_artifact_after_save = persisted
    service = ArtifactLedgerService(uow_factory=lambda: uow)

    first = asyncio.run(service.register_revision(command=_command()))
    second = asyncio.run(
        service.register_revision(
            command=_command(
                content_hash="sha256:" + "e" * 64,
                file_id="file-2",
                source_event_id="event-2",
                tool_call_id="tool-call-2",
            )
        )
    )

    assert first.artifact_id == "artifact-db"
    assert second.artifact_id == "artifact-db"
    assert [revision.revision_no for revision in uow.workspace_artifact_revision.revisions] == [1, 2]


def test_artifact_ledger_mark_historical_revision_should_not_move_current_pointer() -> None:
    uow = _UoW()
    service = ArtifactLedgerService(uow_factory=lambda: uow)
    first = asyncio.run(service.register_revision(command=_command()))
    second = asyncio.run(
        service.register_revision(
            command=_command(
                content_hash="sha256:" + "d" * 64,
                file_id="file-2",
                source_event_id="event-2",
                tool_call_id="tool-call-2",
            )
        )
    )

    updated = asyncio.run(
        service.mark_artifact_revisions_delivery_state(
            scope=_scope(),
            revisions=[
                ArtifactRevisionIdentity(
                    artifact_id=first.artifact_id,
                    revision_id=first.revision_id,
                    content_hash=first.content_hash,
                )
            ],
            delivery_state=ArtifactDeliveryState.SELECTED,
        )
    )

    artifact = next(iter(uow.workspace_artifact.artifacts_by_id.values()))
    assert updated[0].revision_id == first.revision_id
    assert artifact.current_revision_id == second.revision_id
    assert artifact.latest_content_hash == second.content_hash


def test_workspace_runtime_upsert_artifact_should_reject_legacy_kwargs() -> None:
    runtime_service = WorkspaceRuntimeService(
        session_id="session-1",
        user_id="user-1",
        uow_factory=lambda: _UoW(),
        artifact_ledger=ArtifactLedgerService(uow_factory=lambda: _UoW()),
    )

    with pytest.raises(TypeError):
        asyncio.run(
            runtime_service.upsert_artifact(
                path="/workspace/report.md",
                artifact_type="file",
            )
        )


def test_revision_repository_append_should_lock_artifact_and_retry_revision_no_conflict() -> None:
    saved = _revision(revision_no=2, revision_id="revision-new")
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(scalar_one_or_none=lambda: "artifact-1"),
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one=lambda: 0),
                SimpleNamespace(scalar_one_or_none=lambda: "artifact-1"),
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one=lambda: 1),
                SimpleNamespace(scalar_one_or_none=lambda: "artifact-1"),
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one_or_none=lambda: saved.revision_id),
                SimpleNamespace(),
                SimpleNamespace(scalar_one_or_none=lambda: saved),
            ]
        )
    )
    repository = DBWorkspaceArtifactRevisionRepository(db_session=db_session)

    result = asyncio.run(repository.append_revision_for_artifact(_revision()))

    first_statement = db_session.execute.call_args_list[0].args[0]
    assert "FOR UPDATE" in str(first_statement.compile(dialect=postgresql.dialect()))
    assert result.revision_no == 2


def test_revision_repository_delivery_state_update_should_not_move_current_projection() -> None:
    revision = _revision(revision_id="revision-old")
    selected_revision = revision.model_copy(update={"delivery_state": ArtifactDeliveryState.SELECTED})
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(scalar_one_or_none=lambda: selected_revision),
                SimpleNamespace(),
            ]
        )
    )
    repository = DBWorkspaceArtifactRevisionRepository(db_session=db_session)

    updated = asyncio.run(
        repository.update_delivery_state_by_identities(
            user_id="user-1",
            workspace_id="workspace-1",
            session_id="session-1",
            identities=[
                ArtifactRevisionIdentity(
                    artifact_id="artifact-1",
                    revision_id="revision-old",
                    content_hash=revision.content_hash,
                )
            ],
            delivery_state=ArtifactDeliveryState.SELECTED,
        )
    )

    revision_update = db_session.execute.call_args_list[0].args[0]
    artifact_update = db_session.execute.call_args_list[1].args[0]
    compiled_artifact_update = str(artifact_update.compile(dialect=postgresql.dialect()))
    assert updated[0].delivery_state == ArtifactDeliveryState.SELECTED
    assert "UPDATE workspace_artifact_revisions" in str(revision_update.compile(dialect=postgresql.dialect()))
    assert "workspace_artifacts.current_revision_id =" in compiled_artifact_update
    assert "latest_content_hash" not in compiled_artifact_update
    assert "latest_size" not in compiled_artifact_update
    assert "latest_mime_type" not in compiled_artifact_update
