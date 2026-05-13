import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from app.domain.models import Workspace, WorkspaceArtifact
from app.domain.services.workspace_runtime.service import WorkspaceRuntimeService
from app.infrastructure.repositories.db_workspace_artifact_repository import (
    DBWorkspaceArtifactRepository,
)


class _WorkspaceRepo:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace.model_copy(deep=True)

    async def get_by_session_id(self, session_id: str):
        raise AssertionError("WorkspaceRuntimeService 不应使用裸 session_id 查询 workspace")

    async def get_by_session_id_for_user(self, session_id: str, user_id: str):
        if session_id != self._workspace.session_id or user_id != self._workspace.user_id:
            return None
        return self._workspace.model_copy(deep=True)

    async def save(self, workspace: Workspace) -> None:
        self._workspace = workspace.model_copy(deep=True)


class _WorkspaceArtifactRepo:
    def __init__(self) -> None:
        self._artifacts_by_path: dict[str, WorkspaceArtifact] = {}

    async def save(self, artifact: WorkspaceArtifact) -> None:
        self._artifacts_by_path[artifact.path] = artifact.model_copy(deep=True)

    async def insert_current_index_if_absent(self, artifact: WorkspaceArtifact) -> None:
        if artifact.path not in self._artifacts_by_path:
            self._artifacts_by_path[artifact.path] = artifact.model_copy(deep=True)

    async def list_by_user_workspace_id(self, user_id: str, workspace_id: str):
        artifacts = [
            artifact.model_copy(deep=True)
            for artifact in self._artifacts_by_path.values()
            if artifact.workspace_id == workspace_id and artifact.user_id == user_id
        ]
        return sorted(artifacts, key=lambda item: item.updated_at, reverse=True)

    async def list_by_user_workspace_id_and_paths(self, user_id: str, workspace_id: str, paths: list[str]):
        normalized_paths = {
            str(path or "").strip()
            for path in list(paths or [])
            if str(path or "").strip()
        }
        return [
            artifact.model_copy(deep=True)
            for path, artifact in self._artifacts_by_path.items()
            if path in normalized_paths
            and artifact.workspace_id == workspace_id
            and artifact.user_id == user_id
        ]

    async def get_by_user_workspace_id_and_path(self, user_id: str, workspace_id: str, path: str):
        artifact = self._artifacts_by_path.get(path)
        if artifact is None or artifact.workspace_id != workspace_id or artifact.user_id != user_id:
            return None
        return artifact.model_copy(deep=True)

    async def get_by_user_workspace_id_and_id(self, user_id: str, workspace_id: str, artifact_id: str):
        for artifact in self._artifacts_by_path.values():
            if artifact.id == artifact_id and artifact.workspace_id == workspace_id and artifact.user_id == user_id:
                return artifact.model_copy(deep=True)
        return None


class _UoW:
    def __init__(self, workspace_repo: _WorkspaceRepo, workspace_artifact_repo: _WorkspaceArtifactRepo) -> None:
        self.workspace = workspace_repo
        self.workspace_artifact = workspace_artifact_repo

    async def commit(self):
        return None

    async def rollback(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_workspace_artifact_repository_save_should_use_postgresql_on_conflict() -> None:
    db_session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace()))
    repository = DBWorkspaceArtifactRepository(db_session=db_session)

    asyncio.run(
        repository.save(
            WorkspaceArtifact(
                id="artifact-1",
                workspace_id="workspace-1",
                path="/workspace/report.md",
                artifact_type="file",
                summary="report",
            )
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_workspace_artifacts_workspace_id_path" in compiled_sql
    assert "DO UPDATE" in compiled_sql
    assert "current_revision_id =" in compiled_sql
    assert "latest_content_hash =" in compiled_sql
    assert "workspace_artifacts" in compiled_sql


def test_workspace_artifact_repository_insert_current_index_if_absent_should_not_update_projection() -> None:
    db_session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace()))
    repository = DBWorkspaceArtifactRepository(db_session=db_session)

    asyncio.run(
        repository.insert_current_index_if_absent(
            WorkspaceArtifact(
                id="artifact-1",
                workspace_id="workspace-1",
                user_id="user-1",
                session_id="session-1",
                path="/workspace/report.md",
                artifact_type="file",
                current_revision_id=None,
                latest_content_hash=None,
                latest_size=None,
                latest_mime_type=None,
            )
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_workspace_artifacts_workspace_id_path DO NOTHING" in compiled_sql
    assert "DO UPDATE" not in compiled_sql
    assert "current_revision_id =" not in compiled_sql
    assert "latest_content_hash =" not in compiled_sql


def test_workspace_artifact_repository_user_workspace_query_should_filter_user_id() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))
        )
    )
    repository = DBWorkspaceArtifactRepository(db_session=db_session)

    asyncio.run(
        repository.list_by_user_workspace_id_and_paths(
            user_id="user-1",
            workspace_id="workspace-1",
            paths=["/workspace/report.md"],
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "workspace_artifacts.user_id" in compiled_sql
    assert "workspace_artifacts.workspace_id" in compiled_sql


def test_workspace_runtime_service_should_require_user_id() -> None:
    workspace_repo = _WorkspaceRepo(Workspace(id="workspace-1", session_id="session-1", user_id="user-1"))
    workspace_artifact_repo = _WorkspaceArtifactRepo()

    with pytest.raises(ValueError, match="必须提供 user_id"):
        WorkspaceRuntimeService(
            session_id="session-1",
            user_id="",
            uow_factory=lambda: _UoW(
                workspace_repo=workspace_repo,
                workspace_artifact_repo=workspace_artifact_repo,
            ),
        )
