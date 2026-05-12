import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from app.domain.models import Workspace, WorkspaceArtifact
from app.domain.services.workspace_runtime import WorkspaceRuntimeService
from app.infrastructure.repositories.db_workspace_artifact_repository import DBWorkspaceArtifactRepository


def _compile_statement(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def _assert_artifact_sql_filters_user_and_workspace(compiled_sql: str) -> None:
    assert "workspace_artifacts.user_id =" in compiled_sql
    assert "workspace_artifacts.workspace_id =" in compiled_sql


def test_workspace_artifact_list_statement_should_filter_by_user_and_workspace() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])))
    )
    repository = DBWorkspaceArtifactRepository(db_session=db_session)

    result = asyncio.run(repository.list_by_user_workspace_id(user_id="user-1", workspace_id="workspace-1"))

    statement = db_session.execute.call_args.args[0]
    _assert_artifact_sql_filters_user_and_workspace(_compile_statement(statement))
    assert result == []


def test_workspace_artifact_read_statement_should_filter_by_user_workspace_and_path() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None))
    )
    repository = DBWorkspaceArtifactRepository(db_session=db_session)

    result = asyncio.run(
        repository.get_by_user_workspace_id_and_path(
            user_id="user-1",
            workspace_id="workspace-1",
            path="/workspace/report.md",
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = _compile_statement(statement)
    _assert_artifact_sql_filters_user_and_workspace(compiled_sql)
    assert "workspace_artifacts.path =" in compiled_sql
    assert result is None


class _WorkspaceRepo:
    def __init__(self) -> None:
        self.get_by_session_for_user_calls: list[tuple[str, str]] = []

    async def get_by_session_id_for_user(self, session_id: str, user_id: str):
        self.get_by_session_for_user_calls.append((session_id, user_id))
        return Workspace(id="workspace-1", session_id=session_id, user_id=user_id)


class _WorkspaceArtifactRepo:
    def __init__(self) -> None:
        self.list_for_user_workspace_calls: list[tuple[str, str]] = []
        self.list_naked_calls: list[str] = []

    async def list_by_user_workspace_id(self, user_id: str, workspace_id: str):
        self.list_for_user_workspace_calls.append((user_id, workspace_id))
        return [
            WorkspaceArtifact(
                user_id=user_id,
                session_id="session-1",
                workspace_id=workspace_id,
                path="/workspace/report.md",
                artifact_type="file",
            )
        ]

    async def list_by_workspace_id(self, workspace_id: str):
        self.list_naked_calls.append(workspace_id)
        return []


class _UoW:
    def __init__(self) -> None:
        self.workspace = _WorkspaceRepo()
        self.workspace_artifact = _WorkspaceArtifactRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_workspace_runtime_should_list_artifacts_only_with_user_workspace_filter() -> None:
    uow = _UoW()
    service = WorkspaceRuntimeService(
        session_id="session-1",
        user_id="user-1",
        uow_factory=lambda: uow,
    )

    artifacts = asyncio.run(service.list_artifacts())

    assert [artifact.path for artifact in artifacts] == ["/workspace/report.md"]
    assert uow.workspace.get_by_session_for_user_calls == [("session-1", "user-1")]
    assert uow.workspace_artifact.list_for_user_workspace_calls == [("user-1", "workspace-1")]
    assert uow.workspace_artifact.list_naked_calls == []


def test_workspace_runtime_should_reject_missing_user_id() -> None:
    with pytest.raises(ValueError, match="必须提供 user_id"):
        WorkspaceRuntimeService(
            session_id="session-1",
            user_id="",
            uow_factory=lambda: _UoW(),
        )
