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

    async def list_by_workspace_id(self, workspace_id: str):
        raise AssertionError("WorkspaceRuntimeService 不应使用裸 workspace_id 查询 artifact")

    async def list_by_user_workspace_id(self, user_id: str, workspace_id: str):
        artifacts = [
            artifact.model_copy(deep=True)
            for artifact in self._artifacts_by_path.values()
            if artifact.workspace_id == workspace_id and artifact.user_id == user_id
        ]
        return sorted(artifacts, key=lambda item: item.updated_at, reverse=True)

    async def list_by_workspace_id_and_paths(self, workspace_id: str, paths: list[str]):
        raise AssertionError("WorkspaceRuntimeService 不应使用裸 workspace_id + paths 查询 artifact")

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

    async def get_by_workspace_id_and_path(self, workspace_id: str, path: str):
        raise AssertionError("WorkspaceRuntimeService 不应使用裸 workspace_id + path 查询 artifact")

    async def get_by_user_workspace_id_and_path(self, user_id: str, workspace_id: str, path: str):
        artifact = self._artifacts_by_path.get(path)
        if artifact is None or artifact.workspace_id != workspace_id or artifact.user_id != user_id:
            return None
        return artifact.model_copy(deep=True)

    async def update_delivery_state_by_workspace_id_and_paths(
            self,
            *,
            workspace_id: str,
            paths: list[str],
            delivery_state: str,
    ):
        raise AssertionError("WorkspaceRuntimeService 不应使用裸 workspace_id 更新 artifact")

    async def update_delivery_state_by_user_workspace_id_and_paths(
            self,
            *,
            user_id: str,
            workspace_id: str,
            paths: list[str],
            delivery_state: str,
    ):
        updated: list[WorkspaceArtifact] = []
        for path in list(paths or []):
            artifact = self._artifacts_by_path.get(path)
            if artifact is None or artifact.workspace_id != workspace_id or artifact.user_id != user_id:
                continue
            next_artifact = artifact.model_copy(deep=True)
            next_artifact.delivery_state = str(delivery_state or "").strip()
            self._artifacts_by_path[path] = next_artifact
            updated.append(next_artifact.model_copy(deep=True))
        return updated


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
    assert "workspace_artifacts" in compiled_sql


def test_workspace_artifact_repository_update_delivery_state_should_use_single_sql_update() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(),
                SimpleNamespace(
                    scalars=lambda: SimpleNamespace(
                        all=lambda: [
                            SimpleNamespace(
                                to_domain=lambda: WorkspaceArtifact(
                                    id="artifact-1",
                                    workspace_id="workspace-1",
                                    path="/workspace/report.md",
                                    artifact_type="file",
                                    delivery_state="final_delivered",
                                )
                            )
                        ]
                    )
                ),
            ]
        )
    )
    repository = DBWorkspaceArtifactRepository(db_session=db_session)

    updated = asyncio.run(
        repository.update_delivery_state_by_workspace_id_and_paths(
            workspace_id="workspace-1",
            paths=["/workspace/report.md"],
            delivery_state="final_delivered",
        )
    )

    statement = db_session.execute.call_args_list[0].args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "UPDATE workspace_artifacts" in compiled_sql
    assert updated[0].delivery_state == "final_delivered"


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


def test_workspace_runtime_service_should_upsert_artifact_by_workspace_path() -> None:
    workspace_repo = _WorkspaceRepo(Workspace(id="workspace-1", session_id="session-1", user_id="user-1"))
    workspace_artifact_repo = _WorkspaceArtifactRepo()
    runtime_service = WorkspaceRuntimeService(
        session_id="session-1",
        user_id="user-1",
        uow_factory=lambda: _UoW(
            workspace_repo=workspace_repo,
            workspace_artifact_repo=workspace_artifact_repo,
        ),
    )

    first = asyncio.run(
        runtime_service.upsert_artifact(
            path="/workspace/report.md",
            artifact_type="file",
            summary="初始摘要",
            metadata={"source": "write_file"},
        )
    )
    second = asyncio.run(
        runtime_service.upsert_artifact(
            path="/workspace/report.md",
            artifact_type="file",
            summary="最新摘要",
            delivery_state="final_delivered",
            metadata={"size": 128},
        )
    )
    resolved_paths = asyncio.run(
        runtime_service.resolve_authoritative_artifact_paths(
            paths=[
                "/workspace/report.md",
                "/workspace/report.md",
                "/workspace/missing.md",
            ]
        )
    )

    artifacts = asyncio.run(runtime_service.list_artifacts())

    assert first.id == second.id
    assert len(artifacts) == 1
    assert artifacts[0].path == "/workspace/report.md"
    assert artifacts[0].user_id == "user-1"
    assert artifacts[0].session_id == "session-1"
    assert artifacts[0].summary == "最新摘要"
    assert artifacts[0].delivery_state == "final_delivered"
    assert artifacts[0].metadata == {"source": "write_file", "size": 128}
    assert resolved_paths == ["/workspace/report.md"]


def test_workspace_runtime_service_should_mark_delivery_state_without_creating_missing_artifact() -> None:
    workspace_repo = _WorkspaceRepo(Workspace(id="workspace-1", session_id="session-1", user_id="user-1"))
    workspace_artifact_repo = _WorkspaceArtifactRepo()
    runtime_service = WorkspaceRuntimeService(
        session_id="session-1",
        user_id="user-1",
        uow_factory=lambda: _UoW(
            workspace_repo=workspace_repo,
            workspace_artifact_repo=workspace_artifact_repo,
        ),
    )

    asyncio.run(
        runtime_service.upsert_artifact(
            path="/workspace/report.md",
            artifact_type="file",
            summary="report",
        )
    )
    updated = asyncio.run(
        runtime_service.mark_artifacts_delivery_state(
            paths=["/workspace/report.md", "/workspace/missing.md"],
            delivery_state="final_delivered",
        )
    )

    artifacts = asyncio.run(runtime_service.list_artifacts())

    assert [artifact.path for artifact in updated] == ["/workspace/report.md"]
    assert len(artifacts) == 1
    assert artifacts[0].delivery_state == "final_delivered"


def test_workspace_runtime_service_should_not_record_screenshot_artifact_as_changed_file() -> None:
    workspace_repo = _WorkspaceRepo(Workspace(id="workspace-1", session_id="session-1", user_id="user-1"))
    workspace_artifact_repo = _WorkspaceArtifactRepo()
    runtime_service = WorkspaceRuntimeService(
        session_id="session-1",
        user_id="user-1",
        uow_factory=lambda: _UoW(
            workspace_repo=workspace_repo,
            workspace_artifact_repo=workspace_artifact_repo,
        ),
    )

    asyncio.run(
        runtime_service.upsert_artifact(
            path="/.workspace/browser-screenshots/shot.png",
            artifact_type="browser_screenshot",
            summary="截图",
            record_as_changed_file=False,
        )
    )

    workspace = asyncio.run(runtime_service.get_workspace_or_raise())
    artifacts = asyncio.run(runtime_service.list_artifacts())

    assert len(artifacts) == 1
    assert artifacts[0].artifact_type == "browser_screenshot"
    assert workspace.environment_summary.get("recent_changed_files") in (None, [])
