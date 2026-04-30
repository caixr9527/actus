import asyncio
from dataclasses import dataclass

import pytest

from app.application.errors import NotFoundError, error_keys
from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.domain.models import Session, WorkflowRun, Workspace, WorkspaceArtifact
from app.domain.services.runtime.contracts.data_access_contract import DataAccessAction


@dataclass
class _SessionRepo:
    sessions: dict[str, Session]

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        session = self.sessions.get(session_id)
        if session is None:
            return None
        if user_id is not None and session.user_id != user_id:
            return None
        return session


@dataclass
class _WorkspaceRepo:
    workspaces: dict[str, Workspace]

    async def get_by_id_for_user(self, workspace_id: str, user_id: str):
        workspace = self.workspaces.get(workspace_id)
        if workspace is None or workspace.user_id != user_id:
            return None
        return workspace

    async def list_by_session_id(self, session_id: str):
        return [
            workspace
            for workspace in self.workspaces.values()
            if workspace.session_id == session_id
        ]


@dataclass
class _WorkflowRunRepo:
    runs: dict[str, WorkflowRun]

    async def get_by_id_for_user(self, run_id: str, user_id: str):
        run = self.runs.get(run_id)
        if run is None or run.user_id != user_id:
            return None
        return run


@dataclass
class _WorkspaceArtifactRepo:
    artifacts: list[WorkspaceArtifact]

    async def list_by_user_workspace_id_and_paths(self, user_id: str, workspace_id: str, paths: list[str]):
        return [
            artifact
            for artifact in self.artifacts
            if artifact.user_id == user_id
            and artifact.workspace_id == workspace_id
            and artifact.path in paths
        ]


class _FileRepo:
    async def get_by_id_and_user_id(self, file_id: str, user_id: str):
        return None


class _UoW:
    def __init__(
            self,
            *,
            sessions: dict[str, Session],
            workspaces: dict[str, Workspace],
            runs: dict[str, WorkflowRun] | None = None,
            artifacts: list[WorkspaceArtifact] | None = None,
    ) -> None:
        self.session = _SessionRepo(sessions)
        self.workspace = _WorkspaceRepo(workspaces)
        self.workflow_run = _WorkflowRunRepo(runs or {})
        self.workspace_artifact = _WorkspaceArtifactRepo(artifacts or [])
        self.file = _FileRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _build_service(uow: _UoW) -> RuntimeAccessControlService:
    return RuntimeAccessControlService(uow_factory=lambda: uow)


def test_runtime_access_control_should_reject_workspace_bound_to_other_session() -> None:
    session = Session(id="session-1", user_id="user-1", workspace_id="workspace-1")
    workspace = Workspace(id="workspace-1", session_id="session-2", user_id="user-1")
    service = _build_service(
        _UoW(
            sessions={"session-1": session},
            workspaces={"workspace-1": workspace},
        )
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.resolve_session_scope(user_id="user-1", session_id="session-1"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_runtime_access_control_should_reject_current_run_bound_to_other_session() -> None:
    session = Session(id="session-1", user_id="user-1", workspace_id="workspace-1", current_run_id="run-1")
    workspace = Workspace(id="workspace-1", session_id="session-1", user_id="user-1", current_run_id="run-1")
    run = WorkflowRun(id="run-1", session_id="session-2", user_id="user-1")
    service = _build_service(
        _UoW(
            sessions={"session-1": session},
            workspaces={"workspace-1": workspace},
            runs={"run-1": run},
        )
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.resolve_session_scope(user_id="user-1", session_id="session-1"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_runtime_access_control_should_reject_artifact_path_from_other_workspace() -> None:
    session = Session(id="session-1", user_id="user-1", workspace_id="workspace-1")
    workspace = Workspace(id="workspace-1", session_id="session-1", user_id="user-1")
    artifact = WorkspaceArtifact(
        workspace_id="workspace-2",
        user_id="user-1",
        session_id="session-1",
        path="/workspace/report.md",
        artifact_type="file",
    )
    service = _build_service(
        _UoW(
            sessions={"session-1": session},
            workspaces={"workspace-1": workspace},
            artifacts=[artifact],
        )
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(
            service.assert_artifact_access(
                user_id="user-1",
                session_id="session-1",
                path="/workspace/report.md",
                action=DataAccessAction.READ,
            )
        )

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
