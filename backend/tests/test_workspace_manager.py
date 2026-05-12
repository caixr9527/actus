import asyncio

from app.domain.models import Session, Workspace
from app.domain.services.workspace_runtime import WorkspaceManager


class _SessionRepo:
    def __init__(self) -> None:
        self.saved_sessions: list[Session] = []

    async def save(self, session: Session):
        self.saved_sessions.append(session.model_copy(deep=True))


class _WorkspaceRepo:
    def __init__(self) -> None:
        self.workspace_by_id: dict[str, Workspace] = {}
        self.workspace_by_session_id: dict[str, Workspace] = {}

    async def save(self, workspace: Workspace) -> None:
        cloned = workspace.model_copy(deep=True)
        self.workspace_by_id[cloned.id] = cloned
        self.workspace_by_session_id[cloned.session_id] = cloned

    async def get_by_id(self, workspace_id: str):
        return self.workspace_by_id.get(workspace_id)

    async def get_by_id_for_user(self, workspace_id: str, user_id: str):
        workspace = self.workspace_by_id.get(workspace_id)
        if workspace is None or workspace.user_id != user_id:
            return None
        return workspace

    async def get_by_session_id(self, session_id: str):
        return self.workspace_by_session_id.get(session_id)

    async def get_by_session_id_for_user(self, session_id: str, user_id: str):
        workspace = self.workspace_by_session_id.get(session_id)
        if workspace is None or workspace.user_id != user_id:
            return None
        return workspace


class _UoW:
    def __init__(self, session_repo: _SessionRepo, workspace_repo: _WorkspaceRepo):
        self.session = session_repo
        self.workspace = workspace_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _build_manager(
        session_repo: _SessionRepo,
        workspace_repo: _WorkspaceRepo,
) -> WorkspaceManager:
    return WorkspaceManager(
        uow_factory=lambda: _UoW(session_repo=session_repo, workspace_repo=workspace_repo),
    )


def test_workspace_manager_should_create_workspace_and_bind_session() -> None:
    session_repo = _SessionRepo()
    workspace_repo = _WorkspaceRepo()
    manager = _build_manager(session_repo, workspace_repo)
    session = Session(id="session-a", user_id="user-a")

    workspace = asyncio.run(manager.ensure_workspace(session=session))

    assert workspace.session_id == "session-a"
    assert workspace.user_id == "user-a"
    assert session.workspace_id == workspace.id
    assert workspace_repo.workspace_by_id[workspace.id].session_id == "session-a"
    assert session_repo.saved_sessions[0].workspace_id == workspace.id


def test_workspace_manager_should_reuse_workspace_by_session_id_when_session_link_missing() -> None:
    session_repo = _SessionRepo()
    workspace_repo = _WorkspaceRepo()
    manager = _build_manager(session_repo, workspace_repo)
    existing_workspace = Workspace(id="workspace-1", session_id="session-a", user_id="user-a")
    asyncio.run(workspace_repo.save(existing_workspace))
    session = Session(id="session-a", user_id="user-a")

    workspace = asyncio.run(manager.ensure_workspace(session=session))

    assert workspace.id == "workspace-1"
    assert workspace.user_id == "user-a"
    assert session.workspace_id == "workspace-1"
    assert session_repo.saved_sessions[0].workspace_id == "workspace-1"


def test_workspace_manager_should_bind_run_and_environment() -> None:
    session_repo = _SessionRepo()
    workspace_repo = _WorkspaceRepo()
    manager = _build_manager(session_repo, workspace_repo)
    workspace = Workspace(id="workspace-1", session_id="session-a")
    asyncio.run(workspace_repo.save(workspace))
    original_updated_at = workspace.updated_at

    asyncio.run(manager.bind_run(workspace=workspace, run_id="run-1"))
    asyncio.run(
        manager.ensure_environment(
            workspace=workspace,
            sandbox_id="sandbox-1",
            task_id="task-1",
            shell_session_id="shell-1",
        )
    )

    saved_workspace = workspace_repo.workspace_by_id["workspace-1"]
    assert saved_workspace.current_run_id == "run-1"
    assert saved_workspace.sandbox_id == "sandbox-1"
    assert saved_workspace.task_id == "task-1"
    assert saved_workspace.shell_session_id == "shell-1"
    assert saved_workspace.updated_at >= original_updated_at


def test_workspace_manager_should_not_fallback_when_bound_workspace_missing() -> None:
    session_repo = _SessionRepo()
    workspace_repo = _WorkspaceRepo()
    manager = _build_manager(session_repo, workspace_repo)
    existing_workspace = Workspace(id="workspace-other", session_id="session-a", user_id="user-a")
    asyncio.run(workspace_repo.save(existing_workspace))
    session = Session(id="session-a", user_id="user-a", workspace_id="workspace-missing")

    workspace = asyncio.run(manager.get_workspace(session=session))

    assert workspace is None
