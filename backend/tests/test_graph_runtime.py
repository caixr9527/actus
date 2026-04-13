import asyncio

from app.domain.models import Session, SessionStatus, WorkflowRun, Workspace
from app.domain.services.runtime.graph_runtime import DefaultGraphRuntime


class _DummyBrowser:
    pass


class _DummySandbox:
    def __init__(self) -> None:
        self.id = "sandbox-1"
        self.destroy_calls = 0

    async def get_browser(self):
        return _DummyBrowser()

    async def destroy(self):
        self.destroy_calls += 1
        return True


class _DummySandboxCls:
    sandbox = _DummySandbox()

    @classmethod
    async def get(cls, id: str):
        return cls.sandbox if id == cls.sandbox.id else None

    @classmethod
    async def create(cls):
        cls.sandbox = _DummySandbox()
        return cls.sandbox


class _Task:
    def __init__(self, task_id: str) -> None:
        self.id = task_id
        self.cancel_calls = 0

    async def cancel(self) -> bool:
        self.cancel_calls += 1
        return True


class _TaskFactory:
    registry: dict[str, _Task] = {}
    sequence = 0
    destroyed = False

    @classmethod
    def get(cls, task_id: str):
        return cls.registry.get(task_id)

    @classmethod
    def create(cls, task_runner):
        cls.sequence += 1
        task = _Task(task_id=f"task-{cls.sequence}")
        cls.registry[task.id] = task
        return task

    @classmethod
    async def destroy(cls):
        cls.destroyed = True
        cls.registry = {}


class _SessionRepo:
    def __init__(self) -> None:
        self.saved_sessions: list[Session] = []

    async def save(self, session: Session):
        self.saved_sessions.append(session.model_copy(deep=True))


class _WorkflowRunRepo:
    def __init__(self) -> None:
        self.created_for_session_ids: list[str] = []

    async def create_for_session(self, session: Session, *, status, thread_id=None):
        self.created_for_session_ids.append(session.id)
        return WorkflowRun(
            id="run-1",
            session_id=session.id,
            user_id=session.user_id,
            status=status,
            thread_id=thread_id,
        )


class _UoW:
    def __init__(
            self,
            session_repo: _SessionRepo,
            workflow_run_repo: _WorkflowRunRepo,
            workspace_repo,
    ):
        self.session = session_repo
        self.workflow_run = workflow_run_repo
        self.workspace = workspace_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


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

    async def get_by_session_id(self, session_id: str):
        return self.workspace_by_session_id.get(session_id)


def _build_runtime(
        session_repo: _SessionRepo,
        workflow_run_repo: _WorkflowRunRepo,
        workspace_repo: _WorkspaceRepo,
) -> DefaultGraphRuntime:
    return DefaultGraphRuntime(
        sandbox_cls=_DummySandboxCls,
        task_cls=_TaskFactory,
        uow_factory=lambda: _UoW(session_repo, workflow_run_repo, workspace_repo),
        task_runner_factory=lambda **kwargs: object(),
    )


def test_default_graph_runtime_create_task_should_persist_session_links() -> None:
    _TaskFactory.registry = {}
    _TaskFactory.sequence = 0
    session_repo = _SessionRepo()
    workflow_run_repo = _WorkflowRunRepo()
    workspace_repo = _WorkspaceRepo()
    runtime = _build_runtime(session_repo, workflow_run_repo, workspace_repo)
    session = Session(id="session-a", user_id="user-a")

    task = asyncio.run(runtime.create_task(session=session, llm=object()))

    assert task.id == "task-1"
    assert session.workspace_id is not None
    assert session.current_run_id == "run-1"
    assert session.status == SessionStatus.RUNNING
    assert workflow_run_repo.created_for_session_ids == ["session-a"]
    assert session_repo.saved_sessions[-1].current_run_id == "run-1"
    assert session_repo.saved_sessions[-1].status == SessionStatus.RUNNING
    workspace = workspace_repo.workspace_by_session_id["session-a"]
    assert workspace.sandbox_id == "sandbox-1"
    assert workspace.task_id == "task-1"
    assert workspace.current_run_id == "run-1"


def test_default_graph_runtime_get_and_cancel_task_by_session() -> None:
    _TaskFactory.registry = {}
    _TaskFactory.sequence = 0
    session_repo = _SessionRepo()
    workflow_run_repo = _WorkflowRunRepo()
    workspace_repo = _WorkspaceRepo()
    workspace = Workspace(
        id="workspace-1",
        session_id="session-a",
        task_id="task-1",
        sandbox_id="sandbox-1",
    )
    asyncio.run(workspace_repo.save(workspace))
    runtime = _build_runtime(session_repo, workflow_run_repo, workspace_repo)
    session = Session(id="session-a", user_id="user-a", workspace_id="workspace-1")
    _TaskFactory.registry["task-1"] = _Task(task_id="task-1")

    task = asyncio.run(runtime.get_task(session=session))
    cancelled = asyncio.run(runtime.cancel_task(session=session))

    assert task is not None
    assert task.id == "task-1"
    assert cancelled is True
    assert _TaskFactory.registry["task-1"].cancel_calls == 1


def test_default_graph_runtime_destroy_should_delegate_task_factory() -> None:
    _TaskFactory.destroyed = False
    session_repo = _SessionRepo()
    workflow_run_repo = _WorkflowRunRepo()
    workspace_repo = _WorkspaceRepo()
    runtime = _build_runtime(session_repo, workflow_run_repo, workspace_repo)

    asyncio.run(runtime.destroy())

    assert _TaskFactory.destroyed is True


def test_default_graph_runtime_resume_task_should_reuse_current_run() -> None:
    _TaskFactory.registry = {}
    _TaskFactory.sequence = 0
    session_repo = _SessionRepo()
    workflow_run_repo = _WorkflowRunRepo()
    workspace_repo = _WorkspaceRepo()
    existing_workspace = Workspace(
        id="workspace-1",
        session_id="session-a",
        sandbox_id="sandbox-1",
        current_run_id="run-1",
    )
    asyncio.run(workspace_repo.save(existing_workspace))
    runtime = _build_runtime(session_repo, workflow_run_repo, workspace_repo)
    session = Session(
        id="session-a",
        user_id="user-a",
        workspace_id="workspace-1",
        current_run_id="run-1",
        status=SessionStatus.WAITING,
    )

    task = asyncio.run(runtime.resume_task(session=session, llm=object()))

    assert task.id == "task-1"
    assert session.current_run_id == "run-1"
    assert session.status == SessionStatus.RUNNING
    assert workflow_run_repo.created_for_session_ids == []
    assert session_repo.saved_sessions[0].current_run_id == "run-1"
    assert session_repo.saved_sessions[0].status == SessionStatus.RUNNING
    workspace = workspace_repo.workspace_by_id["workspace-1"]
    assert workspace.task_id == "task-1"
    assert workspace.current_run_id == "run-1"


def test_default_graph_runtime_get_task_should_reuse_workspace_found_by_session_id() -> None:
    _TaskFactory.registry = {}
    _TaskFactory.sequence = 0
    session_repo = _SessionRepo()
    workflow_run_repo = _WorkflowRunRepo()
    workspace_repo = _WorkspaceRepo()
    workspace = Workspace(
        id="workspace-1",
        session_id="session-a",
        task_id="task-1",
        sandbox_id="sandbox-1",
    )
    asyncio.run(workspace_repo.save(workspace))
    runtime = _build_runtime(session_repo, workflow_run_repo, workspace_repo)
    session = Session(id="session-a", user_id="user-a")
    _TaskFactory.registry["task-1"] = _Task(task_id="task-1")

    task = asyncio.run(runtime.get_task(session=session))

    assert task is not None
    assert task.id == "task-1"
