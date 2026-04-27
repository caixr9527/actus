import asyncio

from app.application.service.agent_service import AgentService
from app.domain.models import AgentConfig, MCPConfig, A2AConfig, Session, SessionStatus, WorkflowRun, Workspace


class _DummyBrowser:
    pass


class _DummySandbox:
    def __init__(self) -> None:
        self.id = "sandbox-1"

    async def get_browser(self):
        return _DummyBrowser()

    async def destroy(self):
        return None


class _DummySandboxCls:
    @classmethod
    async def get(cls, id: str):
        return None

    @classmethod
    async def create(cls):
        return _DummySandbox()


class _Task:
    def __init__(self, task_runner) -> None:
        self.id = "task-1"
        self.task_runner = task_runner

    async def cancel(self):
        return None


class _TaskFactory:
    created_tasks: list[_Task] = []

    @classmethod
    def create(cls, task_runner):
        task = _Task(task_runner=task_runner)
        cls.created_tasks.append(task)
        return task


class _SessionRepo:
    def __init__(self) -> None:
        self.saved_sessions: list[Session] = []
        self.sessions_by_id: dict[str, Session] = {}

    async def save(self, session: Session) -> None:
        cloned = session.model_copy(deep=True)
        self.saved_sessions.append(cloned)
        self.sessions_by_id[cloned.id] = cloned

    async def get_by_id_for_update(self, session_id: str):
        return self.sessions_by_id.get(session_id)

    async def update_runtime_state(self, session_id: str, *, status: SessionStatus, current_run_id=None, **_kwargs):
        session = self.sessions_by_id[session_id]
        session.status = status
        if current_run_id is not None:
            session.current_run_id = current_run_id


class _WorkflowRunRepo:
    def __init__(self) -> None:
        self.created_for_session_ids: list[str] = []
        self.runs_by_id: dict[str, WorkflowRun] = {}

    async def create_for_session(self, session: Session, *, status, thread_id=None):
        self.created_for_session_ids.append(session.id)
        run = WorkflowRun(id="run-1", session_id=session.id, user_id=session.user_id, status=status, thread_id=thread_id)
        self.runs_by_id[run.id] = run
        return run

    async def get_by_id_for_update(self, run_id: str):
        return self.runs_by_id.get(run_id)

    async def list_event_records_by_session(self, session_id: str):
        return []


class _WorkspaceRepo:
    def __init__(self) -> None:
        self.saved_workspaces: list[Workspace] = []
        self.workspace_by_session_id: dict[str, Workspace] = {}

    async def save(self, workspace: Workspace) -> None:
        cloned = workspace.model_copy(deep=True)
        self.saved_workspaces.append(cloned)
        self.workspace_by_session_id[cloned.session_id] = cloned

    async def get_by_id(self, workspace_id: str):
        for workspace in self.saved_workspaces:
            if workspace.id == workspace_id:
                return workspace
        return None

    async def get_by_session_id(self, session_id: str):
        return self.workspace_by_session_id.get(session_id)


class _UoW:
    def __init__(
            self,
            session_repo: _SessionRepo,
            workflow_run_repo: _WorkflowRunRepo,
            workspace_repo: _WorkspaceRepo,
    ) -> None:
        self.session = session_repo
        self.workflow_run = workflow_run_repo
        self.workspace = workspace_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyResolver:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def resolve(self, session: Session):
        self.calls.append(session.current_model_id)
        from app.domain.models import RuntimeLLMConfig

        return "gpt-5.4", RuntimeLLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="secret",
            model_name="gpt-5.4",
            temperature=0.3,
            max_tokens=4096,
        )


class _DummyLLMFactory:
    def __init__(self) -> None:
        self.configs = []

    def create(self, llm_config):
        self.configs.append(llm_config)
        return object()


class _DummySearchEngine:
    pass


class _DummyFileStorage:
    pass


class _DummyJsonParser:
    pass


class _DummyRunEngine:
    async def invoke(self, message):
        if False:
            yield message


def _dummy_run_engine_factory(**kwargs):
    return _DummyRunEngine()


def test_agent_service_create_task_should_build_llm_from_session_model() -> None:
    session_repo = _SessionRepo()
    workflow_run_repo = _WorkflowRunRepo()
    workspace_repo = _WorkspaceRepo()
    resolver = _DummyResolver()
    llm_factory = _DummyLLMFactory()
    _TaskFactory.created_tasks = []

    service = AgentService(
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        sandbox_cls=_DummySandboxCls,
        task_cls=_TaskFactory,
        json_parser=_DummyJsonParser(),
        search_engine=_DummySearchEngine(),
        file_storage=_DummyFileStorage(),
        uow_factory=lambda: _UoW(session_repo, workflow_run_repo, workspace_repo),
        model_runtime_resolver=resolver,
        llm_factory=llm_factory,
        run_engine_factory=_dummy_run_engine_factory,
    )
    session = Session(id="session-a", user_id="user-a", current_model_id="deepseek")
    asyncio.run(session_repo.save(session))

    task = asyncio.run(service._create_task(session))

    assert task.id == "task-1"
    assert resolver.calls == ["deepseek"]
    assert llm_factory.configs[0].model_name == "gpt-5.4"
    assert llm_factory.configs[0].temperature == 0.3
    assert session_repo.sessions_by_id["session-a"].workspace_id is not None
    assert session_repo.sessions_by_id["session-a"].current_run_id == "run-1"
    assert workspace_repo.workspace_by_session_id["session-a"].task_id == "task-1"
