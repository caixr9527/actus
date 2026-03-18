import asyncio

from app.application.service.agent_service import AgentService
from app.domain.models import AgentConfig, MCPConfig, A2AConfig, Session


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

    def cancel(self):
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

    async def save(self, session: Session) -> None:
        self.saved_sessions.append(session)


class _UoW:
    def __init__(self, session_repo: _SessionRepo) -> None:
        self.session = session_repo

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


def test_agent_service_create_task_should_build_llm_from_session_model() -> None:
    session_repo = _SessionRepo()
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
        uow_factory=lambda: _UoW(session_repo),
        model_runtime_resolver=resolver,
        llm_factory=llm_factory,
    )
    session = Session(id="session-a", user_id="user-a", current_model_id="deepseek")

    task = asyncio.run(service._create_task(session))

    assert task.id == "task-1"
    assert resolver.calls == ["deepseek"]
    assert llm_factory.configs[0].model_name == "gpt-5.4"
    assert llm_factory.configs[0].temperature == 0.3
    assert session_repo.saved_sessions[0].task_id == "task-1"
