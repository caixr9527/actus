import asyncio

from app.application.service.agent_service import AgentService
from app.domain.models import RuntimeLLMConfig, Session, SessionStatus


class _StubTask:
    def __init__(self, task_id: str = "task-1") -> None:
        self.id = task_id


class _StubGraphRuntime:
    def __init__(self) -> None:
        self.get_calls: list[str] = []
        self.create_calls: list[tuple[str, object]] = []
        self.resume_calls: list[tuple[str, object]] = []
        self.cancel_calls: list[str] = []
        self.destroy_calls = 0
        self.task = _StubTask()

    async def get_task(self, session: Session):
        self.get_calls.append(session.id)
        return self.task

    async def create_task(self, session: Session, llm: object):
        self.create_calls.append((session.id, llm))
        return self.task

    async def resume_task(self, session: Session, llm: object):
        self.resume_calls.append((session.id, llm))
        return self.task

    async def cancel_task(self, session: Session) -> bool:
        self.cancel_calls.append(session.id)
        return True

    async def destroy(self) -> None:
        self.destroy_calls += 1


class _DummyResolver:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def resolve(self, session: Session):
        self.calls.append(session.current_model_id)
        return "gpt-5.4", RuntimeLLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="secret",
            model_name="gpt-5.4",
            temperature=0.7,
            max_tokens=1024,
        )


class _DummyLLMFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.last_llm = object()
        self.last_config = None

    def create(self, llm_config):
        self.calls += 1
        self.last_config = llm_config
        return self.last_llm


def test_agent_service_get_task_should_delegate_to_graph_runtime() -> None:
    runtime = _StubGraphRuntime()
    service = object.__new__(AgentService)
    service._graph_runtime = runtime
    session = Session(id="session-a", user_id="user-a")

    task = asyncio.run(service._get_task(session))

    assert task is runtime.task
    assert runtime.get_calls == ["session-a"]


def test_agent_service_create_task_should_delegate_to_graph_runtime() -> None:
    runtime = _StubGraphRuntime()
    resolver = _DummyResolver()
    llm_factory = _DummyLLMFactory()
    service = object.__new__(AgentService)
    service._graph_runtime = runtime
    service._model_runtime_resolver = resolver
    service._llm_factory = llm_factory
    session = Session(id="session-a", user_id="user-a", current_model_id="model-a")

    task = asyncio.run(service._create_task(session))

    assert task is runtime.task
    assert resolver.calls == ["model-a"]
    assert llm_factory.calls == 1
    assert runtime.create_calls == [("session-a", llm_factory.last_llm)]


def test_agent_service_shutdown_should_delegate_graph_runtime_destroy() -> None:
    runtime = _StubGraphRuntime()
    service = object.__new__(AgentService)
    service._graph_runtime = runtime

    asyncio.run(service.shutdown())

    assert runtime.destroy_calls == 1


def test_agent_service_create_task_should_delegate_resume_to_graph_runtime() -> None:
    runtime = _StubGraphRuntime()
    resolver = _DummyResolver()
    llm_factory = _DummyLLMFactory()
    service = object.__new__(AgentService)
    service._graph_runtime = runtime
    service._model_runtime_resolver = resolver
    service._llm_factory = llm_factory
    session = Session(id="session-a", user_id="user-a", current_model_id="model-a", current_run_id="run-1")

    task = asyncio.run(service._create_task(session, reuse_current_run=True))

    assert task is runtime.task
    assert runtime.create_calls == []
    assert runtime.resume_calls == [("session-a", llm_factory.last_llm)]


def test_agent_service_stop_session_should_delegate_cancel_to_graph_runtime() -> None:
    runtime = _StubGraphRuntime()
    session = Session(id="session-a", user_id="user-a", status=SessionStatus.RUNNING)

    class _SessionRepo:
        async def get_by_id(self, session_id: str, user_id: str | None = None):
            return session

    class _UoW:
        def __init__(self) -> None:
            self.session = _SessionRepo()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    service = object.__new__(AgentService)
    service._graph_runtime = runtime
    service._uow_factory = lambda: _UoW()

    asyncio.run(service.stop_session("session-a", "user-a"))

    assert runtime.cancel_calls == ["session-a"]
