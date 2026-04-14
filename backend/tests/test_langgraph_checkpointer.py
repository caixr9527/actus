import asyncio
from types import SimpleNamespace

from app.infrastructure.runtime.langgraph.checkpoint import checkpointer as checkpointer_module
from scripts import bootstrap_langgraph_checkpointer


class _FakeAsyncSaver:
    def __init__(self) -> None:
        self.setup_calls = 0

    async def setup(self) -> None:
        self.setup_calls += 1


class _FakeAsyncSaverContextManager:
    def __init__(self, saver: _FakeAsyncSaver) -> None:
        self._saver = saver
        self.exit_calls = 0

    async def __aenter__(self) -> _FakeAsyncSaver:
        return self._saver

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exit_calls += 1


class _FakeAsyncPostgresSaver:
    last_conn_string: str | None = None
    last_context_manager: _FakeAsyncSaverContextManager | None = None
    last_saver: _FakeAsyncSaver | None = None

    @classmethod
    def from_conn_string(cls, conn_string: str):
        cls.last_conn_string = conn_string
        cls.last_saver = _FakeAsyncSaver()
        cls.last_context_manager = _FakeAsyncSaverContextManager(cls.last_saver)
        return cls.last_context_manager


def test_langgraph_checkpointer_init_should_not_call_setup(monkeypatch) -> None:
    monkeypatch.setattr(checkpointer_module, "AsyncPostgresSaver", _FakeAsyncPostgresSaver)
    checkpointer = checkpointer_module.LangGraphCheckpointer()
    checkpointer._settings = SimpleNamespace(
        sqlalchemy_database_uri="postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/actus"
    )

    asyncio.run(checkpointer.init())
    asyncio.run(checkpointer.close())

    assert _FakeAsyncPostgresSaver.last_conn_string == "postgresql://postgres:postgres@127.0.0.1:5432/actus"
    assert _FakeAsyncPostgresSaver.last_saver is not None
    assert _FakeAsyncPostgresSaver.last_saver.setup_calls == 0
    assert _FakeAsyncPostgresSaver.last_context_manager is not None
    assert _FakeAsyncPostgresSaver.last_context_manager.exit_calls == 1


def test_langgraph_checkpointer_ensure_schema_should_call_setup(monkeypatch) -> None:
    monkeypatch.setattr(checkpointer_module, "AsyncPostgresSaver", _FakeAsyncPostgresSaver)
    checkpointer = checkpointer_module.LangGraphCheckpointer()
    checkpointer._settings = SimpleNamespace(
        sqlalchemy_database_uri="postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/actus"
    )

    asyncio.run(checkpointer.init())
    asyncio.run(checkpointer.ensure_schema())
    asyncio.run(checkpointer.close())

    assert _FakeAsyncPostgresSaver.last_saver is not None
    assert _FakeAsyncPostgresSaver.last_saver.setup_calls == 1


def test_bootstrap_langgraph_checkpointer_main_should_call_init_schema_and_close(monkeypatch) -> None:
    calls: list[str] = []

    class _FakeCheckpointer:
        async def init(self) -> None:
            calls.append("init")

        async def ensure_schema(self) -> None:
            calls.append("ensure_schema")

        async def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(
        bootstrap_langgraph_checkpointer,
        "get_langgraph_checkpointer",
        lambda: _FakeCheckpointer(),
    )

    asyncio.run(bootstrap_langgraph_checkpointer.main())

    assert calls == ["init", "ensure_schema", "close"]
