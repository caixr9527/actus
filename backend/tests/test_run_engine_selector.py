import asyncio

import pytest

from app.application.service.run_engine_selector import build_run_engine
from app.domain.models import AgentConfig, DoneEvent, Message
from app.infrastructure.runtime.langgraph_run_engine import LangGraphRunEngine


class _DummyLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {"role": "assistant", "content": "{}"}


class _DummyTool:
    pass


class _FakeCheckpointerProvider:
    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def get_checkpointer(self):
        return self._checkpointer


class _FakeLangGraph:
    async def ainvoke(self, _state, config=None):
        return {"emitted_events": [DoneEvent()]}


def _raise_langgraph_init_error(**kwargs):
    raise RuntimeError("boom")


def test_build_run_engine_uses_langgraph_when_enabled(monkeypatch) -> None:
    checkpointer = object()
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_settings",
        lambda: type("S", (), {"agent_runtime_engine": "langgraph"})(),
    )
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_langgraph_checkpointer",
        lambda: _FakeCheckpointerProvider(checkpointer),
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: _FakeLangGraph(),
    )

    engine = build_run_engine(
        llm=_DummyLLM(),
        agent_config=AgentConfig(),
        session_id="session-1",
        file_storage=object(),
        uow_factory=lambda: None,
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=_DummyTool(),
        a2a_tool=_DummyTool(),
    )

    assert isinstance(engine, LangGraphRunEngine)

    async def _collect():
        return [event async for event in engine.invoke(Message(message="hello"))]

    events = asyncio.run(_collect())
    assert len(events) == 1
    assert isinstance(events[0], DoneEvent)


def test_build_run_engine_should_pass_persistent_checkpointer(monkeypatch) -> None:
    checkpointer = object()
    captured = {}

    class _FakeEngine:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_settings",
        lambda: type("S", (), {"agent_runtime_engine": "langgraph"})(),
    )
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_langgraph_checkpointer",
        lambda: _FakeCheckpointerProvider(checkpointer),
    )
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.LangGraphRunEngine",
        _FakeEngine,
    )

    build_run_engine(
        llm=_DummyLLM(),
        agent_config=AgentConfig(),
        session_id="session-1",
        file_storage=object(),
        uow_factory=lambda: None,
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=_DummyTool(),
        a2a_tool=_DummyTool(),
    )

    assert captured["checkpointer"] is checkpointer


def test_build_run_engine_should_raise_when_langgraph_init_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_settings",
        lambda: type("S", (), {"agent_runtime_engine": "langgraph"})(),
    )
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_langgraph_checkpointer",
        lambda: _FakeCheckpointerProvider(object()),
    )
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.LangGraphRunEngine",
        _raise_langgraph_init_error,
    )

    with pytest.raises(RuntimeError, match="boom"):
        build_run_engine(
            llm=_DummyLLM(),
            agent_config=AgentConfig(),
            session_id="session-1",
            file_storage=object(),
            uow_factory=lambda: None,
            json_parser=object(),
            browser=object(),
            sandbox=object(),
            search_engine=object(),
            mcp_tool=_DummyTool(),
            a2a_tool=_DummyTool(),
        )


def test_build_run_engine_should_reject_non_langgraph_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_settings",
        lambda: type("S", (), {"agent_runtime_engine": "legacy"})(),
    )

    with pytest.raises(ValueError, match="仅支持 langgraph"):
        build_run_engine(
            llm=_DummyLLM(),
            agent_config=AgentConfig(),
            session_id="session-1",
            file_storage=object(),
            uow_factory=lambda: None,
            json_parser=object(),
            browser=object(),
            sandbox=object(),
            search_engine=object(),
            mcp_tool=_DummyTool(),
            a2a_tool=_DummyTool(),
        )
