import asyncio
from types import SimpleNamespace

from langgraph.types import Command

from app.domain.models import DoneEvent, Message, MessageEvent, WaitEvent
from app.infrastructure.runtime.langgraph_run_engine import LangGraphRunEngine


class _FakeGraph:
    def __init__(self, result=None, checkpoint_state=None) -> None:
        self._result = result or {"emitted_events": []}
        self._checkpoint_state = checkpoint_state
        self.calls = []

    async def ainvoke(self, state, config=None):
        self.calls.append((state, config))
        return self._result

    async def aget_state(self, config):
        if self._checkpoint_state is None:
            return None
        return SimpleNamespace(values=self._checkpoint_state)


def test_langgraph_run_engine_should_inject_checkpointer_into_graph_builder(monkeypatch) -> None:
    captured = {}
    checkpointer = object()

    def _fake_build_graph(**kwargs):
        captured.update(kwargs)
        return _FakeGraph()

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        _fake_build_graph,
    )

    LangGraphRunEngine(
        session_id="session-1",
        llm=object(),
        checkpointer=checkpointer,
    )

    assert captured["checkpointer"] is checkpointer


def test_langgraph_run_engine_invoke_should_emit_wait_event_from_interrupt(monkeypatch) -> None:
    fake_graph = _FakeGraph(
        result={
            "__interrupt__": [
                SimpleNamespace(
                    id="interrupt-1",
                    value={
                        "kind": "input_text",
                        "prompt": "请确认是否继续",
                        "response_key": "message",
                    },
                )
            ]
        },
        checkpoint_state={
            "session_id": "session-1",
            "graph_metadata": {},
            "pending_interrupt": {},
            "emitted_events": [],
        },
    )

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    engine = LangGraphRunEngine(
        session_id="session-1",
        llm=object(),
    )

    async def _collect():
        events = []
        async for event in engine.invoke(Message(message="hello")):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert len(events) == 1
    assert isinstance(events[0], WaitEvent)
    assert events[0].interrupt_id == "interrupt-1"
    assert events[0].payload["prompt"] == "请确认是否继续"


def test_langgraph_run_engine_resume_should_use_command_resume(monkeypatch) -> None:
    fake_graph = _FakeGraph(
        result={"emitted_events": []},
        checkpoint_state={
            "session_id": "session-1",
            "graph_metadata": {
                "pending_interrupts": [
                    {
                        "interrupt_id": "interrupt-1",
                        "payload": {"kind": "input_text", "prompt": "请确认是否继续", "response_key": "message"},
                    }
                ]
            },
            "pending_interrupt": {"kind": "input_text", "prompt": "请确认是否继续", "response_key": "message"},
            "emitted_events": [],
        },
    )

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    engine = LangGraphRunEngine(
        session_id="session-1",
        llm=object(),
    )

    async def _collect():
        events = []
        async for event in engine.resume({"approved": True}):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events == []
    assert len(fake_graph.calls) == 1
    graph_input, config = fake_graph.calls[0]
    assert isinstance(graph_input, Command)
    assert graph_input.resume == {"approved": True}
    assert config == {"configurable": {"thread_id": "session-1"}}


def test_langgraph_run_engine_resume_should_only_emit_incremental_events(monkeypatch) -> None:
    history_event = MessageEvent(id="evt-history", role="assistant", message="历史消息")
    new_event = DoneEvent(id="evt-done")
    fake_graph = _FakeGraph(
        result={
            "session_id": "session-1",
            "graph_metadata": {},
            "pending_interrupt": {},
            "emitted_events": [history_event, new_event],
        },
        checkpoint_state={
            "session_id": "session-1",
            "graph_metadata": {},
            "pending_interrupt": {"kind": "input_text", "prompt": "请确认是否继续", "response_key": "message"},
            "emitted_events": [history_event],
        },
    )

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    engine = LangGraphRunEngine(
        session_id="session-1",
        llm=object(),
    )

    async def _collect():
        events = []
        async for event in engine.resume({"approved": True}):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert len(events) == 1
    assert isinstance(events[0], DoneEvent)
    assert events[0].id == "evt-done"


def test_langgraph_run_engine_inspect_resume_checkpoint_should_report_missing_pending_interrupt(monkeypatch) -> None:
    fake_graph = _FakeGraph(
        result={"emitted_events": []},
        checkpoint_state={
            "session_id": "session-1",
            "graph_metadata": {},
            "pending_interrupt": {},
            "emitted_events": [],
        },
    )

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    engine = LangGraphRunEngine(
        session_id="session-1",
        llm=object(),
    )

    inspection = asyncio.run(engine.inspect_resume_checkpoint())

    assert inspection.run_id is None
    assert inspection.has_checkpoint is True
    assert inspection.pending_interrupt == {}
    assert inspection.is_resumable is False
