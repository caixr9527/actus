from app.infrastructure.runtime.langgraph_run_engine import LangGraphRunEngine


class _FakeGraph:
    async def ainvoke(self, _state, config=None):
        return {"emitted_events": []}


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
