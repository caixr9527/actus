from app.application.service.run_engine_selector import build_run_engine
from app.domain.models import AgentConfig
from app.domain.services.runtime import LegacyPlannerReActRunEngine


class _DummyLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {"role": "assistant", "content": "{}"}


class _DummyTool:
    pass


def test_build_run_engine_falls_back_to_legacy_when_langgraph_not_available(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_settings",
        lambda: type("S", (), {"agent_runtime_engine": "langgraph_poc"})(),
    )

    engine = build_run_engine(
        llm=_DummyLLM(),
        agent_config=AgentConfig(),
        session_id="session-1",
        uow_factory=lambda: None,
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=_DummyTool(),
        a2a_tool=_DummyTool(),
    )

    assert isinstance(engine, LegacyPlannerReActRunEngine)
