from app.domain.models import AgentConfig, MCPConfig, A2AConfig
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.tools import ToolRuntimeAdapter
import pytest


class _DummyRunEngine:
    async def invoke(self, message):
        if False:
            yield message


def test_agent_task_runner_should_build_run_engine_with_task_scoped_tools() -> None:
    captured = {}
    engine = _DummyRunEngine()

    def _factory(**kwargs):
        captured.update(kwargs)
        return engine

    runner = AgentTaskRunner(
        llm=object(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        session_id="session-1",
        user_id="user-1",
        file_storage=object(),
        uow_factory=lambda: None,
        json_parser=object(),
        browser=object(),
        search_engine=object(),
        sandbox=object(),
        run_engine_factory=_factory,
    )

    assert runner._run_engine is engine
    assert captured["session_id"] == "session-1"
    assert captured["mcp_tool"] is runner._mcp_tool
    assert captured["a2a_tool"] is runner._a2a_tool
    assert isinstance(captured["tool_runtime_adapter"], ToolRuntimeAdapter)


def test_agent_task_runner_should_fail_fast_when_run_engine_factory_missing() -> None:
    with pytest.raises(RuntimeError, match="仅支持 LangGraph 运行时"):
        AgentTaskRunner(
            llm=object(),
            agent_config=AgentConfig(),
            mcp_config=MCPConfig(),
            a2a_config=A2AConfig(),
            session_id="session-1",
            user_id="user-1",
            file_storage=object(),
            uow_factory=lambda: None,
            json_parser=object(),
            browser=object(),
            search_engine=object(),
            sandbox=object(),
        )
