import asyncio

from app.domain.models import AgentConfig, MCPConfig, A2AConfig
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.tools import CapabilityRegistry, ToolRuntimeAdapter
import pytest


class _DummyRunEngine:
    async def invoke(self, message):
        if False:
            yield message


class _Recorder:
    async def record_runtime_tool_snapshot(self, *, user_id: str, session_id: str, snapshot):
        return object()


class _WorkspaceRuntime:
    session_id = "session-1"


def test_agent_task_runner_should_build_run_engine_with_task_scoped_tools() -> None:
    captured = {}
    engine = _DummyRunEngine()

    async def _factory(**kwargs):
        captured.update(kwargs)
        return engine

    runner = asyncio.run(
        AgentTaskRunner.create(
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
            tool_runtime_adapter=ToolRuntimeAdapter(
                capability_registry=CapabilityRegistry.default_v1(),
            ),
            runtime_tool_snapshot_recorder=_Recorder(),
            workspace_runtime_factory=lambda: _WorkspaceRuntime(),
        )
    )

    assert runner._run_engine is engine
    assert captured["session_id"] == "session-1"
    assert captured["mcp_tool"] is runner._mcp_tool
    assert captured["a2a_tool"] is runner._a2a_tool
    assert isinstance(captured["tool_runtime_adapter"], ToolRuntimeAdapter)
    assert captured["runtime_tool_snapshot_recorder"] is not None


def test_agent_task_runner_should_fail_fast_when_run_engine_factory_missing() -> None:
    with pytest.raises(RuntimeError, match="仅支持 LangGraph 运行时"):
        asyncio.run(
            AgentTaskRunner.create(
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
                tool_runtime_adapter=ToolRuntimeAdapter(
                    capability_registry=CapabilityRegistry.default_v1(),
                ),
                runtime_tool_snapshot_recorder=_Recorder(),
                workspace_runtime_factory=lambda: _WorkspaceRuntime(),
            )
        )


def test_agent_task_runner_should_fail_fast_when_snapshot_recorder_missing() -> None:
    async def _factory(**kwargs):
        return _DummyRunEngine()

    with pytest.raises(RuntimeError, match="RuntimeToolSnapshotRecorderPort"):
        asyncio.run(
            AgentTaskRunner.create(
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
                tool_runtime_adapter=ToolRuntimeAdapter(
                    capability_registry=CapabilityRegistry.default_v1(),
                ),
                workspace_runtime_factory=lambda: _WorkspaceRuntime(),
            )
        )


def test_agent_task_runner_create_should_require_workspace_runtime_factory() -> None:
    async def _factory(**kwargs):
        return _DummyRunEngine()

    with pytest.raises(RuntimeError, match="ledger-backed WorkspaceRuntimeService factory"):
        asyncio.run(
            AgentTaskRunner.create(
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
                tool_runtime_adapter=ToolRuntimeAdapter(
                    capability_registry=CapabilityRegistry.default_v1(),
                ),
                runtime_tool_snapshot_recorder=_Recorder(),
            )
        )
