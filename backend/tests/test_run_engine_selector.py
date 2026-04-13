import pytest

from app.application.service.run_engine_selector import build_run_engine
from app.domain.models import AgentConfig
from app.domain.services.tools import CapabilityRegistry, ToolRuntimeAdapter
from app.infrastructure.runtime.langgraph_run_engine import LangGraphRunEngine


class _DummyLLM:
    model_name = "base"
    max_tokens = 4096

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {"role": "assistant", "content": "{}"}


class _CloneableDummyLLM(_DummyLLM):
    def __init__(self, *, model_name: str = "base", max_tokens: int = 4096) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens

    def clone_with_overrides(self, model_name=None, temperature=None, max_tokens=None):
        return _CloneableDummyLLM(
            model_name=model_name or self.model_name,
            max_tokens=max_tokens or self.max_tokens,
        )


class _DummyTool:
    pass


class _FakeCheckpointerProvider:
    def __init__(self, checkpointer) -> None:
        self._checkpointer = checkpointer

    def get_checkpointer(self):
        return self._checkpointer


def _raise_langgraph_init_error(**kwargs):
    raise RuntimeError("boom")


class _FakeWorkspaceRuntimeService:
    session_id = "session-1"


def _build_tool_runtime_adapter() -> ToolRuntimeAdapter:
    return ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())


def test_build_run_engine_uses_langgraph_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_settings",
        lambda: type("S", (), {"agent_runtime_engine": "langgraph"})(),
    )
    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_langgraph_checkpointer",
        lambda: _FakeCheckpointerProvider(None),
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
        workspace_runtime_service=_FakeWorkspaceRuntimeService(),
        tool_runtime_adapter=_build_tool_runtime_adapter(),
    )

    assert isinstance(engine, LangGraphRunEngine)


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
        workspace_runtime_service=_FakeWorkspaceRuntimeService(),
        tool_runtime_adapter=_build_tool_runtime_adapter(),
    )

    assert captured["checkpointer"] is checkpointer


def test_build_run_engine_should_pass_stage_llms_and_clamped_iterations(monkeypatch) -> None:
    checkpointer = object()
    captured = {}

    class _FakeEngine:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "app.application.service.run_engine_selector.get_settings",
        lambda: type(
            "S",
            (),
            {
                "agent_runtime_engine": "langgraph",
            },
        )(),
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
        llm=_CloneableDummyLLM(),
        agent_config=AgentConfig(max_iterations=99),
        session_id="session-1",
        file_storage=object(),
        uow_factory=lambda: None,
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=_DummyTool(),
        a2a_tool=_DummyTool(),
        workspace_runtime_service=_FakeWorkspaceRuntimeService(),
        tool_runtime_adapter=_build_tool_runtime_adapter(),
    )

    assert captured["max_tool_iterations"] == 20
    assert set(captured["stage_llms"].keys()) == {"router", "planner", "executor", "replan", "summary"}
    assert captured["stage_llms"]["router"].model_name == "base"
    assert captured["stage_llms"]["router"].max_tokens == 4096
    assert captured["stage_llms"]["planner"].model_name == "base"
    assert captured["stage_llms"]["planner"].max_tokens == 4096
    assert captured["stage_llms"]["executor"].model_name == "base"
    assert captured["stage_llms"]["executor"].max_tokens == 4096
    assert captured["stage_llms"]["replan"].model_name == "base"
    assert captured["stage_llms"]["replan"].max_tokens == 4096
    assert captured["stage_llms"]["summary"].model_name == "base"
    assert captured["stage_llms"]["summary"].max_tokens == 4096
    assert captured["runtime_context_service"] is not None


def test_build_run_engine_should_fallback_to_base_llm_when_clone_unavailable(monkeypatch) -> None:
    checkpointer = object()
    captured = {}
    dummy_llm = _DummyLLM()

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
        llm=dummy_llm,
        agent_config=AgentConfig(max_iterations=3),
        session_id="session-1",
        file_storage=object(),
        uow_factory=lambda: None,
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=_DummyTool(),
        a2a_tool=_DummyTool(),
        workspace_runtime_service=_FakeWorkspaceRuntimeService(),
        tool_runtime_adapter=_build_tool_runtime_adapter(),
    )

    assert set(captured["stage_llms"].keys()) == {"router", "planner", "executor", "replan", "summary"}
    assert all(stage_llm is dummy_llm for stage_llm in captured["stage_llms"].values())


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
            workspace_runtime_service=_FakeWorkspaceRuntimeService(),
            tool_runtime_adapter=_build_tool_runtime_adapter(),
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
            workspace_runtime_service=_FakeWorkspaceRuntimeService(),
            tool_runtime_adapter=_build_tool_runtime_adapter(),
        )
