import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.types import Command

from app.domain.models import (
    Message,
    WaitEvent,
    Session,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowRunSummary,
    SessionContextSnapshot,
    Plan,
    Step,
    ExecutionStatus,
)
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


class _FakeUoW:
    def __init__(self, *, session_repo, workflow_run_repo, workflow_run_summary_repo, session_context_snapshot_repo):
        self.session = session_repo
        self.workflow_run = workflow_run_repo
        self.workflow_run_summary = workflow_run_summary_repo
        self.session_context_snapshot = session_context_snapshot_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


def _build_stage_llms(llm=object()) -> dict[str, object]:
    return {
        "router": llm,
        "planner": llm,
        "executor": llm,
        "replan": llm,
        "summary": llm,
    }


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
        stage_llms=_build_stage_llms(),
        checkpointer=checkpointer,
    )

    assert captured["checkpointer"] is checkpointer


def test_langgraph_run_engine_should_require_complete_stage_llms() -> None:
    with pytest.raises(ValueError, match="缺少必要阶段模型配置"):
        LangGraphRunEngine(
            session_id="session-1",
            stage_llms={"executor": object()},
        )


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
        stage_llms=_build_stage_llms(),
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
    engine = LangGraphRunEngine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
    )
    captured = {}

    async def _fake_run_graph(*, graph_input, invoke_config, run_id, fallback_state):
        captured["graph_input"] = graph_input
        captured["invoke_config"] = invoke_config
        captured["run_id"] = run_id
        captured["fallback_state"] = fallback_state
        if False:
            yield None

    async def _fake_load_checkpoint_state(*, invoke_config):
        captured["checkpoint_invoke_config"] = invoke_config
        return {"session_id": "session-1", "pending_interrupt": {}}

    monkeypatch.setattr(engine, "_run_graph", _fake_run_graph)
    monkeypatch.setattr(engine, "_load_checkpoint_state", _fake_load_checkpoint_state)

    async def _collect():
        events = []
        async for event in engine.resume({"approved": True}):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events == []
    assert isinstance(captured["graph_input"], Command)
    assert captured["graph_input"].resume == {"approved": True}
    assert captured["invoke_config"] == {"configurable": {"thread_id": "session-1"}}
    assert captured["checkpoint_invoke_config"] == {"configurable": {"thread_id": "session-1"}}


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
        stage_llms=_build_stage_llms(),
    )

    inspection = asyncio.run(engine.inspect_resume_checkpoint())

    assert inspection.run_id is None
    assert inspection.has_checkpoint is True
    assert inspection.pending_interrupt == {}
    assert inspection.is_resumable is False


def test_langgraph_run_engine_should_build_initial_state_with_session_snapshot_and_completed_summaries(monkeypatch) -> None:
    fake_graph = _FakeGraph()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    session_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=Session(
                id="session-1",
                user_id="user-1",
                current_run_id="run-1",
            )
        )
    )
    workflow_run_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=WorkflowRun(
                id="run-1",
                session_id="session-1",
                user_id="user-1",
                thread_id="thread-1",
                status=WorkflowRunStatus.RUNNING,
            )
        )
    )
    workflow_run_summary_repo = SimpleNamespace(
        list_by_session_id=AsyncMock(
            side_effect=[
                [
                    WorkflowRunSummary(
                        run_id="run-completed",
                        session_id="session-1",
                        status=WorkflowRunStatus.COMPLETED,
                        title="已完成运行",
                        final_answer_summary="做完了前置分析",
                        open_questions=["还需最终确认"],
                        artifacts=["artifact-1"],
                    ),
                ],
                [
                    WorkflowRunSummary(
                        run_id="run-failed",
                        session_id="session-1",
                        status=WorkflowRunStatus.FAILED,
                        title="失败运行",
                        final_answer_summary="卡在外部依赖",
                        open_questions=["需要补充网络权限"],
                        blockers=["外部接口失败"],
                        artifacts=["artifact-failed"],
                    ),
                ],
            ]
        )
    )
    session_context_snapshot_repo = SimpleNamespace(
        get_by_session_id=AsyncMock(
            return_value=SessionContextSnapshot(
                session_id="session-1",
                summary_text="跨轮会话摘要",
                recent_run_briefs=[{"run_id": "run-snapshot", "title": "快照中的运行"}],
                open_questions=["快照问题"],
                artifact_refs=["snapshot-artifact"],
            )
        )
    )

    engine = LangGraphRunEngine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
        uow_factory=lambda: _FakeUoW(
            session_repo=session_repo,
            workflow_run_repo=workflow_run_repo,
            workflow_run_summary_repo=workflow_run_summary_repo,
            session_context_snapshot_repo=session_context_snapshot_repo,
        ),
    )

    state = asyncio.run(
        engine._build_graph_input_state(
            message=Message(message="hello"),
            run_id="run-1",
            invoke_config={"configurable": {"thread_id": "session-1"}},
        )
    )

    assert state["conversation_summary"] == "跨轮会话摘要"
    assert workflow_run_summary_repo.list_by_session_id.await_args_list[0].kwargs["statuses"] == [WorkflowRunStatus.COMPLETED]
    assert workflow_run_summary_repo.list_by_session_id.await_args_list[1].kwargs["statuses"] == [
        WorkflowRunStatus.FAILED,
        WorkflowRunStatus.CANCELLED,
    ]
    assert state["recent_run_briefs"][0]["run_id"] == "run-completed"
    assert state["recent_attempt_briefs"][0]["run_id"] == "run-failed"
    assert state["session_open_questions"] == ["还需最终确认", "需要补充网络权限", "快照问题"]
    assert state["session_blockers"] == ["外部接口失败"]
    assert state["selected_artifacts"] == []
    assert state["historical_artifact_refs"] == ["snapshot-artifact", "artifact-1", "artifact-failed"]


def test_langgraph_run_engine_should_sync_run_summary_and_session_snapshot(monkeypatch) -> None:
    fake_graph = _FakeGraph()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        thread_id="thread-1",
        status=WorkflowRunStatus.RUNNING,
    )
    workflow_run_repo = SimpleNamespace(
        get_by_id=AsyncMock(return_value=run),
        update_runtime_metadata=AsyncMock(),
    )
    first_summary = WorkflowRunSummary(
        run_id="run-1",
        session_id="session-1",
        user_id="user-1",
        thread_id="thread-1",
        status=WorkflowRunStatus.COMPLETED,
        title="本轮运行",
        final_answer_summary="最终总结",
        artifacts=["artifact-1"],
    )
    workflow_run_summary_repo = SimpleNamespace(
        upsert=AsyncMock(return_value=first_summary),
        list_by_session_id=AsyncMock(return_value=[first_summary]),
    )
    session_context_snapshot_repo = SimpleNamespace(
        upsert=AsyncMock(),
    )

    engine = LangGraphRunEngine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
        uow_factory=lambda: _FakeUoW(
            session_repo=SimpleNamespace(),
            workflow_run_repo=workflow_run_repo,
            workflow_run_summary_repo=workflow_run_summary_repo,
            session_context_snapshot_repo=session_context_snapshot_repo,
        ),
    )

    state = {
        "session_id": "session-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "plan": Plan(
            title="本轮运行",
            goal="整理上下文",
            language="zh",
            steps=[
                Step(
                    id="step-1",
                    title="分析",
                    description="分析现状",
                    objective_key="objective-step-1",
                    success_criteria=["输出分析结论"],
                    status=ExecutionStatus.COMPLETED,
                )
            ],
        ),
        "step_states": [
            {
                "step_id": "step-1",
                "title": "分析",
                "description": "分析现状",
                "objective_key": "objective-step-1",
                "success_criteria": ["输出分析结论"],
                "status": "completed",
                "outcome": {
                    "done": True,
                    "summary": "分析完成",
                    "produced_artifacts": ["artifact-1"],
                    "blockers": [],
                    "facts_learned": ["已经确认问题边界"],
                    "open_questions": ["还需确认最终输出"],
                    "next_hint": None,
                    "reused_from_run_id": None,
                    "reused_from_step_id": None,
                },
            }
        ],
        "working_memory": {
            "goal": "整理上下文",
            "facts_in_session": ["已经确认问题边界"],
            "open_questions": ["还需确认最终输出"],
        },
        "selected_artifacts": ["artifact-1"],
        "artifact_refs": ["artifact-1", "noise-artifact"],
        "graph_metadata": {"projection": {"run_status": "completed"}},
        "final_message": "最终总结",
        "retrieved_memories": [],
        "pending_memory_writes": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "message_window": [],
        "input_parts": [],
        "execution_count": 1,
        "max_execution_steps": 20,
        "current_step_id": None,
        "last_executed_step": None,
        "pending_interrupt": {},
        "emitted_events": [],
        "conversation_summary": "会话摘要",
        "historical_artifact_refs": [],
    }

    asyncio.run(engine._sync_graph_state_contract(run_id="run-1", state=state))

    workflow_run_repo.update_runtime_metadata.assert_awaited_once()
    workflow_run_summary_repo.upsert.assert_awaited_once()
    session_context_snapshot_repo.upsert.assert_awaited_once()

    synced_summary = workflow_run_summary_repo.upsert.await_args.args[0]
    assert synced_summary.run_id == "run-1"
    assert synced_summary.final_answer_summary == "最终总结"
    assert synced_summary.artifacts == ["artifact-1"]

    synced_snapshot = session_context_snapshot_repo.upsert.await_args.args[0]
    assert synced_snapshot.session_id == "session-1"
    assert synced_snapshot.last_run_id == "run-1"
    assert len(synced_snapshot.recent_run_briefs) == 1
    assert synced_snapshot.recent_run_briefs[0]["run_id"] == "run-1"
    assert synced_snapshot.recent_run_briefs[0]["status"] == WorkflowRunStatus.COMPLETED.value
    assert synced_snapshot.artifact_refs == ["artifact-1"]
    assert "artifacts" not in synced_snapshot.recent_run_briefs[0]


def test_langgraph_run_engine_invoke_should_not_sync_episodic_projection_for_waiting_state(monkeypatch) -> None:
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
            "run_id": "run-1",
            "thread_id": "thread-1",
            "graph_metadata": {},
            "pending_interrupt": {},
            "emitted_events": [],
        },
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        thread_id="thread-1",
        status=WorkflowRunStatus.RUNNING,
    )
    session_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=Session(
                id="session-1",
                user_id="user-1",
                current_run_id="run-1",
            )
        )
    )
    workflow_run_repo = SimpleNamespace(
        get_by_id=AsyncMock(return_value=run),
        update_runtime_metadata=AsyncMock(),
    )
    workflow_run_summary_repo = SimpleNamespace(
        list_by_session_id=AsyncMock(return_value=[]),
        upsert=AsyncMock(),
    )
    session_context_snapshot_repo = SimpleNamespace(
        get_by_session_id=AsyncMock(return_value=None),
        upsert=AsyncMock(),
    )

    engine = LangGraphRunEngine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
        uow_factory=lambda: _FakeUoW(
            session_repo=session_repo,
            workflow_run_repo=workflow_run_repo,
            workflow_run_summary_repo=workflow_run_summary_repo,
            session_context_snapshot_repo=session_context_snapshot_repo,
        ),
    )

    async def _collect():
        events = []
        async for event in engine.invoke(Message(message="hello")):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert len(events) == 1
    assert isinstance(events[0], WaitEvent)
    workflow_run_summary_repo.upsert.assert_not_awaited()
    session_context_snapshot_repo.upsert.assert_not_awaited()
