import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.types import Command

from app.domain.models import (
    File,
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
    Workspace,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.engine.run_engine import LangGraphRunEngine


class _FakeGraph:
    def __init__(self, result=None, checkpoint_state=None) -> None:
        self._result = result or {"emitted_events": []}
        self._checkpoint_state = checkpoint_state
        self.calls = []
        self.updated_state = None
        self.updated_config = None

    async def ainvoke(self, state, config=None):
        self.calls.append((state, config))
        return self._result

    async def aget_state(self, config):
        if self._checkpoint_state is None:
            return None
        return SimpleNamespace(values=self._checkpoint_state)

    async def aupdate_state(self, config, values):
        self.updated_config = config
        self.updated_state = values
        next_config = {
            "configurable": {
                **dict(config.get("configurable") or {}),
                "checkpoint_id": "updated-checkpoint-id",
            }
        }
        self._checkpoint_state = values
        return next_config


class _FakeUoW:
    def __init__(
            self,
            *,
            session_repo,
            workflow_run_repo,
            workflow_run_summary_repo,
            session_context_snapshot_repo,
            workspace_repo=None,
    ):
        self.session = session_repo
        self.workflow_run = workflow_run_repo
        self.workflow_run_summary = workflow_run_summary_repo
        self.session_context_snapshot = session_context_snapshot_repo
        self.workspace = workspace_repo or SimpleNamespace(
            get_by_id=AsyncMock(return_value=None),
            get_by_session_id=AsyncMock(return_value=None),
        )

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


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


def _build_run_engine(**kwargs) -> LangGraphRunEngine:
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return LangGraphRunEngine(**kwargs)


def test_langgraph_run_engine_should_inject_checkpointer_into_graph_builder(monkeypatch) -> None:
    captured = {}
    checkpointer = object()

    def _fake_build_graph(**kwargs):
        captured.update(kwargs)
        return _FakeGraph()

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        _fake_build_graph,
    )

    _build_run_engine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
        checkpointer=checkpointer,
    )

    assert captured["checkpointer"] is checkpointer
    assert captured["memory_consolidation_service"] is not None


def test_langgraph_run_engine_should_build_ollama_memory_consolidation_service_when_ollama_model_configured(monkeypatch) -> None:
    captured = {}
    captured_factory_config = {}
    fake_provider = object()

    def _fake_build_graph(**kwargs):
        captured.update(kwargs)
        return _FakeGraph()

    class _FakeFactory:
        def create_memory_consolidation_provider(self, **kwargs):
            captured_factory_config.update(kwargs)
            return fake_provider

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        _fake_build_graph,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.get_settings",
        lambda: SimpleNamespace(
            ollama=SimpleNamespace(
                base_url="http://ollama.test",
                model="qwen2.5:3b",
                timeout_seconds=2.5,
            ),
        ),
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.OllamaLLMFactory",
        _FakeFactory,
    )
    _build_run_engine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
    )

    assert captured["memory_consolidation_service"] is not None
    assert captured_factory_config == {
        "base_url": "http://ollama.test",
        "model": "qwen2.5:3b",
        "timeout_seconds": 2.5,
    }


def test_langgraph_run_engine_should_require_complete_stage_llms() -> None:
    with pytest.raises(ValueError, match="缺少必要阶段模型配置"):
        LangGraphRunEngine(
            session_id="session-1",
            stage_llms={"executor": object()},
            runtime_context_service=_TEST_RUNTIME_CONTEXT_SERVICE,
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
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    engine = _build_run_engine(
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


def test_langgraph_run_engine_invoke_should_not_crash_when_graph_finishes_without_wait(monkeypatch) -> None:
    fake_graph = _FakeGraph(
        result={
            "session_id": "session-1",
            "thread_id": "thread-1",
            "emitted_events": [],
        }
    )

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    engine = _build_run_engine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
    )

    async def _collect():
        events = []
        async for event in engine.invoke(Message(message="hello")):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events == []


class _InputPartsSessionRepo:
    def __init__(self, files_by_path: dict[str, File]) -> None:
        self._files_by_path = {
            path: file.model_copy(deep=True)
            for path, file in files_by_path.items()
        }

    async def get_file_by_path(self, session_id: str, filepath: str):
        file = self._files_by_path.get(filepath)
        return file.model_copy(deep=True) if file is not None else None


class _InputPartsFileStorage:
    def __init__(self, payloads_by_id: dict[str, bytes]) -> None:
        self._payloads_by_id = dict(payloads_by_id)

    async def download_file(self, file_id: str, user_id=None):
        return io.BytesIO(self._payloads_by_id[file_id]), File(id=file_id)

    def get_file_url(self, file: File) -> str:
        return f"https://cdn.example.com/{file.id}"


def test_langgraph_run_engine_build_input_parts_should_distinguish_same_name_attachments() -> None:
    engine = _build_run_engine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
        file_storage=_InputPartsFileStorage(
            {
                "file-1": b"content-a",
                "file-2": b"content-b",
            }
        ),
    )
    uow = SimpleNamespace(
        session=_InputPartsSessionRepo(
            {
                "/home/ubuntu/upload/file-1/same.txt": File(
                    id="file-1",
                    filename="same.txt",
                    filepath="/home/ubuntu/upload/file-1/same.txt",
                    mime_type="text/plain",
                ),
                "/home/ubuntu/upload/file-2/same.txt": File(
                    id="file-2",
                    filename="same.txt",
                    filepath="/home/ubuntu/upload/file-2/same.txt",
                    mime_type="text/plain",
                ),
            }
        )
    )

    parts = asyncio.run(
        engine._build_input_parts(
            Message(
                message="读取两个同名附件",
                attachments=[
                    "/home/ubuntu/upload/file-1/same.txt",
                    "/home/ubuntu/upload/file-2/same.txt",
                ],
            ),
            uow=uow,
        )
    )

    assert [part["sandbox_filepath"] for part in parts] == [
        "/home/ubuntu/upload/file-1/same.txt",
        "/home/ubuntu/upload/file-2/same.txt",
    ]
    assert [part["file_url"] for part in parts] == [
        "https://cdn.example.com/file-1",
        "https://cdn.example.com/file-2",
    ]
    assert parts[0]["base64_payload"] != parts[1]["base64_payload"]


def test_langgraph_run_engine_resume_should_use_command_resume(monkeypatch) -> None:
    fake_graph = _FakeGraph()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )
    engine = _build_run_engine(session_id="session-1", stage_llms=_build_stage_llms())
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
    assert captured["invoke_config"] == {
        "configurable": {
            "thread_id": "session-1",
            "checkpoint_id": "updated-checkpoint-id",
        }
    }
    assert captured["checkpoint_invoke_config"] == {"configurable": {"thread_id": "session-1"}}
    assert fake_graph.updated_state == {"session_id": "session-1", "pending_interrupt": {}}


def test_langgraph_run_engine_resume_should_write_normalized_checkpoint_before_resume(monkeypatch) -> None:
    fake_graph = _FakeGraph(
        checkpoint_state={
            "session_id": "session-1",
            "plan": {
                "title": "测试计划",
                "steps": [
                    {
                        "id": "step-1",
                        "title": "执行步骤",
                        "description": "执行步骤",
                        "objective_key": "objective-step-1",
                        "success_criteria": ["执行步骤完成"],
                        "status": ExecutionStatus.PENDING.value,
                    }
                ],
            },
            "step_states": [
                {
                    "step_id": "stale-step",
                    "title": "旧步骤",
                    "description": "旧步骤",
                    "status": ExecutionStatus.COMPLETED.value,
                }
            ],
            "current_step_id": "stale-step",
            "pending_interrupt": {},
        }
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )
    engine = _build_run_engine(session_id="session-1", stage_llms=_build_stage_llms())
    captured = {}

    async def _fake_run_graph(*, graph_input, invoke_config, run_id, fallback_state):
        captured["invoke_config"] = invoke_config
        captured["fallback_state"] = fallback_state
        if False:
            yield None

    monkeypatch.setattr(engine, "_run_graph", _fake_run_graph)
    async def _collect():
        events = []
        async for event in engine.resume({"approved": True}):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events == []
    assert fake_graph.updated_state["plan"].steps[0].id == "step-1"
    assert fake_graph.updated_state["step_states"][0]["step_id"] == "step-1"
    assert fake_graph.updated_state["current_step_id"] == "step-1"
    assert captured["invoke_config"]["configurable"]["checkpoint_id"] == "updated-checkpoint-id"


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
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    engine = _build_run_engine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
    )

    inspection = asyncio.run(engine.inspect_resume_checkpoint())

    assert inspection.run_id is None
    assert inspection.has_checkpoint is True
    assert inspection.pending_interrupt == {}
    assert inspection.is_resumable is False


def test_langgraph_run_engine_should_normalize_checkpoint_state_on_load(monkeypatch) -> None:
    fake_graph = _FakeGraph(
        checkpoint_state={
            "session_id": "session-1",
            "user_id": "user-1",
            "run_id": "run-1",
            "thread_id": "thread-1",
            "user_message": "继续执行",
            "message_window": [
                {
                    "role": "assistant",
                    "message": "已有中间结果",
                    "attachment_paths": ["artifact-id-1", "/tmp/message.md"],
                }
            ],
            "selected_artifacts": ["artifact-id-1", "/tmp/final.md"],
            "plan": {
                "title": "测试计划",
                "steps": [
                    {
                        "id": "step-1",
                        "title": "整理结果",
                        "description": "整理结果",
                        "status": "completed",
                        "outcome": {
                            "done": True,
                            "summary": "已完成",
                            "produced_artifacts": ["artifact-id-2", "/tmp/plan.md"],
                        },
                    }
                ],
            },
            "last_executed_step": {
                "id": "step-1",
                "title": "整理结果",
                "description": "整理结果",
                "status": "completed",
                "outcome": {
                    "done": True,
                    "summary": "已完成",
                    "produced_artifacts": ["artifact-id-3", "/tmp/last-step.md"],
                },
            },
            "step_states": [
                {
                    "step_id": "step-1",
                    "title": "整理结果",
                    "description": "整理结果",
                    "status": "completed",
                    "outcome": {
                        "summary": "已完成",
                        "produced_artifacts": ["artifact-id-4", "/tmp/step-state.md"],
                    },
                }
            ],
            "pending_interrupt": {},
            "graph_metadata": {},
            "emitted_events": [],
        }
    )

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    engine = _build_run_engine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
    )

    state = asyncio.run(
        engine._load_checkpoint_state(
            invoke_config={"configurable": {"thread_id": "session-1"}},
        )
    )

    assert state is not None
    assert state["message_window"][0]["attachment_paths"] == ["/tmp/message.md"]
    assert state["selected_artifacts"] == ["/tmp/final.md"]
    assert state["plan"].steps[0].outcome.produced_artifacts == ["/tmp/plan.md"]
    assert state["last_executed_step"].outcome.produced_artifacts == ["/tmp/last-step.md"]
    assert state["step_states"][0]["outcome"]["produced_artifacts"] == ["/tmp/plan.md"]
    assert state["current_step_id"] is None


def test_langgraph_run_engine_should_build_initial_state_with_session_snapshot_and_completed_summaries(monkeypatch) -> None:
    fake_graph = _FakeGraph()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    session_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=Session(
                id="session-1",
                user_id="user-1",
                workspace_id="workspace-1",
                current_run_id="run-1",
            )
        )
    )
    workspace_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=Workspace(
                id="workspace-1",
                session_id="session-1",
                current_run_id="run-1",
            )
        ),
        get_by_session_id=AsyncMock(return_value=None),
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
                        final_answer_text="这是上一轮真正交付给用户的完整正文。",
                        open_questions=["还需最终确认"],
                        artifacts=["/tmp/artifact-1.md"],
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
                        artifacts=["/tmp/artifact-failed.md"],
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
                artifact_paths=["/tmp/snapshot-artifact.md"],
            )
        )
    )

    engine = _build_run_engine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
        uow_factory=lambda: _FakeUoW(
            session_repo=session_repo,
            workflow_run_repo=workflow_run_repo,
            workflow_run_summary_repo=workflow_run_summary_repo,
            session_context_snapshot_repo=session_context_snapshot_repo,
            workspace_repo=workspace_repo,
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
    assert state["workspace_id"] == "workspace-1"
    assert workflow_run_summary_repo.list_by_session_id.await_args_list[0].kwargs["statuses"] == [WorkflowRunStatus.COMPLETED]
    assert workflow_run_summary_repo.list_by_session_id.await_args_list[1].kwargs["statuses"] == [
        WorkflowRunStatus.FAILED,
        WorkflowRunStatus.CANCELLED,
    ]
    assert state["recent_run_briefs"][0]["run_id"] == "run-completed"
    assert state["recent_run_briefs"][0]["final_answer_text_excerpt"] == "这是上一轮真正交付给用户的完整正文。"
    assert state["recent_attempt_briefs"][0]["run_id"] == "run-failed"
    assert state["session_open_questions"] == ["还需最终确认", "需要补充网络权限", "快照问题"]
    assert state["session_blockers"] == ["外部接口失败"]
    assert state["selected_artifacts"] == []
    assert state["historical_artifact_paths"] == [
        "/tmp/snapshot-artifact.md",
        "/tmp/artifact-1.md",
        "/tmp/artifact-failed.md",
    ]


def test_langgraph_run_engine_should_sync_run_summary_and_session_snapshot(monkeypatch) -> None:
    fake_graph = _FakeGraph()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
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
        final_answer_text="这是最终交付正文。",
        artifacts=["/tmp/final-output.md"],
    )
    workflow_run_summary_repo = SimpleNamespace(
        upsert=AsyncMock(return_value=first_summary),
        list_by_session_id=AsyncMock(return_value=[first_summary]),
    )
    session_context_snapshot_repo = SimpleNamespace(
        upsert=AsyncMock(),
    )

    engine = _build_run_engine(
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
                        "produced_artifacts": ["/tmp/final-output.md"],
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
            "final_delivery_payload": {
                "text": "这是最终交付正文。",
                "sections": [],
                "source_refs": [],
            },
        },
        "selected_artifacts": ["/tmp/final-output.md"],
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
        "historical_artifact_paths": [],
    }

    asyncio.run(engine._sync_graph_state_contract(run_id="run-1", state=state))

    workflow_run_repo.update_runtime_metadata.assert_awaited_once()
    workflow_run_summary_repo.upsert.assert_awaited_once()
    session_context_snapshot_repo.upsert.assert_awaited_once()

    synced_summary = workflow_run_summary_repo.upsert.await_args.args[0]
    assert synced_summary.run_id == "run-1"
    assert synced_summary.final_answer_summary == "最终总结"
    assert synced_summary.final_answer_text == "这是最终交付正文。"
    assert synced_summary.artifacts == ["/tmp/final-output.md"]

    synced_snapshot = session_context_snapshot_repo.upsert.await_args.args[0]
    assert synced_snapshot.session_id == "session-1"
    assert synced_snapshot.last_run_id == "run-1"
    assert synced_snapshot.summary_text == "会话摘要"
    assert len(synced_snapshot.recent_run_briefs) == 1
    assert synced_snapshot.recent_run_briefs[0]["run_id"] == "run-1"
    assert synced_snapshot.recent_run_briefs[0]["status"] == WorkflowRunStatus.COMPLETED.value
    assert synced_snapshot.recent_run_briefs[0]["final_answer_text_excerpt"] == "这是最终交付正文。"
    assert synced_snapshot.artifact_paths == ["/tmp/final-output.md"]
    assert "artifacts" not in synced_snapshot.recent_run_briefs[0]


def test_langgraph_run_engine_session_snapshot_should_prefer_conversation_summary() -> None:
    long_delivery_text = "这是很长的最终交付正文，包含大量细节。" * 30
    summary = WorkflowRunSummary(
        run_id="run-1",
        session_id="session-1",
        status=WorkflowRunStatus.COMPLETED,
        title="本轮运行",
        final_answer_summary="轻量最终摘要",
        final_answer_text=long_delivery_text,
    )

    snapshot = LangGraphRunEngine._build_session_context_snapshot_projection(
        session_id="session-1",
        user_id="user-1",
        summaries=[summary],
        conversation_summary="真实会话主题摘要",
    )

    assert snapshot.summary_text == "真实会话主题摘要"
    assert snapshot.recent_run_briefs[0]["final_answer_summary"] == "轻量最终摘要"
    assert snapshot.recent_run_briefs[0]["final_answer_text_excerpt"] == long_delivery_text[:200]


def test_langgraph_run_engine_session_snapshot_should_fallback_to_recent_run_briefs_without_conversation_summary() -> None:
    summary = WorkflowRunSummary(
        run_id="run-1",
        session_id="session-1",
        status=WorkflowRunStatus.COMPLETED,
        title="本轮运行",
        final_answer_summary="轻量最终摘要",
        final_answer_text="这是最终交付正文。",
    )

    snapshot = LangGraphRunEngine._build_session_context_snapshot_projection(
        session_id="session-1",
        user_id="user-1",
        summaries=[summary],
        conversation_summary="",
    )

    assert snapshot.summary_text == "轻量最终摘要"


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
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
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

    engine = _build_run_engine(
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


def test_langgraph_run_engine_should_not_fallback_to_session_current_run_id_when_workspace_run_missing(monkeypatch) -> None:
    fake_graph = _FakeGraph()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: fake_graph,
    )

    session_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=Session(
                id="session-1",
                user_id="user-1",
                workspace_id="workspace-1",
                current_run_id="run-legacy",
            )
        )
    )
    workspace_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=Workspace(
                id="workspace-1",
                session_id="session-1",
                current_run_id=None,
            )
        ),
        get_by_session_id=AsyncMock(return_value=None),
    )
    workflow_run_repo = SimpleNamespace(get_by_id=AsyncMock(return_value=None))
    workflow_run_summary_repo = SimpleNamespace(list_by_session_id=AsyncMock(side_effect=[[], []]))
    session_context_snapshot_repo = SimpleNamespace(get_by_session_id=AsyncMock(return_value=None))

    engine = _build_run_engine(
        session_id="session-1",
        stage_llms=_build_stage_llms(),
        uow_factory=lambda: _FakeUoW(
            session_repo=session_repo,
            workflow_run_repo=workflow_run_repo,
            workflow_run_summary_repo=workflow_run_summary_repo,
            session_context_snapshot_repo=session_context_snapshot_repo,
            workspace_repo=workspace_repo,
        ),
    )

    state = asyncio.run(
        engine._build_graph_input_state(
            message=Message(message="hello"),
            run_id=None,
            invoke_config={"configurable": {"thread_id": "session-1"}},
        )
    )

    assert state["run_id"] is None
    workflow_run_repo.get_by_id.assert_not_awaited()
