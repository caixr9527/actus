import asyncio

from app.domain.models import (
    Message,
    TitleEvent,
    PlanEvent,
    StepEvent,
    MessageEvent,
    DoneEvent,
    Plan,
    Step,
    ExecutionStatus,
    PlanEventStatus,
    StepEventStatus,
    Session,
    WorkflowRun,
)
from app.domain.services.runtime.langgraph_state import GRAPH_STATE_CONTRACT_SCHEMA_VERSION
from app.infrastructure.runtime.langgraph_graphs.planner_react_poc import build_planner_react_poc_graph
from app.infrastructure.runtime.langgraph_run_engine import LangGraphRunEngine


class _FakeGraph:
    async def ainvoke(self, _state, config=None):
        return {
            "emitted_events": [
                TitleEvent(title="POC 标题"),
                PlanEvent(
                    plan=Plan(
                        title="x",
                        goal="",
                        language="zh",
                        steps=[],
                        message="",
                        status=ExecutionStatus.PENDING,
                    ),
                    status=PlanEventStatus.CREATED,
                ),
                StepEvent(
                    step=Step(
                        id="s1",
                        description="step",
                        status=ExecutionStatus.COMPLETED,
                        success=True,
                    ),
                    status=StepEventStatus.COMPLETED,
                ),
                MessageEvent(role="assistant", message="完成"),
                DoneEvent(),
            ]
        }


class _CheckpointTuple:
    def __init__(self, config):
        self.config = config


class _FakeCheckpointer:
    def __init__(self, checkpoint_id: str) -> None:
        self._checkpoint_id = checkpoint_id

    async def aget_tuple(self, _config):
        return _CheckpointTuple(
            {
                "configurable": {
                    "thread_id": "thread-1",
                    "checkpoint_ns": "",
                    "checkpoint_id": self._checkpoint_id,
                }
            }
        )


class _FakeGraphWithCheckpoint(_FakeGraph):
    def __init__(self) -> None:
        self.checkpointer = _FakeCheckpointer(checkpoint_id="cp-new")
        self.last_config = None
        self.last_state = None

    async def ainvoke(self, _state, config=None):
        self.last_state = _state
        self.last_config = config
        return await super().ainvoke(_state, config=config)


class _SessionRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_by_id(self, session_id: str, user_id=None):
        if session_id != self._session.id:
            return None
        return self._session


class _WorkflowRunRepo:
    def __init__(self, run: WorkflowRun) -> None:
        self._run = run
        self.updated_runtime_metadata: list[tuple[str, dict, str | None]] = []

    async def get_by_id(self, run_id: str):
        if run_id != self._run.id:
            return None
        return self._run

    async def update_checkpoint_ref(self, run_id: str, checkpoint_namespace: str, checkpoint_id: str) -> None:
        if run_id != self._run.id:
            raise ValueError("run not found")
        self._run.checkpoint_namespace = checkpoint_namespace
        self._run.checkpoint_id = checkpoint_id

    async def update_runtime_metadata(
            self,
            run_id: str,
            runtime_metadata: dict,
            current_step_id: str | None,
    ) -> None:
        if run_id != self._run.id:
            raise ValueError("run not found")
        merged_metadata = dict(self._run.runtime_metadata or {})
        merged_metadata.update(runtime_metadata or {})
        self._run.runtime_metadata = merged_metadata
        self._run.current_step_id = current_step_id
        self.updated_runtime_metadata.append((run_id, runtime_metadata, current_step_id))


class _UoW:
    def __init__(self, session_repo: _SessionRepo, workflow_run_repo: _WorkflowRunRepo) -> None:
        self.session = session_repo
        self.workflow_run = workflow_run_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _POCLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        prompt = messages[0]["content"]
        if "任务拆成最小POC计划" in prompt:
            return {"role": "assistant", "content": "{\"title\": \"POC\", \"steps\": [\"步骤1\"]}"}
        return {"role": "assistant", "content": "{\"success\": true, \"result\": \"完成\"}"}


class _V1GraphLLM:
    def __init__(self) -> None:
        self._execute_count = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        prompt = messages[0]["content"]
        if "创建一个计划" in prompt:
            return {
                "role": "assistant",
                "content": (
                    "{"
                    "\"message\": \"已生成计划\","
                    "\"goal\": \"完成两步任务\","
                    "\"title\": \"V1 计划\","
                    "\"language\": \"zh\","
                    "\"steps\": ["
                    "{\"id\": \"step-1\", \"description\": \"执行第一步\"},"
                    "{\"id\": \"step-2\", \"description\": \"执行第二步\"}"
                    "]"
                    "}"
                ),
            }
        if "你正在更新计划" in prompt:
            return {
                "role": "assistant",
                "content": "{\"steps\": [{\"id\": \"step-2\", \"description\": \"执行第二步\"}]}",
            }
        if "任务已完成，你需要将最终结果交付给用户" in prompt:
            return {
                "role": "assistant",
                "content": "{\"message\": \"最终总结\", \"attachments\": [\"/home/ubuntu/report.md\"]}",
            }
        if "你正在执行任务" in prompt:
            self._execute_count += 1
            return {
                "role": "assistant",
                "content": (
                    "{"
                    "\"success\": true,"
                    f"\"result\": \"步骤{self._execute_count}完成\","
                    "\"attachments\": []"
                    "}"
                ),
            }
        return {"role": "assistant", "content": "{}"}


def test_langgraph_run_engine_yields_emitted_events(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_poc_graph",
        lambda llm: _FakeGraph(),
    )

    engine = LangGraphRunEngine(session_id="session-1", llm=object())

    async def _collect():
        return [event async for event in engine.invoke(Message(message="hello"))]

    events = asyncio.run(_collect())

    assert len(events) == 5
    assert isinstance(events[0], TitleEvent)
    assert isinstance(events[1], PlanEvent)
    assert isinstance(events[2], StepEvent)
    assert isinstance(events[3], MessageEvent)
    assert isinstance(events[4], DoneEvent)


def test_langgraph_run_engine_should_sync_checkpoint_ref_to_workflow_run(monkeypatch) -> None:
    fake_graph = _FakeGraphWithCheckpoint()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_poc_graph",
        lambda llm: fake_graph,
    )

    session = Session(id="session-1", current_run_id="run-1")
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-old",
    )
    uow_factory = lambda: _UoW(_SessionRepo(session), _WorkflowRunRepo(run))
    engine = LangGraphRunEngine(session_id="session-1", llm=object(), uow_factory=uow_factory)

    async def _collect():
        return [event async for event in engine.invoke(Message(message="hello"))]

    events = asyncio.run(_collect())

    assert len(events) == 5
    assert fake_graph.last_state is not None
    assert fake_graph.last_state["schema_version"] == GRAPH_STATE_CONTRACT_SCHEMA_VERSION
    assert fake_graph.last_state["run_id"] == "run-1"
    assert fake_graph.last_state["thread_id"] == "thread-1"
    assert fake_graph.last_config == {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": "cp-old",
        }
    }
    assert run.checkpoint_namespace == ""
    assert run.checkpoint_id == "cp-new"
    assert run.runtime_metadata.get("graph_state_contract") is not None
    assert run.runtime_metadata["graph_state_contract"]["schema_version"] == GRAPH_STATE_CONTRACT_SCHEMA_VERSION


def test_planner_react_poc_graph_should_execute_async_nodes_without_coroutine_error() -> None:
    graph = build_planner_react_poc_graph(llm=_POCLLM())

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": "帮我做个计划",
                "emitted_events": [],
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())
    events = state.get("emitted_events", [])

    assert any(isinstance(event, TitleEvent) for event in events)
    assert any(isinstance(event, PlanEvent) for event in events)
    assert any(isinstance(event, StepEvent) for event in events)
    assert any(isinstance(event, DoneEvent) for event in events)


def test_planner_react_v1_graph_should_cover_execute_replan_summarize_path() -> None:
    graph = build_planner_react_poc_graph(llm=_V1GraphLLM())

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": "请完成一个两步任务",
                "emitted_events": [],
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())
    events = state.get("emitted_events", [])

    plan_events = [event for event in events if isinstance(event, PlanEvent)]
    step_events = [event for event in events if isinstance(event, StepEvent)]
    message_events = [event for event in events if isinstance(event, MessageEvent)]

    assert any(event.status == PlanEventStatus.CREATED for event in plan_events)
    assert any(event.status == PlanEventStatus.UPDATED for event in plan_events)
    assert any(event.status == PlanEventStatus.COMPLETED for event in plan_events)
    assert len([event for event in step_events if event.status == StepEventStatus.COMPLETED]) >= 2
    assert any(event.message == "最终总结" for event in message_events)
    assert any(isinstance(event, DoneEvent) for event in events)
