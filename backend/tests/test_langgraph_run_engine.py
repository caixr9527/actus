import asyncio

from pydantic import BaseModel, Field

from app.domain.models import (
    Message,
    TitleEvent,
    PlanEvent,
    StepEvent,
    MessageEvent,
    ToolEvent,
    ToolEventStatus,
    DoneEvent,
    Plan,
    Step,
    ExecutionStatus,
    PlanEventStatus,
    StepEventStatus,
    Session,
    WorkflowRun,
    ToolResult,
)
from app.domain.services.tools import BaseTool
from app.domain.services.runtime.langgraph_state import GRAPH_STATE_CONTRACT_SCHEMA_VERSION
from app.domain.services.prompts import EXECUTION_PROMPT
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph import build_planner_react_langgraph_graph
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.live_events import emit_live_events
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes import execute_step_node
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.tools import (
    build_execution_prompt,
    execute_step_with_prompt,
)
from app.infrastructure.runtime.langgraph_run_engine import LangGraphRunEngine


class _FakeGraph:
    async def ainvoke(self, _state, config=None):
        return {
            "emitted_events": [
                TitleEvent(title="LangGraph 标题"),
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


class _FakeLiveGraph:
    def __init__(self) -> None:
        self.returned = asyncio.Event()

    async def ainvoke(self, _state, config=None):
        # 复用同一个事件实例：验证 RunEngine 在“实时事件 + 最终state”双来源下不会重复输出。
        live_event = MessageEvent(role="assistant", message="实时事件")
        await emit_live_events(live_event)
        await asyncio.sleep(0.05)
        self.returned.set()
        return {
            "emitted_events": [
                live_event,
                DoneEvent(),
            ]
        }


class _FakeGraphWithInjectedCheckpointer(_FakeGraph):
    def __init__(self, checkpointer) -> None:
        self.checkpointer = checkpointer


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


class _LangGraphLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        prompt = messages[0]["content"]
        if "创建一个计划" in prompt:
            return {"role": "assistant", "content": "{\"title\": \"LangGraph\", \"steps\": [\"步骤1\"]}"}
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


class _SingleStepHallucinatedSummaryLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        prompt = messages[0]["content"]
        if "创建一个计划" in prompt:
            return {
                "role": "assistant",
                "content": (
                    "{"
                    "\"message\": \"你好\","
                    "\"goal\": \"回复用户问候\","
                    "\"title\": \"问候任务\","
                    "\"language\": \"zh\","
                    "\"steps\": [{\"id\": \"step-1\", \"description\": \"向用户问好\"}]"
                    "}"
                ),
            }
        if "你正在更新计划" in prompt:
            return {"role": "assistant", "content": "{\"steps\": []}"}
        if "任务已完成，你需要将最终结果交付给用户" in prompt:
            return {
                "role": "assistant",
                "content": (
                    "{"
                    "\"message\": \"# 任务完成总结报告\\n我已完成数据清洗、异常检测、趋势预测并输出报告。"
                    "以下是详细分析：1. 增长率 2. 异常值 3. 下季度预测......\","
                    "\"attachments\": [\"/home/ubuntu/report.md\"]"
                    "}"
                ),
            }
        if "你正在执行任务" in prompt:
            return {
                "role": "assistant",
                "content": "{\"success\": true, \"result\": \"你好！很高兴见到你。\", \"attachments\": []}",
            }
        return {"role": "assistant", "content": "{}"}


class _SingleStepRepeatedSummaryLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        prompt = messages[0]["content"]
        if "创建一个计划" in prompt:
            return {
                "role": "assistant",
                "content": (
                    "{"
                    "\"message\": \"你好\","
                    "\"goal\": \"回复用户问候\","
                    "\"title\": \"问候任务\","
                    "\"language\": \"zh\","
                    "\"steps\": [{\"id\": \"step-1\", \"description\": \"向用户问好\"}]"
                    "}"
                ),
            }
        if "你正在更新计划" in prompt:
            return {"role": "assistant", "content": "{\"steps\": []}"}
        if "任务已完成，你需要将最终结果交付给用户" in prompt:
            return {
                "role": "assistant",
                "content": "{\"message\": \"你好！很高兴见到你。\", \"attachments\": []}",
            }
        if "你正在执行任务" in prompt:
            return {
                "role": "assistant",
                "content": "{\"success\": true, \"result\": \"你好！很高兴见到你。\", \"attachments\": []}",
            }
        return {"role": "assistant", "content": "{}"}


class _InvalidPlanJSONLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        prompt = messages[0]["content"]
        if "创建一个计划" in prompt:
            return {"role": "assistant", "content": "```json\n{invalid}\n```"}
        if "你正在更新计划" in prompt:
            return {"role": "assistant", "content": "{\"steps\": []}"}
        if "任务已完成，你需要将最终结果交付给用户" in prompt:
            return {"role": "assistant", "content": "{\"message\": \"总结完成\", \"attachments\": []}"}
        if "你正在执行任务" in prompt:
            return {
                "role": "assistant",
                "content": "{\"success\": true, \"result\": \"步骤执行完成\", \"attachments\": []}",
            }
        return {"role": "assistant", "content": "{}"}


class _NeverInvokeLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        raise AssertionError("纯问候分支不应调用LLM")


class _SkillStepOutput(BaseModel):
    success: bool = True
    result: str
    attachments: list[str] = Field(default_factory=list)


class _FakeSearchTool(BaseTool):
    name = "search"

    def __init__(self) -> None:
        super().__init__()
        self.invocations: list[tuple[str, dict]] = []

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "search web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    def has_tool(self, tool_name: str) -> bool:
        return tool_name == "search_web"

    async def invoke(self, tool_name: str, **kwargs):
        self.invocations.append((tool_name, kwargs))
        return ToolResult(success=True, data={"query": kwargs.get("query", ""), "results": []})


class _FakeBrowserTool(BaseTool):
    name = "browser"

    def __init__(self) -> None:
        super().__init__()
        self.invocations: list[tuple[str, dict]] = []

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "navigate browser",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                        },
                        "required": ["url"],
                    },
                },
            }
        ]

    def has_tool(self, tool_name: str) -> bool:
        return tool_name == "browser_navigate"

    async def invoke(self, tool_name: str, **kwargs):
        self.invocations.append((tool_name, kwargs))
        return ToolResult(success=True, data={"url": kwargs.get("url", "")})


class _FakeWriteFileTool(BaseTool):
    name = "file"

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "write file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["filepath", "content"],
                    },
                },
            }
        ]

    def has_tool(self, tool_name: str) -> bool:
        return tool_name == "write_file"

    async def invoke(self, tool_name: str, **kwargs):
        filepath = str(kwargs.get("filepath") or "")
        content = str(kwargs.get("content") or "")
        return ToolResult(success=True, data={"filepath": filepath, "bytes_written": len(content)})


class _ToolCallingLLM:
    def __init__(self) -> None:
        self._execute_round = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        prompt = messages[0]["content"]
        if "创建一个计划" in prompt:
            return {
                "role": "assistant",
                "content": (
                    "{"
                    "\"message\": \"已生成计划\","
                    "\"goal\": \"做一次联网检索\","
                    "\"title\": \"检索任务\","
                    "\"language\": \"zh\","
                    "\"steps\": [{\"id\": \"step-1\", \"description\": \"检索AI Agent课程\"}]"
                    "}"
                ),
            }
        if "你正在更新计划" in prompt:
            return {"role": "assistant", "content": "{\"steps\": []}"}
        if "任务已完成，你需要将最终结果交付给用户" in prompt:
            return {"role": "assistant", "content": "{\"message\": \"最终总结\", \"attachments\": []}"}
        if "你正在执行任务" in prompt:
            self._execute_round += 1
            if self._execute_round == 1:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": "{\"query\":\"AI Agent 体系课\"}",
                            },
                        }
                    ],
                }
            return {
                "role": "assistant",
                "content": "{\"success\": true, \"result\": \"已完成检索\", \"attachments\": []}",
            }
        return {"role": "assistant", "content": "{}"}


class _WriteFileToolCallingLLM:
    def __init__(self) -> None:
        self._execute_round = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        prompt = messages[0]["content"]
        if "创建一个计划" in prompt:
            return {
                "role": "assistant",
                "content": (
                    "{"
                    "\"message\": \"已生成计划\","
                    "\"goal\": \"输出推荐文档\","
                    "\"title\": \"推荐任务\","
                    "\"language\": \"zh\","
                    "\"steps\": [{\"id\": \"step-1\", \"description\": \"生成推荐文档\"}]"
                    "}"
                ),
            }
        if "你正在更新计划" in prompt:
            return {"role": "assistant", "content": "{\"steps\": []}"}
        if "任务已完成，你需要将最终结果交付给用户" in prompt:
            # 故意不返回附件，验证 summarize 节点会从步骤产物兜底附件。
            return {"role": "assistant", "content": "{\"message\": \"详细信息已保存为推荐文档。\", \"attachments\": []}"}
        if "你正在执行任务" in prompt:
            self._execute_round += 1
            if self._execute_round == 1:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-write-1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": "{\"filepath\":\"/home/ubuntu/recommendation.md\",\"content\":\"# report\"}",
                            },
                        }
                    ],
                }
            return {
                "role": "assistant",
                "content": "{\"success\": true, \"result\": \"详细信息已保存为推荐文档。\", \"attachments\": []}",
            }
        return {"role": "assistant", "content": "{}"}


class _MultiToolCallsLLM:
    def __init__(self) -> None:
        self._execute_round = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        prompt = messages[0]["content"]
        if "你正在执行任务" in prompt:
            self._execute_round += 1
            if self._execute_round == 1:
                # 同轮返回 browser + search，由运行时仲裁优先执行 search。
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-browser-1",
                            "type": "function",
                            "function": {
                                "name": "browser_navigate",
                                "arguments": "{\"url\":\"https://example.com\"}",
                            },
                        },
                        {
                            "id": "call-search-1",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": "{\"query\":\"AI Agent 课程\"}",
                            },
                        },
                    ],
                }
            return {
                "role": "assistant",
                "content": "{\"success\": true, \"result\": \"已完成检索\", \"attachments\": []}",
            }
        return {"role": "assistant", "content": "{}"}


def test_execute_step_with_prompt_should_emit_tool_events_incrementally_via_callback() -> None:
    llm = _ToolCallingLLM()
    step = Step(
        id="step-1",
        description="检索AI Agent课程",
        status=ExecutionStatus.PENDING,
    )
    observed_statuses: list[ToolEventStatus] = []
    execution_prompt = build_execution_prompt(
        execution_prompt_template=EXECUTION_PROMPT,
        user_message="帮我检索AI Agent课程",
        step_description=step.description,
        language="zh",
        attachments=[],
    )

    async def _invoke():
        result, events = await execute_step_with_prompt(
            llm=llm,
            execution_prompt=execution_prompt,
            step=step,
            runtime_tools=[_FakeSearchTool()],
            max_tool_iterations=3,
            on_tool_event=lambda event: observed_statuses.append(event.status),
        )
        return result, events

    result, events = asyncio.run(_invoke())

    assert result["success"] is True
    assert observed_statuses == [ToolEventStatus.CALLING, ToolEventStatus.CALLED]
    assert [event.status for event in events] == [ToolEventStatus.CALLING, ToolEventStatus.CALLED]


def test_execute_step_with_prompt_should_prefer_search_tool_when_model_returns_multiple_tool_calls() -> None:
    llm = _MultiToolCallsLLM()
    step = Step(
        id="step-1",
        description="推荐3门Java课程",
        status=ExecutionStatus.PENDING,
    )
    search_tool = _FakeSearchTool()
    browser_tool = _FakeBrowserTool()
    execution_prompt = build_execution_prompt(
        execution_prompt_template=EXECUTION_PROMPT,
        user_message="看下慕课网的Java课程推荐3门",
        step_description=step.description,
        language="zh",
        attachments=[],
    )

    async def _invoke():
        result, events = await execute_step_with_prompt(
            llm=llm,
            execution_prompt=execution_prompt,
            step=step,
            runtime_tools=[search_tool, browser_tool],
            max_tool_iterations=3,
        )
        return result, events

    result, events = asyncio.run(_invoke())

    assert result["success"] is True
    assert search_tool.invocations == [("search_web", {"query": "AI Agent 课程"})]
    assert browser_tool.invocations == []
    assert [event.function_name for event in events if event.status == ToolEventStatus.CALLING] == ["search_web"]


def test_execute_step_node_should_emit_started_and_final_events_in_separate_batches(monkeypatch) -> None:
    emitted_batches: list[list[str]] = []

    async def _capture_emit(*events):
        emitted_batches.append([event.type for event in events])

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes.emit_live_events",
        _capture_emit,
    )

    plan = Plan(
        title="检索计划",
        goal="推荐课程",
        language="zh",
        steps=[
            Step(
                id="step-1",
                description="检索AI Agent课程",
                status=ExecutionStatus.PENDING,
            )
        ],
        message="开始执行",
        status=ExecutionStatus.PENDING,
    )
    state = {
        "session_id": "session-1",
        "user_message": "帮我检索AI Agent课程",
        "plan": plan,
        "execution_count": 0,
        "emitted_events": [],
    }

    async def _invoke():
        return await execute_step_node(
            state=state,
            llm=_ToolCallingLLM(),
            runtime_tools=[_FakeSearchTool()],
            max_tool_iterations=3,
        )

    next_state = asyncio.run(_invoke())

    assert isinstance(next_state.get("plan"), Plan)
    assert emitted_batches[0] == ["step"]
    assert any(batch == ["tool"] for batch in emitted_batches[1:-1])
    assert emitted_batches[-1][0] == "step"


def test_langgraph_run_engine_yields_emitted_events(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
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
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
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


def test_langgraph_run_engine_should_stream_live_events_before_graph_completion(monkeypatch) -> None:
    fake_graph = _FakeLiveGraph()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda llm: fake_graph,
    )

    engine = LangGraphRunEngine(session_id="session-1", llm=object())

    async def _collect():
        events = []
        streamed_before_completion = False
        async for event in engine.invoke(Message(message="hello")):
            if not fake_graph.returned.is_set():
                streamed_before_completion = True
            events.append(event)
        return events, streamed_before_completion

    events, streamed_before_completion = asyncio.run(_collect())

    assert streamed_before_completion is True
    assert len(events) == 2
    assert isinstance(events[0], MessageEvent)
    assert isinstance(events[1], DoneEvent)
    assert len({event.id for event in events}) == 2


def test_langgraph_run_engine_should_not_duplicate_event_when_consumer_mutates_event_id(monkeypatch) -> None:
    fake_graph = _FakeLiveGraph()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        lambda llm: fake_graph,
    )

    engine = LangGraphRunEngine(session_id="session-1", llm=object())

    async def _collect():
        events = []
        async for event in engine.invoke(Message(message="hello")):
            # 模拟下游（如 TaskRunner 持久化）改写 event.id 的历史行为。
            if isinstance(event, MessageEvent):
                event.id = "mutated-by-consumer"
            events.append(event)
        return events

    events = asyncio.run(_collect())
    message_events = [event for event in events if isinstance(event, MessageEvent)]

    assert len(events) == 2
    assert len(message_events) == 1
    assert any(isinstance(event, DoneEvent) for event in events)


def test_langgraph_run_engine_should_inject_checkpointer_into_graph_builder(monkeypatch) -> None:
    captured_kwargs = {}
    llm = object()

    def _fake_build_graph(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeGraphWithInjectedCheckpointer(kwargs.get("checkpointer"))

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_langgraph_graph",
        _fake_build_graph,
    )

    engine = LangGraphRunEngine(session_id="session-1", llm=llm)

    assert captured_kwargs["llm"] is llm
    assert captured_kwargs["checkpointer"] is not None
    assert getattr(engine._graph, "checkpointer", None) is captured_kwargs["checkpointer"]


def test_planner_react_langgraph_graph_should_execute_async_nodes_without_coroutine_error() -> None:
    graph = build_planner_react_langgraph_graph(llm=_LangGraphLLM())

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
    graph = build_planner_react_langgraph_graph(llm=_V1GraphLLM())

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


def test_planner_react_v1_graph_should_build_readable_fallback_title_when_plan_json_invalid() -> None:
    graph = build_planner_react_langgraph_graph(llm=_InvalidPlanJSONLLM())
    user_message = "请帮我分析线上故障日志并给出完整修复方案以及回归验证步骤"

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": user_message,
                "emitted_events": [],
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())
    events = state.get("emitted_events", [])
    title_events = [event for event in events if isinstance(event, TitleEvent)]

    assert len(title_events) >= 1
    assert title_events[0].title != "LangGraph 任务"
    assert title_events[0].title.startswith(user_message[:6])
    assert len(title_events[0].title) <= 27
    assert any(isinstance(event, DoneEvent) for event in events)


def test_planner_react_v1_graph_should_execute_step_via_skill_runtime(monkeypatch) -> None:
    skill_calls: list[tuple[str, dict]] = []

    class _SpySkillRuntime:
        async def execute_skill(self, *, skill_id: str, payload: dict, version: str | None = None):
            skill_calls.append((skill_id, payload))
            return _SkillStepOutput(
                success=True,
                result=f"skill执行:{payload.get('step_description', '')}",
                attachments=[],
            )

    class _SpySkillRegistry:
        def create_runtime(self, llm):
            return _SpySkillRuntime()

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.build_default_skill_graph_registry",
        lambda: _SpySkillRegistry(),
    )

    graph = build_planner_react_langgraph_graph(llm=_V1GraphLLM())

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
    message_events = [event for event in events if isinstance(event, MessageEvent)]

    assert len(skill_calls) >= 1
    assert all(call[0] == "planner_react.execute_step" for call in skill_calls)
    assert any(event.message.startswith("skill执行:") for event in message_events)


def test_planner_react_v1_graph_should_fallback_to_step_result_when_single_step_summary_is_irrelevant() -> None:
    graph = build_planner_react_langgraph_graph(llm=_SingleStepHallucinatedSummaryLLM())

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": "请向我问好",
                "emitted_events": [],
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())
    events = state.get("emitted_events", [])
    assistant_messages = [event for event in events if isinstance(event, MessageEvent) and event.role == "assistant"]

    # summarize 产出与步骤结果无关时，应回退到步骤结果。
    final_greeting_messages = [event.message for event in assistant_messages if event.message == "你好！很高兴见到你。"]
    assert len(final_greeting_messages) == 1
    assert all("数据清洗" not in event.message for event in assistant_messages)


def test_planner_react_v1_graph_should_not_emit_duplicate_message_when_summary_equals_last_step_result() -> None:
    graph = build_planner_react_langgraph_graph(llm=_SingleStepRepeatedSummaryLLM())

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": "请向我问好",
                "emitted_events": [],
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())
    events = state.get("emitted_events", [])
    message_events = [event for event in events if isinstance(event, MessageEvent) and event.role == "assistant"]

    greeting_message = "你好！很高兴见到你。"
    # summarize 与上一步结果一致且无附件时，应避免重复追加 assistant message。
    assert len([event for event in message_events if event.message == greeting_message]) == 1
    assert state.get("final_message") == greeting_message
    assert any(isinstance(event, DoneEvent) for event in events)


def test_planner_react_v1_graph_should_fallback_attachments_from_write_file_tool_events() -> None:
    graph = build_planner_react_langgraph_graph(
        llm=_WriteFileToolCallingLLM(),
        runtime_tools=[_FakeWriteFileTool()],
    )

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": "请推荐3门Java课程并保存成文档",
                "emitted_events": [],
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())
    events = state.get("emitted_events", [])
    assistant_messages = [event for event in events if isinstance(event, MessageEvent) and event.role == "assistant"]
    completed_steps = [
        event.step for event in events
        if isinstance(event, StepEvent) and event.status == StepEventStatus.COMPLETED
    ]

    assert len(completed_steps) >= 1
    assert "/home/ubuntu/recommendation.md" in completed_steps[-1].attachments
    duplicated_summary_messages = [
        message
        for message in assistant_messages
        if message.message == "详细信息已保存为推荐文档。"
    ]
    # summarize 与步骤消息在文本和附件一致时，不应重复输出两条最终内容。
    assert len(duplicated_summary_messages) == 1
    assert any(
        any(str(getattr(attachment, "filepath", "")) == "/home/ubuntu/recommendation.md" for attachment in message.attachments)
        for message in assistant_messages
    )
    assert any(isinstance(event, DoneEvent) for event in events)


def test_planner_react_v1_graph_should_emit_tool_events_when_runtime_tools_available() -> None:
    graph = build_planner_react_langgraph_graph(
        llm=_ToolCallingLLM(),
        runtime_tools=[_FakeSearchTool()],
    )

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": "帮我检索AI Agent课程",
                "emitted_events": [],
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())
    events = state.get("emitted_events", [])
    tool_events = [event for event in events if isinstance(event, ToolEvent)]

    assert any(event.status == ToolEventStatus.CALLING for event in tool_events)
    assert any(event.status == ToolEventStatus.CALLED for event in tool_events)
    assert any(event.function_name == "search_web" for event in tool_events)
    assert any(isinstance(event, DoneEvent) for event in events)


def test_planner_react_v1_graph_should_skip_plan_and_step_for_simple_greeting() -> None:
    graph = build_planner_react_langgraph_graph(llm=_NeverInvokeLLM())

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": "你好",
                "emitted_events": [],
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())
    events = state.get("emitted_events", [])

    assistant_messages = [event for event in events if isinstance(event, MessageEvent) and event.role == "assistant"]
    plan_events = [event for event in events if isinstance(event, PlanEvent)]
    step_events = [event for event in events if isinstance(event, StepEvent)]

    assert len(assistant_messages) == 1
    assert assistant_messages[0].message == "你好！我是助手，很高兴为您服务。"
    assert len(plan_events) == 0
    assert len(step_events) == 0
    assert any(isinstance(event, DoneEvent) for event in events)
