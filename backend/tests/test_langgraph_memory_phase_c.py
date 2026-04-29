import asyncio
import json
from datetime import datetime

from app.domain.models import (
    ExecutionStatus,
    LongTermMemory,
    LongTermMemorySearchMode,
    LongTermMemorySearchQuery,
    Plan,
    Step,
    StepOutcome,
    StepTaskModeHint,
    MessageEvent,
    TextStreamDeltaEvent,
    TextStreamEndEvent,
    TextStreamStartEvent,
    ToolResult,
    ToolEvent,
    ToolEventStatus,
    Workspace,
    WorkspaceArtifact,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.entry import EntryCompiler
from app.domain.services.workspace_runtime import WorkspaceEnvironmentSnapshot
from app.domain.services.tools import BaseTool, MessageTool
from app.domain.services.tools.base import tool
from app.domain.services.runtime.langgraph_state import GraphStateContractMapper
from app.infrastructure.runtime.langgraph.graphs import bind_live_event_sink, unbind_live_event_sink
from app.infrastructure.runtime.langgraph.graphs.planner_react.graph import (
    build_planner_react_langgraph_graph as _build_planner_react_langgraph_graph,
)
from app.domain.services.runtime.contracts.step_evidence_contracts import STEP_DRAFT_FACT_PREFIX
from app.domain.services.runtime.contracts.data_access_contract import (
    DataClassificationResult,
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes import (
    consolidate_memory_node,
    direct_wait_node,
    execute_step_node as _execute_step_node,
    guard_step_reuse_node,
    recall_memory_context_node,
    replan_node as _replan_node,
    summarize_node as _summarize_node,
    wait_for_human_node,
)


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


class _FakeDataRetentionPolicyService:
    def __init__(
            self,
            *,
            tenant_id: str | None = None,
            trust_level: DataTrustLevel = DataTrustLevel.SYSTEM_GENERATED,
            privacy_level: PrivacyLevel = PrivacyLevel.SENSITIVE,
            retention_policy: RetentionPolicyKind = RetentionPolicyKind.USER_MEMORY,
    ) -> None:
        self.tenant_id = tenant_id
        self.trust_level = trust_level
        self.privacy_level = privacy_level
        self.retention_policy = retention_policy
        self.calls = []

    def classify_data(
            self,
            *,
            tenant_id: str,
            origin: DataOrigin,
            requested_privacy_level: PrivacyLevel | None = None,
            retention_policy: RetentionPolicyKind | None = None,
    ) -> DataClassificationResult:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "origin": origin,
                "requested_privacy_level": requested_privacy_level,
                "retention_policy": retention_policy,
            }
        )
        return DataClassificationResult(
            tenant_id=self.tenant_id or tenant_id,
            origin=origin,
            trust_level=self.trust_level,
            privacy_level=self.privacy_level,
            retention_policy=self.retention_policy,
        )


_FAKE_RETENTION_POLICY_SERVICE = _FakeDataRetentionPolicyService()


async def _consolidate_memory_node(*args, **kwargs):
    kwargs.setdefault("data_retention_policy_service", _FAKE_RETENTION_POLICY_SERVICE)
    return await consolidate_memory_node(*args, **kwargs)


def _entry_contract_control(user_message: str):
    contract = EntryCompiler().compile_state(
        {
            "user_message": user_message,
            "input_parts": [],
            "message_window": [],
            "recent_run_briefs": [],
            "conversation_summary": "",
        }
    )
    return {"entry_contract": contract.model_dump(mode="json")}


def _planned_entry_contract_control(user_message: str):
    return _entry_contract_control(f"请规划并分阶段执行：{user_message}")


async def execute_step_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _execute_step_node(
        *args,
        **kwargs,
    )


async def replan_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _replan_node(
        *args,
        **kwargs,
    )


async def summarize_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _summarize_node(
        *args,
        **kwargs,
    )


def build_planner_react_langgraph_graph(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return _build_planner_react_langgraph_graph(
        *args,
        **kwargs,
    )


class _FakeLongTermMemoryRepository:
    def __init__(self, search_results=None, search_results_by_type=None) -> None:
        self.search_results = list(search_results or [])
        self.search_results_by_type = {
            str(memory_type): list(results)
            for memory_type, results in dict(search_results_by_type or {}).items()
        }
        self.search_calls = []
        self.upserted = []

    async def search(
            self,
            query: LongTermMemorySearchQuery,
    ):
        self.search_calls.append(query.model_dump(mode="json"))
        if len(self.search_results_by_type) == 0:
            return list(self.search_results)[:query.limit]

        recalled_memories: list[LongTermMemory] = []
        for memory_type in list(query.memory_types or []):
            recalled_memories.extend(self.search_results_by_type.get(str(memory_type), []))
        return recalled_memories[:query.limit]

    async def upsert(self, memory: LongTermMemory) -> LongTermMemory:
        persisted = memory.model_copy(
            update={
                "id": memory.id or f"mem-{len(self.upserted) + 1}",
            }
        )
        self.upserted.append(persisted)
        return persisted


class _FakeLLM:
    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "message": "最终总结",
                    "final_answer_text": "最终整理后的正文",
                    "attachments": [],
                    "facts_in_session": [],
                    "user_preferences": {},
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        }


class _FakeStructuredMemoryLLM:
    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "message": "任务完成，后续都只需要关注 backend 并保持中文简洁回复。",
                    "final_answer_text": "任务已完成，后续只需要关注 backend，并保持中文简洁回复。",
                    "attachments": [],
                    "facts_in_session": ["当前任务后续都只需要关注 backend"],
                    "user_preferences": {"language": "zh", "response_style": "concise"},
                    "memory_candidates": [
                        {
                            "memory_type": "instruction",
                            "summary": "后续任务只关注 backend",
                            "content": {"text": "后续任务只关注 backend"},
                            "tags": ["backend"],
                            "confidence": 0.85,
                        }
                    ],
                },
                ensure_ascii=False,
            )
        }


class _CaptureSummaryPromptLLM:
    def __init__(self) -> None:
        self.last_prompt = ""

    async def invoke(self, messages, tools, response_format):
        self.last_prompt = str(messages[0]["content"])
        return {
            "content": json.dumps(
                {
                    "message": "最终总结",
                    "final_answer_text": "最终整理后的正文",
                    "attachments": [],
                    "facts_in_session": [],
                    "user_preferences": {},
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        }


class _FakeLightSummaryLLM:
    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "message": "轻量总结",
                    "final_answer_text": "最终整理后的正文",
                    "attachments": [],
                    "facts_in_session": [],
                    "user_preferences": {},
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        }


class _FakeLegacyFinalMessageOnlySummaryLLM:
    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "message": "轻量总结",
                    "final_message": "旧合同里的完整正文，不应再被当作最终正文。",
                    "attachments": [],
                    "facts_in_session": [],
                    "user_preferences": {},
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        }


class _FakeInvalidSummaryJsonLLM:
    async def invoke(self, messages, tools, response_format):
        return {"content": '{"\\n  ":", ","\\n  ":"'}


class _FakeSummaryAttachmentPathDisclosureLLM:
    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "message": "已完成调研并附上报告。",
                    "final_answer_text": (
                        "LangGraph Human-in-the-Loop 实现模式调研报告\n\n"
                        "完整报告文件：/home/ubuntu/langgraph_hitl_report.md"
                    ),
                    "attachments": ["/home/ubuntu/langgraph_hitl_report.md"],
                    "facts_in_session": [],
                    "user_preferences": {},
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        }


class _FailIfSummaryLLMCalled:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools, response_format):
        self.calls += 1
        raise AssertionError("命中确定性交付正文时不应调用总结模型")


class _FailIfCalledSummaryLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools, response_format):
        self.calls += 1
        raise AssertionError("direct_wait 未执行原任务时不应调用总结模型")


class _FakeReplanLLM:
    def __init__(self, steps) -> None:
        self.steps = list(steps)

    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "steps": self.steps,
                },
                ensure_ascii=False,
            )
        }


class _CaptureReplanPromptLLM:
    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.last_prompt = ""

    async def invoke(self, messages, tools, response_format):
        self.last_prompt = str(messages[0]["content"])
        return {
            "content": json.dumps(
                {
                    "steps": self.steps,
                },
                ensure_ascii=False,
            )
        }


class _FakeWriteFileTool(BaseTool):
    name = "file"

    @tool(
        name="write_file",
        description="写入文件",
        parameters={
            "filepath": {"type": "string"},
            "content": {"type": "string"},
        },
        required=["filepath", "content"],
    )
    async def write_file(self, filepath: str, content: str) -> ToolResult:
        return ToolResult(
            success=True,
            data={
                "filepath": filepath,
                "content_length": len(content),
            },
        )


class _FakeSearchTool(BaseTool):
    name = "search"

    @tool(
        name="search_web",
        description="搜索网页",
        parameters={
            "query": {"type": "string"},
        },
        required=["query"],
    )
    async def search_web(self, query: str) -> ToolResult:
        return ToolResult(
            success=True,
            data={
                "query": query,
            },
        )


class _FakeToolLoopLLM:
    def __init__(self) -> None:
        self.tool_name_snapshots: list[list[str]] = []

    async def invoke(self, messages, tools, tool_choice=None, response_format=None):
        tool_names = [str((tool.get("function") or {}).get("name") or "") for tool in list(tools or [])]
        self.tool_name_snapshots.append(tool_names)
        call_index = len(self.tool_name_snapshots) - 1
        if call_index == 0:
            assert "message_notify_user" not in tool_names
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-write",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {
                                    "filepath": "/home/ubuntu/report.md",
                                    "content": "# 报告\n\n已生成。",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        if call_index == 1:
            assert "message_notify_user" not in tool_names
            return {
                "content": json.dumps(
                    {
                        "success": True,
                        "result": "报告已生成",
                        "attachments": [],
                    },
                    ensure_ascii=False,
                ),
                "tool_calls": [],
            }
        raise AssertionError("unexpected invoke count")


class _BlockedThenCompleteLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools, tool_choice=None, response_format=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-write",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {
                                    "filepath": "/home/ubuntu/report.md",
                                    "content": "# 报告\n\n测试内容。",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已整理完成",
                    "attachments": [],
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
        }


class _SearchThenCompleteLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools, tool_choice=None, response_format=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-search",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps(
                                {
                                    "query": "workspace runtime p3",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "检索完成",
                    "attachments": [],
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
        }


def _build_plan(*, step_status: ExecutionStatus = ExecutionStatus.COMPLETED) -> Plan:
    return Plan(
        title="记忆阶段测试",
        goal="验证长期记忆边界",
        language="zh",
        message="开始执行",
        steps=[
            Step(
                id="step-1",
                title="执行阶段",
                description="执行阶段",
                objective_key="objective-step-1",
                success_criteria=["执行阶段完成"],
                status=step_status,
                outcome=(
                    StepOutcome(
                        done=True,
                        summary="步骤完成",
                    )
                    if step_status == ExecutionStatus.COMPLETED
                    else None
                ),
            )
        ],
        status=ExecutionStatus.PENDING,
    )


def test_recall_memory_context_node_should_search_long_term_memory() -> None:
    repository = _FakeLongTermMemoryRepository(
        search_results_by_type={
            "profile": [
                LongTermMemory(
                    user_id="user-1",
                    id="mem-1",
                    namespace="user/user-1/profile",
                    memory_type="profile",
                    summary="用户偏好中文",
                    content={"language": "zh", "style": "concise"},
                )
            ],
            "instruction": [
                LongTermMemory(
                    user_id="user-1",
                    id="mem-2",
                    namespace="agent/planner_react/instruction",
                    memory_type="instruction",
                    summary="保持中文回复",
                    content={"text": "保持中文回复"},
                )
            ],
        }
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "user_message": "帮我整理长期记忆",
        "conversation_summary": "用户之前一直要求中文简洁回复",
        "working_memory": {},
        "graph_metadata": {},
    }

    next_state = asyncio.run(
        recall_memory_context_node(
            state,
            long_term_memory_repository=repository,
        )
    )

    assert len(repository.search_calls) == 3
    assert repository.search_calls[0]["namespace_prefixes"] == [
        "user/user-1/",
        "session/session-1/",
        "agent/planner_react/",
    ]
    assert repository.search_calls[0]["query_text"] == ""
    assert repository.search_calls[0]["memory_types"] == ["profile"]
    assert repository.search_calls[0]["mode"] == LongTermMemorySearchMode.RECENT.value
    assert repository.search_calls[1]["memory_types"] == ["instruction"]
    assert repository.search_calls[1]["query_text"] == ""
    assert repository.search_calls[1]["mode"] == LongTermMemorySearchMode.RECENT.value
    assert repository.search_calls[2]["memory_types"] == ["fact"]
    assert "帮我整理长期记忆" in repository.search_calls[2]["query_text"]
    assert repository.search_calls[2]["mode"] == LongTermMemorySearchMode.HYBRID.value
    assert next_state["retrieved_memories"][0]["id"] == "mem-1"
    assert next_state["retrieved_memories"][1]["id"] == "mem-2"
    assert next_state["working_memory"]["user_preferences"] == {}


def test_recall_memory_context_node_should_skip_long_term_memory_for_first_turn_without_history() -> None:
    repository = _FakeLongTermMemoryRepository(
        search_results=[
            LongTermMemory(
                user_id="user-1",
                id="mem-1",
                namespace="user/user-1/fact",
                memory_type="fact",
                summary="不应在首轮召回",
                content={"text": "不应在首轮召回"},
            )
        ]
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "user_message": "给我设计一份周末出行计划",
        "conversation_summary": "",
        "message_window": [
            {"role": "user", "message": "给我设计一份周末出行计划", "attachment_paths": []},
        ],
        "recent_run_briefs": [],
        "working_memory": {},
        "graph_metadata": {},
    }

    next_state = asyncio.run(
        recall_memory_context_node(
            state,
            long_term_memory_repository=repository,
        )
    )

    assert repository.search_calls == []
    assert next_state["retrieved_memories"] == []


def test_execute_step_node_should_not_write_string_none_when_no_executor_path_available() -> None:
    state = {
        "session_id": "session-1",
        "user_message": "继续执行",
        "plan": Plan(
            title="执行测试",
            goal="验证结果归一化",
            language="zh",
            message="执行当前步骤",
            steps=[
                Step(
                    id="step-1",
                    title="执行阶段",
                    description="执行阶段",
                    objective_key="objective-step-1",
                    success_criteria=["执行阶段完成"],
                    status=ExecutionStatus.PENDING,
                )
            ],
        ),
        "input_parts": [],
        "working_memory": {},
        "graph_metadata": {},
        "emitted_events": [],
    }

    next_state = asyncio.run(
        execute_step_node(
            state,
            llm=object(),
        )
    )

    assert next_state["plan"].steps[0].outcome is not None
    assert next_state["plan"].steps[0].outcome.summary == "步骤执行失败：执行阶段"
    assert next_state["final_message"] == ""
    assert next_state["working_memory"]["decisions"] == ["步骤执行失败：执行阶段"]
    assert next_state["step_states"][0]["step_id"] == "step-1"


def test_execute_step_node_should_capture_write_file_artifact_and_limit_notify_tool() -> None:
    llm = _FakeToolLoopLLM()
    state = {
        "session_id": "session-1",
        "user_message": "继续执行",
        "plan": Plan(
            title="执行测试",
            goal="验证工具循环治理",
            language="zh",
            message="执行当前步骤",
            steps=[
                Step(
                    id="step-1",
                    title="生成报告",
                    description="生成报告",
                    objective_key="objective-step-1",
                    success_criteria=["报告生成完成"],
                    status=ExecutionStatus.PENDING,
                )
            ],
        ),
        "input_parts": [],
        "working_memory": {},
        "graph_metadata": {},
        "selected_artifacts": [],
        "emitted_events": [],
    }

    next_state = asyncio.run(
        execute_step_node(
            state,
            llm=llm,
            runtime_tools=[MessageTool(), _FakeWriteFileTool()],
            max_tool_iterations=5,
        )
    )

    executed_step = next_state["plan"].steps[0]
    assert executed_step.outcome is not None
    assert executed_step.outcome.produced_artifacts == ["/home/ubuntu/report.md"]
    assert next_state["selected_artifacts"] == []
    assert "message_notify_user" not in llm.tool_name_snapshots[0]
    assert "message_notify_user" not in llm.tool_name_snapshots[1]
    assert next_state["step_states"][0]["status"] == ExecutionStatus.COMPLETED.value
    assert next_state["graph_metadata"]["control"]["step_reuse_hit"] is False


def test_guard_step_reuse_node_should_reuse_completed_step_in_current_run() -> None:
    completed_step = Step(
        id="step-1",
        title="生成报告",
        description="生成报告",
        objective_key="objective-report",
        success_criteria=["产出报告"],
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(
            done=True,
            summary="报告已经生成",
            produced_artifacts=["/tmp/report.md"],
            facts_learned=["报告结构已确定"],
        ),
    )
    pending_duplicate_step = Step(
        id="step-2",
        title="再次生成报告",
        description="再次生成报告",
        objective_key="objective-report",
        success_criteria=["产出报告"],
        status=ExecutionStatus.PENDING,
    )
    state = {
        "run_id": "run-1",
        "plan": Plan(
            title="复用测试",
            goal="避免重复执行",
            language="zh",
            steps=[completed_step, pending_duplicate_step],
        ),
        "working_memory": {},
        "graph_metadata": {},
        "selected_artifacts": [],
        "emitted_events": [],
        "execution_count": 0,
    }

    next_state = asyncio.run(guard_step_reuse_node(state))

    reused_step = next_state["plan"].steps[1]
    assert reused_step.status == ExecutionStatus.COMPLETED
    assert reused_step.outcome is not None
    assert reused_step.outcome.reused_from_run_id == "run-1"
    assert reused_step.outcome.reused_from_step_id == "step-1"
    assert reused_step.outcome.produced_artifacts == ["/tmp/report.md"]
    assert next_state["step_states"][-1]["step_id"] == "step-2"
    assert next_state["step_states"][-1]["status"] == ExecutionStatus.COMPLETED.value
    assert next_state["graph_metadata"]["control"]["step_reuse_hit"] is True
    assert next_state["selected_artifacts"] == []
    assert next_state["working_memory"]["facts_in_session"] == ["报告结构已确定"]
    assert next_state.get("final_message", "") == ""


def test_guard_step_reuse_node_should_not_reuse_historical_projection() -> None:
    pending_step = Step(
        id="step-2",
        title="整理结论",
        description="整理结论",
        objective_key="objective-summary",
        success_criteria=["输出结论"],
        status=ExecutionStatus.PENDING,
    )
    state = {
        "run_id": "run-current",
        "plan": Plan(
            title="跨 run 复用测试",
            goal="避免重复执行",
            language="zh",
            steps=[pending_step],
        ),
        "working_memory": {},
        "graph_metadata": {},
        "selected_artifacts": [],
        "recent_run_briefs": [
            {
                "run_id": "run-history",
                "title": "整理结论",
                "goal": "避免重复执行",
                "status": "completed",
                "final_answer_summary": "历史结论已经整理完成",
            }
        ],
        "emitted_events": [],
        "execution_count": 1,
    }

    next_state = asyncio.run(guard_step_reuse_node(state))

    guarded_step = next_state["plan"].steps[0]
    assert guarded_step.status == ExecutionStatus.PENDING
    assert guarded_step.outcome is None
    assert next_state["graph_metadata"]["control"]["step_reuse_hit"] is False


def test_runtime_context_service_should_build_structured_packet_with_separated_context() -> None:
    context_service = RuntimeContextService()
    state = {
        "conversation_summary": "会话摘要",
        "retrieved_memories": [
            {
                "id": "mem-1",
                "memory_type": "profile",
                "summary": "偏好中文",
                "content": {
                    "language": "zh",
                    "embedding": [0.1, 0.2, 0.3],
                    "source": {"kind": "vector"},
                },
                "tags": ["language", "profile"],
            }
        ],
        "session_open_questions": ["问题1", "问题2"],
        "session_blockers": ["阻塞1"],
        "selected_artifacts": [f"/tmp/current-{index}.md" for index in range(12)],
        "historical_artifact_paths": [f"/tmp/history-{index}.md" for index in range(12)],
        "step_states": [
            {
                "step_id": "step-a",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "produced_artifacts": [f"/tmp/generated-{index}.md" for index in range(8)],
                },
            }
        ],
        "recent_run_briefs": [
            {
                "run_id": f"run-{index}",
                "title": f"完成运行{index}",
                "goal": "整理上下文",
                "status": "completed",
                "final_answer_summary": f"总结{index}" * 60,
                "final_answer_text_excerpt": f"重交付正文摘录{index}" * 40,
            }
            for index in range(6)
        ],
        "recent_attempt_briefs": [
            {
                "run_id": f"attempt-{index}",
                "title": f"失败运行{index}",
                "goal": "尝试执行",
                "status": "failed",
                "final_answer_summary": f"失败总结{index}" * 60,
                "final_answer_text_excerpt": f"失败重交付摘录{index}" * 40,
            }
            for index in range(6)
        ],
        "working_memory": {"open_questions": ["问题2", "问题3"]},
        "message_window": [
            {"role": "user", "message": "第一轮问题"},
            {"role": "assistant", "message": "第一轮回答"},
        ],
        "pending_interrupt": {
            "kind": "confirm",
            "prompt": "请确认是否继续调研",
            "confirm_label": "继续",
            "cancel_label": "取消",
        },
    }

    context_packet = context_service.build_packet(
        stage="execute",
        state=state,
        task_mode="research",
    )

    assert context_packet["stage"] == "execute"
    assert context_packet["task_mode"] == "research"
    assert context_packet["open_questions"] == ["问题1", "问题2", "问题3"]
    assert context_packet["pending_confirmation"]["prompt"] == "请确认是否继续调研"
    assert len(context_packet["stable_background"]["recent_run_briefs"]) == 3
    assert len(context_packet["stable_background"]["recent_attempt_briefs"]) == 3
    assert "总结0总结0" in context_packet["stable_background"]["recent_run_briefs"][0]["final_answer_summary"]
    assert len(context_packet["stable_background"]["recent_run_briefs"][0]["final_answer_summary"]) <= 120
    assert "重交付正文摘录0" in context_packet["stable_background"]["recent_run_briefs"][0][
        "final_answer_text_excerpt"
    ]
    assert len(context_packet["stable_background"]["recent_run_briefs"][0]["final_answer_text_excerpt"]) <= 160
    assert "recent_messages" not in context_packet["stable_background"]
    assert context_packet["retrieved_memory_digest"][0]["memory_type"] == "profile"
    assert context_packet["retrieved_memory_digest"][0]["summary"] == "偏好中文"
    assert context_packet["retrieved_memory_digest"][0]["content_preview"] == "language: zh"
    assert "embedding" not in context_packet["retrieved_memory_digest"][0]


def test_runtime_context_service_should_clear_previous_mode_digest_when_switching_to_coding() -> None:
    context_service = RuntimeContextService()
    state = {
        "task_mode": StepTaskModeHint.RESEARCH.value,
        "working_memory": {"goal": "继续完成 backend 改造"},
        "selected_artifacts": ["/tmp/result.md"],
        "environment_digest": {
            "task_mode": StepTaskModeHint.RESEARCH.value,
            "payload": {
                "candidate_links": [{"title": "文档", "url": "https://example.com"}],
                "read_page_summaries": [{"title": "调研页", "url": "https://example.com/doc"}],
                "current_page": {"url": "https://example.com/doc", "title": "调研页"},
            },
        },
        "observation_digest": {
            "task_mode": StepTaskModeHint.RESEARCH.value,
            "payload": {
                "latest_fetch_page": {"title": "调研页"},
                "latest_browser_observation": {"url": "https://example.com/doc"},
            },
        },
        "recent_action_digest": {
            "task_mode": StepTaskModeHint.RESEARCH.value,
            "payload": {
                "recent_search_queries": ["runtime context p2"],
                "last_failed_action": {
                    "function_name": "search_web",
                    "message": "搜索失败",
                },
                "last_blocked_tool_call": {
                    "function_name": "fetch_page",
                    "reason": "research_route_fetch_required",
                    "message": "请先读取候选链接",
                },
                "last_no_progress_reason": "还没确认实现边界",
            },
        },
    }

    context_packet = context_service.build_packet(
        stage="execute",
        state=state,
        task_mode=StepTaskModeHint.CODING.value,
    )
    state_updates = context_service.extract_state_updates(context_packet)

    assert context_packet["task_mode"] == StepTaskModeHint.CODING.value
    assert context_packet["environment_digest"] == {}
    assert "latest_fetch_page" not in context_packet["observation_digest"]
    assert "latest_browser_observation" not in context_packet["observation_digest"]
    assert "recent_search_queries" not in context_packet["recent_action_digest"]
    assert "last_failed_action" not in context_packet["recent_action_digest"]
    assert "last_blocked_tool_call" not in context_packet["recent_action_digest"]
    assert state_updates["environment_digest"] == {
        "task_mode": StepTaskModeHint.CODING.value,
        "payload": {},
    }


def test_runtime_context_service_should_inject_current_temporal_context() -> None:
    context_service = RuntimeContextService()

    context_packet = context_service.build_packet(
        stage="planner",
        state={"user_message": "查询最新政策"},
        task_mode=StepTaskModeHint.GENERAL.value,
    )

    temporal_context = dict(context_packet.get("temporal_context") or {})
    assert temporal_context["timezone"] == "Asia/Shanghai"
    assert temporal_context["utc_offset"] == "+08:00"
    assert temporal_context["current_date"]
    datetime.fromisoformat(str(temporal_context["current_datetime"]))
    assert "模型训练年份" in temporal_context["relative_time_rule"]


def test_runtime_context_service_should_hide_observation_digest_in_human_wait_mode() -> None:
    context_service = RuntimeContextService()
    state = {
        "task_mode": StepTaskModeHint.BROWSER_INTERACTION.value,
        "working_memory": {"goal": "等待用户确认后继续"},
        "pending_interrupt": {
            "kind": "select",
            "prompt": "请选择下一步",
            "options": [
                {"label": "继续调研", "resume_value": "research"},
                {"label": "直接总结", "resume_value": "summary"},
            ],
        },
        "observation_digest": {
            "task_mode": StepTaskModeHint.BROWSER_INTERACTION.value,
            "payload": {
                "latest_fetch_page": {"title": "调研页"},
                "latest_browser_observation": {"url": "https://example.com/doc"},
                "latest_shell_result": {"function_name": "shell_exec"},
            },
        },
        "recent_action_digest": {
            "task_mode": StepTaskModeHint.BROWSER_INTERACTION.value,
            "payload": {
                "last_failed_action": {
                    "function_name": "browser_click",
                    "message": "点击失败",
                },
                "last_blocked_tool_call": {
                    "function_name": "browser_click",
                    "reason": "browser_route_blocked",
                    "message": "当前页面不允许直接点击",
                },
            },
        },
    }

    context_packet = context_service.build_packet(
        stage="execute",
        state=state,
        task_mode=StepTaskModeHint.HUMAN_WAIT.value,
    )

    assert context_packet["task_mode"] == StepTaskModeHint.HUMAN_WAIT.value
    assert "observation_digest" not in context_packet
    assert context_packet["environment_digest"]["wait_kind"] == "select"
    assert context_packet["recent_action_digest"]["last_user_wait_reason"] == "请选择下一步"
    assert "last_failed_action" not in context_packet["recent_action_digest"]
    assert "last_blocked_tool_call" not in context_packet["recent_action_digest"]


def test_runtime_context_service_should_not_inherit_previous_task_mode_in_planner_stage() -> None:
    context_service = RuntimeContextService()
    state = {
        "task_mode": StepTaskModeHint.CODING.value,
        "working_memory": {"goal": "重新规划下一批任务"},
        "environment_digest": {
            "task_mode": StepTaskModeHint.CODING.value,
            "payload": {"cwd": "/workspace/project"},
        },
        "retrieved_memories": [
            {
                "id": "mem-1",
                "memory_type": "instruction",
                "summary": "保持 backend 结构清晰",
                "content": {"text": "保持 backend 结构清晰"},
                "tags": ["backend"],
            }
        ],
    }

    context_packet = context_service.build_packet(
        stage="planner",
        state=state,
    )

    assert context_packet["task_mode"] == StepTaskModeHint.GENERAL.value
    assert "environment_digest" not in context_packet
    assert "observation_digest" not in context_packet
    assert context_packet["retrieved_memory_digest"][0]["memory_type"] == "instruction"


def test_runtime_context_service_should_exclude_profile_memory_in_planner_stage() -> None:
    context_service = RuntimeContextService()
    state = {
        "task_mode": StepTaskModeHint.GENERAL.value,
        "retrieved_memories": [
            {
                "id": "mem-profile-1",
                "memory_type": "profile",
                "summary": "常从上海出发",
                "content": {"text": "常从上海出发"},
            },
            {
                "id": "mem-fact-1",
                "memory_type": "fact",
                "summary": "预算上限 2000",
                "content": {"text": "预算上限 2000"},
            },
            {
                "id": "mem-instruction-1",
                "memory_type": "instruction",
                "summary": "用户要求先确认再继续",
                "content": {"text": "用户要求先确认再继续"},
            },
        ],
    }

    context_packet = context_service.build_packet(
        stage="planner",
        state=state,
    )

    memory_types = [item["memory_type"] for item in context_packet["retrieved_memory_digest"]]
    assert "profile" not in memory_types
    assert memory_types == ["fact", "instruction"]


def test_runtime_context_service_should_filter_retrieved_memories_by_stage_and_task_mode_policy() -> None:
    context_service = RuntimeContextService()
    state = {
        "task_mode": StepTaskModeHint.BROWSER_INTERACTION.value,
        "retrieved_memories": [
            {"id": "mem-fact-1", "memory_type": "fact", "summary": "事实1", "content": {"text": "事实1"}},
            {"id": "mem-profile-1", "memory_type": "profile", "summary": "偏好1", "content": {"text": "偏好1"}},
            {
                "id": "mem-instruction-1",
                "memory_type": "instruction",
                "summary": "指令1",
                "content": {"text": "指令1"},
            },
            {"id": "mem-profile-2", "memory_type": "profile", "summary": "偏好2", "content": {"text": "偏好2"}},
            {
                "id": "mem-instruction-2",
                "memory_type": "instruction",
                "summary": "指令2",
                "content": {"text": "指令2"},
            },
        ],
    }

    context_packet = context_service.build_packet(
        stage="execute",
        state=state,
        task_mode=StepTaskModeHint.BROWSER_INTERACTION.value,
    )

    memory_types = [item["memory_type"] for item in context_packet["retrieved_memory_digest"]]
    assert memory_types == ["profile", "instruction", "profile"]
    assert len(context_packet["retrieved_memory_digest"]) == 3


def test_runtime_context_service_should_hide_execute_only_fields_in_summary_stage() -> None:
    context_service = RuntimeContextService()
    state = {
        "task_mode": StepTaskModeHint.RESEARCH.value,
        "working_memory": {
            "goal": "输出总结",
        },
        "environment_digest": {
            "task_mode": StepTaskModeHint.RESEARCH.value,
            "payload": {"candidate_links": [{"title": "文档"}]},
        },
        "observation_digest": {
            "task_mode": StepTaskModeHint.RESEARCH.value,
            "payload": {"latest_browser_observation": {"url": "https://example.com/doc"}},
        },
        "recent_action_digest": {
            "task_mode": StepTaskModeHint.RESEARCH.value,
            "payload": {"recent_search_queries": ["runtime context p2"]},
        },
        "selected_artifacts": ["/tmp/final.md"],
        "execution_count": 2,
    }

    context_packet = context_service.build_packet(
        stage="summary",
        state=state,
    )

    assert context_packet["stage"] == "summary"
    assert "environment_digest" not in context_packet
    assert "observation_digest" not in context_packet
    assert "recent_action_digest" not in context_packet


class _FakeWorkspaceRuntimeService:
    def __init__(self, snapshot: WorkspaceEnvironmentSnapshot) -> None:
        self._snapshot = snapshot

    async def build_environment_snapshot(self) -> WorkspaceEnvironmentSnapshot:
        return self._snapshot


def _build_runtime_context_service_with_workspace_artifacts(*paths: str) -> RuntimeContextService:
    return RuntimeContextService(
        workspace_runtime_service=_FakeWorkspaceRuntimeService(
            WorkspaceEnvironmentSnapshot(
                workspace=Workspace(
                    id="workspace-1",
                    session_id="session-1",
                ),
                artifacts=[
                    WorkspaceArtifact(
                        workspace_id="workspace-1",
                        path=path,
                        artifact_type="file",
                    )
                    for path in paths
                ],
            )
        )
    )


def test_runtime_context_service_should_prefer_workspace_snapshot_for_coding_digest() -> None:
    context_service = RuntimeContextService(
        workspace_runtime_service=_FakeWorkspaceRuntimeService(
            WorkspaceEnvironmentSnapshot(
                workspace=Workspace(
                    id="workspace-1",
                    session_id="session-1",
                    shell_session_id="shell-workspace-1",
                    cwd="/workspace/project",
                    environment_summary={
                        "shell_session_status": "active",
                        "latest_shell_result": {
                            "function_name": "shell_execute",
                            "message": "命令执行完成",
                            "console": "pytest -q\n24 passed",
                        },
                        "recent_changed_files": ["/workspace/project/app.py"],
                        "file_tree_summary": ["src/: 3 files"],
                    },
                ),
                artifacts=[
                    WorkspaceArtifact(
                        workspace_id="workspace-1",
                        path="/workspace/project/report.md",
                        artifact_type="report",
                    )
                ],
            )
        )
    )
    state = {
        "session_id": "session-1",
        "task_mode": StepTaskModeHint.CODING.value,
        "working_memory": {"goal": "继续完成 backend 改造"},
    }

    context_packet = asyncio.run(
        context_service.build_packet_async(
            stage="execute",
            state=state,
            task_mode=StepTaskModeHint.CODING.value,
        )
    )

    assert context_packet["environment_digest"]["cwd"] == "/workspace/project"
    assert context_packet["environment_digest"]["shell_session_status"] == "active"
    assert context_packet["environment_digest"]["available_artifacts"] == ["/workspace/project/report.md"]
    assert context_packet["observation_digest"]["latest_shell_result"]["console"] == "pytest -q\n24 passed"


def test_runtime_context_service_should_project_search_evidence_from_workspace_snapshot() -> None:
    context_service = RuntimeContextService(
        workspace_runtime_service=_FakeWorkspaceRuntimeService(
            WorkspaceEnvironmentSnapshot(
                workspace=Workspace(
                    id="workspace-1",
                    session_id="session-1",
                    environment_summary={
                        "candidate_links": [
                            {
                                "title": "OpenAI 文档",
                                "url": "https://example.com/docs",
                                "snippet": "OpenAI 文档的搜索摘要，可用于先判断是否已经有足够信息。",
                            }
                        ],
                    },
                ),
                artifacts=[],
            )
        )
    )
    state = {
        "session_id": "session-1",
        "task_mode": StepTaskModeHint.RESEARCH.value,
        "working_memory": {"goal": "继续调研"},
    }

    context_packet = asyncio.run(
        context_service.build_packet_async(
            stage="execute",
            state=state,
            task_mode=StepTaskModeHint.RESEARCH.value,
        )
    )

    assert context_packet["environment_digest"]["search_evidence_summaries"] == [
        {
            "title": "OpenAI 文档",
            "snippet": "OpenAI 文档的搜索摘要，可用于先判断是否已经有足够信息。",
            "url": "https://example.com/docs",
        }
    ]


def test_runtime_context_service_should_ignore_emitted_events_for_environment_truth() -> None:
    context_service = RuntimeContextService(
        workspace_runtime_service=_FakeWorkspaceRuntimeService(
            WorkspaceEnvironmentSnapshot(
                workspace=Workspace(
                    id="workspace-1",
                    session_id="session-1",
                ),
                artifacts=[],
            )
        )
    )
    state = {
        "session_id": "session-1",
        "task_mode": StepTaskModeHint.RESEARCH.value,
        "working_memory": {"goal": "继续调研"},
        "emitted_events": [
            ToolEvent(
                tool_name="search",
                function_name="search_web",
                function_args={"query": "should be ignored"},
                function_result=ToolResult(success=True, data={}),
                status=ToolEventStatus.CALLED,
            )
        ],
    }

    context_packet = asyncio.run(
        context_service.build_packet_async(
            stage="execute",
            state=state,
            task_mode=StepTaskModeHint.RESEARCH.value,
        )
    )

    assert context_packet["environment_digest"] == {}
    assert context_packet["observation_digest"]["stage"] == "execute"
    assert "latest_fetch_page" not in context_packet["observation_digest"]
    assert "latest_browser_observation" not in context_packet["observation_digest"]


def test_runtime_context_service_should_not_fallback_to_stale_digest_when_workspace_snapshot_is_empty() -> None:
    context_service = RuntimeContextService(
        workspace_runtime_service=_FakeWorkspaceRuntimeService(
            WorkspaceEnvironmentSnapshot(
                workspace=Workspace(
                    id="workspace-1",
                    session_id="session-1",
                ),
                artifacts=[],
            )
        )
    )
    state = {
        "session_id": "session-1",
        "task_mode": StepTaskModeHint.CODING.value,
        "working_memory": {"goal": "继续完成 backend 改造"},
        "environment_digest": {
            "task_mode": StepTaskModeHint.CODING.value,
            "payload": {
                "cwd": "/stale/workspace",
                "shell_session_status": "stale-active",
                "recent_changed_files": ["/stale/workspace/app.py"],
                "file_tree_summary": ["stale tree"],
                "available_artifacts": ["/stale/workspace/report.md"],
            },
        },
        "observation_digest": {
            "task_mode": StepTaskModeHint.CODING.value,
            "payload": {
                "latest_shell_result": {
                    "console": "stale result",
                }
            },
        },
    }

    context_packet = asyncio.run(
        context_service.build_packet_async(
            stage="execute",
            state=state,
            task_mode=StepTaskModeHint.CODING.value,
        )
    )

    assert context_packet["environment_digest"] == {}
    assert "latest_shell_result" not in context_packet["observation_digest"]
    assert context_packet["observation_digest"]["stage"] == "execute"


def test_execute_step_node_should_persist_current_mode_recent_action_digest() -> None:
    llm = _BlockedThenCompleteLLM()
    state = {
        "session_id": "session-1",
        "user_message": "请继续调研并整理结果",
        "plan": Plan(
            title="执行测试",
            goal="验证 recent action 沉淀",
            language="zh",
            message="执行当前步骤",
            steps=[
                Step(
                    id="step-1",
                    title="整理调研结论",
                    description="整理调研结论",
                    objective_key="objective-step-1",
                    success_criteria=["整理完成"],
                    task_mode_hint=StepTaskModeHint.RESEARCH,
                    status=ExecutionStatus.PENDING,
                )
            ],
        ),
        "input_parts": [],
        "working_memory": {},
        "graph_metadata": {},
        "selected_artifacts": [],
        "emitted_events": [],
        "recent_action_digest": {},
    }

    next_state = asyncio.run(
        execute_step_node(
            state,
            llm=llm,
            runtime_tools=[_FakeWriteFileTool()],
            max_tool_iterations=3,
        )
    )

    assert next_state["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert next_state["recent_action_digest"]["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert next_state["recent_action_digest"]["payload"]["last_failed_action"]["function_name"] == "write_file"
    assert next_state["recent_action_digest"]["payload"]["last_blocked_tool_call"]["function_name"] == "write_file"
    assert next_state["recent_action_digest"]["payload"]["last_blocked_tool_call"]["reason"] == "task_mode_tool_blocked"


def test_execute_step_node_should_persist_recent_search_queries_into_recent_action_digest() -> None:
    llm = _SearchThenCompleteLLM()
    state = {
        "session_id": "session-1",
        "user_message": "请继续调研并整理结果",
        "plan": Plan(
            title="执行测试",
            goal="验证搜索查询沉淀",
            language="zh",
            message="执行当前步骤",
            steps=[
                Step(
                    id="step-1",
                    title="检索资料",
                    description="检索资料",
                    objective_key="objective-step-1",
                    success_criteria=["检索完成"],
                    task_mode_hint=StepTaskModeHint.RESEARCH,
                    status=ExecutionStatus.PENDING,
                )
            ],
        ),
        "input_parts": [],
        "working_memory": {},
        "graph_metadata": {},
        "selected_artifacts": [],
        "emitted_events": [],
        "recent_action_digest": {},
    }

    next_state = asyncio.run(
        execute_step_node(
            state,
            llm=llm,
            runtime_tools=[_FakeSearchTool()],
            max_tool_iterations=3,
        )
    )

    assert next_state["recent_action_digest"]["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert next_state["recent_action_digest"]["payload"]["recent_search_queries"] == ["workspace runtime p3"]


def test_replan_node_should_regenerate_conflicting_step_ids_without_numeric_assumption() -> None:
    completed_step = Step(
        id="step-a",
        title="完成已有步骤",
        description="完成已有步骤",
        objective_key="objective-step-a",
        success_criteria=["完成已有步骤"],
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="已完成"),
    )
    plan = Plan(
        title="重规划测试",
        goal="验证 step_id 去重",
        language="zh",
        message="开始重规划",
        steps=[completed_step],
    )
    state = {
        "plan": plan,
        "last_executed_step": completed_step.model_copy(deep=True),
        "emitted_events": [],
    }

    next_state = asyncio.run(
        replan_node(
            state,
            _FakeReplanLLM(
                steps=[
                    {"id": "step-a", "description": "新的分析步骤"},
                    {"id": "step-c", "description": "新的交付步骤"},
                ]
            ),
        )
    )

    replanned_steps = next_state["plan"].steps
    step_ids = [step.id for step in replanned_steps]

    assert len(replanned_steps) == 3
    assert step_ids[0] == "step-a"
    assert step_ids[1] != "step-a"
    assert len(set(step_ids)) == 3
    assert replanned_steps[1].description == "新的分析步骤"
    assert replanned_steps[2].id == "step-c"
    assert next_state["current_step_id"] == step_ids[1]


def test_replan_node_should_clear_consumed_entry_upgrade_signal() -> None:
    completed_step = Step(
        id="atomic-action-step",
        title="完成原子动作",
        description="完成原子动作",
        objective_key="objective-atomic-action",
        success_criteria=["完成原子动作"],
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="原子动作需要升级"),
    )
    plan = Plan(
        title="重规划测试",
        goal="验证 entry_upgrade 消费",
        language="zh",
        message="开始重规划",
        steps=[completed_step],
    )
    state = {
        "plan": plan,
        "last_executed_step": completed_step.model_copy(deep=True),
        "graph_metadata": {
            "control": {
                **_planned_entry_contract_control("验证 entry_upgrade 消费"),
                "entry_upgrade": {
                    "reason_code": "open_questions_require_planner",
                    "source_route": "atomic_action",
                    "target_route": "planned_task",
                    "evidence": {"open_questions": ["需要继续规划"]},
                },
            }
        },
        "emitted_events": [],
    }

    next_state = asyncio.run(
        replan_node(
            state,
            _FakeReplanLLM(
                steps=[
                    {"id": "step-next", "description": "继续规划后的步骤"},
                ]
            ),
        )
    )

    control = (next_state.get("graph_metadata") or {}).get("control") or {}
    assert "entry_upgrade" not in control
    assert next_state["current_step_id"] == "step-next"


def test_replan_node_should_filter_meta_tool_validation_steps_and_retry_once() -> None:
    class _DriftThenValidReplanLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def invoke(self, messages, tools, response_format):
            self.calls += 1
            if self.calls == 1:
                return {
                    "content": json.dumps(
                        {
                            "steps": [
                                {
                                    "id": "step-meta",
                                    "title": "测试 shell 工具可用性",
                                    "description": "先验证 shell_execute 是否可用",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                }
            return {
                "content": json.dumps(
                    {
                        "steps": [
                            {
                                "id": "step-real",
                                "title": "读取目录并整理结果",
                                "description": "读取当前目录并输出文件列表摘要",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }

    completed_step = Step(
        id="step-a",
        title="完成已有步骤",
        description="完成已有步骤",
        objective_key="objective-step-a",
        success_criteria=["完成已有步骤"],
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="已完成"),
    )
    plan = Plan(
        title="重规划测试",
        goal="继续执行目录信息收集",
        language="zh",
        message="开始重规划",
        steps=[completed_step],
    )
    llm = _DriftThenValidReplanLLM()
    state = {
        "plan": plan,
        "last_executed_step": completed_step.model_copy(deep=True),
        "user_message": "继续读取目录并给我结果，不要测试工具",
        "emitted_events": [],
    }

    next_state = asyncio.run(replan_node(state, llm))

    assert llm.calls == 2
    replanned_steps = next_state["plan"].steps
    assert len(replanned_steps) == 2
    assert "工具可用性" not in replanned_steps[1].description
    assert "验证 shell_execute" not in replanned_steps[1].description
    assert "读取当前目录" in replanned_steps[1].description


def test_replan_node_should_drop_semantically_duplicated_steps_after_replan() -> None:
    completed_step = Step(
        id="step-completed",
        title="初始化目录",
        description="初始化目录",
        objective_key="objective-step-completed",
        success_criteria=["初始化目录"],
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="已完成"),
    )
    plan = Plan(
        title="重规划语义去重测试",
        goal="收集目录信息并输出",
        language="zh",
        message="开始重规划",
        steps=[completed_step],
    )
    state = {
        "plan": plan,
        "last_executed_step": completed_step.model_copy(deep=True),
        "user_message": "继续收集目录信息并输出结果",
        "emitted_events": [],
    }
    llm = _FakeReplanLLM(
        steps=[
            {
                "id": "step-a",
                "title": "读取当前目录并输出结果",
                "description": "读取当前目录并输出结果",
                "success_criteria": ["读取目录", "输出结果"],
            },
            {
                "id": "step-b",
                "title": "读取当前目录并输出结果",
                "description": "读取当前目录并输出结果",
                "success_criteria": ["读取目录", "输出结果"],
            },
            {
                "id": "step-c",
                "title": "读取 hello.txt 内容",
                "description": "读取 hello.txt 内容并确认",
                "success_criteria": ["读取 hello.txt"],
            },
        ]
    )

    next_state = asyncio.run(replan_node(state, llm))

    replanned_steps = next_state["plan"].steps
    assert len(replanned_steps) == 3
    assert replanned_steps[1].description == "读取当前目录并输出结果"
    assert replanned_steps[2].description == "读取 hello.txt 内容并确认"


def test_replan_node_should_use_summarized_prompt_inputs_instead_of_full_plan_json() -> None:
    completed_step = Step(
        id="step-a",
        title="完成已有步骤",
        description="完成已有步骤",
        objective_key="objective-step-a",
        success_criteria=["完成已有步骤"],
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="已完成"),
    )
    plan = Plan(
        title="重规划测试",
        goal="验证 replan 输入摘要化",
        language="zh",
        message="开始重规划",
        steps=[completed_step],
    )
    llm = _CaptureReplanPromptLLM(
        steps=[
            {"id": "step-b", "description": "继续执行下一步"},
        ]
    )
    state = {
        "plan": plan,
        "last_executed_step": completed_step.model_copy(deep=True),
        "emitted_events": [],
        "task_mode": StepTaskModeHint.RESEARCH.value,
    }

    asyncio.run(replan_node(state, llm))

    assert "当前步骤摘要" in llm.last_prompt
    assert "当前计划快照" in llm.last_prompt
    context_prompt = llm.last_prompt.split("已知上下文:", 1)[-1]
    assert '"success_criteria"' not in context_prompt


def test_summarize_should_not_generate_memory_candidates_and_consolidate_should_persist_existing_writes() -> None:
    repository = _FakeLongTermMemoryRepository()
    llm = _FakeLLM()
    pending_memory_writes = [
        {
            "namespace": "user/user-1/profile",
            "memory_type": "profile",
            "summary": "用户偏好中文回复",
            "content": {"language": "zh"},
            "tags": ["language"],
            "confidence": 0.9,
        },
        {
            "namespace": "session/session-1/fact",
            "memory_type": "fact",
            "summary": "本会话只关注 backend",
            "content": {"text": "本会话只关注 backend"},
            "tags": ["backend"],
            "confidence": 0.8,
        },
    ]
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "帮我完成总结，并且后续请用中文回复",
        "plan": _build_plan(),
        "execution_count": 1,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证长期记忆边界",
            "user_preferences": {"language": "zh"},
            "facts_in_session": ["本会话只关注 backend"],
        },
        "pending_memory_writes": pending_memory_writes,
        "message_window": [
            {"role": "user", "message": "帮我完成总结，并且后续请用中文回复", "attachment_paths": []},
        ],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["pending_memory_writes"] == pending_memory_writes
    assert summarized_state["selected_artifacts"] == []

    consolidated_state = asyncio.run(
        _consolidate_memory_node(
            summarized_state,
            long_term_memory_repository=repository,
        )
    )

    assert len(repository.upserted) == 2
    assert {item.memory_type for item in repository.upserted} == {"profile", "fact"}
    assert consolidated_state["pending_memory_writes"] == []


def test_summarize_should_use_compacted_plan_snapshot_in_prompt() -> None:
    llm = _CaptureSummaryPromptLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终总结",
        "plan": _build_plan(),
        "execution_count": 2,
        "selected_artifacts": ["/home/ubuntu/final.md"],
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证总结快照",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    asyncio.run(summarize_node(state, llm))

    assert "上下文数据包(JSON)" in llm.last_prompt
    assert '"plan_snapshot"' in llm.last_prompt
    assert '"selected_artifacts": ["/home/ubuntu/final.md"]' in llm.last_prompt
    assert '"objective_key"' not in llm.last_prompt
    assert '"success_criteria"' not in llm.last_prompt


def test_summarize_should_not_emit_false_success_when_last_step_failed() -> None:
    llm = _CaptureSummaryPromptLLM()
    failed_step = Step(
        id="step-3",
        title="整理 5 条要点",
        description="整理 LangGraph human-in-the-loop 的常见实现模式，归纳为 5 条要点，并标注对应来源链接",
        objective_key="objective-step-3",
        success_criteria=["产出 5 条要点"],
        status=ExecutionStatus.FAILED,
        outcome=StepOutcome(
            done=False,
            summary="步骤执行失败：整理 5 条要点",
            blockers=["达到最大工具调用轮次，当前步骤仍未形成可交付结果。"],
        ),
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "调研 LangGraph human-in-the-loop 常见实现模式，给我 5 条要点和来源链接",
        "plan": Plan(
            title="LangGraph Human-in-the-Loop 实现模式调研",
            goal="调研 LangGraph human-in-the-loop 常见实现模式，给出 5 条要点和来源链接",
            language="zh",
            message="开始执行",
            steps=[failed_step],
            status=ExecutionStatus.FAILED,
        ),
        "execution_count": 3,
        "last_executed_step": failed_step,
        "step_states": [
            {
                "step_id": "step-3",
                "status": ExecutionStatus.FAILED.value,
                "outcome": {
                    "done": False,
                    "summary": "步骤执行失败：整理 5 条要点",
                    "blockers": ["达到最大工具调用轮次，当前步骤仍未形成可交付结果。"],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "调研 LangGraph human-in-the-loop 常见实现模式，给出 5 条要点和来源链接",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert "未完整完成" in summarized_state["final_answer_text"]
    assert "执行失败" in summarized_state["final_answer_text"]
    assert "未完整完成" in summarized_state["final_message"]


def test_summarize_should_use_unified_summary_prompt_for_preview_requests() -> None:
    llm = _CaptureSummaryPromptLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "先给我一个北京三日游草稿",
        "plan": _build_plan(),
        "execution_count": 2,
        "last_executed_step": Step(
            id="step-3",
            title="北京三日游草稿",
            description="生成北京三日游草稿",
            objective_key="objective-step-3",
            success_criteria=["草稿生成完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成北京三日游草稿",
                facts_learned=["这里是一整段很长的草稿证据。"],
            ),
        ),
        "step_states": [
            {
                "step_id": "step-3",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成北京三日游草稿",
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "生成北京三日游草稿",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    asyncio.run(summarize_node(state, llm))

    assert "final_answer_text" in llm.last_prompt
    assert "你是多步执行链里最终面向用户正文的唯一整理者" in llm.last_prompt
    assert "最后一步属于“预览/草稿”步骤" not in llm.last_prompt
    assert "如果最终正文已经足够完整、结构化、可直接使用，优先只交付正文" in llm.last_prompt
    assert "对攻略、方案、教程、清单、对比类任务，默认优先把核心可执行内容放进 `final_answer_text`" in llm.last_prompt


def test_summarize_should_emit_heavy_delivery_to_user_but_keep_light_summary_in_state() -> None:
    llm = _FakeLightSummaryLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证轻重分轨",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最近一步结果的短摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["final_message"] == "轻量总结"
    assert summarized_state["final_answer_text"] == "最终整理后的正文"
    message_event = summarized_state["emitted_events"][0]
    assert message_event.type == "message"
    assert message_event.stage == "final"
    assert message_event.message == "最终整理后的正文"


def test_summarize_should_always_generate_final_answer_text_via_summary_llm() -> None:
    llm = _FakeLightSummaryLLM()
    final_step = Step(
        id="step-final",
        title="输出最终结果",
        description="输出最终结果",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(
            done=True,
            summary="步骤完成",
            facts_learned=["这是步骤阶段的旧草稿证据，不应直接作为最终答案。"],
        ),
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "last_executed_step": final_step,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证最终正文由 summary 生成",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "轻量摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["final_message"] == "轻量总结"
    assert summarized_state["final_answer_text"] == "最终整理后的正文"
    message_event = summarized_state["emitted_events"][0]
    assert message_event.type == "message"
    assert message_event.stage == "final"
    assert message_event.message == "最终整理后的正文"


def test_summarize_should_not_use_legacy_final_message_as_final_answer_text() -> None:
    llm = _FakeLegacyFinalMessageOnlySummaryLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证旧 final_message 不再作为最终正文",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "旧的状态摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["final_message"] == "轻量总结"
    assert summarized_state["final_answer_text"] == "轻量总结"
    message_event = summarized_state["emitted_events"][0]
    assert message_event.type == "message"
    assert message_event.stage == "final"
    assert message_event.message == "轻量总结"


def test_summarize_should_fallback_to_step_draft_fact_when_summary_json_invalid() -> None:
    llm = _FakeInvalidSummaryJsonLLM()
    draft_text = "## LangChain Human-in-the-Loop 能力概述\n\n支持人工审核、编辑和拒绝工具调用。"
    final_step = Step(
        id="step-final",
        title="整理网页结果",
        description="整理网页结果",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(
            done=True,
            summary="已读取并分析文档",
            facts_learned=[
                "普通事实不应作为最终正文",
                f"{STEP_DRAFT_FACT_PREFIX}{draft_text}",
            ],
        ),
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "阅读文档并总结",
        "plan": _build_plan(),
        "execution_count": 2,
        "last_executed_step": final_step,
        "step_states": [
            {
                "step_id": "step-final",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已读取并分析文档",
                    "facts_learned": [
                        "普通事实不应作为最终正文",
                        f"{STEP_DRAFT_FACT_PREFIX}{draft_text}",
                    ],
                    "blockers": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证 summary JSON 失败时使用正文草稿兜底",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["final_message"] == "已读取并分析文档"
    assert summarized_state["final_answer_text"] == draft_text
    message_event = summarized_state["emitted_events"][0]
    assert message_event.message == draft_text


def test_summarize_should_not_reuse_generic_step_summary_as_final_message() -> None:
    llm = _FakeLightSummaryLLM()
    final_step = Step(
        id="step-final",
        title="导出最终报告",
        description="导出最终报告",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(
            done=True,
            summary="当前步骤已完成",
            facts_learned=["这里有一份候选报告证据。"],
        ),
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终报告",
        "plan": _build_plan(),
        "execution_count": 2,
        "last_executed_step": final_step,
        "working_memory": {
            "goal": "验证泛化步骤摘要不会污染最终轻总结",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "旧的轻量摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["final_message"] == "轻量总结"
    assert summarized_state["emitted_events"][0].message == "最终整理后的正文"


def test_summarize_should_emit_final_message_stream_before_final_message_event() -> None:
    llm = _FakeLightSummaryLLM()
    captured_events = []
    final_step = Step(
        id="step-final",
        title="输出最终结果",
        description="输出最终结果",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(
            done=True,
            summary="步骤完成",
            facts_learned=["这里有一份候选最终结论。"],
        ),
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "last_executed_step": final_step,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证 final_message 流式输出",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "轻量摘要",
    }

    async def _sink(event):
        captured_events.append(event)

    token = bind_live_event_sink(_sink)
    try:
        summarized_state = asyncio.run(summarize_node(state, llm))
    finally:
        unbind_live_event_sink(token)

    assert isinstance(captured_events[0], TextStreamStartEvent)
    assert isinstance(captured_events[1], TextStreamDeltaEvent)
    assert isinstance(captured_events[2], TextStreamEndEvent)
    assert isinstance(captured_events[-2], MessageEvent)
    assert captured_events[-2].stage == "final"
    assert captured_events[-2].message == "最终整理后的正文"
    assert [event.type for event in summarized_state["emitted_events"]] == ["message", "plan"]


def test_summarize_should_emit_final_message_stream_for_direct_wait_summary_error() -> None:
    llm = _FakeLLM()
    captured_events = []
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "继续",
        "plan": _build_plan(),
        "execution_count": 1,
        "step_states": [
            {
                "step_id": "direct-wait-confirm",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证 direct_wait 错误总结流式输出",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {"control": _entry_contract_control("先让我确认后再继续搜索课程")},
        "emitted_events": [],
        "final_message": "",
    }

    async def _sink(event):
        captured_events.append(event)

    token = bind_live_event_sink(_sink)
    try:
        summarized_state = asyncio.run(summarize_node(state, llm))
    finally:
        unbind_live_event_sink(token)

    assert isinstance(captured_events[0], TextStreamStartEvent)
    assert isinstance(captured_events[1], TextStreamDeltaEvent)
    assert isinstance(captured_events[2], TextStreamEndEvent)
    assert not any(getattr(event, "type", "") == "error" for event in captured_events)
    assert any(
        getattr(event, "type", "") == "message"
        and getattr(event, "stage", "") == "final"
        for event in captured_events
    )
    assert [event.type for event in summarized_state["emitted_events"]] == ["message", "plan"]


def test_summarize_should_build_summary_context_for_final_round(monkeypatch) -> None:
    llm = _FakeLightSummaryLLM()
    final_step = Step(
        id="step-final",
        title="输出最终结果",
        description="输出最终结果",
        status=ExecutionStatus.COMPLETED,
    )

    called = {"value": False}

    async def _capture_call(*args, **kwargs):
        called["value"] = True
        return {"stable_background": {}, "current_turn": {}}

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes._build_prompt_context_packet_async",
        _capture_call,
    )

    state = {
        "session_id": "session-ctx-skip",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "last_executed_step": final_step,
        "working_memory": {
            "goal": "验证跳过总结时不构建上下文",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "轻量摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert called["value"] is True
    assert summarized_state["final_message"] == "轻量总结"


def test_summarize_should_still_generate_final_answer_text_for_preview_requests() -> None:
    llm = _FakeLightSummaryLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "先给我一个草稿",
        "plan": _build_plan(),
        "execution_count": 2,
        "last_executed_step": Step(
            id="step-3",
            title="候选草稿",
            description="生成候选草稿",
            objective_key="objective-step-3",
            success_criteria=["草稿生成完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成候选草稿",
                facts_learned=["已生成候选草稿，可供最终总结参考。"],
            ),
        ),
        "step_states": [
            {
                "step_id": "step-3",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成候选草稿",
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "生成候选草稿",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "已生成候选草稿",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["final_message"] == "轻量总结"
    assert summarized_state["final_answer_text"] == "最终整理后的正文"
    message_event = summarized_state["emitted_events"][0]
    assert message_event.stage == "final"
    assert message_event.message == "最终整理后的正文"


def test_summarize_should_emit_heavy_delivery_with_resolved_attachments_and_keep_light_summary() -> None:
    llm = _FakeSummaryAttachmentLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "last_executed_step": Step(
            id="step-1",
            title="生成最终文档",
            description="生成最终文档",
            objective_key="objective-step-1",
            success_criteria=["最终文档生成完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成最终文档",
                produced_artifacts=["/home/ubuntu/final-output.md"],
            ),
        ),
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成最终文档",
                    "produced_artifacts": ["/home/ubuntu/final-output.md"],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证最终交付正文与附件一并输出",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "selected_artifacts": ["/home/ubuntu/final-output.md"],
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最近一步结果的短摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["final_message"] == "最终总结"
    assert summarized_state["final_answer_text"] == "最终整理后的正文"
    assert summarized_state["selected_artifacts"] == ["/home/ubuntu/final-output.md"]
    message_event = summarized_state["emitted_events"][0]
    assert message_event.type == "message"
    assert message_event.stage == "final"
    assert message_event.message == "最终整理后的正文"
    assert [attachment.filepath for attachment in message_event.attachments] == ["/home/ubuntu/final-output.md"]


def test_summarize_should_not_expose_attachment_path_in_final_answer_text_by_default() -> None:
    llm = _FakeSummaryAttachmentPathDisclosureLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "调研 LangGraph Human-in-the-Loop 实现模式，给我总结。",
        "plan": _build_plan(),
        "execution_count": 2,
        "last_executed_step": Step(
            id="step-1",
            title="生成调研报告",
            description="生成调研报告",
            objective_key="objective-step-1",
            success_criteria=["报告生成完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成调研报告",
                produced_artifacts=["/home/ubuntu/langgraph_hitl_report.md"],
            ),
        ),
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成调研报告",
                    "produced_artifacts": ["/home/ubuntu/langgraph_hitl_report.md"],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证默认不暴露附件路径",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "selected_artifacts": ["/home/ubuntu/langgraph_hitl_report.md"],
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert "/home/ubuntu/langgraph_hitl_report.md" not in summarized_state["final_answer_text"]
    assert "完整内容已作为附件交付。" in summarized_state["final_answer_text"]
    message_event = summarized_state["emitted_events"][0]
    assert "/home/ubuntu/langgraph_hitl_report.md" not in message_event.message
    assert [attachment.filepath for attachment in message_event.attachments] == [
        "/home/ubuntu/langgraph_hitl_report.md"
    ]


def test_summarize_should_use_current_run_artifacts_as_attachment_truth_source() -> None:
    llm = _FakeLightSummaryLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成最终文档",
                    "produced_artifacts": [
                        "/home/ubuntu/final-output.md",
                        "/home/ubuntu/final-checklist.md",
                    ],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证当前运行产物附件",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "selected_artifacts": [],
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最近一步结果的短摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == []


def test_summarize_should_not_fallback_to_last_step_artifacts_without_final_delivery_source_refs() -> None:
    llm = _FakeUnknownSummaryAttachmentLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "整理成 md 文档",
        "plan": _build_plan(),
        "execution_count": 1,
        "last_executed_step": Step(
            id="step-1",
            title="生成文档",
            description="生成课程目录文档",
            objective_key="objective-step-1",
            success_criteria=["文档生成完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成真实附件",
                produced_artifacts=["/home/ubuntu/intermediate.md"],
            ),
        ),
        "selected_artifacts": ["/home/ubuntu/intermediate.md"],
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成真实附件",
                    "produced_artifacts": ["/home/ubuntu/intermediate.md"],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证模型返回未知附件时回退到当前运行附件",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == []


def test_summarize_should_filter_non_file_refs_from_final_delivery_source_refs() -> None:
    llm = _FakeLightSummaryLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成最终文档",
                    "produced_artifacts": [
                        "/home/ubuntu/final-output.md",
                        "/home/ubuntu/final-checklist.md",
                    ],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证最终附件过滤",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "selected_artifacts": [
            "artifact-id-1",
            "https://example.com/final.md",
            "final-output.md",
            "/home/ubuntu/final-output.md",
        ],
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最近一步结果的短摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == ["/home/ubuntu/final-output.md"]
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == ["/home/ubuntu/final-output.md"]


def test_summarize_should_not_emit_non_file_refs_as_attachments() -> None:
    llm = _FakeLightSummaryLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "step_states": [],
        "working_memory": {
            "goal": "验证非文件引用不会变成附件",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "selected_artifacts": ["artifact-id-2"],
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最近一步结果的短摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert message_event.attachments == []


def test_summarize_should_filter_final_delivery_refs_by_workspace_artifact_index() -> None:
    llm = _FakeLightSummaryLLM()
    runtime_context_service = _build_runtime_context_service_with_workspace_artifacts(
        "/home/ubuntu/final-output.md",
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": _build_plan(),
        "execution_count": 2,
        "step_states": [],
        "working_memory": {
            "goal": "验证 workspace artifact 过滤",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "selected_artifacts": [
            "/home/ubuntu/final-output.md",
            "/home/ubuntu/not-indexed.md",
        ],
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最近一步结果的短摘要",
    }

    summarized_state = asyncio.run(
        summarize_node(
            state,
            llm,
            runtime_context_service=runtime_context_service,
        )
    )

    assert summarized_state["selected_artifacts"] == ["/home/ubuntu/final-output.md"]
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == ["/home/ubuntu/final-output.md"]


def test_summarize_should_filter_explicit_summary_attachments_by_workspace_artifact_index() -> None:
    llm = _FakeSummaryAttachmentLLM()
    runtime_context_service = _build_runtime_context_service_with_workspace_artifacts(
        "/home/ubuntu/final-output.md",
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "整理成 md 文档",
        "plan": _build_plan(),
        "execution_count": 1,
        "last_executed_step": Step(
            id="step-1",
            title="生成文档",
            description="生成课程目录文档",
            objective_key="objective-step-1",
            success_criteria=["文档生成完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成中间文档",
                produced_artifacts=["/home/ubuntu/intermediate.md"],
            ),
        ),
        "selected_artifacts": ["/home/ubuntu/intermediate.md"],
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成中间文档与最终文档",
                    "produced_artifacts": [
                        "/home/ubuntu/intermediate.md",
                        "/home/ubuntu/final-output.md",
                    ],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证显式附件过滤",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(
        summarize_node(
            state,
            llm,
            runtime_context_service=runtime_context_service,
        )
    )

    # P3-一次性收口：显式 attachments 仅可选取真相源(source_refs)子集；未声明真相源时应全部丢弃。
    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == []


def test_summarize_should_keep_all_current_run_artifacts_when_falling_back() -> None:
    llm = _FakeUnknownSummaryAttachmentLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "整理成 md 文档",
        "plan": _build_plan(),
        "execution_count": 1,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成真实附件",
                    "produced_artifacts": [
                        "/home/ubuntu/final-output.md",
                        "/home/ubuntu/final-checklist.md",
                    ],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证当前 run 多附件回退",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "selected_artifacts": [],
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == []


def test_summarize_should_block_direct_wait_without_original_execution(monkeypatch) -> None:
    llm = _FakeLLM()
    captured_events = []

    async def _capture_events(*events):
        captured_events.extend(events)

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.emit_live_events",
        _capture_events,
    )

    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "继续",
        "plan": _build_plan(),
        "execution_count": 1,
        "step_states": [
            {
                "step_id": "direct-wait-confirm",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证 direct_wait 错误总结阻断",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {"control": _entry_contract_control("先让我确认后再继续搜索课程")},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["plan"].status == ExecutionStatus.COMPLETED
    assert summarized_state["final_message"] == "最终总结"
    assert not any(getattr(event, "type", "") == "error" for event in captured_events)
    assert any(
        getattr(event, "type", "") == "message"
        and getattr(event, "stage", "") == "final"
        for event in captured_events
    )


def test_summarize_should_not_block_after_direct_wait_cancel_and_replan(monkeypatch) -> None:
    llm = _FakeLLM()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: False,
    )

    waiting_state = asyncio.run(
        direct_wait_node(
            {
                "session_id": "session-1",
                "user_id": "user-1",
                "run_id": "run-1",
                "thread_id": "thread-1",
                "user_message": "先让我确认后再继续搜索课程",
                "graph_metadata": {"control": _entry_contract_control("先让我确认后再继续搜索课程")},
                "message_window": [],
                "working_memory": {},
                "execution_count": 0,
            }
        )
    )
    cancelled_state = asyncio.run(wait_for_human_node(waiting_state))
    replanned_state = asyncio.run(
        replan_node(
            cancelled_state,
            _FakeReplanLLM(
                steps=[
                    {
                        "title": "按新的计划继续执行",
                        "description": "按新的计划继续执行",
                    }
                ]
            ),
        )
    )
    replanned_plan = _build_plan()
    replanned_plan.steps[0].status = ExecutionStatus.COMPLETED
    replanned_plan.steps[0].outcome = StepOutcome(done=True, summary="重规划后的步骤已完成")
    replanned_state["plan"] = replanned_plan
    replanned_state["user_message"] = "按新的计划继续执行"
    replanned_state["final_message"] = "重规划后的步骤已完成"

    summarized_state = asyncio.run(summarize_node(replanned_state, llm))

    control = (summarized_state.get("graph_metadata") or {}).get("control") or {}
    assert summarized_state["plan"].status == ExecutionStatus.COMPLETED
    assert summarized_state["final_message"] == "最终总结"
    assert "wait_resume_action" not in control
    assert "entry_strategy" not in control


def test_consolidate_memory_should_trim_message_window_and_update_summary() -> None:
    long_final_message = "最终总结" + ("x" * 700)
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请整理上下文",
        "plan": _build_plan(),
        "execution_count": 1,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证消息压缩策略",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "selected_artifacts": [f"/tmp/artifact-{index}.md" for index in range(12)],
        "pending_memory_writes": [],
        "message_window": [
            {"role": "user", "message": f"历史消息 {index}", "attachment_paths": []}
            for index in range(105)
        ],
        "conversation_summary": "已有历史摘要",
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": long_final_message,
    }

    consolidated_state = asyncio.run(_consolidate_memory_node(state))

    assert len(consolidated_state["message_window"]) == 100
    assert "裁剪:6条消息" in consolidated_state["conversation_summary"]
    assert consolidated_state["message_window"][-1]["message"] == long_final_message[:500]
    assert len(consolidated_state["message_window"][-1]["attachment_paths"]) == 8


def test_consolidate_memory_should_govern_candidates_before_persisting() -> None:
    repository = _FakeLongTermMemoryRepository()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请整理记忆",
        "plan": _build_plan(),
        "execution_count": 1,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证候选治理",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "用户偏好",
                "content": {"language": "zh"},
                "tags": ["language"],
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "用户偏好补充",
                "content": {"response_style": "concise"},
                "tags": ["response_style"],
                "confidence": 0.7,
            },
            {
                "namespace": "agent/planner_react/instruction",
                "memory_type": "instruction",
                "summary": "低置信度候选",
                "content": {"text": "仅供参考"},
                "tags": ["low-confidence"],
                "confidence": 0.1,
            },
            {
                "namespace": "session/session-1/fact",
                "memory_type": "fact",
                "summary": "",
                "content": {},
                "confidence": 0.8,
            },
            {
                "namespace": "session/session-1/fact",
                "memory_type": "fact",
                "summary": "当前任务只关注 backend",
                "content": {"text": "当前任务只关注 backend"},
                "tags": ["fact"],
                "confidence": 0.6,
            },
        ],
        "message_window": [],
        "conversation_summary": "",
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最终总结",
    }

    consolidated_state = asyncio.run(
        _consolidate_memory_node(
            state,
            long_term_memory_repository=repository,
        )
    )

    assert len(repository.upserted) == 2
    persisted_profile = next(item for item in repository.upserted if item.memory_type == "profile")
    persisted_fact = next(item for item in repository.upserted if item.memory_type == "fact")
    assert persisted_profile.content == {
        "language": "zh",
        "response_style": "concise",
    }
    assert persisted_fact.content == {"text": "当前任务只关注 backend"}
    assert consolidated_state["pending_memory_writes"] == []


def test_consolidate_memory_should_redact_pii_and_write_memory_governance_fields() -> None:
    repository = _FakeLongTermMemoryRepository()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请整理记忆",
        "plan": _build_plan(),
        "execution_count": 1,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证隐私治理",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "用户邮箱 test@example.com",
                "content": {"email": "test@example.com", "phone": "13812345678"},
                "tags": ["contact"],
                "confidence": 0.9,
                "source": {"kind": "summary"},
            },
        ],
        "message_window": [],
        "conversation_summary": "",
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最终总结",
    }

    consolidated_state = asyncio.run(
        _consolidate_memory_node(
            state,
            long_term_memory_repository=repository,
        )
    )

    assert consolidated_state["pending_memory_writes"] == []
    assert len(repository.upserted) == 1
    persisted_memory = repository.upserted[0]
    assert persisted_memory.user_id == "user-1"
    assert persisted_memory.tenant_id == "user-1"
    assert persisted_memory.source == {"kind": "summary"}
    assert persisted_memory.privacy_level == PrivacyLevel.SENSITIVE
    assert persisted_memory.trust_level == DataTrustLevel.SYSTEM_GENERATED
    assert persisted_memory.retention_policy == RetentionPolicyKind.USER_MEMORY
    assert "test@example.com" not in persisted_memory.summary
    assert persisted_memory.content == {
        "email": "[REDACTED_EMAIL]",
        "phone": "[REDACTED_PHONE]",
    }


def test_consolidate_memory_should_use_injected_data_retention_policy_service() -> None:
    repository = _FakeLongTermMemoryRepository()
    policy_service = _FakeDataRetentionPolicyService(
        tenant_id="tenant-from-policy",
        trust_level=DataTrustLevel.EXTERNAL_UNTRUSTED,
        privacy_level=PrivacyLevel.INTERNAL,
        retention_policy=RetentionPolicyKind.SESSION_BOUND,
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请整理记忆",
        "plan": _build_plan(),
        "working_memory": {"goal": "验证分类策略注入"},
        "pending_memory_writes": [
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "策略注入事实",
                "content": {"text": "策略注入事实"},
                "confidence": 0.9,
            },
        ],
        "message_window": [],
        "conversation_summary": "",
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最终总结",
    }

    asyncio.run(
        _consolidate_memory_node(
            state,
            long_term_memory_repository=repository,
            data_retention_policy_service=policy_service,
        )
    )

    assert len(policy_service.calls) == 1
    assert policy_service.calls[0]["tenant_id"] == "user-1"
    assert policy_service.calls[0]["origin"] == DataOrigin.LONG_TERM_MEMORY
    persisted_memory = repository.upserted[0]
    assert persisted_memory.tenant_id == "tenant-from-policy"
    assert persisted_memory.trust_level == DataTrustLevel.EXTERNAL_UNTRUSTED
    assert persisted_memory.privacy_level == PrivacyLevel.INTERNAL
    assert persisted_memory.retention_policy == RetentionPolicyKind.SESSION_BOUND


def test_consolidate_memory_should_not_write_plain_secret_memory_body() -> None:
    repository = _FakeLongTermMemoryRepository()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请整理记忆",
        "plan": _build_plan(),
        "execution_count": 1,
        "step_states": [],
        "working_memory": {
            "goal": "验证 secret 拒写",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "用户 API key",
                "content": {"text": "api_key=abcdefghi123456789"},
                "confidence": 0.9,
            },
        ],
        "message_window": [],
        "conversation_summary": "",
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最终总结",
    }

    consolidated_state = asyncio.run(
        _consolidate_memory_node(
            state,
            long_term_memory_repository=repository,
        )
    )

    assert repository.upserted == []
    assert consolidated_state["pending_memory_writes"] == []


def test_summarize_should_ignore_structured_memory_extraction_when_working_memory_empty() -> None:
    llm = _FakeStructuredMemoryLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "后续任务只看 backend，并用中文简洁回复",
        "plan": _build_plan(),
        "last_executed_step": Step(
            id="step-1",
            title="整理 backend 结论",
            description="整理 backend 结论",
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="当前任务后续都只需要关注 backend",
                facts_learned=["后续仅关注 backend。"],
            ),
        ),
        "execution_count": 1,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证长期记忆提炼",
            "decisions": ["已经完成分析并准备总结"],
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["working_memory"]["facts_in_session"] == []
    assert summarized_state["working_memory"]["user_preferences"] == {}
    assert summarized_state["pending_memory_writes"] == []


def test_summarize_should_ignore_memory_fields_even_when_model_returns_them() -> None:
    class _HallucinatedMemoryLLM:
        async def invoke(self, messages, tools, response_format):
            return {
                "content": json.dumps(
                    {
                        "message": "总结完成",
                        "attachments": [],
                        "facts_in_session": ["项目已经迁移到 golang"],
                        "user_preferences": {"budget": "5000"},
                        "memory_candidates": [
                            {
                                "memory_type": "fact",
                                "summary": "项目已经迁移到 golang",
                                "content": {"text": "项目已经迁移到 golang"},
                                "confidence": 0.9,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }

    llm = _HallucinatedMemoryLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请总结当前任务",
        "plan": _build_plan(),
        "last_executed_step": Step(
            id="step-1",
            title="整理 Python 结果",
            description="整理 Python 结果",
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="当前项目仍是 Python 后端",
            ),
        ),
        "execution_count": 1,
        "working_memory": {
            "goal": "验证证据过滤",
            "decisions": [],
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["working_memory"]["facts_in_session"] == []
    assert summarized_state["working_memory"]["user_preferences"] == {}
    assert summarized_state["pending_memory_writes"] == []


def test_summarize_should_not_fallback_to_task_outcome_memory_candidate() -> None:
    llm = _FakeLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "帮我完成任务",
        "plan": _build_plan(),
        "last_executed_step": Step(
            id="step-1",
            title="收敛任务结果",
            description="收敛任务结果",
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="最终总结",
            ),
        ),
        "execution_count": 1,
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证兜底候选生成",
            "decisions": ["已经完成任务并准备输出最终结果"],
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["pending_memory_writes"] == []


def test_summarize_should_not_fallback_to_last_step_artifacts_when_model_returns_empty_attachments() -> None:
    llm = _FakeLLM()
    last_executed_step = Step(
        id="step-1",
        title="生成文档",
        description="生成课程目录文档",
        objective_key="objective-step-1",
        success_criteria=["文档生成完成"],
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(
            done=True,
            summary="已生成课程目录文档",
            produced_artifacts=["/home/ubuntu/course_directory.md"],
        ),
    )
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "整理成 md 文档",
        "plan": _build_plan(),
        "execution_count": 1,
        "last_executed_step": last_executed_step,
        "selected_artifacts": ["artifact-id-1", "/home/ubuntu/course_directory.md"],
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证总结附件回填",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == []


def test_summarize_should_not_fallback_attachments_when_last_step_disables_attachment_delivery() -> None:
    llm = _FakeLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "创建文档但不要作为最终附件交付",
        "plan": _build_plan(),
        "execution_count": 1,
        "last_executed_step": Step(
            id="step-1",
            title="生成文档",
            description="生成课程目录文档",
            objective_key="objective-step-1",
            success_criteria=["文档生成完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成课程目录文档",
                produced_artifacts=["/home/ubuntu/course_directory.md"],
                deliver_result_as_attachment=False,
            ),
        ),
        "selected_artifacts": ["/home/ubuntu/course_directory.md"],
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "working_memory": {
            "goal": "验证总结附件硬门禁",
            "user_preferences": {},
            "facts_in_session": [],
            "delivery_controls": {
                "source_step_id": "step-1",
                "deliver_result_as_attachment": False,
            },
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert message_event.attachments == []


def test_summarize_should_ignore_failed_step_artifacts_when_resolving_summary_attachments() -> None:
    llm = _FakeUnknownSummaryAttachmentLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "整理成 md 文档",
        "plan": _build_plan(),
        "execution_count": 1,
        "last_executed_step": Step(
            id="step-2",
            title="失败步骤",
            description="失败步骤",
            objective_key="objective-step-2",
            success_criteria=["失败步骤完成"],
            status=ExecutionStatus.FAILED,
            outcome=StepOutcome(
                done=False,
                summary="执行失败",
                produced_artifacts=["/home/ubuntu/sensitive-temp.json"],
            ),
        ),
        "selected_artifacts": [
            "/home/ubuntu/sensitive-temp.json",
            "/home/ubuntu/final-output.md",
        ],
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "成功步骤",
                    "produced_artifacts": ["/home/ubuntu/final-output.md"],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            },
            {
                "step_id": "step-2",
                "status": ExecutionStatus.FAILED.value,
                "outcome": {
                    "done": False,
                    "summary": "失败步骤",
                    "produced_artifacts": ["/home/ubuntu/sensitive-temp.json"],
                    "blockers": ["失败"],
                    "facts_learned": [],
                    "open_questions": [],
                },
            },
        ],
        "working_memory": {
            "goal": "验证失败产物隔离",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == []
    assert all(
        attachment.filepath != "/home/ubuntu/sensitive-temp.json"
        for attachment in message_event.attachments
    )


class _FakeSummaryAttachmentLLM:
    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "message": "最终总结",
                    "final_answer_text": "最终整理后的正文",
                    "attachments": ["/home/ubuntu/final-output.md"],
                    "facts_in_session": [],
                    "user_preferences": {},
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        }


def test_summarize_should_prefer_explicit_summary_attachments_over_previous_artifacts() -> None:
    llm = _FakeSummaryAttachmentLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "整理成 md 文档",
        "plan": _build_plan(),
        "execution_count": 1,
        "last_executed_step": Step(
            id="step-1",
            title="生成文档",
            description="生成课程目录文档",
            objective_key="objective-step-1",
            success_criteria=["文档生成完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成中间文档",
                produced_artifacts=["/home/ubuntu/intermediate.md"],
            ),
        ),
        "selected_artifacts": ["/home/ubuntu/intermediate.md", "/home/ubuntu/older.md"],
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成中间文档与最终文档",
                    "produced_artifacts": [
                        "/home/ubuntu/intermediate.md",
                        "/home/ubuntu/final-output.md",
                    ],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证总结附件优先级",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == []


class _FakeUnknownSummaryAttachmentLLM:
    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "message": "最终总结",
                    "final_answer_text": "最终整理后的正文",
                    "attachments": ["/home/ubuntu/non-existent-final-output.md"],
                    "facts_in_session": [],
                    "user_preferences": {},
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        }


def test_summarize_should_fallback_when_model_returns_unknown_summary_attachment() -> None:
    llm = _FakeUnknownSummaryAttachmentLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "整理成 md 文档",
        "plan": _build_plan(),
        "execution_count": 1,
        "last_executed_step": Step(
            id="step-1",
            title="生成文档",
            description="生成课程目录文档",
            objective_key="objective-step-1",
            success_criteria=["文档生成完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成真实附件",
                produced_artifacts=["/home/ubuntu/intermediate.md"],
            ),
        ),
        "selected_artifacts": ["/home/ubuntu/intermediate.md"],
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
                "outcome": {
                    "done": True,
                    "summary": "已生成真实附件",
                    "produced_artifacts": ["/home/ubuntu/intermediate.md"],
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证总结附件未知路径回退",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == []
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == []


def test_planner_react_graph_should_only_inject_repository_into_boundary_nodes(monkeypatch) -> None:
    repository = _FakeLongTermMemoryRepository()

    def _append_trace(state: dict, marker: str) -> dict:
        working_memory = dict(state.get("working_memory") or {})
        trace = list(working_memory.get("trace") or [])
        trace.append(marker)
        working_memory["trace"] = trace
        return {
            **state,
            "working_memory": working_memory,
        }

    async def _recall(state, long_term_memory_repository=None):
        assert long_term_memory_repository is repository
        return _append_trace(state, "recall")

    async def _entry_router(state):
        next_state = _append_trace(state, "entry_router")
        graph_metadata = {
            "control": _planned_entry_contract_control("验证长期记忆边界"),
        }
        return {
            **next_state,
            "graph_metadata": graph_metadata,
        }

    async def _plan(state, _llm, runtime_context_service=None):
        plan = Plan(
            title="记忆阶段测试",
            goal="验证长期记忆边界",
            language="zh",
            message="开始执行",
            steps=[
                Step(
                    id="step-1",
                    title="执行阶段一",
                    description="执行阶段一",
                    objective_key="objective-step-1",
                    success_criteria=["执行阶段一完成"],
                    status=ExecutionStatus.PENDING,
                ),
                Step(
                    id="step-2",
                    title="执行阶段二",
                    description="执行阶段二",
                    objective_key="objective-step-2",
                    success_criteria=["执行阶段二完成"],
                    status=ExecutionStatus.PENDING,
                ),
            ],
        )
        next_state = _append_trace(state, "plan")
        return {
            **next_state,
            "plan": plan,
            "current_step_id": "step-1",
        }

    async def _execute(
            state,
            _llm,
            runtime_context_service=None,
            skill_runtime=None,
            runtime_tools=None,
            max_tool_iterations=5,
    ):
        plan = state["plan"].model_copy(deep=True)
        current_step = next(step for step in plan.steps if not step.done)
        current_step.status = ExecutionStatus.COMPLETED
        current_step.outcome = StepOutcome(
            done=True,
            summary="步骤执行完成",
        )
        next_step = plan.get_next_step()
        next_state = _append_trace(state, "execute")
        return {
            **next_state,
            "plan": plan,
            "last_executed_step": current_step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
        }

    async def _guard(state):
        return _append_trace(state, "guard")

    async def _replan(state, _llm, runtime_context_service=None):
        next_state = _append_trace(state, "replan")
        return {
            **next_state,
            "plan": state["plan"],
            "current_step_id": None,
        }

    async def _summarize(state, _llm, runtime_context_service=None):
        next_state = _append_trace(state, "summarize")
        return {
            **next_state,
            "final_message": "最终总结",
        }

    async def _consolidate(
            state,
            long_term_memory_repository=None,
            memory_consolidation_service=None,
            data_retention_policy_service=None,
    ):
        assert long_term_memory_repository is repository
        assert memory_consolidation_service is None
        assert data_retention_policy_service is not None
        assert callable(data_retention_policy_service.classify_data)
        return _append_trace(state, "consolidate")

    async def _finalize(state):
        return _append_trace(state, "finalize")

    async def _direct_answer(state, _llm, runtime_context_service=None):
        return _append_trace(state, "direct_answer")

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.entry_router_node",
        _entry_router,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.recall_memory_context_node",
        _recall,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.direct_answer_node",
        _direct_answer,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.direct_wait_node",
        lambda state: state,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.atomic_action_node",
        lambda state: state,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.create_or_reuse_plan_node",
        _plan,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.execute_step_node",
        _execute,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.guard_step_reuse_node",
        _guard,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.replan_node",
        _replan,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.summarize_node",
        _summarize,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.consolidate_memory_node",
        _consolidate,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.graph.finalize_node",
        _finalize,
    )

    graph = build_planner_react_langgraph_graph(
        stage_llms={
            "router": object(),
            "planner": object(),
            "executor": object(),
            "replan": object(),
            "summary": object(),
        },
        long_term_memory_repository=repository,
    )

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": "请规划并分阶段整理记忆任务",
                "graph_metadata": {},
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())

    assert state["working_memory"]["trace"] == [
        "entry_router",
        "recall",
        "plan",
        "guard",
        "execute",
        "guard",
        "execute",
        "replan",
        "summarize",
        "consolidate",
        "finalize",
    ]


def test_graph_state_contract_should_include_user_id_in_runtime_metadata() -> None:
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "retrieved_memories": [{"id": "mem-0", "summary": "历史记忆"}],
        "pending_memory_writes": [{"id": "mem-1"}, {"id": "mem-2"}],
        "graph_metadata": {},
        "message_window": [],
        "conversation_summary": "",
        "working_memory": {},
        "execution_count": 0,
        "max_execution_steps": 20,
        "step_states": [],
        "pending_interrupt": {},
        "emitted_events": [],
    }

    runtime_metadata = GraphStateContractMapper.build_runtime_metadata(state)

    assert runtime_metadata["graph_state_contract"]["graph_state"]["user_id"] == "user-1"
    assert runtime_metadata["memory"]["recall_count"] == 1
    assert runtime_metadata["memory"]["recall_ids"] == ["mem-0"]
    assert runtime_metadata["memory"]["write_count"] == 2
    assert runtime_metadata["memory"]["write_ids"] == ["mem-1", "mem-2"]
