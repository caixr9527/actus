import asyncio
import json

from app.domain.models import (
    ExecutionStatus,
    LongTermMemory,
    LongTermMemorySearchMode,
    LongTermMemorySearchQuery,
    Plan,
    Step,
    StepOutcome,
    ToolResult,
)
from app.domain.services.tools import BaseTool, MessageTool
from app.domain.services.tools.base import tool
from app.domain.services.runtime.langgraph_state import GraphStateContractMapper
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph import (
    build_planner_react_langgraph_graph,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes import (
    _build_execution_context_block,
    consolidate_memory_node,
    direct_wait_node,
    execute_step_node,
    guard_step_reuse_node,
    recall_memory_context_node,
    replan_node,
    summarize_node,
    wait_for_human_node,
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
                    "attachments": [],
                    "facts_in_session": [],
                    "user_preferences": {},
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        }


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


class _FakeToolLoopLLM:
    def __init__(self) -> None:
        self.tool_name_snapshots: list[list[str]] = []

    async def invoke(self, messages, tools, tool_choice=None, response_format=None):
        tool_names = [str((tool.get("function") or {}).get("name") or "") for tool in list(tools or [])]
        self.tool_name_snapshots.append(tool_names)
        call_index = len(self.tool_name_snapshots) - 1
        if call_index == 0:
            assert "message_notify_user" in tool_names
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-notify",
                        "function": {
                            "name": "message_notify_user",
                            "arguments": json.dumps({"text": "开始生成报告"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        if call_index == 1:
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
                    id="mem-1",
                    namespace="user/user-1/profile",
                    memory_type="profile",
                    summary="用户偏好中文",
                    content={"language": "zh", "style": "concise"},
                )
            ],
            "instruction": [
                LongTermMemory(
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
    assert next_state["working_memory"]["user_preferences"] == {
        "language": "zh",
        "style": "concise",
    }


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
    assert next_state["final_message"] == "步骤执行失败：执行阶段"
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
        "artifact_refs": [],
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
    assert next_state["artifact_refs"] == ["/home/ubuntu/report.md"]
    assert llm.tool_name_snapshots[0].count("message_notify_user") == 1
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
    assert next_state["artifact_refs"] == ["/tmp/report.md"]
    assert next_state["working_memory"]["facts_in_session"] == ["报告结构已确定"]


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


def test_execution_context_block_should_separate_current_and_historical_context() -> None:
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
        "historical_artifact_refs": [f"/tmp/history-{index}.md" for index in range(12)],
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
    }

    context_block = _build_execution_context_block(state)

    assert context_block["selected_artifacts"] == [f"/tmp/current-{index}.md" for index in range(6)]
    assert context_block["current_run_artifacts"] == [f"/tmp/generated-{index}.md" for index in range(2, 8)]
    assert context_block["historical_artifact_refs"] == [f"/tmp/history-{index}.md" for index in range(6)]
    assert len(context_block["recent_run_briefs"]) == 3
    assert len(context_block["recent_attempt_briefs"]) == 3
    assert "总结0总结0" in context_block["recent_run_briefs"][0]["final_answer_summary"]
    assert len(context_block["recent_run_briefs"][0]["final_answer_summary"]) <= 120
    assert "重交付正文摘录0" in context_block["recent_run_briefs"][0]["final_answer_text_excerpt"]
    assert len(context_block["recent_run_briefs"][0]["final_answer_text_excerpt"]) <= 160
    assert context_block["blockers"] == ["阻塞1"]
    assert context_block["open_questions"] == ["问题1", "问题2", "问题3"]
    assert context_block["recent_messages"] == [
        {"role": "user", "message": "第一轮问题"},
        {"role": "assistant", "message": "第一轮回答"},
    ]
    assert context_block["retrieved_memories"][0]["memory_type"] == "profile"
    assert context_block["retrieved_memories"][0]["summary"] == "偏好中文"
    assert context_block["retrieved_memories"][0]["content_preview"] == "language: zh"
    assert "embedding" not in context_block["retrieved_memories"][0]


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


def test_summarize_and_consolidate_should_generate_and_persist_memory_candidates() -> None:
    repository = _FakeLongTermMemoryRepository()
    llm = _FakeLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "帮我完成总结",
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
        "pending_memory_writes": [],
        "message_window": [
            {"role": "user", "message": "帮我完成总结", "attachment_paths": []},
        ],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert len(summarized_state["pending_memory_writes"]) == 2
    assert summarized_state["selected_artifacts"] == []

    consolidated_state = asyncio.run(
        consolidate_memory_node(
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
            "final_delivery_payload": {
                "text": "这是一段需要压缩后再喂给总结模型的最终交付正文。",
                "sections": [
                    {
                        "title": "行程建议",
                        "content": "第一天先去古城墙，第二天安排博物馆。",
                    }
                ],
                "source_refs": ["/home/ubuntu/final.md"],
            },
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    asyncio.run(summarize_node(state, llm))

    assert "计划摘要快照(JSON)" in llm.last_prompt
    assert "最终交付载荷(JSON)" in llm.last_prompt
    assert '"completed_step_summaries"' in llm.last_prompt
    assert '"selected_artifacts": ["/home/ubuntu/final.md"]' in llm.last_prompt
    assert '"objective_key"' not in llm.last_prompt
    assert '"success_criteria"' not in llm.last_prompt


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
            "final_delivery_payload": {
                "text": "这是用户最终应该看到的完整攻略正文。",
                "sections": [],
                "source_refs": [],
            },
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最近一步结果的短摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["final_message"] == "轻量总结"
    message_event = summarized_state["emitted_events"][0]
    assert message_event.type == "message"
    assert message_event.stage == "final"
    assert message_event.message == "这是用户最终应该看到的完整攻略正文。"


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
            "final_delivery_payload": {
                "text": "这是用户最终应该看到的完整攻略正文。",
                "sections": [],
                "source_refs": ["/home/ubuntu/final-output.md"],
            },
        },
        "selected_artifacts": ["/home/ubuntu/final-output.md"],
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "artifact_refs": ["/home/ubuntu/final-output.md"],
        "emitted_events": [],
        "final_message": "最近一步结果的短摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["final_message"] == "最终总结"
    assert summarized_state["selected_artifacts"] == ["/home/ubuntu/final-output.md"]
    message_event = summarized_state["emitted_events"][0]
    assert message_event.type == "message"
    assert message_event.stage == "final"
    assert message_event.message == "这是用户最终应该看到的完整攻略正文。"
    assert [attachment.filepath for attachment in message_event.attachments] == ["/home/ubuntu/final-output.md"]


def test_summarize_should_use_final_delivery_source_refs_as_attachment_truth_source() -> None:
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
            "goal": "验证最终交付来源附件",
            "user_preferences": {},
            "facts_in_session": [],
            "final_delivery_payload": {
                "text": "这是用户最终应该看到的完整攻略正文。",
                "sections": [],
                "source_refs": [
                    "/home/ubuntu/final-output.md",
                    "/home/ubuntu/final-checklist.md",
                ],
            },
        },
        "selected_artifacts": [],
        "artifact_refs": [],
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "最近一步结果的短摘要",
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == [
        "/home/ubuntu/final-output.md",
        "/home/ubuntu/final-checklist.md",
    ]
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == [
        "/home/ubuntu/final-output.md",
        "/home/ubuntu/final-checklist.md",
    ]


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
        "step_states": [],
        "working_memory": {
            "goal": "验证最终附件过滤",
            "user_preferences": {},
            "facts_in_session": [],
            "final_delivery_payload": {
                "text": "这是用户最终应该看到的完整攻略正文。",
                "sections": [],
                "source_refs": [
                    "artifact-id-1",
                    "https://example.com/final.md",
                    "final-output.md",
                    "/home/ubuntu/final-output.md",
                ],
            },
        },
        "selected_artifacts": [],
        "artifact_refs": ["artifact-id-1", "https://example.com/final.md", "/home/ubuntu/final-output.md"],
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
            "final_delivery_payload": {
                "text": "这是用户最终应该看到的完整攻略正文。",
                "sections": [],
                "source_refs": [
                    "artifact-id-1",
                    "https://example.com/final.md",
                    "final-output.md",
                ],
            },
        },
        "selected_artifacts": ["artifact-id-2"],
        "artifact_refs": ["artifact-id-1", "https://example.com/final.md"],
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
        "artifact_refs": [],
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == [
        "/home/ubuntu/final-output.md",
        "/home/ubuntu/final-checklist.md",
    ]
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == [
        "/home/ubuntu/final-output.md",
        "/home/ubuntu/final-checklist.md",
    ]


def test_summarize_should_block_direct_wait_without_original_execution(monkeypatch) -> None:
    llm = _FailIfCalledSummaryLLM()
    captured_events = []

    async def _capture_events(*events):
        captured_events.extend(events)

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes.emit_live_events",
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
        "graph_metadata": {
            "control": {
                "entry_strategy": "direct_wait",
                "direct_wait_original_task_executed": False,
            }
        },
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert llm.calls == 0
    assert summarized_state["plan"].status == ExecutionStatus.FAILED
    assert summarized_state["plan"].error == "运行时异常：direct_wait 已完成确认，但原始任务尚未执行，已阻止错误总结。"
    assert summarized_state["final_message"] == "运行时异常：direct_wait 已完成确认，但原始任务尚未执行，已阻止错误总结。"
    assert any(getattr(event, "type", "") == "error" for event in captured_events)
    assert any(
        getattr(event, "type", "") == "error"
        and getattr(event, "error_key", "") == "direct_wait_unexecuted"
        for event in captured_events
    )
    assert any(
        getattr(event, "type", "") == "message"
        and getattr(event, "stage", "") == "final"
        for event in captured_events
    )


def test_summarize_should_not_block_after_direct_wait_cancel_and_replan(monkeypatch) -> None:
    llm = _FakeLLM()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes.interrupt",
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
                "graph_metadata": {},
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

    consolidated_state = asyncio.run(consolidate_memory_node(state))

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
        consolidate_memory_node(
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


def test_summarize_should_generate_candidates_from_structured_extraction_when_working_memory_empty() -> None:
    llm = _FakeStructuredMemoryLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "后续任务只看 backend，并用中文简洁回复",
        "plan": _build_plan(),
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

    assert summarized_state["working_memory"]["facts_in_session"] == [
        "当前任务后续都只需要关注 backend"
    ]
    assert summarized_state["working_memory"]["user_preferences"] == {
        "language": "zh",
        "response_style": "concise",
    }
    assert len(summarized_state["pending_memory_writes"]) == 3
    assert {item["memory_type"] for item in summarized_state["pending_memory_writes"]} == {
        "profile",
        "fact",
        "instruction",
    }


def test_summarize_should_fallback_to_task_outcome_candidate_when_no_structured_memory_available() -> None:
    llm = _FakeLLM()
    state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "帮我完成任务",
        "plan": _build_plan(),
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

    assert len(summarized_state["pending_memory_writes"]) == 1
    assert summarized_state["pending_memory_writes"][0]["memory_type"] == "fact"
    assert summarized_state["pending_memory_writes"][0]["tags"] == ["task_outcome"]


def test_summarize_should_fallback_to_last_step_artifacts_when_model_returns_empty_attachments() -> None:
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

    assert summarized_state["selected_artifacts"] == ["/home/ubuntu/course_directory.md"]
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == ["/home/ubuntu/course_directory.md"]


class _FakeSummaryAttachmentLLM:
    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "message": "最终总结",
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
        "artifact_refs": ["/home/ubuntu/final-output.md"],
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert summarized_state["selected_artifacts"] == ["/home/ubuntu/final-output.md"]
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == ["/home/ubuntu/final-output.md"]


class _FakeUnknownSummaryAttachmentLLM:
    async def invoke(self, messages, tools, response_format):
        return {
            "content": json.dumps(
                {
                    "message": "最终总结",
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

    assert summarized_state["selected_artifacts"] == ["/home/ubuntu/intermediate.md"]
    message_event = summarized_state["emitted_events"][0]
    assert [attachment.filepath for attachment in message_event.attachments] == ["/home/ubuntu/intermediate.md"]


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
            "control": {
                "entry_strategy": "recall_memory_context",
            }
        }
        return {
            **next_state,
            "graph_metadata": graph_metadata,
        }

    async def _plan(state, _llm):
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

    async def _execute(state, _llm, skill_runtime=None, runtime_tools=None, max_tool_iterations=5):
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

    async def _replan(state, _llm):
        next_state = _append_trace(state, "replan")
        return {
            **next_state,
            "plan": state["plan"],
            "current_step_id": None,
        }

    async def _summarize(state, _llm):
        next_state = _append_trace(state, "summarize")
        return {
            **next_state,
            "final_message": "最终总结",
        }

    async def _consolidate(state, long_term_memory_repository=None):
        assert long_term_memory_repository is repository
        return _append_trace(state, "consolidate")

    async def _finalize(state):
        return _append_trace(state, "finalize")

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.entry_router_node",
        _entry_router,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.recall_memory_context_node",
        _recall,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.direct_answer_node",
        lambda state, llm: state,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.direct_wait_node",
        lambda state: state,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.direct_execute_node",
        lambda state: state,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.create_or_reuse_plan_node",
        _plan,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.execute_step_node",
        _execute,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.guard_step_reuse_node",
        _guard,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.replan_node",
        _replan,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.summarize_node",
        _summarize,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.consolidate_memory_node",
        _consolidate,
    )
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.finalize_node",
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
        "artifact_refs": [],
        "emitted_events": [],
    }

    runtime_metadata = GraphStateContractMapper.build_runtime_metadata(state)

    assert runtime_metadata["graph_state_contract"]["graph_state"]["user_id"] == "user-1"
    assert runtime_metadata["memory"]["recall_count"] == 1
    assert runtime_metadata["memory"]["recall_ids"] == ["mem-0"]
    assert runtime_metadata["memory"]["write_count"] == 2
    assert runtime_metadata["memory"]["write_ids"] == ["mem-1", "mem-2"]
