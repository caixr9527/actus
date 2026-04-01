import asyncio
import json

from app.domain.models import ExecutionStatus, LongTermMemory, Plan, Step
from app.domain.services.runtime.langgraph_state import GraphStateContractMapper
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph import (
    build_planner_react_langgraph_graph,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes import (
    consolidate_memory_node,
    execute_step_node,
    recall_memory_context_node,
    replan_node,
    summarize_node,
)


class _FakeLongTermMemoryRepository:
    def __init__(self, search_results=None) -> None:
        self.search_results = list(search_results or [])
        self.search_calls = []
        self.upserted = []

    async def search(
            self,
            *,
            namespace_prefixes,
            query="",
            limit=10,
            memory_types=None,
            tags=None,
    ):
        self.search_calls.append(
            {
                "namespace_prefixes": list(namespace_prefixes),
                "query": query,
                "limit": limit,
                "memory_types": list(memory_types or []),
                "tags": list(tags or []),
            }
        )
        return list(self.search_results)

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


class _FakeSkillRuntime:
    async def execute_skill(self, skill_id, payload):
        return type(
            "SkillResult",
            (),
            {
                "success": True,
                "result": None,
                "attachments": [],
            },
        )()


def _build_plan(*, step_status: ExecutionStatus = ExecutionStatus.COMPLETED) -> Plan:
    return Plan(
        title="记忆阶段测试",
        goal="验证长期记忆边界",
        language="zh",
        message="开始执行",
        steps=[
            Step(
                id="step-1",
                description="执行阶段",
                status=step_status,
                success=step_status == ExecutionStatus.COMPLETED,
                result="步骤完成" if step_status == ExecutionStatus.COMPLETED else None,
            )
        ],
        status=ExecutionStatus.PENDING,
    )


def test_recall_memory_context_node_should_search_long_term_memory() -> None:
    repository = _FakeLongTermMemoryRepository(
        search_results=[
            LongTermMemory(
                id="mem-1",
                namespace="user/user-1/profile",
                memory_type="profile",
                summary="用户偏好中文",
                content={"language": "zh", "style": "concise"},
            )
        ]
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

    assert repository.search_calls[0]["namespace_prefixes"] == [
        "user/user-1/",
        "session/session-1/",
        "agent/planner_react/",
    ]
    assert "帮我整理长期记忆" in repository.search_calls[0]["query"]
    assert next_state["retrieved_memories"][0]["id"] == "mem-1"
    assert next_state["working_memory"]["user_preferences"] == {
        "language": "zh",
        "style": "concise",
    }
    assert next_state["graph_metadata"]["memory_recall_count"] == 1


def test_execute_step_node_should_not_write_string_none_when_skill_result_missing() -> None:
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
                    description="执行阶段",
                    status=ExecutionStatus.PENDING,
                )
            ],
        ),
        "input_parts": [],
        "working_memory": {},
        "step_local_memory": {},
        "graph_metadata": {},
        "emitted_events": [],
    }

    next_state = asyncio.run(
        execute_step_node(
            state,
            llm=object(),
            skill_runtime=_FakeSkillRuntime(),
        )
    )

    assert next_state["plan"].steps[0].result == "已完成步骤：执行阶段"
    assert next_state["final_message"] == "已完成步骤：执行阶段"
    assert next_state["working_memory"]["decisions"] == ["已完成步骤：执行阶段"]


def test_replan_node_should_regenerate_conflicting_step_ids_without_numeric_assumption() -> None:
    completed_step = Step(
        id="step-a",
        description="完成已有步骤",
        status=ExecutionStatus.COMPLETED,
        success=True,
        result="已完成",
    )
    pending_step = Step(
        id="step-b",
        description="待执行步骤",
        status=ExecutionStatus.PENDING,
    )
    plan = Plan(
        title="重规划测试",
        goal="验证 step_id 去重",
        language="zh",
        message="开始重规划",
        steps=[completed_step, pending_step],
    )
    state = {
        "plan": plan,
        "last_executed_step": completed_step.model_copy(deep=True),
        "planner_local_memory": {},
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
    assert next_state["planner_local_memory"]["replan_rationale"] == "已完成"


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
        "summary_local_memory": {},
        "pending_memory_writes": [],
        "message_window": [
            {"role": "user", "message": "帮我完成总结", "attachment_paths": []},
        ],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert len(summarized_state["pending_memory_writes"]) == 2
    assert summarized_state["summary_local_memory"]["memory_candidates_reason"] != ""

    consolidated_state = asyncio.run(
        consolidate_memory_node(
            summarized_state,
            long_term_memory_repository=repository,
        )
    )

    assert len(repository.upserted) == 2
    assert {item.memory_type for item in repository.upserted} == {"profile", "fact"}
    assert consolidated_state["pending_memory_writes"] == []
    assert consolidated_state["graph_metadata"]["memory_write_count"] == 2
    assert len(consolidated_state["graph_metadata"]["memory_write_ids"]) == 2
    assert consolidated_state["summary_local_memory"] == {}


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
        "summary_local_memory": {
            "selected_artifacts": [f"/tmp/artifact-{index}.md" for index in range(12)],
        },
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
    assert consolidated_state["graph_metadata"]["memory_trimmed_message_count"] == 6
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
        "summary_local_memory": {},
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
    assert consolidated_state["graph_metadata"]["memory_candidate_input_count"] == 5
    assert consolidated_state["graph_metadata"]["memory_candidate_kept_count"] == 2
    assert consolidated_state["graph_metadata"]["memory_candidate_dropped_invalid_count"] == 1
    assert consolidated_state["graph_metadata"]["memory_candidate_dropped_low_confidence_count"] == 1
    assert consolidated_state["graph_metadata"]["memory_candidate_profile_merge_count"] == 1


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
        "summary_local_memory": {},
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
    assert summarized_state["summary_local_memory"]["memory_candidates_reason"] != ""


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
        "summary_local_memory": {},
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
    }

    summarized_state = asyncio.run(summarize_node(state, llm))

    assert len(summarized_state["pending_memory_writes"]) == 1
    assert summarized_state["pending_memory_writes"][0]["memory_type"] == "fact"
    assert summarized_state["pending_memory_writes"][0]["tags"] == ["task_outcome"]


def test_planner_react_graph_should_only_inject_repository_into_boundary_nodes(monkeypatch) -> None:
    repository = _FakeLongTermMemoryRepository()

    def _append_trace(state: dict, marker: str) -> dict:
        graph_metadata = dict(state.get("graph_metadata") or {})
        trace = list(graph_metadata.get("trace") or [])
        trace.append(marker)
        graph_metadata["trace"] = trace
        return {
            **state,
            "graph_metadata": graph_metadata,
        }

    async def _recall(state, long_term_memory_repository=None):
        assert long_term_memory_repository is repository
        return _append_trace(state, "recall")

    async def _plan(state, _llm):
        plan = _build_plan(step_status=ExecutionStatus.PENDING)
        next_state = _append_trace(state, "plan")
        return {
            **next_state,
            "plan": plan,
            "current_step_id": "step-1",
        }

    async def _execute(state, _llm, skill_runtime=None, runtime_tools=None, max_tool_iterations=5):
        plan = state["plan"].model_copy(deep=True)
        plan.steps[0].status = ExecutionStatus.COMPLETED
        plan.steps[0].success = True
        plan.steps[0].result = "步骤执行完成"
        next_state = _append_trace(state, "execute")
        return {
            **next_state,
            "plan": plan,
            "last_executed_step": plan.steps[0].model_copy(deep=True),
            "current_step_id": None,
        }

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
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph.recall_memory_context_node",
        _recall,
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
        llm=object(),
        long_term_memory_repository=repository,
    )

    async def _invoke():
        return await graph.ainvoke(
            {
                "session_id": "session-1",
                "user_message": "帮我整理记忆",
                "graph_metadata": {},
            },
            config={"configurable": {"thread_id": "session-1"}},
        )

    state = asyncio.run(_invoke())

    assert state["graph_metadata"]["trace"] == [
        "recall",
        "plan",
        "execute",
        "replan",
        "summarize",
        "consolidate",
        "finalize",
    ]


def test_graph_state_contract_should_include_user_id_in_runtime_metadata() -> None:
    state = {
        "schema_version": "be-lg-04.v4",
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "retrieved_memories": [],
        "pending_memory_writes": [],
        "graph_metadata": {"memory_write_count": 2, "memory_write_ids": ["mem-1", "mem-2"]},
        "message_window": [],
        "conversation_summary": "",
        "working_memory": {},
        "planner_local_memory": {},
        "step_local_memory": {},
        "summary_local_memory": {},
        "memory_context_version": "ctx-v3",
        "execution_count": 0,
        "max_execution_steps": 20,
        "step_states": [],
        "pending_interrupt": {},
        "tool_invocations": {},
        "artifact_refs": [],
        "emitted_events": [],
    }

    runtime_metadata = GraphStateContractMapper.build_runtime_metadata(state)

    assert runtime_metadata["graph_state_contract"]["graph_state"]["user_id"] == "user-1"
    assert runtime_metadata["memory"]["persisted_write_count"] == 2
    assert runtime_metadata["memory"]["persisted_write_ids"] == ["mem-1", "mem-2"]
