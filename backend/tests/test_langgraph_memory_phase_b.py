import asyncio

from app.domain.models import ExecutionStatus, Plan, Step
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.graph import (
    build_planner_react_langgraph_graph,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes import (
    consolidate_memory_node,
    recall_memory_context_node,
)


def _build_plan(*, step_status: ExecutionStatus = ExecutionStatus.PENDING) -> Plan:
    return Plan(
        title="记忆整理计划",
        goal="整理线程级短期记忆",
        language="zh",
        message="开始整理",
        steps=[
            Step(
                id="step-1",
                description="整理记忆",
                status=step_status,
                success=step_status == ExecutionStatus.COMPLETED,
            )
        ],
        status=ExecutionStatus.PENDING,
    )


def test_recall_memory_context_node_should_prepare_working_memory() -> None:
    state = {
        "thread_id": "thread-1",
        "execution_count": 0,
        "message_window": [{"role": "user", "message": "历史消息"}],
        "user_message": "帮我整理记忆",
        "plan": _build_plan(),
        "retrieved_memories": [
            {
                "id": "mem-1",
                "memory_type": "profile",
                "content": {"language": "zh", "style": "concise"},
            }
        ],
        "graph_metadata": {},
    }

    next_state = asyncio.run(recall_memory_context_node(state))

    assert next_state["working_memory"]["goal"] == "整理线程级短期记忆"
    assert next_state["working_memory"]["user_preferences"] == {
        "language": "zh",
        "style": "concise",
    }
    assert next_state["memory_context_version"] == "thread-1:0:1:1"
    assert next_state["graph_metadata"]["memory_recall_count"] == 1
    assert "memory_recall_prepared_at" in next_state["graph_metadata"]


def test_consolidate_memory_node_should_compact_window_and_clear_local_memory() -> None:
    state = {
        "user_message": "帮我整理记忆",
        "final_message": "这是最终答复",
        "plan": _build_plan(step_status=ExecutionStatus.COMPLETED),
        "step_states": [
            {
                "step_id": "step-1",
                "status": ExecutionStatus.COMPLETED.value,
            }
        ],
        "message_window": [
            {"role": "user", "message": f"历史消息{i}", "attachment_paths": []}
            for i in range(7)
        ],
        "working_memory": {"goal": "整理线程级短期记忆"},
        "summary_local_memory": {"selected_artifacts": ["/tmp/result.md"]},
        "planner_local_memory": {"plan_brief": "旧规划"},
        "step_local_memory": {"current_step_id": "step-1"},
        "graph_metadata": {},
    }

    next_state = asyncio.run(consolidate_memory_node(state))

    assert len(next_state["message_window"]) == 6
    assert next_state["message_window"][-1]["message"] == "这是最终答复"
    assert next_state["message_window"][-1]["attachment_paths"] == ["/tmp/result.md"]
    assert "结果:这是最终答复" in next_state["conversation_summary"]
    assert next_state["graph_metadata"]["memory_compacted"] is True
    assert "memory_last_compaction_at" in next_state["graph_metadata"]
    assert next_state["planner_local_memory"] == {}
    assert next_state["step_local_memory"] == {}
    assert next_state["summary_local_memory"] == {}


def test_planner_react_graph_should_include_memory_boundary_nodes(monkeypatch) -> None:
    def _append_trace(state: dict, marker: str) -> dict:
        graph_metadata = dict(state.get("graph_metadata") or {})
        trace = list(graph_metadata.get("trace") or [])
        trace.append(marker)
        graph_metadata["trace"] = trace
        return {
            **state,
            "graph_metadata": graph_metadata,
        }

    async def _recall(state):
        return _append_trace(state, "recall")

    async def _plan(state, _llm):
        plan = _build_plan()
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

    async def _consolidate(state):
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

    graph = build_planner_react_langgraph_graph(llm=object())

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
