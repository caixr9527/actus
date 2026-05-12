import asyncio
import json

from app.domain.models import (
    ExecutionStatus,
    MessageEvent,
    Plan,
    Step,
    StepEvent,
    StepOutcome,
)
from app.domain.services.runtime.normalizers import normalize_execution_response
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes import (
    execute_step_node as _execute_step_node,
    guard_step_reuse_node,
    wait_for_human_node,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.routing import route_after_execute


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


async def execute_step_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _execute_step_node(*args, **kwargs)


class _LeakyExecuteLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已完成执行步骤，保留为步骤摘要。",
                    "facts_learned": ["执行阶段只沉淀事实"],
                    "attachments": ["/tmp/step-output.md"],
                    "final_answer_text": "这段文本不能成为最终正文。",
                    "final_message": "这段轻量总结不能写入最终字段。",
                    "selected_artifacts": ["/tmp/final.md"],
                    "final_delivery": {
                        "text": "最终交付载荷不能由 execute 写入。",
                        "source_refs": ["/tmp/final.md"],
                    },
                },
                ensure_ascii=False,
            )
        }


def test_normalize_execution_response_should_drop_final_output_fields() -> None:
    normalized = normalize_execution_response(
        {
            "success": True,
            "summary": "执行阶段摘要",
            "attachments": ["/tmp/step-output.md"],
            "final_answer_text": "越权最终正文",
            "final_message": "越权轻量总结",
            "selected_artifacts": ["/tmp/final.md"],
            "final_delivery": {"text": "越权最终交付"},
        }
    )

    assert normalized["summary"] == "执行阶段摘要"
    assert normalized["attachments"] == ["/tmp/step-output.md"]
    assert "final_answer_text" not in normalized
    assert "final_message" not in normalized
    assert "selected_artifacts" not in normalized
    assert "final_delivery" not in normalized


def test_execute_step_node_should_not_promote_model_final_fields_to_final_output() -> None:
    plan = Plan(
        title="执行泄漏字段测试",
        goal="验证 execute 纯执行化",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="生成步骤候选产物",
                description="生成步骤候选产物",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    state = {
        "session_id": "session-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "执行当前步骤",
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": ["/tmp/existing-final.md"],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "current_step_id": None,
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
        "conversation_summary": "",
    }

    next_state = asyncio.run(execute_step_node(state, _LeakyExecuteLLM(), runtime_tools=[]))

    executed_step = next_state["last_executed_step"]
    assert executed_step.outcome is not None
    assert executed_step.outcome.summary == "已完成执行步骤，保留为步骤摘要。"
    assert executed_step.outcome.produced_artifacts == ["/tmp/step-output.md"]
    assert next_state["final_message"] == "已有轻量总结"
    assert next_state["final_answer_text"] == "已有最终正文"
    assert next_state["selected_artifacts"] == ["/tmp/existing-final.md"]
    assert not any(
        isinstance(event, MessageEvent) and event.stage == "final"
        for event in next_state["emitted_events"]
    )
    assert route_after_execute(next_state) == "replan"


def test_wait_resume_should_not_emit_final_event_or_final_answer_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: "选择 A",
    )
    plan = Plan(
        title="等待恢复测试",
        goal="验证 wait 纯执行化",
        language="zh",
        steps=[
            Step(
                id="wait-step",
                title="等待用户选择",
                description="等待用户选择",
                status=ExecutionStatus.RUNNING,
            )
        ],
    )
    state = {
        "plan": plan,
        "current_step_id": "wait-step",
        "pending_interrupt": {"kind": "input_text", "prompt": "请选择"},
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
    }

    next_state = asyncio.run(wait_for_human_node(state))

    assert next_state["last_executed_step"].outcome is not None
    assert next_state["final_message"] == "已有轻量总结"
    assert next_state["final_answer_text"] == "已有最终正文"
    assert not any(
        isinstance(event, MessageEvent) and event.stage == "final"
        for event in next_state["emitted_events"]
    )


def test_wait_cancel_should_not_emit_final_event_or_final_answer_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: False,
    )
    plan = Plan(
        title="等待取消测试",
        goal="验证 wait cancel 纯执行化",
        language="zh",
        steps=[
            Step(
                id="wait-step",
                title="等待用户确认",
                description="等待用户确认",
                status=ExecutionStatus.RUNNING,
            )
        ],
    )
    state = {
        "plan": plan,
        "current_step_id": "wait-step",
        "pending_interrupt": {
            "kind": "confirm",
            "prompt": "是否继续？",
            "confirm_resume_value": True,
            "cancel_resume_value": False,
        },
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
    }

    next_state = asyncio.run(wait_for_human_node(state))

    assert next_state["last_executed_step"].status == ExecutionStatus.CANCELLED
    assert next_state["final_message"] == "已有轻量总结"
    assert next_state["final_answer_text"] == "已有最终正文"
    assert not any(
        isinstance(event, MessageEvent) and event.stage == "final"
        for event in next_state["emitted_events"]
    )


def test_reuse_hit_should_not_promote_reused_summary_or_artifacts_to_final_output() -> None:
    completed_step = Step(
        id="step-1",
        title="生成报告",
        description="生成报告",
        objective_key="objective-report",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(
            done=True,
            summary="历史步骤摘要不能成为最终正文",
            produced_artifacts=["/tmp/reused-candidate.md"],
            facts_learned=["复用事实"],
        ),
    )
    pending_duplicate_step = Step(
        id="step-2",
        title="再次生成报告",
        description="再次生成报告",
        objective_key="objective-report",
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
        "selected_artifacts": ["/tmp/existing-final.md"],
        "emitted_events": [],
        "execution_count": 0,
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
    }

    next_state = asyncio.run(guard_step_reuse_node(state))

    assert next_state["plan"].steps[1].outcome is not None
    assert next_state["plan"].steps[1].outcome.produced_artifacts == ["/tmp/reused-candidate.md"]
    assert next_state["final_message"] == "已有轻量总结"
    assert next_state["final_answer_text"] == "已有最终正文"
    assert next_state["selected_artifacts"] == ["/tmp/existing-final.md"]
    assert all(isinstance(event, StepEvent) for event in next_state["emitted_events"])
