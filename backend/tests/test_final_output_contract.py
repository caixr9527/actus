import asyncio
import json

from app.domain.models import (
    DoneEvent,
    ExecutionStatus,
    MessageEvent,
    Plan,
    Step,
    StepOutcome,
)
from app.domain.services.runtime.contracts.final_output_contract import (
    FINAL_ATTACHMENT_FIELDS,
    FINAL_EVENT_STAGES,
    FINAL_TEXT_FIELDS,
    RuntimeOutputStage,
    assert_state_update_allowed,
    can_emit_final_message_event,
    can_write_final_answer_text,
    can_write_final_message,
    can_write_selected_artifacts,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.entry import EntryCompiler
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes import (
    direct_answer_node,
    execute_step_node,
    finalize_node,
    guard_step_reuse_node,
    summarize_node,
    wait_for_human_node,
)


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


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


class _DirectAnswerLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {"message": "你好，我在。"},
                ensure_ascii=False,
            )
        }


class _ExecuteLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已完成搜索步骤",
                    "facts_learned": ["步骤事实"],
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _SummaryLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "message": "轻量总结",
                    "final_answer_text": "最终整理后的正文",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


def test_final_output_contract_should_define_owned_final_fields() -> None:
    assert FINAL_TEXT_FIELDS == frozenset({"final_message", "final_answer_text"})
    assert FINAL_ATTACHMENT_FIELDS == frozenset({"selected_artifacts"})
    assert FINAL_EVENT_STAGES == frozenset({"final"})


def test_final_output_contract_should_allow_summary_to_write_final_output() -> None:
    stage = RuntimeOutputStage.SUMMARY

    assert can_write_final_message(stage) is True
    assert can_write_final_answer_text(stage) is True
    assert can_write_selected_artifacts(stage) is True
    assert can_emit_final_message_event(stage) is True

    assert_state_update_allowed(
        stage=stage,
        before_state={
            "final_message": "",
            "final_answer_text": "",
            "selected_artifacts": [],
        },
        updates={
            "final_message": "轻量总结",
            "final_answer_text": "最终正文",
            "selected_artifacts": ["/tmp/final.md"],
        },
    )


def test_final_output_contract_should_allow_direct_answer_without_attachments() -> None:
    stage = RuntimeOutputStage.DIRECT_ANSWER

    assert can_write_final_message(stage) is True
    assert can_write_final_answer_text(stage) is True
    assert can_write_selected_artifacts(stage) is False
    assert can_emit_final_message_event(stage) is True

    assert_state_update_allowed(
        stage=stage,
        before_state={
            "final_message": "",
            "final_answer_text": "",
            "selected_artifacts": [],
        },
        updates={
            "final_message": "直接回答",
            "final_answer_text": "直接回答",
        },
    )


def test_final_output_contract_should_allow_planner_direct_fallback_without_attachments() -> None:
    stage = RuntimeOutputStage.PLANNER_DIRECT_FALLBACK

    assert can_write_final_message(stage) is True
    assert can_write_final_answer_text(stage) is True
    assert can_write_selected_artifacts(stage) is False
    assert can_emit_final_message_event(stage) is True

    assert_state_update_allowed(
        stage=stage,
        before_state={
            "final_message": "",
            "final_answer_text": "",
            "selected_artifacts": [],
        },
        updates={
            "final_message": "兼容直答",
            "final_answer_text": "兼容直答",
        },
    )


def test_final_output_contract_should_allow_plan_only_planner_final_text() -> None:
    stage = RuntimeOutputStage.PLANNER

    assert can_write_final_message(stage) is False
    assert can_write_final_answer_text(stage) is False
    assert can_emit_final_message_event(stage) is False
    assert can_write_final_message(stage, plan_only=True) is True
    assert can_write_final_answer_text(stage, plan_only=True) is True
    assert can_emit_final_message_event(stage, plan_only=True) is True
    assert can_write_selected_artifacts(stage) is False

    assert_state_update_allowed(
        stage=stage,
        plan_only=True,
        before_state={
            "final_message": "",
            "final_answer_text": "",
            "selected_artifacts": [],
        },
        updates={
            "final_message": "执行计划",
            "final_answer_text": "执行计划",
        },
    )


def test_final_output_contract_should_reject_regular_planner_final_text() -> None:
    before_state = {
        "final_message": "",
        "final_answer_text": "",
        "selected_artifacts": [],
    }

    try:
        assert_state_update_allowed(
            stage=RuntimeOutputStage.PLANNER,
            before_state=before_state,
            updates={"final_answer_text": "越权最终正文"},
        )
    except ValueError as exc:
        assert "planner cannot write final_answer_text" in str(exc)
    else:
        raise AssertionError("普通 planner 不应允许写 final_answer_text")


def test_final_output_contract_should_allow_execute_wait_reuse_same_value_retention() -> None:
    before_state = {
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
        "selected_artifacts": ["/tmp/final.md"],
    }

    for stage in (
        RuntimeOutputStage.EXECUTE,
        RuntimeOutputStage.WAIT,
        RuntimeOutputStage.REUSE,
    ):
        assert can_write_final_message(stage) is False
        assert can_write_final_answer_text(stage) is False
        assert can_write_selected_artifacts(stage) is False
        assert can_emit_final_message_event(stage) is False
        assert_state_update_allowed(
            stage=stage,
            before_state=before_state,
            updates={
                "final_message": "已有轻量总结",
                "final_answer_text": "已有最终正文",
                "selected_artifacts": ["/tmp/final.md"],
            },
        )


def test_final_output_contract_should_reject_execute_wait_reuse_new_final_output() -> None:
    before_state = {
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
        "selected_artifacts": ["/tmp/final.md"],
    }
    cases = [
        (RuntimeOutputStage.EXECUTE, {"final_answer_text": "新的最终正文"}, "final_answer_text"),
        (RuntimeOutputStage.WAIT, {"final_message": "等待恢复文本"}, "final_message"),
        (RuntimeOutputStage.REUSE, {"selected_artifacts": ["/tmp/reused.md"]}, "selected_artifacts"),
    ]

    for stage, updates, field_name in cases:
        try:
            assert_state_update_allowed(
                stage=stage,
                before_state=before_state,
                updates=updates,
            )
        except ValueError as exc:
            assert f"{stage.value} cannot write {field_name}" in str(exc)
        else:
            raise AssertionError(f"{stage.value} 不应允许写 {field_name}")


def test_final_output_contract_should_reject_finalize_and_memory_final_output() -> None:
    before_state = {
        "final_message": "",
        "final_answer_text": "",
        "selected_artifacts": [],
    }

    for stage in (RuntimeOutputStage.FINALIZE, RuntimeOutputStage.MEMORY):
        assert can_write_final_message(stage) is False
        assert can_write_final_answer_text(stage) is False
        assert can_write_selected_artifacts(stage) is False
        assert can_emit_final_message_event(stage) is False
        try:
            assert_state_update_allowed(
                stage=stage,
                before_state=before_state,
                updates={"final_message": "越权最终总结"},
            )
        except ValueError as exc:
            assert f"{stage.value} cannot write final_message" in str(exc)
        else:
            raise AssertionError(f"{stage.value} 不应允许写 final_message")


def test_direct_answer_node_should_follow_final_output_contract() -> None:
    before_state = {
        "user_message": "你好",
        "graph_metadata": {"control": _entry_contract_control("你好")},
        "selected_artifacts": ["/tmp/existing-final.md"],
    }

    state = asyncio.run(
        direct_answer_node(
            before_state,
            _DirectAnswerLLM(),
            runtime_context_service=_TEST_RUNTIME_CONTEXT_SERVICE,
        )
    )

    assert state["final_message"] == "你好，我在。"
    assert state["final_answer_text"] == "你好，我在。"
    assert state.get("selected_artifacts", []) == []
    assert isinstance(state["emitted_events"][-1], MessageEvent)
    assert state["emitted_events"][-1].stage == "final"
    assert_state_update_allowed(
        stage=RuntimeOutputStage.DIRECT_ANSWER,
        before_state=before_state,
        updates={
            "final_message": state["final_message"],
            "final_answer_text": state["final_answer_text"],
        },
    )


def test_execute_step_node_should_follow_final_output_contract() -> None:
    plan = Plan(
        title="搜索课程",
        goal="搜索课程",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="搜索课程",
                description="搜索课程信息",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    before_state = {
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": [],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "user_message": "请先搜索课程",
        "current_step_id": None,
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }

    state = asyncio.run(
        execute_step_node(
            before_state,
            _ExecuteLLM(),
            runtime_tools=[],
            runtime_context_service=_TEST_RUNTIME_CONTEXT_SERVICE,
        )
    )

    assert state["last_executed_step"].outcome.summary == "已完成搜索步骤"
    assert state["final_message"] == "已有轻量总结"
    assert state["selected_artifacts"] == []
    assert "final_answer_text" not in {
        key
        for key, value in state.items()
        if before_state.get(key) != value
    }
    assert_state_update_allowed(
        stage=RuntimeOutputStage.EXECUTE,
        before_state=before_state,
        updates={
            "final_message": state["final_message"],
            "selected_artifacts": state["selected_artifacts"],
        },
    )


def test_wait_for_human_node_should_follow_final_output_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: "AI 人工智能算法工程师体系课",
    )
    plan = Plan(
        title="课程推荐",
        goal="推荐课程并等待用户选择",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="等待用户选择课程",
                description="展示三门课程并等待用户选择",
                status=ExecutionStatus.RUNNING,
            ),
            Step(
                id="step-2",
                title="继续处理",
                description="根据用户选择继续后续步骤",
                status=ExecutionStatus.PENDING,
            ),
        ],
    )
    before_state = {
        "plan": plan,
        "current_step_id": "step-1",
        "pending_interrupt": {
            "kind": "select",
            "prompt": "请选择一门课程",
            "options": [
                {"label": "AI 人工智能算法工程师体系课", "resume_value": "AI 人工智能算法工程师体系课"},
            ],
        },
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "final_message": "",
    }

    state = asyncio.run(wait_for_human_node(before_state))

    assert state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert state["final_message"] == ""
    assert_state_update_allowed(
        stage=RuntimeOutputStage.WAIT,
        before_state=before_state,
        updates={"final_message": state["final_message"]},
    )


def test_reuse_node_should_follow_final_output_contract() -> None:
    completed_step = Step(
        id="step-1",
        title="生成报告",
        description="生成报告",
        objective_key="objective-report",
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
        status=ExecutionStatus.PENDING,
    )
    before_state = {
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

    state = asyncio.run(guard_step_reuse_node(before_state))

    assert state["plan"].steps[1].status == ExecutionStatus.COMPLETED
    assert state["selected_artifacts"] == []
    assert state.get("final_message", "") == ""
    assert_state_update_allowed(
        stage=RuntimeOutputStage.REUSE,
        before_state=before_state,
        updates={
            "final_message": state.get("final_message", ""),
            "selected_artifacts": state["selected_artifacts"],
        },
    )


def test_summary_node_should_follow_final_output_contract() -> None:
    final_step = Step(
        id="step-final",
        title="输出最终结果",
        description="输出最终结果",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="步骤完成"),
    )
    before_state = {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "请输出最终结果",
        "plan": Plan(
            title="总结测试",
            goal="验证 summary 最终出口",
            language="zh",
            steps=[final_step],
            status=ExecutionStatus.PENDING,
        ),
        "execution_count": 1,
        "last_executed_step": final_step,
        "step_states": [],
        "working_memory": {
            "goal": "验证最终正文由 summary 生成",
            "user_preferences": {},
            "facts_in_session": [],
        },
        "pending_memory_writes": [],
        "message_window": [],
        "graph_metadata": {},
        "emitted_events": [],
        "final_message": "旧轻量摘要",
        "selected_artifacts": [],
    }

    state = asyncio.run(
        summarize_node(
            before_state,
            _SummaryLLM(),
            runtime_context_service=_TEST_RUNTIME_CONTEXT_SERVICE,
        )
    )

    assert state["final_message"] == "轻量总结"
    assert state["final_answer_text"] == "最终整理后的正文"
    assert state["selected_artifacts"] == []
    assert isinstance(state["emitted_events"][0], MessageEvent)
    assert state["emitted_events"][0].stage == "final"
    assert_state_update_allowed(
        stage=RuntimeOutputStage.SUMMARY,
        before_state=before_state,
        updates={
            "final_message": state["final_message"],
            "final_answer_text": state["final_answer_text"],
            "selected_artifacts": state["selected_artifacts"],
        },
    )


def test_finalize_node_should_only_emit_done_event_without_final_output() -> None:
    before_state = {
        "session_id": "session-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "graph_metadata": {},
        "emitted_events": [
            MessageEvent(
                role="assistant",
                message="最终整理后的正文",
                stage="final",
            )
        ],
        "final_message": "轻量总结",
        "final_answer_text": "最终整理后的正文",
        "selected_artifacts": ["/home/ubuntu/final.md"],
    }

    state = asyncio.run(finalize_node(before_state))

    assert state["final_message"] == "轻量总结"
    assert state["final_answer_text"] == "最终整理后的正文"
    assert state["selected_artifacts"] == ["/home/ubuntu/final.md"]
    assert isinstance(state["emitted_events"][-2], MessageEvent)
    assert state["emitted_events"][-2].stage == "final"
    assert isinstance(state["emitted_events"][-1], DoneEvent)
    assert state["emitted_events"][-1].type == "done"
    assert not hasattr(state["emitted_events"][-1], "message")
    assert_state_update_allowed(
        stage=RuntimeOutputStage.FINALIZE,
        before_state=before_state,
        updates={
            "final_message": state["final_message"],
            "final_answer_text": state["final_answer_text"],
            "selected_artifacts": state["selected_artifacts"],
        },
    )
