from types import SimpleNamespace

from app.domain.models import (
    ExecutionStatus,
    Plan,
    Step,
    StepOutcome,
    WorkflowRunStatus,
)
from app.infrastructure.runtime.langgraph.engine.run_engine import LangGraphRunEngine


def _completed_run():
    return SimpleNamespace(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        thread_id="thread-1",
        status=WorkflowRunStatus.COMPLETED,
    )


def _completed_state(*, final_message: str, include_final_answer_text: bool, final_answer_text: str = "") -> dict:
    final_step = Step(
        id="step-1",
        title="整理结果",
        description="整理结果",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="步骤执行摘要"),
    )
    state = {
        "plan": Plan(
            title="最终输出投影测试",
            goal="验证最终正文投影",
            language="zh",
            steps=[final_step],
            status=ExecutionStatus.COMPLETED,
        ),
        "step_states": [
            {
                "step_id": "step-1",
                "title": "整理结果",
                "description": "整理结果",
                "status": "completed",
                "outcome": {
                    "done": True,
                    "summary": "步骤执行摘要",
                    "blockers": [],
                    "facts_learned": [],
                    "open_questions": [],
                },
            }
        ],
        "working_memory": {
            "goal": "验证最终正文投影",
            "facts_in_session": [],
            "open_questions": [],
        },
        "selected_artifacts": [],
        "graph_metadata": {"projection": {"run_status": "completed"}},
        "final_message": final_message,
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
        "last_executed_step": final_step,
        "pending_interrupt": {},
        "emitted_events": [],
        "conversation_summary": "",
        "historical_artifact_paths": [],
    }
    if include_final_answer_text:
        state["final_answer_text"] = final_answer_text
    return state


def test_run_summary_projection_should_use_final_answer_text_when_present() -> None:
    summary = LangGraphRunEngine._build_run_summary_projection(
        run=_completed_run(),
        state=_completed_state(
            final_message="轻量总结",
            include_final_answer_text=True,
            final_answer_text="真正交付给用户的最终正文。",
        ),
    )

    assert summary.final_answer_summary == "轻量总结"
    assert summary.final_answer_text == "真正交付给用户的最终正文。"


def test_run_summary_projection_should_not_fallback_for_new_completed_state_without_final_answer_text() -> None:
    summary = LangGraphRunEngine._build_run_summary_projection(
        run=_completed_run(),
        state=_completed_state(
            final_message="轻量总结不应被当成最终正文",
            include_final_answer_text=True,
            final_answer_text="",
        ),
    )

    assert summary.final_answer_summary == "轻量总结不应被当成最终正文"
    assert summary.final_answer_text == ""


def test_run_summary_projection_should_keep_legacy_fallback_when_final_answer_text_key_missing() -> None:
    summary = LangGraphRunEngine._build_run_summary_projection(
        run=_completed_run(),
        state=_completed_state(
            final_message="旧运行只有 final_message",
            include_final_answer_text=False,
        ),
    )

    assert summary.final_answer_summary == "旧运行只有 final_message"
    assert summary.final_answer_text == "旧运行只有 final_message"


def test_run_summary_projection_should_not_use_previous_final_message_as_current_final_text() -> None:
    state = _completed_state(
        final_message="当前轻量总结",
        include_final_answer_text=True,
        final_answer_text="",
    )
    state["previous_final_message"] = "上一轮真正交付的正文，不能污染当前 run。"

    summary = LangGraphRunEngine._build_run_summary_projection(
        run=_completed_run(),
        state=state,
    )

    assert summary.final_answer_text == ""
    assert "上一轮" not in summary.final_answer_summary
