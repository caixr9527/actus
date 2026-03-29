from datetime import datetime

from app.domain.models import (
    ErrorEvent,
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Step,
    WaitEvent,
)
from app.interfaces.schemas.event import (
    BaseEventData,
    CommonEventData,
    EVENT_COMPAT_SCHEMA_VERSION,
    EventCompatContext,
    EventMapper,
)


def test_base_event_data_created_at_serialized_as_timestamp() -> None:
    created_at = datetime(2026, 3, 11, 12, 0, 1)
    data = BaseEventData(event_id="evt-1", created_at=created_at)

    payload = data.model_dump(mode="json")

    assert payload["created_at"] == int(created_at.timestamp())


def test_common_event_data_keeps_extra_fields_and_timestamp_serializer() -> None:
    created_at = datetime(2026, 3, 11, 12, 0, 2)
    data = CommonEventData.model_validate(
        {
            "event_id": "evt-2",
            "created_at": created_at,
            "custom_key": "custom-value",
        }
    )

    payload = data.model_dump(mode="json")

    assert payload["created_at"] == int(created_at.timestamp())
    assert payload["custom_key"] == "custom-value"


def test_event_mapper_preserves_timestamp_json_shape() -> None:
    created_at = datetime(2026, 3, 11, 12, 0, 3)
    event = MessageEvent(
        id="evt-3",
        created_at=created_at,
        role="assistant",
        message="hello",
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["data"]["created_at"] == int(created_at.timestamp())


def test_event_mapper_should_serialize_error_event_key_and_params() -> None:
    event = ErrorEvent(
        id="evt-4",
        created_at=datetime(2026, 3, 11, 12, 0, 4),
        error="任务会话不存在",
        error_key="error.session.not_found",
        error_params={"session_id": "session-1"},
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["event"] == "error"
    assert payload["data"]["error"] == "任务会话不存在"
    assert payload["data"]["error_key"] == "error.session.not_found"
    assert payload["data"]["error_params"] == {"session_id": "session-1"}


def test_event_mapper_should_attach_v2_extensions_with_context() -> None:
    event = MessageEvent(
        id="evt-ctx-1",
        created_at=datetime(2026, 3, 11, 12, 0, 5),
        role="assistant",
        message="兼容层测试",
    )

    sse_event = EventMapper.event_to_sse_event(
        event,
        context=EventCompatContext(
            session_id="session-1",
            run_id="run-1",
            channel="chat_stream",
            runtime_extensions={
                "input_part_summary": {"total": 2, "by_type": {"text": 1, "file": 1}},
                "downgrade_reason": "model_multimodal_disabled",
                "unsupported_parts": [{"type": "image", "reason": "model_multimodal_disabled"}],
            },
        ),
    )
    payload = sse_event.model_dump(mode="json")

    assert payload["data"]["extensions"]["compat"]["schema_version"] == EVENT_COMPAT_SCHEMA_VERSION
    assert payload["data"]["extensions"]["compat"]["semantic_type"] == "message"
    assert payload["data"]["extensions"]["runtime"]["session_id"] == "session-1"
    assert payload["data"]["extensions"]["runtime"]["run_id"] == "run-1"
    assert payload["data"]["extensions"]["runtime"]["channel"] == "chat_stream"
    assert payload["data"]["extensions"]["runtime"]["input_part_summary"]["total"] == 2
    assert payload["data"]["extensions"]["runtime"]["downgrade_reason"] == "model_multimodal_disabled"
    assert payload["data"]["extensions"]["runtime"]["unsupported_parts"][0]["type"] == "image"


def test_plan_sse_event_should_preserve_richer_plan_fields() -> None:
    event = PlanEvent(
        id="evt-plan-1",
        created_at=datetime(2026, 3, 11, 12, 0, 6),
        status=PlanEventStatus.UPDATED,
        plan=Plan(
            title="任务标题",
            goal="任务目标",
            language="zh",
            message="计划说明",
            status=ExecutionStatus.RUNNING,
            steps=[
                Step(
                    id="step-1",
                    description="执行步骤1",
                    status=ExecutionStatus.COMPLETED,
                    success=True,
                )
            ],
        ),
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["event"] == "plan"
    assert payload["data"]["title"] == "任务标题"
    assert payload["data"]["goal"] == "任务目标"
    assert payload["data"]["message"] == "计划说明"
    assert payload["data"]["language"] == "zh"
    assert payload["data"]["status"] == PlanEventStatus.UPDATED.value
    assert payload["data"]["plan_status"] == ExecutionStatus.RUNNING.value
    assert payload["data"]["steps"][0]["event_status"] == "completed"


def test_wait_sse_event_should_include_human_task_fields() -> None:
    event = WaitEvent.build_for_user_input(
        session_id="session-1",
        question="请确认是否继续",
        reason="ask_user",
        attachments=["/tmp/spec.md"],
        suggest_user_takeover="browser",
        timeout_seconds=600,
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-1",
        current_step_id="step-1",
        resume_token="resume-token-1",
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["event"] == "wait"
    assert payload["data"]["reason"] == "ask_user"
    assert payload["data"]["question"] == "请确认是否继续"
    assert payload["data"]["attachments"] == ["/tmp/spec.md"]
    assert payload["data"]["suggest_user_takeover"] == "browser"
    assert payload["data"]["resume_token"] == "resume-token-1"
    assert payload["data"]["resume_command"]["type"] == "chat_message"
    assert payload["data"]["resume_point"]["run_id"] == "run-1"
    assert payload["data"]["timeout_seconds"] == 600
    assert payload["data"]["status"] == "waiting"
