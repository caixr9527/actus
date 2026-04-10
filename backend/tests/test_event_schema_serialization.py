from datetime import datetime

from app.domain.models import (
    ErrorEvent,
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
    Step,
    StepOutcome,
    WaitEvent,
)
from app.interfaces.schemas.event import (
    BaseEventData,
    CommonEventData,
    EventMapper,
)
from app.infrastructure.models.workflow_run_event import WorkflowRunEventModel


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
                    title="执行步骤1",
                    description="执行步骤1",
                    objective_key="objective-step-1",
                    success_criteria=["执行步骤1完成"],
                    status=ExecutionStatus.COMPLETED,
                    outcome=StepOutcome(done=True, summary="执行步骤1完成"),
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
    assert "event_status" not in payload["data"]["steps"][0]
    assert payload["data"]["steps"][0]["outcome"]["summary"] == "执行步骤1完成"


def test_step_sse_event_should_include_step_outcome() -> None:
    event = StepEvent(
        id="evt-step-1",
        created_at=datetime(2026, 3, 11, 12, 0, 7),
        status=StepEventStatus.FAILED,
        step=Step(
            id="step-1",
            title="执行步骤1",
            description="执行步骤1",
            objective_key="objective-step-1",
            success_criteria=["执行步骤1完成"],
            status=ExecutionStatus.FAILED,
            outcome=StepOutcome(
                done=False,
                summary="步骤执行超时：执行步骤1",
                blockers=["当前步骤超过 180 秒未完成"],
                next_hint="请缩小当前步骤范围后重试",
            ),
        ),
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["event"] == "step"
    assert payload["data"]["status"] == ExecutionStatus.FAILED.value
    assert payload["data"]["outcome"]["summary"] == "步骤执行超时：执行步骤1"
    assert payload["data"]["outcome"]["blockers"] == ["当前步骤超过 180 秒未完成"]
    assert payload["data"]["outcome"]["next_hint"] == "请缩小当前步骤范围后重试"


def test_event_mapper_should_normalize_live_step_event_outcome_before_sse() -> None:
    event = StepEvent(
        id="evt-step-live",
        created_at=datetime(2026, 3, 11, 12, 0, 7),
        status=StepEventStatus.COMPLETED,
        step=Step(
            id="step-1",
            title="生成文件",
            description="生成文件",
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成",
                produced_artifacts=["artifact-id-1", "https://example.com/file.md", "/tmp/final.md"],
            ),
        ),
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["data"]["outcome"]["produced_artifacts"] == ["/tmp/final.md"]


def test_event_mapper_should_normalize_live_plan_event_outcomes_before_sse() -> None:
    event = PlanEvent(
        id="evt-plan-live",
        created_at=datetime(2026, 3, 11, 12, 0, 7),
        status=PlanEventStatus.UPDATED,
        plan=Plan(
            title="测试计划",
            steps=[
                Step(
                    id="step-1",
                    title="生成文件",
                    description="生成文件",
                    status=ExecutionStatus.COMPLETED,
                    outcome=StepOutcome(
                        done=True,
                        summary="已生成",
                        produced_artifacts=["artifact-id-1", "/tmp/final.md"],
                    ),
                )
            ],
        ),
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["data"]["steps"][0]["outcome"]["produced_artifacts"] == ["/tmp/final.md"]


def test_workflow_run_event_model_should_normalize_historical_step_event_outcome_on_read() -> None:
    event = StepEvent(
        id="evt-step-dirty",
        step=Step(
            id="step-1",
            title="生成文件",
            description="生成文件",
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="已生成",
                produced_artifacts=["artifact-id-1", "https://example.com/final.md", "/tmp/final.md"],
            ),
        ),
        status=StepEventStatus.COMPLETED,
    )
    record = WorkflowRunEventModel(
        id="record-1",
        run_id="run-1",
        session_id="session-1",
        event_id=event.id,
        event_type=event.type,
        event_payload=event.model_dump(mode="json"),
        created_at=datetime(2026, 3, 11, 12, 0, 8),
    )

    domain_record = record.to_domain()

    assert isinstance(domain_record.event_payload, StepEvent)
    assert domain_record.event_payload.step.outcome.produced_artifacts == ["/tmp/final.md"]


def test_workflow_run_event_model_should_normalize_historical_plan_event_outcome_on_read() -> None:
    event = PlanEvent(
        id="evt-plan-dirty",
        plan=Plan(
            title="测试计划",
            steps=[
                Step(
                    id="step-1",
                    title="生成文件",
                    description="生成文件",
                    status=ExecutionStatus.COMPLETED,
                    outcome=StepOutcome(
                        done=True,
                        summary="已生成",
                        produced_artifacts=["artifact-id-1", "/tmp/final.md"],
                    ),
                )
            ],
        ),
        status=PlanEventStatus.COMPLETED,
    )
    record = WorkflowRunEventModel(
        id="record-2",
        run_id="run-1",
        session_id="session-1",
        event_id=event.id,
        event_type=event.type,
        event_payload=event.model_dump(mode="json"),
        created_at=datetime(2026, 3, 11, 12, 0, 9),
    )

    domain_record = record.to_domain()

    assert isinstance(domain_record.event_payload, PlanEvent)
    assert domain_record.event_payload.plan.steps[0].outcome.produced_artifacts == ["/tmp/final.md"]


def test_wait_sse_event_should_include_interrupt_payload() -> None:
    event = WaitEvent.from_interrupt(
        interrupt_id="interrupt-1",
        payload={
            "kind": "select",
            "title": "需要你的选择",
            "prompt": "请确认是否继续",
            "attachments": ["/tmp/spec.md"],
            "suggest_user_takeover": "browser",
            "options": [
                {"label": "继续", "resume_value": {"approved": True}},
                {"label": "取消", "resume_value": {"approved": False}},
            ],
        },
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["event"] == "wait"
    assert payload["data"]["interrupt_id"] == "interrupt-1"
    assert payload["data"]["payload"]["kind"] == "select"
    assert payload["data"]["payload"]["prompt"] == "请确认是否继续"
    assert payload["data"]["payload"]["attachments"] == ["/tmp/spec.md"]
    assert payload["data"]["payload"]["suggest_user_takeover"] == "browser"
    assert payload["data"]["payload"]["options"][0]["label"] == "继续"
