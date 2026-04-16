import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.domain.models import (
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Session,
    Step,
    StepArtifactPolicy,
    StepDeliveryContextState,
    StepDeliveryRole,
    StepEvent,
    StepEventStatus,
    StepOutcome,
    StepOutputMode,
    StepTaskModeHint,
    WorkflowRunEventRecord,
)
from app.infrastructure.models.workflow_run_step import WorkflowRunStepModel
from app.infrastructure.repositories.db_workflow_run_repository import DBWorkflowRunRepository


def _build_repo(execute_result) -> DBWorkflowRunRepository:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=execute_result),
        add=MagicMock(),
    )
    return DBWorkflowRunRepository(db_session=db_session)


def test_add_event_if_absent_should_sync_step_projection_for_step_event() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    repo._refresh_run_status_by_event = AsyncMock()
    repo.upsert_step_from_event = AsyncMock()
    repo.replace_steps_from_plan = AsyncMock()
    event = StepEvent(
        id="evt-step-1",
        step=Step(
            id="step-1",
            title="执行步骤",
            description="执行步骤",
            objective_key="objective-step-1",
            success_criteria=["执行步骤完成"],
            status=ExecutionStatus.RUNNING,
        ),
        status=StepEventStatus.STARTED,
    )

    inserted = asyncio.run(
        repo.add_event_if_absent(
            session_id="session-1",
            run_id="run-1",
            event=event,
        )
    )

    assert inserted is True
    repo.upsert_step_from_event.assert_awaited_once()
    repo.replace_steps_from_plan.assert_not_awaited()


def test_add_event_if_absent_should_sync_step_projection_for_plan_event() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    repo._refresh_run_status_by_event = AsyncMock()
    repo.upsert_step_from_event = AsyncMock()
    repo.replace_steps_from_plan = AsyncMock()
    event = PlanEvent(
        id="evt-plan-1",
        status=PlanEventStatus.CREATED,
        plan=Plan(
            title="计划",
            goal="目标",
            language="zh",
            message="说明",
            steps=[
                Step(
                    id="step-1",
                    title="步骤1",
                    description="步骤1",
                    objective_key="objective-step-1",
                    success_criteria=["步骤1完成"],
                )
            ],
            status=ExecutionStatus.PENDING,
        ),
    )

    inserted = asyncio.run(
        repo.add_event_if_absent(
            session_id="session-1",
            run_id="run-1",
            event=event,
        )
    )

    assert inserted is True
    repo.replace_steps_from_plan.assert_awaited_once()
    repo.upsert_step_from_event.assert_not_awaited()


def test_upsert_step_from_event_should_create_step_snapshot_and_update_current_step() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    run_record = SimpleNamespace(
        id="run-1",
        current_step_id=None,
    )
    repo._get_record_with_lock = AsyncMock(return_value=run_record)
    repo._get_step_record_with_lock = AsyncMock(return_value=None)
    repo._resolve_step_index = AsyncMock(return_value=2)
    event = StepEvent(
        id="evt-step-2",
        step=Step(
            id="step-2",
            title="增量步骤",
            description="增量步骤",
            execution_template="根据{{topic}}执行增量步骤",
            required_slots=["topic"],
            execution_slots={"topic": "runtime"},
            task_mode_hint=StepTaskModeHint.RESEARCH,
            output_mode=StepOutputMode.NONE,
            artifact_policy=StepArtifactPolicy.FORBID_FILE_OUTPUT,
            delivery_role=StepDeliveryRole.FINAL,
            delivery_context_state=StepDeliveryContextState.NEEDS_PREPARATION,
            objective_key="objective-step-2",
            success_criteria=["增量步骤完成"],
            status=ExecutionStatus.RUNNING,
        ),
        status=StepEventStatus.STARTED,
    )

    asyncio.run(repo.upsert_step_from_event(run_id="run-1", event=event))

    assert run_record.current_step_id == "step-2"
    assert repo.db_session.add.call_count == 1
    step_record = repo.db_session.add.call_args[0][0]
    assert step_record.run_id == "run-1"
    assert step_record.step_id == "step-2"
    assert step_record.step_index == 2
    assert step_record.objective_key == "objective-step-2"
    assert step_record.execution_template == "根据{{topic}}执行增量步骤"
    assert step_record.required_slots == ["topic"]
    assert step_record.execution_slots == {"topic": "runtime"}
    assert step_record.status == ExecutionStatus.RUNNING.value
    assert step_record.task_mode_hint == StepTaskModeHint.RESEARCH.value
    assert step_record.output_mode == StepOutputMode.NONE.value
    assert step_record.artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT.value
    assert step_record.delivery_role == StepDeliveryRole.FINAL.value
    assert step_record.delivery_context_state == StepDeliveryContextState.NEEDS_PREPARATION.value


def test_upsert_step_from_event_should_update_existing_snapshot_and_clear_current_step() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    run_record = SimpleNamespace(
        id="run-1",
        current_step_id="step-3",
    )
    existing_step_record = SimpleNamespace(
        step_index=0,
        title="旧标题",
        description="旧描述",
        objective_key="objective-old",
        success_criteria=[],
        status=ExecutionStatus.RUNNING.value,
        task_mode_hint=None,
        output_mode=None,
        artifact_policy=None,
        delivery_role=None,
        delivery_context_state=None,
        outcome=None,
        error=None,
    )
    repo._get_record_with_lock = AsyncMock(return_value=run_record)
    repo._get_step_record_with_lock = AsyncMock(return_value=existing_step_record)
    event = StepEvent(
        id="evt-step-3",
        step=Step(
            id="step-3",
            title="新描述",
            description="新描述",
            execution_template="根据{{city}}生成新描述",
            required_slots=["city"],
            execution_slots={"city": "上海"},
            task_mode_hint=StepTaskModeHint.GENERAL,
            output_mode=StepOutputMode.INLINE,
            artifact_policy=StepArtifactPolicy.DEFAULT,
            delivery_role=StepDeliveryRole.FINAL,
            delivery_context_state=StepDeliveryContextState.READY,
            objective_key="objective-step-3",
            success_criteria=["新描述完成"],
            status=ExecutionStatus.COMPLETED,
            outcome=StepOutcome(
                done=True,
                summary="完成",
                produced_artifacts=["artifact-id-1", "https://example.com/file.md", "file-1", "/tmp/file-1.md"],
            ),
        ),
        status=StepEventStatus.COMPLETED,
    )

    asyncio.run(repo.upsert_step_from_event(run_id="run-1", event=event))

    assert run_record.current_step_id is None
    assert existing_step_record.title == "新描述"
    assert existing_step_record.description == "新描述"
    assert existing_step_record.execution_template == "根据{{city}}生成新描述"
    assert existing_step_record.required_slots == ["city"]
    assert existing_step_record.execution_slots == {"city": "上海"}
    assert existing_step_record.objective_key == "objective-step-3"
    assert existing_step_record.status == ExecutionStatus.COMPLETED.value
    assert existing_step_record.task_mode_hint == StepTaskModeHint.GENERAL.value
    assert existing_step_record.output_mode == StepOutputMode.INLINE.value
    assert existing_step_record.artifact_policy == StepArtifactPolicy.DEFAULT.value
    assert existing_step_record.delivery_role == StepDeliveryRole.FINAL.value
    assert existing_step_record.delivery_context_state == StepDeliveryContextState.READY.value
    assert existing_step_record.outcome == {
        "done": True,
        "summary": "完成",
        "delivery_text": "",
        "produced_artifacts": ["/tmp/file-1.md"],
        "blockers": [],
        "facts_learned": [],
        "open_questions": [],
        "deliver_result_as_attachment": None,
        "next_hint": None,
        "reused_from_run_id": None,
        "reused_from_step_id": None,
    }


def test_workflow_run_step_model_to_domain_should_filter_non_file_artifacts_from_outcome() -> None:
    record = WorkflowRunStepModel(
        id="record-1",
        run_id="run-1",
        step_id="step-1",
        step_index=0,
        title="步骤1",
        description="步骤1",
        execution_template="根据{{slot}}执行步骤1",
        required_slots=["slot"],
        execution_slots={"slot": "value"},
        objective_key="objective-step-1",
        success_criteria=["步骤1完成"],
        status=ExecutionStatus.COMPLETED.value,
        updated_at=datetime(2026, 4, 10, 12, 0, 0),
        created_at=datetime(2026, 4, 10, 12, 0, 0),
        outcome={
            "done": True,
            "summary": "完成",
            "delivery_text": "",
            "produced_artifacts": [
                "artifact-id-1",
                "https://example.com/file.md",
                "file-1",
                "/tmp/file-1.md",
            ],
            "blockers": [],
            "facts_learned": [],
            "open_questions": [],
            "next_hint": None,
            "reused_from_run_id": None,
            "reused_from_step_id": None,
        },
    )

    domain_record = record.to_domain()

    assert domain_record.execution_template == "根据{{slot}}执行步骤1"
    assert domain_record.required_slots == ["slot"]
    assert domain_record.execution_slots == {"slot": "value"}
    assert domain_record.outcome is not None
    assert domain_record.outcome.produced_artifacts == ["/tmp/file-1.md"]


def test_list_events_should_return_empty_when_run_id_missing() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    repo._list_events_by_run_id = AsyncMock()

    events = asyncio.run(repo.list_events(run_id=None))

    assert events == []
    repo._list_events_by_run_id.assert_not_awaited()


def test_list_events_should_return_workflow_run_events() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    runtime_event = MessageEvent(id="runtime-event-1", role="assistant", message="runtime")
    repo._list_events_by_run_id = AsyncMock(return_value=[runtime_event])

    events = asyncio.run(repo.list_events(run_id="run-2"))

    assert len(events) == 1
    assert events[0].id == "runtime-event-1"
    repo._list_events_by_run_id.assert_awaited_once_with("run-2")


def test_list_events_by_session_should_query_all_workflow_run_events() -> None:
    event_one = MessageEvent(id="evt-1", role="assistant", message="first")
    event_two = MessageEvent(id="evt-2", role="assistant", message="second")
    execute_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(
            all=lambda: [
                SimpleNamespace(to_domain=lambda: SimpleNamespace(event_payload=event_one)),
                SimpleNamespace(to_domain=lambda: SimpleNamespace(event_payload=event_two)),
            ]
        )
    )
    repo = _build_repo(execute_result)

    events = asyncio.run(repo.list_events_by_session(session_id="session-1"))

    assert [event.id for event in events] == ["evt-1", "evt-2"]


def test_list_event_records_by_session_should_return_full_event_records() -> None:
    event_one = MessageEvent(id="evt-1", role="assistant", message="first")
    event_two = MessageEvent(id="evt-2", role="assistant", message="second")
    execute_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(
            all=lambda: [
                SimpleNamespace(
                    to_domain=lambda: WorkflowRunEventRecord(
                        id="record-1",
                        run_id="run-1",
                        session_id="session-1",
                        event_id="evt-1",
                        event_type="message",
                        event_payload=event_one,
                    )
                ),
                SimpleNamespace(
                    to_domain=lambda: WorkflowRunEventRecord(
                        id="record-2",
                        run_id="run-2",
                        session_id="session-1",
                        event_id="evt-2",
                        event_type="message",
                        event_payload=event_two,
                    )
                ),
            ]
        )
    )
    repo = _build_repo(execute_result)

    records = asyncio.run(repo.list_event_records_by_session(session_id="session-1"))

    assert [(record.run_id, record.event_payload.id) for record in records] == [
        ("run-1", "evt-1"),
        ("run-2", "evt-2"),
    ]
