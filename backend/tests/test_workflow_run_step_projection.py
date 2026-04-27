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
    Step,
    StepArtifactPolicy,
    StepEvent,
    StepEventStatus,
    StepOutcome,
    StepOutputMode,
    StepTaskModeHint,
    WorkflowRunEventRecord,
    WorkflowRunStatus,
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
    repo._refresh_run_status_by_event.assert_not_awaited()
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
    repo._refresh_run_status_by_event.assert_not_awaited()
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
            task_mode_hint=StepTaskModeHint.RESEARCH,
            output_mode=StepOutputMode.NONE,
            artifact_policy=StepArtifactPolicy.FORBID_FILE_OUTPUT,
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
    assert step_record.status == ExecutionStatus.RUNNING.value
    assert step_record.task_mode_hint == StepTaskModeHint.RESEARCH.value
    assert step_record.output_mode == StepOutputMode.NONE.value
    assert step_record.artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT.value


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
            task_mode_hint=StepTaskModeHint.GENERAL,
            output_mode=StepOutputMode.FILE,
            artifact_policy=StepArtifactPolicy.DEFAULT,
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
    assert existing_step_record.objective_key == "objective-step-3"
    assert existing_step_record.status == ExecutionStatus.COMPLETED.value
    assert existing_step_record.task_mode_hint == StepTaskModeHint.GENERAL.value
    assert existing_step_record.output_mode == StepOutputMode.FILE.value
    assert existing_step_record.artifact_policy == StepArtifactPolicy.DEFAULT.value
    assert existing_step_record.outcome == {
        "done": True,
        "summary": "完成",
        "produced_artifacts": ["/tmp/file-1.md"],
        "blockers": [],
        "facts_learned": [],
        "open_questions": [],
        "deliver_result_as_attachment": None,
        "next_hint": None,
        "reused_from_run_id": None,
        "reused_from_step_id": None,
    }


def test_mark_unfinished_steps_cancelled_should_only_cancel_non_terminal_steps() -> None:
    repo = _build_repo(SimpleNamespace())
    step_records = [
        SimpleNamespace(status=ExecutionStatus.PENDING.value),
        SimpleNamespace(status=ExecutionStatus.RUNNING.value),
        SimpleNamespace(status=ExecutionStatus.COMPLETED.value),
        SimpleNamespace(status=ExecutionStatus.FAILED.value),
        SimpleNamespace(status=ExecutionStatus.CANCELLED.value),
    ]
    repo.db_session.execute = AsyncMock(
        return_value=SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: step_records),
        )
    )

    asyncio.run(repo.mark_unfinished_steps_cancelled(run_id="run-1"))

    assert [record.status for record in step_records] == [
        ExecutionStatus.CANCELLED.value,
        ExecutionStatus.CANCELLED.value,
        ExecutionStatus.COMPLETED.value,
        ExecutionStatus.FAILED.value,
        ExecutionStatus.CANCELLED.value,
    ]


def test_workflow_run_step_model_to_domain_should_filter_non_file_artifacts_from_outcome() -> None:
    record = WorkflowRunStepModel(
        id="record-1",
        run_id="run-1",
        step_id="step-1",
        step_index=0,
        title="步骤1",
        description="步骤1",
        objective_key="objective-step-1",
        success_criteria=["步骤1完成"],
        status=ExecutionStatus.COMPLETED.value,
        updated_at=datetime(2026, 4, 10, 12, 0, 0),
        created_at=datetime(2026, 4, 10, 12, 0, 0),
        outcome={
            "done": True,
            "summary": "完成",
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

    assert domain_record.outcome is not None
    assert domain_record.outcome.produced_artifacts == ["/tmp/file-1.md"]


def test_list_events_should_return_empty_when_run_id_missing() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    repo._list_events_by_run_id = AsyncMock()

    events = asyncio.run(repo.list_events(run_id=None))

    assert events == []
    repo._list_events_by_run_id.assert_not_awaited()


def test_update_status_should_not_clear_current_step_when_current_step_is_unset() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    run_record = SimpleNamespace(
        status=WorkflowRunStatus.RUNNING.value,
        finished_at=None,
        last_event_at=None,
        current_step_id="step-1",
    )
    repo._get_record_with_lock = AsyncMock(return_value=run_record)

    asyncio.run(
        repo.update_status(
            "run-1",
            status=WorkflowRunStatus.WAITING,
        )
    )

    assert run_record.status == WorkflowRunStatus.WAITING.value
    assert run_record.current_step_id == "step-1"


def test_update_status_should_clear_current_step_when_current_step_is_explicit_none() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    run_record = SimpleNamespace(
        status=WorkflowRunStatus.RUNNING.value,
        finished_at=None,
        last_event_at=None,
        current_step_id="step-1",
    )
    repo._get_record_with_lock = AsyncMock(return_value=run_record)

    asyncio.run(
        repo.update_status(
            "run-1",
            status=WorkflowRunStatus.COMPLETED,
            current_step_id=None,
        )
    )

    assert run_record.status == WorkflowRunStatus.COMPLETED.value
    assert run_record.current_step_id is None


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
