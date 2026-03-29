import asyncio
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
    StepEvent,
    StepEventStatus,
)
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
            description="执行步骤",
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
            steps=[Step(id="step-1", description="步骤1")],
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
            description="增量步骤",
            status=ExecutionStatus.RUNNING,
            success=False,
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
    assert step_record.status == ExecutionStatus.RUNNING.value


def test_upsert_step_from_event_should_update_existing_snapshot_and_clear_current_step() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    run_record = SimpleNamespace(
        id="run-1",
        current_step_id="step-3",
    )
    existing_step_record = SimpleNamespace(
        step_index=0,
        description="旧描述",
        status=ExecutionStatus.RUNNING.value,
        result=None,
        error=None,
        success=False,
        attachments=[],
    )
    repo._get_record_with_lock = AsyncMock(return_value=run_record)
    repo._get_step_record_with_lock = AsyncMock(return_value=existing_step_record)
    event = StepEvent(
        id="evt-step-3",
        step=Step(
            id="step-3",
            description="新描述",
            status=ExecutionStatus.COMPLETED,
            result="完成",
            success=True,
            attachments=["file-1"],
        ),
        status=StepEventStatus.COMPLETED,
    )

    asyncio.run(repo.upsert_step_from_event(run_id="run-1", event=event))

    assert run_record.current_step_id is None
    assert existing_step_record.description == "新描述"
    assert existing_step_record.status == ExecutionStatus.COMPLETED.value
    assert existing_step_record.result == "完成"
    assert existing_step_record.success is True
    assert existing_step_record.attachments == ["file-1"]


def test_get_events_with_compat_should_fallback_to_session_events_when_run_events_empty() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    repo._list_events_by_run_id = AsyncMock(return_value=[])
    session = Session(
        id="session-1",
        current_run_id="run-1",
        events=[MessageEvent(id="legacy-event-1", role="assistant", message="legacy")],
    )

    events = asyncio.run(repo.get_events_with_compat(session=session))

    assert len(events) == 1
    assert events[0].id == "legacy-event-1"
    repo._list_events_by_run_id.assert_awaited_once_with("run-1")


def test_get_events_with_compat_should_prefer_workflow_run_events() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)
    runtime_event = MessageEvent(id="runtime-event-1", role="assistant", message="runtime")
    repo._list_events_by_run_id = AsyncMock(return_value=[runtime_event])
    session = Session(
        id="session-2",
        current_run_id="run-2",
        events=[MessageEvent(id="legacy-event-2", role="assistant", message="legacy")],
    )

    events = asyncio.run(repo.get_events_with_compat(session=session))

    assert len(events) == 1
    assert events[0].id == "runtime-event-1"
    repo._list_events_by_run_id.assert_awaited_once_with("run-2")
