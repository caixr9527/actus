import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import postgresql

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
    FeedbackEvent,
    WorkflowRunEventRecord,
    WorkflowRunStatus,
)
from app.domain.models.feedback import FeedbackEventPayloadResult
from app.infrastructure.models.workflow_run import WorkflowRunModel
from app.infrastructure.models.workflow_run_event import WorkflowRunEventModel
from app.infrastructure.models.workflow_run_step import WorkflowRunStepModel
from app.infrastructure.repositories.db_workflow_run_repository import DBWorkflowRunRepository


class _NestedTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _ExecuteQueue:
    def __init__(self, *results) -> None:
        self._results = list(results)

    async def __call__(self, statement):
        if not self._results:
            return SimpleNamespace(scalar_one_or_none=lambda: None)
        return self._results.pop(0)


def _build_repo(execute_result) -> DBWorkflowRunRepository:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=execute_result),
        add=MagicMock(),
    )
    return DBWorkflowRunRepository(db_session=db_session)


def _build_event_insert_repo() -> DBWorkflowRunRepository:
    run_record = SimpleNamespace(
        id="run-1",
        user_id="user-1",
        session_id="session-1",
        current_step_id=None,
    )
    db_session = SimpleNamespace(
        execute=_ExecuteQueue(
            SimpleNamespace(scalar_one_or_none=lambda: None),
            SimpleNamespace(scalar_one_or_none=lambda: run_record),
        ),
        add=MagicMock(),
        begin_nested=MagicMock(return_value=_NestedTransaction()),
        flush=AsyncMock(),
    )
    return DBWorkflowRunRepository(db_session=db_session)


def test_add_event_if_absent_should_sync_step_projection_for_step_event() -> None:
    repo = _build_event_insert_repo()
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
    event_record = repo.db_session.add.call_args.args[0]
    assert event_record.user_id == "user-1"
    repo._refresh_run_status_by_event.assert_not_awaited()
    repo.upsert_step_from_event.assert_awaited_once()
    repo.replace_steps_from_plan.assert_not_awaited()


def test_add_event_if_absent_should_sync_step_projection_for_plan_event() -> None:
    repo = _build_event_insert_repo()
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
    event_record = repo.db_session.add.call_args.args[0]
    assert event_record.user_id == "user-1"
    repo._refresh_run_status_by_event.assert_not_awaited()
    repo.replace_steps_from_plan.assert_awaited_once()
    repo.upsert_step_from_event.assert_not_awaited()


def test_upsert_feedback_event_record_should_preserve_row_and_payload_created_at_on_replace() -> None:
    first_created_at = datetime(2026, 5, 20, 9, 0, 0)
    replacement_created_at = datetime(2026, 5, 20, 9, 5, 0)
    run_record = WorkflowRunModel(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        status=WorkflowRunStatus.RUNNING.value,
    )
    existing_event = FeedbackEvent(
        id="feedback:run-1:evt-source",
        created_at=first_created_at,
        payload=FeedbackEventPayloadResult(
            feedback_refs=["fb-1"],
            counts={"feedback_count": 1},
            severity_counts={"error": 1},
            status_counts={"open": 1},
            kind_counts={"runtime_feedback": 1},
            summary="first",
            source_event_ids=["evt-source"],
            runtime_metadata={"schema_version": "feedback_event.v1"},
        ),
    )
    event_record = WorkflowRunEventModel(
        id="workflow-event-row-1",
        run_id="run-1",
        session_id="session-1",
        user_id="user-1",
        event_id=existing_event.id,
        event_type="feedback",
        event_payload=existing_event.model_dump(mode="json"),
        created_at=first_created_at,
    )
    db_session = SimpleNamespace(
        execute=_ExecuteQueue(
            SimpleNamespace(scalar_one_or_none=lambda: run_record),
            SimpleNamespace(scalar_one_or_none=lambda: event_record),
        ),
        flush=AsyncMock(),
    )
    repo = DBWorkflowRunRepository(db_session=db_session)
    replacement = FeedbackEvent(
        id=existing_event.id,
        created_at=replacement_created_at,
        payload=FeedbackEventPayloadResult(
            feedback_refs=["fb-1", "fb-2"],
            counts={"feedback_count": 2},
            severity_counts={"error": 2},
            status_counts={"open": 2},
            kind_counts={"runtime_feedback": 2},
            summary="replacement",
            source_event_ids=["evt-source"],
            runtime_metadata={"schema_version": "feedback_event.v1"},
        ),
    )

    result = asyncio.run(
        repo.upsert_feedback_event_record(
            session_id="session-1",
            run_id="run-1",
            event=replacement,
        )
    )

    assert result is not None
    assert result.created_at == first_created_at
    assert result.event_payload.created_at == first_created_at
    assert result.event_payload.payload.feedback_refs == ["fb-1", "fb-2"]
    assert event_record.created_at == first_created_at
    assert event_record.event_payload["created_at"] == first_created_at.isoformat()


def test_upsert_feedback_event_record_should_reject_session_mismatch_without_write() -> None:
    run_record = WorkflowRunModel(
        id="run-1",
        session_id="other-session",
        user_id="user-1",
        status=WorkflowRunStatus.RUNNING.value,
    )
    db_session = SimpleNamespace(
        execute=_ExecuteQueue(SimpleNamespace(scalar_one_or_none=lambda: run_record)),
        add=MagicMock(),
        begin_nested=MagicMock(return_value=_NestedTransaction()),
        flush=AsyncMock(),
    )
    repo = DBWorkflowRunRepository(db_session=db_session)
    event = FeedbackEvent(
        id="feedback:run-1:evt-source",
        created_at=datetime(2026, 5, 20, 9, 0, 0),
        payload=FeedbackEventPayloadResult(
            feedback_refs=["fb-1"],
            counts={"feedback_count": 1},
            severity_counts={"error": 1},
            status_counts={"open": 1},
            kind_counts={"runtime_feedback": 1},
            summary="feedback",
            source_event_ids=["evt-source"],
            runtime_metadata={"schema_version": "feedback_event.v1"},
        ),
    )

    result = asyncio.run(
        repo.upsert_feedback_event_record(
            session_id="session-1",
            run_id="run-1",
            event=event,
        )
    )

    assert result is None
    db_session.add.assert_not_called()
    db_session.flush.assert_not_awaited()


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
        "evidence_backed_facts": [],
        "facts_learned": [],
        "open_questions": [],
        "deliver_result_as_attachment": None,
        "next_hint": None,
        "reused_from_run_id": None,
        "reused_from_step_id": None,
        "evidence_reconcile_metadata": {},
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


def test_get_event_record_by_type_and_hash_should_filter_input_hash_in_sql() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = _build_repo(execute_result)

    record = asyncio.run(
        repo.get_event_record_by_type_and_hash(
            user_id="user-1",
            session_id="session-1",
            run_id="run-1",
            event_type="feedback_input",
            input_hash="feedback_input:hash-1",
        )
    )

    assert record is None
    statement = repo.db_session.execute.await_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "workflow_run_events.user_id" in compiled_sql
    assert "workflow_run_events.session_id" in compiled_sql
    assert "workflow_run_events.run_id" in compiled_sql
    assert "workflow_run_events.event_type" in compiled_sql
    assert "#>>" in compiled_sql
    assert statement.compile(dialect=postgresql.dialect()).params["event_payload_1"] == ("payload", "input_hash")
    assert "LIMIT" in compiled_sql


def test_get_latest_event_record_by_session_should_return_single_record() -> None:
    latest_event = MessageEvent(id="evt-latest", role="assistant", message="latest")
    execute_result = SimpleNamespace(
        scalar_one_or_none=lambda: SimpleNamespace(
            to_domain=lambda: WorkflowRunEventRecord(
                id="record-latest",
                run_id="run-2",
                session_id="session-1",
                event_id="evt-latest",
                event_type="message",
                event_payload=latest_event,
            )
        )
    )
    repo = _build_repo(execute_result)

    record = asyncio.run(repo.get_latest_event_record_by_session(
        session_id="session-1",
        event_type="message",
        run_id="run-2",
    ))

    assert record is not None
    assert record.event_id == "evt-latest"
