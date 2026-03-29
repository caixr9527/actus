import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.domain.models import (
    ExecutionStatus,
    MessageEvent,
    SessionStatus,
    Step,
    StepEvent,
    StepEventStatus,
)
from app.infrastructure.repositories.db_session_repository import DBSessionRepository


def _build_repository(record, *, event_inserted: bool = True):
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: record)
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=execute_result),
        info={},
    )
    repository = DBSessionRepository(db_session=db_session)
    repository._workflow_run_repository = SimpleNamespace(
        add_event_if_absent=AsyncMock(return_value=event_inserted),
        replace_steps_from_plan=AsyncMock(),
        upsert_step_from_event=AsyncMock(),
    )
    return repository, db_session


def test_add_event_if_absent_should_write_only_workflow_run_events() -> None:
    record = SimpleNamespace(
        id="session-1",
        current_run_id="run-1",
    )
    repository, _ = _build_repository(record=record, event_inserted=True)
    event = MessageEvent(id="evt-1", role="assistant", message="hello")

    inserted = asyncio.run(repository.add_event_if_absent(session_id="session-1", event=event))

    assert inserted is True
    repository._workflow_run_repository.add_event_if_absent.assert_awaited_once_with(
        session_id="session-1",
        run_id="run-1",
        event=event,
    )


def test_add_event_with_snapshot_if_absent_should_reconcile_projection_on_replay() -> None:
    record = SimpleNamespace(
        id="session-1",
        current_run_id="run-1",
        title="old-title",
        latest_message="old",
        latest_message_at=None,
        unread_message_count=3,
        status=SessionStatus.PENDING.value,
    )
    repository, db_session = _build_repository(record=record, event_inserted=False)
    event = StepEvent(
        id="evt-step-1",
        step=Step(
            id="step-1",
            description="执行步骤",
            status=ExecutionStatus.COMPLETED,
        ),
        status=StepEventStatus.COMPLETED,
    )

    inserted = asyncio.run(
        repository.add_event_with_snapshot_if_absent(
            session_id="session-1",
            event=event,
            title="new-title",
            latest_message="new-message",
            increment_unread=True,
            status=SessionStatus.RUNNING,
        )
    )

    assert inserted is False
    assert record.title == "new-title"
    assert record.latest_message == "new-message"
    assert record.unread_message_count == 3
    assert record.status == SessionStatus.RUNNING.value
    assert db_session.info["session_list_changed_ids"] == {"session-1"}
    assert len(db_session.info["post_commit_hooks"]) == 1
    repository._workflow_run_repository.upsert_step_from_event.assert_awaited_once()


def test_add_event_with_snapshot_if_absent_should_increment_unread_on_first_insert() -> None:
    record = SimpleNamespace(
        id="session-2",
        current_run_id="run-2",
        title="title",
        latest_message="latest",
        latest_message_at=None,
        unread_message_count=0,
        status=SessionStatus.PENDING.value,
    )
    repository, _ = _build_repository(record=record, event_inserted=True)
    event = MessageEvent(id="evt-2", role="assistant", message="reply")

    inserted = asyncio.run(
        repository.add_event_with_snapshot_if_absent(
            session_id="session-2",
            event=event,
            increment_unread=True,
            status=SessionStatus.RUNNING,
        )
    )

    assert inserted is True
    assert record.unread_message_count == 1
    assert record.status == SessionStatus.RUNNING.value


def test_add_event_if_absent_should_fail_when_current_run_id_missing() -> None:
    record = SimpleNamespace(
        id="session-3",
        current_run_id=None,
    )
    repository, _ = _build_repository(record=record, event_inserted=True)
    event = MessageEvent(id="evt-3", role="assistant", message="reply")

    with pytest.raises(RuntimeError):
        asyncio.run(repository.add_event_if_absent(session_id="session-3", event=event))

    repository._workflow_run_repository.add_event_if_absent.assert_not_awaited()
