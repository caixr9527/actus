import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.domain.models import MessageEvent
from app.infrastructure.repositories.db_session_repository import DBSessionRepository


class _FakeSessionRecord:
    def __init__(self) -> None:
        self.id = "session-1"
        self.user_id = "user-1"
        self.current_run_id = "run-active"

    def to_domain(self):
        return SimpleNamespace(
            id=self.id,
            user_id=self.user_id,
            current_run_id=self.current_run_id,
            events=[],
            files=[],
        )


def test_get_by_id_should_hydrate_events_across_all_runs_in_session() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: _FakeSessionRecord())
    db_session = SimpleNamespace(execute=AsyncMock(return_value=execute_result), info={})
    repository = DBSessionRepository(db_session=db_session)
    cross_run_events = [
        MessageEvent(id="evt-run-1", role="assistant", message="history-1"),
        MessageEvent(id="evt-run-2", role="assistant", message="history-2"),
    ]
    repository._workflow_run_repository = SimpleNamespace(
        list_events_by_session=AsyncMock(return_value=cross_run_events),
    )

    session = asyncio.run(repository.get_by_id(session_id="session-1", user_id="user-1"))

    assert session is not None
    assert [event.id for event in session.events] == ["evt-run-1", "evt-run-2"]
    repository._workflow_run_repository.list_events_by_session.assert_awaited_once_with("session-1")
