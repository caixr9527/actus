import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.domain.models import Memory
from app.infrastructure.repositories.db_session_repository import DBSessionRepository


def test_save_memory_should_only_persist_session_memories() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(rowcount=1)),
        info={},
    )
    repository = DBSessionRepository(db_session=db_session)
    repository._workflow_run_repository = SimpleNamespace(
        upsert_memory_snapshot=AsyncMock(),
    )

    asyncio.run(
        repository.save_memory(
            session_id="session-1",
            agent_name="planner",
            memory=Memory(messages=[{"role": "assistant", "content": "hello"}]),
        )
    )

    repository._workflow_run_repository.upsert_memory_snapshot.assert_not_awaited()


def test_get_memory_should_not_read_workflow_run_snapshot() -> None:
    record = SimpleNamespace(
        current_run_id="run-1",
        memories={
            "planner": {
                "messages": [
                    {"role": "assistant", "content": "session-memory"},
                ]
            }
        },
    )
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: record)),
        info={},
    )
    repository = DBSessionRepository(db_session=db_session)
    repository._workflow_run_repository = SimpleNamespace(
        get_by_id=AsyncMock(),
    )

    memory = asyncio.run(repository.get_memory(session_id="session-1", agent_name="planner"))

    assert memory.messages == [{"role": "assistant", "content": "session-memory"}]
    repository._workflow_run_repository.get_by_id.assert_not_awaited()
