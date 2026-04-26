import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy.dialects import postgresql

from app.domain.models import SessionContextSnapshot, WorkflowRunSummary, WorkflowRunStatus
from app.infrastructure.repositories.db_session_context_snapshot_repository import (
    DBSessionContextSnapshotRepository,
)
from app.infrastructure.repositories.db_workflow_run_summary_repository import DBWorkflowRunSummaryRepository


def test_workflow_run_summary_upsert_should_use_postgresql_on_conflict() -> None:
    inserted_result = SimpleNamespace()
    selected_result = SimpleNamespace(
        scalar_one_or_none=lambda: SimpleNamespace(
            to_domain=lambda: WorkflowRunSummary(
                id="summary-1",
                run_id="run-1",
                session_id="session-1",
                status=WorkflowRunStatus.COMPLETED,
            )
        )
    )
    db_session = SimpleNamespace(execute=AsyncMock(side_effect=[inserted_result, selected_result]))
    repository = DBWorkflowRunSummaryRepository(db_session=db_session)

    asyncio.run(
        repository.upsert(
            WorkflowRunSummary(
                id="summary-1",
                run_id="run-1",
                session_id="session-1",
                title="title",
                updated_at=datetime(2026, 4, 1, 10, 0, 0),
            )
        )
    )

    statement = db_session.execute.call_args_list[0].args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT" in compiled_sql
    assert "workflow_run_summaries" in compiled_sql


def test_session_context_snapshot_upsert_should_use_postgresql_on_conflict() -> None:
    inserted_result = SimpleNamespace()
    selected_result = SimpleNamespace(
        scalar_one_or_none=lambda: SimpleNamespace(
            to_domain=lambda: SessionContextSnapshot(
                session_id="session-1",
                summary_text="summary",
            )
        )
    )
    db_session = SimpleNamespace(execute=AsyncMock(side_effect=[inserted_result, selected_result]))
    repository = DBSessionContextSnapshotRepository(db_session=db_session)

    asyncio.run(
        repository.upsert(
            SessionContextSnapshot(
                session_id="session-1",
                summary_text="summary",
                updated_at=datetime(2026, 4, 1, 10, 0, 0),
            )
        )
    )

    statement = db_session.execute.call_args_list[0].args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT" in compiled_sql
    assert "session_context_snapshots" in compiled_sql
