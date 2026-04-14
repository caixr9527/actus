import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

from app.domain.models import Workspace
from app.infrastructure.repositories.db_workspace_repository import DBWorkspaceRepository


def test_db_workspace_repository_save_should_use_upsert_statement() -> None:
    db_session = AsyncMock()
    repo = DBWorkspaceRepository(db_session=db_session)
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        shell_session_id="shell-1",
        cwd="/home/ubuntu/workspace",
        browser_snapshot={},
        environment_summary={},
        last_active_at=datetime.now(),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    asyncio.run(repo.save(workspace=workspace))

    db_session.execute.assert_awaited_once()
    stmt = db_session.execute.await_args.args[0]
    rendered = str(stmt).upper()
    assert "INSERT INTO WORKSPACES" in rendered
    assert "ON CONFLICT" in rendered
