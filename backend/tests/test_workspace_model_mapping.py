from datetime import datetime, timedelta

from app.domain.models import Workspace
from app.infrastructure.models import WorkspaceModel


def test_workspace_model_from_domain_should_keep_datetime_fields() -> None:
    now = datetime.now()
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        shell_session_id="shell-1",
        cwd="/home/ubuntu/workspace",
        browser_snapshot={"url": "https://example.com"},
        environment_summary={"latest_shell_result": {"message": "ok"}},
        last_active_at=now,
        expires_at=now + timedelta(hours=1),
        created_at=now,
        updated_at=now + timedelta(seconds=1),
    )

    record = WorkspaceModel.from_domain(workspace)

    assert isinstance(record.last_active_at, datetime)
    assert isinstance(record.expires_at, datetime)
    assert isinstance(record.created_at, datetime)
    assert isinstance(record.updated_at, datetime)
    assert record.browser_snapshot["url"] == "https://example.com"


def test_workspace_model_update_from_domain_should_keep_datetime_fields() -> None:
    now = datetime.now()
    record = WorkspaceModel(
        id="workspace-2",
        session_id="session-2",
        status="active",
        cwd="",
        browser_snapshot={},
        environment_summary={},
        last_active_at=now,
        expires_at=None,
        created_at=now,
        updated_at=now,
    )

    updated = Workspace(
        id="workspace-2",
        session_id="session-2",
        current_run_id="run-2",
        sandbox_id="sandbox-2",
        task_id="task-2",
        shell_session_id="shell-2",
        cwd="/home/ubuntu/workspace/new",
        browser_snapshot={"title": "test"},
        environment_summary={"shell_session_status": "ready"},
        status="active",
        last_active_at=now + timedelta(minutes=1),
        expires_at=now + timedelta(days=1),
        created_at=now,
        updated_at=now + timedelta(minutes=1),
    )

    record.update_from_domain(updated)

    assert record.current_run_id == "run-2"
    assert record.cwd == "/home/ubuntu/workspace/new"
    assert record.environment_summary["shell_session_status"] == "ready"
    assert isinstance(record.last_active_at, datetime)
    assert isinstance(record.expires_at, datetime)
    assert isinstance(record.updated_at, datetime)
