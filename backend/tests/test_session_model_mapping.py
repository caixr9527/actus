from datetime import datetime

from app.domain.models import Session
from app.infrastructure.models import SessionModel


def test_session_model_from_domain_should_include_current_model_id() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        current_model_id="gpt-5.4",
        workspace_id="workspace-1",
        current_run_id="run-1",
        title="测试会话",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    record = SessionModel.from_domain(session)

    assert record.current_model_id == "gpt-5.4"
    assert record.workspace_id == "workspace-1"
    assert record.current_run_id == "run-1"


def test_session_model_to_domain_should_map_current_model_id() -> None:
    now = datetime.now()
    record = SessionModel(
        id="session-2",
        user_id="user-2",
        current_model_id="auto",
        workspace_id="workspace-2",
        current_run_id="run-2",
        title="测试会话2",
        unread_message_count=0,
        latest_message="",
        latest_message_at=None,
        files=[],
        status="pending",
        created_at=now,
        updated_at=now,
    )

    session = record.to_domain()

    assert session.current_model_id == "auto"
    assert session.workspace_id == "workspace-2"
    assert session.current_run_id == "run-2"
