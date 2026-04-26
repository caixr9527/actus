import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.errors import error_keys
from app.domain.models import ErrorEvent, Session, SessionStatus, User
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.services import get_agent_service, get_session_service
from app.interfaces.endpoints.session_routes import router as session_router
from app.interfaces.errors.exception_handlers import register_exception_handlers


class _RouteWaitingSessionService:
    async def get_session(self, user_id: str, session_id: str):
        return Session(
            id=session_id,
            user_id=user_id,
            status=SessionStatus.WAITING,
            current_run_id="run-1",
        )


class _RouteResumeInvalidAgentService:
    def __init__(self) -> None:
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        yield ErrorEvent(
            error="当前恢复输入与等待态要求不匹配，请按界面提示重新提交",
            error_key=error_keys.SESSION_RESUME_VALUE_INVALID,
            error_params={"session_id": kwargs["session_id"]},
        )


def _build_current_user() -> User:
    return User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )


def test_session_chat_route_should_stream_error_event_when_resume_value_invalid() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(session_router, prefix="/api")

    agent_service = _RouteResumeInvalidAgentService()
    app.dependency_overrides[get_session_service] = lambda: _RouteWaitingSessionService()
    app.dependency_overrides[get_agent_service] = lambda: agent_service
    app.dependency_overrides[get_current_user] = _build_current_user

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/sessions/session-1/chat",
            json={"resume": {"value": {"message": "wrong-key"}}},
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line]

    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["resume"] == {"message": "wrong-key"}

    normalized_lines = [line.decode("utf-8") if isinstance(line, bytes) else line for line in lines]
    assert "event: error" in normalized_lines

    data_line = next(line for line in normalized_lines if line.startswith("data: "))
    payload = json.loads(data_line[6:])
    assert payload["error_key"] == error_keys.SESSION_RESUME_VALUE_INVALID
    assert payload["error_params"] == {"session_id": "session-1"}
