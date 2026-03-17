import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.errors import NotFoundError
from app.application.errors import error_keys
from app.domain.models import ErrorEvent
from app.interfaces.endpoints.session_routes import router as session_router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from app.interfaces.dependencies.services import get_agent_service, get_session_service
from app.application.service.agent_service import AgentService
from app.application.service.session_service import SessionService


class _MissingSessionRepo:
    async def get_by_id(self, session_id: str):
        return None


class _MissingSessionUoW:
    def __init__(self) -> None:
        self.session = _MissingSessionRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _missing_session_uow_factory() -> _MissingSessionUoW:
    return _MissingSessionUoW()


class _DummySandbox:
    @classmethod
    async def get(cls, id: str):
        return None


def _build_session_service() -> SessionService:
    return SessionService(
        uow_factory=_missing_session_uow_factory,
        sandbox_cls=_DummySandbox,
    )


def test_session_service_get_session_files_missing_session_raises_not_found() -> None:
    service = _build_session_service()

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.get_session_files("session-1"))
    assert "任务会话不存在" in exc.value.msg
    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_session_service_read_file_missing_session_raises_not_found() -> None:
    service = _build_session_service()

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.read_file("session-1", "/tmp/a.txt"))
    assert "任务会话不存在" in exc.value.msg
    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_session_service_read_shell_output_missing_session_raises_not_found() -> None:
    service = _build_session_service()

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.read_shell_output("session-1", "shell-1"))
    assert "任务会话不存在" in exc.value.msg
    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_session_service_get_vnc_url_missing_session_raises_not_found() -> None:
    service = _build_session_service()

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.get_vnc_url("session-1"))
    assert "任务会话不存在" in exc.value.msg
    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_agent_service_stop_session_missing_session_raises_not_found() -> None:
    service = object.__new__(AgentService)
    service._uow_factory = _missing_session_uow_factory

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.stop_session("session-1"))
    assert exc.value.msg == "会话session-1不存在"
    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_agent_service_chat_missing_session_should_yield_error_event_with_error_key() -> None:
    service = object.__new__(AgentService)
    service._uow_factory = _missing_session_uow_factory

    async def _collect_first_event():
        async for event in service.chat("session-1", message="hello"):
            return event
        return None

    event = asyncio.run(_collect_first_event())

    assert isinstance(event, ErrorEvent)
    assert event.error_key == error_keys.SESSION_NOT_FOUND
    assert event.error_params == {"session_id": "session-1"}


class _RouteMissingSessionService:
    async def get_session(self, session_id: str):
        return None


class _RouteAgentService:
    async def chat(self, *args, **kwargs):
        raise AssertionError("chat should not be called when session does not exist")


def test_session_chat_route_returns_404_when_session_not_found() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(session_router, prefix="/api")
    app.dependency_overrides[get_session_service] = lambda: _RouteMissingSessionService()
    app.dependency_overrides[get_agent_service] = lambda: _RouteAgentService()

    with TestClient(app) as client:
        response = client.post("/api/sessions/session-1/chat", json={})

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == 404
    assert payload["msg"] == "该会话不存在，请核实后重试"
    assert payload["error_key"] == error_keys.SESSION_NOT_FOUND
