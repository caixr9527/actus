from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from app.domain.models import HealthStatus
from app.interfaces.endpoints.routes import router as api_router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from app.interfaces.schemas.auth import RefreshResult
from app.interfaces.dependencies.services import get_status_service, get_auth_service


class _FakeStatusService:
    async def check_all(self):
        return [HealthStatus(service="api", status="ok", details="ready")]


class _FakeAuthService:
    async def refresh_tokens(self, refresh_token: str):
        return RefreshResult(
            access_token="next-access-token",
            refresh_token="next-refresh-token",
            access_token_expires_in=1800,
            refresh_token_expires_in=604800,
        )


def test_protected_http_routes_should_require_authorization_header() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api")

    with TestClient(app) as client:
        sessions_response = client.get("/api/sessions")
        files_response = client.get("/api/files/file-1")
        app_config_response = client.get("/api/app-config/llm")

    assert sessions_response.status_code == 401
    assert sessions_response.json()["code"] == 401
    assert files_response.status_code == 401
    assert app_config_response.status_code == 401


def test_whitelisted_routes_should_not_require_authorization_header() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api")
    app.dependency_overrides[get_status_service] = lambda: _FakeStatusService()
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        status_response = client.get("/api/status")
        refresh_response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": "rt-1"},
        )

    assert status_response.status_code == 200
    assert status_response.json()["code"] == 200
    assert refresh_response.status_code == 200
    assert refresh_response.json()["code"] == 200


def test_session_vnc_websocket_should_require_authorization_header() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api")

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/api/sessions/session-1/vnc"):
                pass

    assert exc.value.code == 1008
