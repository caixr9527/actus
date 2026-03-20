from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException

from app.application.errors import AppException, NotFoundError
from app.application.errors import error_keys
from app.domain.models import HealthStatus
from app.interfaces.endpoints.status_routes import router as status_router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from app.interfaces.dependencies.services import get_status_service


class _FakeStatusService:
    def __init__(self, statuses: list[HealthStatus]):
        self._statuses = statuses

    async def check_all(self) -> list[HealthStatus]:
        return self._statuses


def _build_status_app(statuses: list[HealthStatus]) -> FastAPI:
    app = FastAPI()
    app.include_router(status_router, prefix="/api")
    app.dependency_overrides[get_status_service] = lambda: _FakeStatusService(statuses)
    return app


def test_status_route_returns_503_when_any_checker_is_error() -> None:
    app = _build_status_app(
        statuses=[
            HealthStatus(service="postgres", status="ok", details=""),
            HealthStatus(service="redis", status="ERROR", details="down"),
        ]
    )
    with TestClient(app) as client:
        response = client.get("/api/status")

    assert response.status_code == 503
    payload = response.json()
    assert payload["code"] == 503
    assert payload["msg"] == "系统服务存在异常"
    assert payload["error_key"] == error_keys.STATUS_UNHEALTHY


def test_status_route_returns_200_when_all_services_are_healthy() -> None:
    app = _build_status_app(
        statuses=[
            HealthStatus(service="postgres", status="ok", details=""),
            HealthStatus(service="redis", status="ok", details=""),
        ]
    )
    with TestClient(app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "系统服务正常"


def test_app_exception_handler_maps_not_found_to_404() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-not-found")
    async def raise_not_found() -> None:
        raise NotFoundError("资源不存在")

    with TestClient(app) as client:
        response = client.get("/raise-not-found")

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == 404
    assert payload["msg"] == "资源不存在"
    assert payload["error_key"] == "error.common.not_found"
    assert payload["error_params"] is None


def test_app_exception_handler_should_return_custom_error_key_and_params() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-app-error")
    async def raise_app_error() -> None:
        raise AppException(
            code=409,
            status_code=409,
            msg="状态冲突",
            error_key="error.session.conflict",
            error_params={"session_id": "session-1"},
        )

    with TestClient(app) as client:
        response = client.get("/raise-app-error")

    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == 409
    assert payload["msg"] == "状态冲突"
    assert payload["error_key"] == "error.session.conflict"
    assert payload["error_params"] == {"session_id": "session-1"}


def test_app_exception_handler_should_allow_msg_to_be_optional() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-app-error-without-msg")
    async def raise_app_error_without_msg() -> None:
        raise AppException(
            code=409,
            status_code=409,
            msg=None,
            error_key="error.session.conflict",
        )

    with TestClient(app) as client:
        response = client.get("/raise-app-error-without-msg")

    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == 409
    assert payload["msg"] is None
    assert payload["error_key"] == "error.session.conflict"
    assert payload["error_params"] is None


def test_http_exception_handler_should_return_http_error_key() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-http-error")
    async def raise_http_error() -> None:
        raise HTTPException(status_code=403, detail="禁止访问")

    with TestClient(app) as client:
        response = client.get("/raise-http-error")

    assert response.status_code == 403
    payload = response.json()
    assert payload["code"] == 403
    assert payload["msg"] == "禁止访问"
    assert payload["error_key"] == "error.http.403"
    assert payload["error_params"] is None


def test_unhandled_exception_handler_should_return_internal_error_key() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-exception")
    async def raise_exception() -> None:
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/raise-exception")

    assert response.status_code == 500
    payload = response.json()
    assert payload["code"] == 500
    assert payload["msg"] == "服务器异常"
    assert payload["error_key"] == "error.common.internal_server_error"
    assert payload["error_params"] is None
