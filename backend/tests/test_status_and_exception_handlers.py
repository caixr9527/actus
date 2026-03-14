from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.errors import NotFoundError
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
