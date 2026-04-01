from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.service import ModelConfigService
from app.domain.models import LLMModelConfig, MessageEvent, Session, SessionStatus, User, WorkflowRunEventRecord
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.services import get_model_config_service, get_session_service
from app.interfaces.endpoints.app_config_routes import router as app_config_router
from app.interfaces.endpoints.session_routes import router as session_router
from app.interfaces.errors.exception_handlers import register_exception_handlers


def _build_current_user() -> User:
    return User(
        id="user-a",
        email="user-a@example.com",
        password="hashed-password",
    )


class _RouteModelConfigService(ModelConfigService):
    def __init__(self) -> None:
        pass

    async def get_public_models(self):
        return "gpt-5.4", [
            LLMModelConfig(
                id="gpt-5.4",
                provider="openai",
                display_name="GPT-5.4",
                base_url="https://api.openai.com/v1",
                api_key="secret-key",
                model_name="gpt-5.4",
                enabled=True,
                sort_order=1,
                config={
                    "temperature": 0.7,
                    "max_tokens": 8192,
                    "description": "复杂推理",
                    "badge": "Reasoning",
                },
            )
        ]


class _RouteSessionService:
    async def set_current_model(self, user_id: str, session_id: str, model_id: str):
        return Session(
            id=session_id,
            user_id=user_id,
            title="我的会话",
            status=SessionStatus.PENDING,
            current_model_id=model_id,
        )

    async def get_session(self, user_id: str, session_id: str):
        return Session(
            id=session_id,
            user_id=user_id,
            title="我的会话",
            status=SessionStatus.RUNNING,
            current_model_id="auto",
        )

    async def get_session_detail(self, user_id: str, session_id: str):
        session = await self.get_session(user_id=user_id, session_id=session_id)
        return session, [
            WorkflowRunEventRecord(
                id="record-1",
                run_id="run-1",
                session_id=session_id,
                event_id="evt-1",
                event_type="message",
                event_payload=MessageEvent(id="evt-1", role="assistant", message="first"),
            ),
            WorkflowRunEventRecord(
                id="record-2",
                run_id="run-2",
                session_id=session_id,
                event_id="evt-2",
                event_type="message",
                event_payload=MessageEvent(id="evt-2", role="assistant", message="second"),
            ),
        ], {
            "run-1": {"downgrade_reason": "reason-a"},
            "run-2": {"downgrade_reason": "reason-b"},
        }

    async def get_runtime_extensions(self, user_id: str, session_id: str):
        return None, {}


def test_get_models_route_should_return_public_model_fields() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(app_config_router, prefix="/api")
    app.dependency_overrides[get_model_config_service] = lambda: _RouteModelConfigService()

    with TestClient(app) as client:
        response = client.get("/api/app-config/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["default_model_id"] == "gpt-5.4"
    assert payload["data"]["models"][0]["id"] == "gpt-5.4"
    assert payload["data"]["models"][0]["config"]["badge"] == "Reasoning"
    assert "api_key" not in payload["data"]["models"][0]
    assert "base_url" not in payload["data"]["models"][0]


def test_session_model_routes_should_update_and_return_current_model_id() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(session_router, prefix="/api")
    app.dependency_overrides[get_current_user] = _build_current_user
    app.dependency_overrides[get_session_service] = lambda: _RouteSessionService()

    with TestClient(app) as client:
        update_response = client.post(
            "/api/sessions/session-a/model",
            json={"model_id": "gpt-5.4"},
        )
        detail_response = client.get("/api/sessions/session-a")

    assert update_response.status_code == 200
    assert update_response.json()["data"]["current_model_id"] == "gpt-5.4"

    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["current_model_id"] == "auto"
    detail_events = detail_response.json()["data"]["events"]
    assert detail_events[0]["data"]["extensions"]["runtime"]["run_id"] == "run-1"
    assert detail_events[0]["data"]["extensions"]["runtime"]["downgrade_reason"] == "reason-a"
    assert detail_events[1]["data"]["extensions"]["runtime"]["run_id"] == "run-2"
    assert detail_events[1]["data"]["extensions"]["runtime"]["downgrade_reason"] == "reason-b"
