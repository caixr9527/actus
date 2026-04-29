import asyncio

import pytest

from app.application.errors import ValidationError, error_keys
from app.application.service.session_service import SessionService
from app.domain.models import LLMModelConfig, Session


class _OwnedSessionRepo:
    def __init__(self) -> None:
        self.session = Session(id="session-a", user_id="user-a", title="测试会话")
        self.updated_current_model_id: str | None = None

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        if session_id == self.session.id and user_id == self.session.user_id:
            return self.session
        return None

    async def update_current_model_id(self, session_id: str, current_model_id: str | None) -> None:
        self.updated_current_model_id = current_model_id
        self.session.current_model_id = current_model_id


class _SessionUoW:
    def __init__(self, session_repo: _OwnedSessionRepo) -> None:
        self.session = session_repo
        self.workspace = _EmptyWorkspaceRepo()
        self.workflow_run = _EmptyWorkflowRunRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeModelConfigService:
    def __init__(self, enabled_model_ids: set[str]) -> None:
        self._enabled_model_ids = enabled_model_ids

    async def get_enabled_model_by_id(self, model_id: str):
        if model_id not in self._enabled_model_ids:
            return None
        return LLMModelConfig(
            id=model_id,
            provider="openai",
            display_name=model_id.upper(),
            base_url="https://api.example.com/v1",
            api_key="secret",
            model_name=model_id,
        )


class _DummySandbox:
    @classmethod
    async def get(cls, id: str):
        return None


class _EmptyWorkspaceRepo:
    async def get_by_id(self, workspace_id: str):
        return None

    async def get_by_session_id(self, session_id: str):
        return None

    async def list_by_session_id(self, session_id: str):
        return []


class _EmptyWorkflowRunRepo:
    async def get_by_id(self, run_id: str):
        return None


def test_session_service_set_current_model_should_accept_auto() -> None:
    session_repo = _OwnedSessionRepo()
    service = SessionService(
        uow_factory=lambda: _SessionUoW(session_repo),
        sandbox_cls=_DummySandbox,
        model_config_service=_FakeModelConfigService(enabled_model_ids=set()),
    )

    session = asyncio.run(service.set_current_model("user-a", "session-a", "auto"))

    assert session.current_model_id == "auto"
    assert session_repo.updated_current_model_id == "auto"


def test_session_service_set_current_model_should_reject_invalid_model() -> None:
    session_repo = _OwnedSessionRepo()
    service = SessionService(
        uow_factory=lambda: _SessionUoW(session_repo),
        sandbox_cls=_DummySandbox,
        model_config_service=_FakeModelConfigService(enabled_model_ids={"gpt-5.4"}),
    )

    with pytest.raises(ValidationError) as exc:
        asyncio.run(service.set_current_model("user-a", "session-a", "kimi"))

    assert exc.value.error_key == error_keys.SESSION_MODEL_ID_INVALID
    assert exc.value.error_params == {"model_id": "kimi"}
