import asyncio

import pytest

from app.application.errors import ValidationError, error_keys
from app.application.service.session_service import SessionService
from app.domain.models import LLMModelConfig, MessageEvent, Session, WorkflowRunEventRecord


class _OwnedSessionRepo:
    def __init__(self) -> None:
        self.session = Session(id="session-a", user_id="user-a", title="测试会话")
        self.updated_current_model_id: str | None = None
        self.get_by_id_calls = 0
        self.get_by_id_without_events_calls = 0

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        self.get_by_id_calls += 1
        self.session.events = [
            MessageEvent(id="evt-hydrated", role="assistant", message="should not be used"),
        ]
        if session_id == self.session.id and user_id == self.session.user_id:
            return self.session
        return None

    async def get_by_id_without_events(self, session_id: str, user_id: str | None = None):
        self.get_by_id_without_events_calls += 1
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
    def __init__(self) -> None:
        self.event_records = [
            WorkflowRunEventRecord(
                run_id="run-a",
                session_id="session-a",
                event_id="evt-1",
                event_type="message",
                event_payload=MessageEvent(
                    id="evt-1",
                    role="assistant",
                    message="hello",
                ),
            ),
        ]
        self.list_event_records_by_session_calls = 0

    async def get_by_id(self, run_id: str):
        return None

    async def list_event_records_by_session(self, session_id: str):
        self.list_event_records_by_session_calls += 1
        return [
            record for record in self.event_records
            if record.session_id == session_id
        ]


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


def test_get_session_detail_should_not_hydrate_session_events_twice() -> None:
    session_repo = _OwnedSessionRepo()
    uow = _SessionUoW(session_repo)
    service = SessionService(
        uow_factory=lambda: uow,
        sandbox_cls=_DummySandbox,
        model_config_service=_FakeModelConfigService(enabled_model_ids=set()),
    )

    session, records = asyncio.run(service.get_session_detail("user-a", "session-a"))

    assert session is session_repo.session
    assert session.events == []
    assert [record.event_id for record in records] == ["evt-1"]
    assert session_repo.get_by_id_without_events_calls == 1
    assert session_repo.get_by_id_calls == 0
    assert uow.workflow_run.list_event_records_by_session_calls == 1
