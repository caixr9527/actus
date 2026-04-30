import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.application.errors import NotFoundError, error_keys
from app.application.service.agent_service import AgentService
from app.application.service.session_service import SessionService
from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.domain.models import ErrorEvent, Session
from app.domain.services.runtime.contracts.data_access_contract import DataAccessAction


@dataclass
class _SessionRepo:
    session: Session | None
    add_event_calls: list[tuple[str, ErrorEvent]]
    update_model_calls: list[tuple[str, str]]

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        if self.session is None or self.session.id != session_id:
            return None
        if user_id is not None and self.session.user_id != user_id:
            return None
        return self.session

    async def add_event(self, session_id: str, event: ErrorEvent) -> None:
        self.add_event_calls.append((session_id, event))

    async def update_current_model_id(self, session_id: str, current_model_id: str) -> None:
        self.update_model_calls.append((session_id, current_model_id))


class _WorkspaceRepo:
    async def get_by_id_for_user(self, workspace_id: str, user_id: str):
        return None

    async def list_by_session_id(self, session_id: str):
        return []


class _WorkflowRunRepo:
    async def get_by_id_for_user(self, run_id: str, user_id: str):
        return None


class _FileRepo:
    async def get_by_id_and_user_id(self, file_id: str, user_id: str):
        return None


class _WorkspaceArtifactRepo:
    async def list_by_user_workspace_id_and_paths(self, user_id: str, workspace_id: str, paths: list[str]):
        return []


class _UoW:
    def __init__(self, session: Session | None) -> None:
        self.add_event_calls: list[tuple[str, ErrorEvent]] = []
        self.update_model_calls: list[tuple[str, str]] = []
        self.session = _SessionRepo(
            session=session,
            add_event_calls=self.add_event_calls,
            update_model_calls=self.update_model_calls,
        )
        self.workspace = _WorkspaceRepo()
        self.workflow_run = _WorkflowRunRepo()
        self.file = _FileRepo()
        self.workspace_artifact = _WorkspaceArtifactRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SpyCoordinator:
    def __init__(self) -> None:
        self.reconcile_calls: list[tuple[str, str]] = []
        self.cancel_calls: list[tuple[str, str]] = []

    async def reconcile_current_run(self, *, session_id: str, reason: str):
        self.reconcile_calls.append((session_id, reason))
        return SimpleNamespace(warnings=[], snapshot_after=None)

    async def cancel_current_run(self, *, session_id: str, reason: str):
        self.cancel_calls.append((session_id, reason))


class _SpyGraphRuntime:
    def __init__(self) -> None:
        self.cancel_calls: list[Session] = []

    async def cancel_task(self, *, session: Session) -> bool:
        self.cancel_calls.append(session)
        return True


def _build_agent_service(uow: _UoW, coordinator: _SpyCoordinator, graph_runtime: _SpyGraphRuntime) -> AgentService:
    service = object.__new__(AgentService)
    service._uow_factory = lambda: uow
    service._access_control_service = RuntimeAccessControlService(uow_factory=lambda: uow)
    service._runtime_state_coordinator = coordinator
    service._graph_runtime = graph_runtime
    return service


def test_chat_sse_should_reject_cross_user_session_before_runtime_side_effects() -> None:
    uow = _UoW(session=Session(id="session-1", user_id="user-2"))
    coordinator = _SpyCoordinator()
    graph_runtime = _SpyGraphRuntime()
    service = _build_agent_service(uow, coordinator, graph_runtime)

    async def _collect_first_error() -> ErrorEvent:
        async for event in service.chat(session_id="session-1", user_id="user-1", message="hello"):
            return event
        raise AssertionError("chat() did not yield an event")

    event = asyncio.run(_collect_first_error())

    assert isinstance(event, ErrorEvent)
    assert event.error_key == error_keys.SESSION_NOT_FOUND
    assert coordinator.reconcile_calls == []
    assert graph_runtime.cancel_calls == []
    assert len(uow.add_event_calls) == 1
    assert uow.add_event_calls[0][1].error_key == error_keys.SESSION_NOT_FOUND


def test_stop_session_should_reject_cross_user_session_before_runtime_cancel() -> None:
    uow = _UoW(session=Session(id="session-1", user_id="user-2"))
    coordinator = _SpyCoordinator()
    graph_runtime = _SpyGraphRuntime()
    service = _build_agent_service(uow, coordinator, graph_runtime)

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.stop_session(session_id="session-1", user_id="user-1"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
    assert graph_runtime.cancel_calls == []
    assert coordinator.cancel_calls == []


class _Sandbox:
    pass


def test_set_current_model_should_reject_cross_user_session_before_update() -> None:
    uow = _UoW(session=Session(id="session-1", user_id="user-2"))
    service = SessionService(
        uow_factory=lambda: uow,
        sandbox_cls=_Sandbox,
        model_config_service=None,
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.set_current_model("user-1", "session-1", "auto"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
    assert uow.update_model_calls == []
