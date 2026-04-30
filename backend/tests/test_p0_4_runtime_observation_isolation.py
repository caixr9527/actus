import asyncio
from dataclasses import dataclass

import pytest

from app.application.errors import NotFoundError, error_keys
from app.application.service.runtime_observation_service import RuntimeObservationService
from app.domain.models import MessageEvent, Session, SessionStatus, WorkflowRun, WorkflowRunEventRecord, WorkflowRunStatus, Workspace
from app.domain.repositories.workflow_run_repository import UNSET_CURRENT_STEP_ID


@dataclass
class _SessionRepo:
    sessions: dict[str, Session]

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        session = self.sessions.get(session_id)
        if session is None:
            return None
        if user_id is not None and session.user_id != user_id:
            return None
        return session

    async def get_by_id_for_update(self, session_id: str):
        return self.sessions.get(session_id)

    async def update_runtime_state(
            self,
            session_id: str,
            *,
            status: SessionStatus,
            current_run_id: str | None = None,
            title: str | None = None,
            latest_message: str | None = None,
            latest_message_at=None,
            increment_unread: bool = False,
    ) -> None:
        session = self.sessions[session_id]
        session.status = status
        if current_run_id is not None:
            session.current_run_id = current_run_id


@dataclass
class _WorkspaceRepo:
    workspaces: dict[str, Workspace]

    async def get_by_id_for_user(self, workspace_id: str, user_id: str):
        workspace = self.workspaces.get(workspace_id)
        if workspace is None or workspace.user_id != user_id:
            return None
        return workspace

    async def list_by_session_id(self, session_id: str):
        return [
            workspace
            for workspace in self.workspaces.values()
            if workspace.session_id == session_id
        ]


@dataclass
class _WorkflowRunRepo:
    runs: dict[str, WorkflowRun]
    records: list[WorkflowRunEventRecord]

    async def get_by_id_for_user(self, run_id: str, user_id: str):
        run = self.runs.get(run_id)
        if run is None or run.user_id != user_id:
            return None
        return run

    async def get_by_id_for_update(self, run_id: str):
        return self.runs.get(run_id)

    async def list_event_records_by_session(self, session_id: str):
        return [record for record in self.records if record.session_id == session_id]

    async def update_status(
            self,
            run_id: str,
            *,
            status: WorkflowRunStatus,
            finished_at=None,
            last_event_at=None,
            current_step_id=UNSET_CURRENT_STEP_ID,
    ) -> None:
        run = self.runs.get(run_id)
        if run is None:
            return
        run.status = status
        if current_step_id is not UNSET_CURRENT_STEP_ID:
            run.current_step_id = current_step_id


class _UoW:
    def __init__(
            self,
            *,
            sessions: dict[str, Session],
            workspaces: dict[str, Workspace],
            runs: dict[str, WorkflowRun],
            records: list[WorkflowRunEventRecord],
    ) -> None:
        self.session = _SessionRepo(sessions)
        self.workspace = _WorkspaceRepo(workspaces)
        self.workflow_run = _WorkflowRunRepo(runs, records)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self):
        return None

    async def rollback(self):
        return None


def _record(*, session_id: str, run_id: str, event_id: str, message: str) -> WorkflowRunEventRecord:
    event = MessageEvent(id=event_id, role="assistant", message=message)
    return WorkflowRunEventRecord(
        run_id=run_id,
        session_id=session_id,
        user_id="user-1",
        event_id=event_id,
        event_type=event.type,
        event_payload=event,
    )


def _build_service(uow: _UoW) -> RuntimeObservationService:
    return RuntimeObservationService(uow_factory=lambda: uow)


def test_session_detail_observation_should_hide_cross_user_session() -> None:
    session = Session(id="session-1", user_id="user-2", status=SessionStatus.RUNNING)
    service = _build_service(_UoW(sessions={"session-1": session}, workspaces={}, runs={}, records=[]))

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.build_session_observation(user_id="user-1", session_id="session-1"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_empty_replay_should_hide_cross_user_session() -> None:
    session = Session(id="session-1", user_id="user-2", status=SessionStatus.RUNNING)
    service = _build_service(_UoW(sessions={"session-1": session}, workspaces={}, runs={}, records=[]))

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(
            service.list_persistent_events_after_cursor(
                user_id="user-1",
                session_id="session-1",
                cursor_event_id=None,
            )
        )

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_invalid_cursor_should_replay_only_current_session_records() -> None:
    session = Session(id="session-1", user_id="user-1", status=SessionStatus.RUNNING)
    current_record = _record(session_id="session-1", run_id="run-1", event_id="evt-current", message="current")
    foreign_record = _record(session_id="session-2", run_id="run-2", event_id="evt-other", message="other")
    service = _build_service(
        _UoW(
            sessions={"session-1": session},
            workspaces={},
            runs={},
            records=[current_record, foreign_record],
        )
    )

    replay = asyncio.run(
        service.list_persistent_events_after_cursor(
            user_id="user-1",
            session_id="session-1",
            cursor_event_id="evt-other",
        )
    )

    assert replay.cursor_invalid is True
    assert [record.event_id for record in replay.records] == ["evt-current"]
    assert all(record.session_id == "session-1" for record in replay.records)
