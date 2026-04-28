import asyncio
from dataclasses import dataclass, field

from app.application.service.runtime_observation_service import RuntimeObservationService
from app.domain.models import (
    Plan,
    PlanEvent,
    Session,
    SessionStatus,
    Step,
    WaitEvent,
    WorkflowRun,
    WorkflowRunEventRecord,
    WorkflowRunStatus,
    Workspace,
    ExecutionStatus,
)
from app.domain.repositories.workflow_run_repository import UNSET_CURRENT_STEP_ID


@dataclass
class _SessionRepo:
    session: Session
    updates: list[dict] = field(default_factory=list)

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        if session_id != self.session.id:
            return None
        if user_id is not None and user_id != self.session.user_id:
            return None
        return self.session

    async def get_by_id_for_update(self, session_id: str):
        return self.session if session_id == self.session.id else None

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
        self.session.status = status
        if current_run_id is not None:
            self.session.current_run_id = current_run_id
        self.updates.append({"session_id": session_id, "status": status})


@dataclass
class _WorkflowRunRepo:
    run: WorkflowRun | None
    records: list[WorkflowRunEventRecord] = field(default_factory=list)

    async def get_by_id_for_update(self, run_id: str):
        return self.run if self.run and self.run.id == run_id else None

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
        if self.run is None:
            return
        self.run.status = status
        if current_step_id is not UNSET_CURRENT_STEP_ID:
            self.run.current_step_id = current_step_id


@dataclass
class _WorkspaceRepo:
    workspace: Workspace | None = None

    async def get_by_id(self, workspace_id: str):
        if self.workspace and self.workspace.id == workspace_id:
            return self.workspace
        return None

    async def get_by_session_id(self, session_id: str):
        if self.workspace and self.workspace.session_id == session_id:
            return self.workspace
        return None


class _UoW:
    def __init__(
            self,
            *,
            session: Session,
            run: WorkflowRun | None,
            workspace: Workspace | None,
            records: list[WorkflowRunEventRecord] | None = None,
    ) -> None:
        self.session = _SessionRepo(session=session)
        self.workflow_run = _WorkflowRunRepo(run=run, records=records or [])
        self.workspace = _WorkspaceRepo(workspace=workspace)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def commit(self):
        return None

    async def rollback(self):
        return None


def _build_service(uow: _UoW) -> RuntimeObservationService:
    return RuntimeObservationService(uow_factory=lambda: uow)


def test_session_observation_should_allow_message_when_no_current_run() -> None:
    session = Session(id="session-1", user_id="user-1", status=SessionStatus.PENDING)
    uow = _UoW(session=session, run=None, workspace=None)
    service = _build_service(uow)

    result = asyncio.run(
        service.build_session_observation(user_id="user-1", session_id="session-1")
    )

    assert result.run_id is None
    assert result.status == SessionStatus.PENDING
    assert result.capabilities.can_send_message is True
    assert result.capabilities.can_cancel is False
    assert result.capabilities.can_resume is False


def test_session_observation_should_return_wait_interaction_and_capabilities() -> None:
    wait_event = WaitEvent(
        id="evt-wait-1",
        interrupt_id="interrupt-1",
        payload={"kind": "confirm", "prompt": "继续？"},
    )
    record = WorkflowRunEventRecord(
        run_id="run-1",
        session_id="session-1",
        event_id=wait_event.id,
        event_type=wait_event.type,
        event_payload=wait_event,
    )
    session = Session(
        id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
        current_run_id="run-1",
        status=SessionStatus.WAITING,
    )
    run = WorkflowRun(id="run-1", session_id="session-1", status=WorkflowRunStatus.WAITING)
    workspace = Workspace(id="workspace-1", session_id="session-1", current_run_id="run-1")
    uow = _UoW(session=session, run=run, workspace=workspace, records=[record])
    service = _build_service(uow)

    result = asyncio.run(
        service.build_session_observation(user_id="user-1", session_id="session-1")
    )

    assert result.run_id == "run-1"
    assert result.status == SessionStatus.WAITING
    assert result.cursor.latest_event_id == "evt-wait-1"
    assert result.interaction.kind == "wait"
    assert result.interaction.interrupt_id == "interrupt-1"
    assert result.capabilities.can_resume is True
    assert result.capabilities.can_cancel is True


def test_session_observation_should_expose_continue_cancelled_capability() -> None:
    plan_event = PlanEvent(
        id="evt-plan-cancelled",
        plan=Plan(
            status=ExecutionStatus.CANCELLED,
            steps=[Step(id="step-1", status=ExecutionStatus.CANCELLED)],
        ),
    )
    record = WorkflowRunEventRecord(
        run_id="run-1",
        session_id="session-1",
        event_id=plan_event.id,
        event_type=plan_event.type,
        event_payload=plan_event,
    )
    session = Session(
        id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
        current_run_id="run-1",
        status=SessionStatus.CANCELLED,
    )
    run = WorkflowRun(id="run-1", session_id="session-1", status=WorkflowRunStatus.CANCELLED)
    workspace = Workspace(id="workspace-1", session_id="session-1", current_run_id="run-1")
    uow = _UoW(session=session, run=run, workspace=workspace, records=[record])
    service = _build_service(uow)

    result = asyncio.run(
        service.build_session_observation(user_id="user-1", session_id="session-1")
    )

    assert result.status == SessionStatus.CANCELLED
    assert result.capabilities.can_send_message is True
    assert result.capabilities.can_continue_cancelled is True
