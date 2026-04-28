import asyncio
import logging
from dataclasses import dataclass, field

from app.application.service.runtime_observation_service import (
    RuntimeCapabilityResult,
    RuntimeCursorResult,
    RuntimeInteractionResult,
    RuntimeObservationResult,
    RuntimeObservationService,
)
from app.domain.models import (
    DoneEvent,
    ErrorEvent,
    Plan,
    PlanEvent,
    Session,
    SessionStatus,
    Step,
    StepEvent,
    TextStreamChannel,
    TextStreamDeltaEvent,
    ToolEvent,
    ToolEventStatus,
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


def test_session_observation_cursor_should_ignore_trailing_live_only_event() -> None:
    records = [
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-message-1",
            event_type="message",
            event_payload=WaitEvent(id="evt-message-1"),
        ),
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-ts-1",
            event_type="text_stream_delta",
            event_payload=TextStreamDeltaEvent(
                id="evt-ts-1",
                stream_id="stream-1",
                channel=TextStreamChannel.FINAL_MESSAGE,
                text="draft",
                sequence=1,
            ),
        ),
    ]
    session = Session(id="session-1", user_id="user-1")
    service = _build_service(_UoW(
        session=session,
        run=None,
        workspace=None,
        records=records,
    ))

    result = asyncio.run(service.build_session_observation(
        user_id="user-1",
        session_id="session-1",
    ))

    assert result.cursor.latest_event_id == "evt-message-1"


def test_session_observation_cursor_should_be_empty_when_only_live_only_events() -> None:
    records = [
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-ts-1",
            event_type="text_stream_delta",
            event_payload=TextStreamDeltaEvent(
                id="evt-ts-1",
                stream_id="stream-1",
                channel=TextStreamChannel.FINAL_MESSAGE,
                text="draft",
                sequence=1,
            ),
        ),
    ]
    session = Session(id="session-1", user_id="user-1")
    service = _build_service(_UoW(
        session=session,
        run=None,
        workspace=None,
        records=records,
    ))

    result = asyncio.run(service.build_session_observation(
        user_id="user-1",
        session_id="session-1",
    ))

    assert result.cursor.latest_event_id is None


def test_capabilities_should_follow_runtime_status_matrix() -> None:
    service = _build_service(_UoW(
        session=Session(id="session-1", user_id="user-1"),
        run=None,
        workspace=None,
    ))

    pending = asyncio.run(service.build_capabilities(
        session_id="session-1",
        status=SessionStatus.PENDING,
        latest_wait_event=None,
    ))
    running = asyncio.run(service.build_capabilities(
        session_id="session-1",
        status=SessionStatus.RUNNING,
        latest_wait_event=None,
    ))
    waiting_without_event = asyncio.run(service.build_capabilities(
        session_id="session-1",
        status=SessionStatus.WAITING,
        latest_wait_event=None,
    ))
    waiting_with_event = asyncio.run(service.build_capabilities(
        session_id="session-1",
        status=SessionStatus.WAITING,
        latest_wait_event=WaitEvent(id="evt-wait"),
    ))
    completed = asyncio.run(service.build_capabilities(
        session_id="session-1",
        status=SessionStatus.COMPLETED,
        latest_wait_event=None,
    ))
    failed = asyncio.run(service.build_capabilities(
        session_id="session-1",
        status=SessionStatus.FAILED,
        latest_wait_event=None,
    ))
    cancelled_without_plan = asyncio.run(service.build_capabilities(
        session_id="session-1",
        status=SessionStatus.CANCELLED,
        latest_wait_event=None,
    ))
    cancelled_with_plan = asyncio.run(service.build_capabilities(
        session_id="session-1",
        status=SessionStatus.CANCELLED,
        latest_wait_event=None,
        has_continuable_cancelled_plan=True,
    ))

    assert pending.can_send_message is False
    assert running.can_cancel is True
    assert running.can_send_message is False
    assert waiting_without_event.can_resume is False
    assert waiting_with_event.can_resume is True
    assert waiting_with_event.can_cancel is True
    assert completed.can_send_message is True
    assert failed.can_send_message is True
    assert cancelled_without_plan.can_send_message is True
    assert cancelled_without_plan.can_continue_cancelled is False
    assert cancelled_with_plan.can_continue_cancelled is True


def test_observable_event_should_project_runtime_metadata_from_context() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
        current_run_id="run-1",
        status=SessionStatus.RUNNING,
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        status=WorkflowRunStatus.RUNNING,
        current_step_id="step-1",
    )
    workspace = Workspace(id="workspace-1", session_id="session-1", current_run_id="run-1")
    service = _build_service(_UoW(session=session, run=run, workspace=workspace))
    context = asyncio.run(service.build_event_context(user_id="user-1", session_id="session-1"))

    tool_envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=ToolEvent(
            id="evt-tool-1",
            tool_call_id="tool-call-1",
            tool_name="workspace",
            function_name="read_file",
            function_args={"path": "/tmp/a.md"},
            status=ToolEventStatus.CALLING,
        ),
        run_id=context.run_id,
        source_event_id="evt-tool-1",
        cursor_event_id="evt-tool-1",
        source="sse",
        context=context,
    ))
    wait_envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=WaitEvent(id="evt-wait-1"),
        run_id=context.run_id,
        source_event_id="evt-wait-1",
        cursor_event_id="evt-wait-1",
        source="sse",
        context=context,
    ))
    done_envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=DoneEvent(id="evt-done-1"),
        run_id=context.run_id,
        source_event_id="evt-done-1",
        cursor_event_id="evt-done-1",
        source="sse",
        context=context,
    ))
    error_envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=ErrorEvent(id="evt-error-1", error="boom"),
        run_id=context.run_id,
        source_event_id="evt-error-1",
        cursor_event_id="evt-error-1",
        source="sse",
        context=context,
    ))

    assert tool_envelope.runtime.current_step_id == "step-1"
    assert tool_envelope.runtime.status_after_event is None
    assert wait_envelope.runtime.status_after_event == SessionStatus.WAITING
    assert done_envelope.runtime.status_after_event == SessionStatus.COMPLETED
    assert error_envelope.runtime.status_after_event == SessionStatus.FAILED


def test_observable_event_should_keep_plain_message_without_completed_status() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
        current_run_id="run-1",
        status=SessionStatus.COMPLETED,
    )
    run = WorkflowRun(id="run-1", session_id="session-1", status=WorkflowRunStatus.COMPLETED)
    workspace = Workspace(id="workspace-1", session_id="session-1", current_run_id="run-1")
    service = _build_service(_UoW(session=session, run=run, workspace=workspace))
    context = asyncio.run(service.build_event_context(user_id="user-1", session_id="session-1"))
    envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=StepEvent(id="evt-step-1", step=Step(id="step-1", status=ExecutionStatus.COMPLETED)),
        run_id=context.run_id,
        source_event_id="evt-step-1",
        cursor_event_id="evt-step-1",
        source="snapshot",
        context=context,
    ))

    assert envelope.runtime.status_after_event is None
    assert envelope.runtime.current_step_id == "step-1"


def test_observable_event_should_mark_text_stream_live_only() -> None:
    service = _build_service(_UoW(
        session=Session(id="session-1", user_id="user-1"),
        run=None,
        workspace=None,
    ))
    envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=TextStreamDeltaEvent(
            id="evt-ts-1",
            stream_id="stream-1",
            channel=TextStreamChannel.FINAL_MESSAGE,
            text="draft",
            sequence=1,
        ),
        run_id="run-1",
        source_event_id="evt-ts-1",
        cursor_event_id="evt-ts-1",
        source="sse",
    ))

    assert envelope.runtime.durability == "live_only"
    assert envelope.runtime.visibility == "draft"
    assert envelope.runtime.source_event_id is None
    assert envelope.runtime.cursor_event_id is None


def test_snapshot_event_context_should_not_reuse_snapshot_current_step_for_earlier_events() -> None:
    observation = RuntimeObservationResult(
        session_id="session-1",
        run_id="run-1",
        status=SessionStatus.RUNNING,
        current_step_id="step-latest",
        cursor=RuntimeCursorResult(),
        capabilities=RuntimeCapabilityResult(can_cancel=True),
        interaction=RuntimeInteractionResult(),
    )
    service = _build_service(_UoW(
        session=Session(id="session-1", user_id="user-1"),
        run=None,
        workspace=None,
    ))
    context = service.context_from_observation(observation)

    assert context.current_step_id is None


def test_replay_should_return_records_after_valid_persistent_cursor() -> None:
    events = [
        WaitEvent(id="evt-1"),
        DoneEvent(id="evt-2"),
        ErrorEvent(id="evt-3", error="boom"),
    ]
    records = [
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id=event.id,
            event_type=event.type,
            event_payload=event,
        )
        for event in events
    ]
    service = _build_service(_UoW(
        session=Session(id="session-1", user_id="user-1"),
        run=None,
        workspace=None,
        records=records,
    ))

    replay = asyncio.run(service.list_persistent_events_after_cursor(
        user_id="user-1",
        session_id="session-1",
        cursor_event_id="evt-1",
    ))

    assert replay.cursor_invalid is False
    assert [record.event_id for record in replay.records] == ["evt-2", "evt-3"]
    assert replay.live_attach_after_event_id == "evt-3"


def test_replay_should_return_all_records_for_empty_or_invalid_cursor(caplog) -> None:
    records = [
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-1",
            event_type="wait",
            event_payload=WaitEvent(id="evt-1"),
        ),
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-2",
            event_type="done",
            event_payload=DoneEvent(id="evt-2"),
        ),
    ]
    service = _build_service(_UoW(
        session=Session(id="session-1", user_id="user-1"),
        run=None,
        workspace=None,
        records=records,
    ))

    empty_cursor_replay = asyncio.run(service.list_persistent_events_after_cursor(
        user_id="user-1",
        session_id="session-1",
        cursor_event_id=None,
    ))
    with caplog.at_level(logging.WARNING):
        invalid_cursor_replay = asyncio.run(service.list_persistent_events_after_cursor(
            user_id="user-1",
            session_id="session-1",
            cursor_event_id="missing-event",
        ))

    assert empty_cursor_replay.cursor_invalid is False
    assert [record.event_id for record in empty_cursor_replay.records] == ["evt-1", "evt-2"]
    assert empty_cursor_replay.live_attach_after_event_id == "evt-2"
    assert invalid_cursor_replay.cursor_invalid is True
    assert [record.event_id for record in invalid_cursor_replay.records] == ["evt-1", "evt-2"]
    assert invalid_cursor_replay.live_attach_after_event_id == "evt-2"
    assert any(
        record.__dict__.get("reason") == "cursor_invalid"
        for record in caplog.records
    )


def test_replay_should_not_leak_records_when_cursor_belongs_to_another_session(caplog) -> None:
    records = [
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-1",
            event_type="wait",
            event_payload=WaitEvent(id="evt-1"),
        ),
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-2",
            event_type="done",
            event_payload=DoneEvent(id="evt-2"),
        ),
        WorkflowRunEventRecord(
            run_id="run-other",
            session_id="session-other",
            event_id="evt-other",
            event_type="done",
            event_payload=DoneEvent(id="evt-other"),
        ),
    ]
    service = _build_service(_UoW(
        session=Session(id="session-1", user_id="user-1"),
        run=None,
        workspace=None,
        records=records,
    ))

    with caplog.at_level(logging.WARNING):
        replay = asyncio.run(service.list_persistent_events_after_cursor(
            user_id="user-1",
            session_id="session-1",
            cursor_event_id="evt-other",
        ))

    assert replay.cursor_invalid is True
    assert [record.event_id for record in replay.records] == ["evt-1", "evt-2"]
    assert replay.live_attach_after_event_id == "evt-2"
    assert all(record.session_id == "session-1" for record in replay.records)
    assert any(
        record.__dict__.get("reason") == "cursor_invalid"
        and record.__dict__.get("requested_event_id") == "evt-other"
        for record in caplog.records
    )


def test_replay_should_use_valid_cursor_as_live_attach_boundary_when_no_new_records() -> None:
    records = [
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-1",
            event_type="wait",
            event_payload=WaitEvent(id="evt-1"),
        ),
    ]
    service = _build_service(_UoW(
        session=Session(id="session-1", user_id="user-1"),
        run=None,
        workspace=None,
        records=records,
    ))

    replay = asyncio.run(service.list_persistent_events_after_cursor(
        user_id="user-1",
        session_id="session-1",
        cursor_event_id="evt-1",
    ))

    assert replay.records == []
    assert replay.cursor_invalid is False
    assert replay.live_attach_after_event_id == "evt-1"


def test_replay_should_leave_live_attach_boundary_empty_without_db_history() -> None:
    service = _build_service(_UoW(
        session=Session(id="session-1", user_id="user-1"),
        run=None,
        workspace=None,
        records=[],
    ))

    replay = asyncio.run(service.list_persistent_events_after_cursor(
        user_id="user-1",
        session_id="session-1",
        cursor_event_id=None,
    ))

    assert replay.records == []
    assert replay.cursor_invalid is False
    assert replay.live_attach_after_event_id is None


def test_replay_should_filter_live_only_text_stream_records() -> None:
    records = [
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-1",
            event_type="message",
            event_payload=WaitEvent(id="evt-1"),
        ),
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-ts-1",
            event_type="text_stream_delta",
            event_payload=TextStreamDeltaEvent(
                id="evt-ts-1",
                stream_id="stream-1",
                channel=TextStreamChannel.FINAL_MESSAGE,
                text="draft",
                sequence=1,
            ),
        ),
        WorkflowRunEventRecord(
            run_id="run-1",
            session_id="session-1",
            event_id="evt-2",
            event_type="done",
            event_payload=DoneEvent(id="evt-2"),
        ),
    ]
    service = _build_service(_UoW(
        session=Session(id="session-1", user_id="user-1"),
        run=None,
        workspace=None,
        records=records,
    ))

    replay = asyncio.run(service.list_persistent_events_after_cursor(
        user_id="user-1",
        session_id="session-1",
        cursor_event_id=None,
    ))

    assert [record.event_id for record in replay.records] == ["evt-1", "evt-2"]
