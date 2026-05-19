import asyncio
from types import SimpleNamespace

import pytest

from app.application.errors import BadRequestError, error_keys
from app.application.service.runtime_state_coordinator import RuntimeStateCoordinator
from app.application.service.agent_service import AgentService
from app.application.service.feedback_ledger_service import FeedbackRequiredRecordError
from app.application.service.user_feedback_ingress_service import (
    UserFeedbackCapturePolicy,
    UserFeedbackIngressService,
)
from app.domain.models.feedback import (
    FeedbackReasonCode,
    FeedbackScopeKind,
    UserFeedbackIntentKind,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.models import (
    ContinueCancelledTaskInput,
    ErrorEvent,
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    RuntimeInput,
    Session,
    SessionStatus,
    Step,
    StepEvent,
    WaitEvent,
    WorkflowRun,
    WorkflowRunEventRecord,
    WorkflowRunStatus,
    Workspace,
)


class _TaskFactory:
    @classmethod
    def get(cls, task_id: str):
        return None


class _NoTaskGraphRuntime:
    async def get_task(self, session: Session):
        return None


class _ReconcileOnlyCoordinator:
    def __init__(self) -> None:
        self.reconcile_calls: list[tuple[str, str]] = []

    async def reconcile_current_run(self, session_id: str, *, reason: str):
        self.reconcile_calls.append((session_id, reason))
        return SimpleNamespace(warnings=[], snapshot_after=None)


class _InputStream:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.deleted: list[str] = []

    async def put(self, message: str) -> str:
        self.messages.append(message)
        return "resume-msg-1"

    async def delete_message(self, event_id: str) -> None:
        self.deleted.append(event_id)


class _Task:
    def __init__(self) -> None:
        self.input_stream = _InputStream()
        self.invoked = False

    async def invoke(self) -> None:
        self.invoked = True


class _IdleOutputStream:
    async def get(self, start_id=None, block_ms=0):
        return None, None


class _IdleTask(_Task):
    def __init__(self) -> None:
        super().__init__()
        self.output_stream = _IdleOutputStream()

    @property
    def done(self) -> bool:
        return True


class _CancellableTask:
    def __init__(self, session: Session) -> None:
        self.cancelled = False
        self._session = session

    async def cancel(self) -> bool:
        self.cancelled = True
        self._session.status = SessionStatus.CANCELLED
        return True


class _SessionRepo:
    def __init__(self, session: Session, *, fail_on_update_status: bool = False) -> None:
        self._session = session
        self._fail_on_update_status = fail_on_update_status

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        return self._session

    async def get_by_id_for_update(self, session_id: str):
        return self._session

    async def add_event(self, session_id: str, event) -> None:
        return None

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        if self._fail_on_update_status:
            raise RuntimeError("update status failed")
        self._session.status = status

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
        if self._fail_on_update_status:
            raise RuntimeError("update status failed")
        self._session.status = status
        if current_run_id is not None:
            self._session.current_run_id = current_run_id


class _WorkflowRunRepo:
    def __init__(
            self,
            run: WorkflowRun | None = None,
            *,
            events: list[object] | None = None,
            event_records: list[WorkflowRunEventRecord] | None = None,
    ) -> None:
        self._run = run
        if self._run is not None and self._run.user_id is None:
            self._run.user_id = "user-1"
        self._events = list(events or [])
        self._event_records = list(event_records or [])
        self.cancelled_run_ids: list[str] = []
        self.replaced_plan_count = 0
        self.upserted_step_count = 0
        self.unfinished_steps_cancelled_count = 0

    async def get_by_id(self, run_id: str):
        if self._run is None:
            return None
        return self._run if self._run.id == run_id else None

    async def get_by_id_for_user(self, run_id: str, user_id: str):
        run = await self.get_by_id(run_id)
        if run is None or run.user_id != user_id:
            return None
        return run

    async def get_by_id_for_user_session(self, *, run_id: str, user_id: str, session_id: str):
        run = await self.get_by_id_for_user(run_id, user_id)
        if run is None or run.session_id != session_id:
            return None
        return run

    async def get_by_id_for_update(self, run_id: str):
        return await self.get_by_id(run_id)

    async def update_status(self, run_id: str, *, status: WorkflowRunStatus, **_kwargs) -> None:
        if self._run is not None and self._run.id == run_id:
            self._run.status = status

    async def add_event_record_if_absent(self, session_id: str, run_id: str, event) -> bool:
        self._events.append(event)
        self._event_records.append(
            WorkflowRunEventRecord(
                run_id=run_id,
                session_id=session_id,
                user_id=self._run.user_id if self._run is not None else "user-1",
                event_id=event.id,
                event_type=event.type,
                event_payload=event,
            )
        )
        return True

    async def replace_steps_from_plan(self, run_id: str, plan: Plan) -> None:
        self.replaced_plan_count += 1

    async def upsert_step_from_event(self, run_id: str, event) -> None:
        self.upserted_step_count += 1

    async def mark_unfinished_steps_cancelled(self, run_id: str) -> None:
        self.unfinished_steps_cancelled_count += 1

    async def list_event_records_by_session(self, session_id: str):
        if self._event_records:
            return [record for record in self._event_records if record.session_id == session_id]
        return [
            SimpleNamespace(
                session_id=session_id,
                run_id=self._run.id if self._run else "run-1",
                event_id=getattr(event, "id", ""),
                event_type=getattr(event, "type", ""),
                event_payload=event,
            )
            for event in self._events
        ]

    async def get_latest_event_record_by_session(
            self,
            session_id: str,
            *,
            event_type: str | None = None,
            run_id: str | None = None,
    ):
        records = await self.list_event_records_by_session(session_id)
        filtered = [
            record for record in records
            if (event_type is None or record.event_type == event_type)
            and (run_id is None or record.run_id == run_id)
        ]
        return filtered[-1] if filtered else None

    async def get_event_record_by_type_and_hash(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            event_type: str,
            input_hash: str,
    ):
        for record in reversed(await self.list_event_records_by_session(session_id)):
            if record.run_id != run_id or record.event_type != event_type or record.user_id != user_id:
                continue
            payload = getattr(record.event_payload, "payload", None)
            if getattr(payload, "input_hash", None) == input_hash:
                return record
        return None

    async def get_event_record_by_event_id(self, *, user_id: str, session_id: str, run_id: str, event_id: str):
        for record in await self.list_event_records_by_session(session_id):
            if (
                    record.user_id == user_id
                    and record.run_id == run_id
                    and record.event_id == event_id
            ):
                return record
        return None

    async def cancel_run(self, run_id: str) -> None:
        self.cancelled_run_ids.append(run_id)
        if self._run is not None and self._run.id == run_id:
            self._run.status = WorkflowRunStatus.CANCELLED

    async def list_events(self, run_id: str | None):
        if self._run is None or run_id != self._run.id:
            return []
        return list(self._events)


class _WorkspaceRepo:
    def __init__(self, workspace: Workspace | None = None) -> None:
        self._workspace = workspace
        if self._workspace is not None and self._workspace.user_id is None:
            self._workspace.user_id = "user-1"

    async def get_by_id(self, workspace_id: str):
        if self._workspace is None or workspace_id != self._workspace.id:
            return None
        return self._workspace

    async def get_by_id_for_user(self, workspace_id: str, user_id: str):
        workspace = await self.get_by_id(workspace_id)
        if workspace is None or workspace.user_id != user_id:
            return None
        return workspace

    async def get_by_session_id(self, session_id: str):
        if self._workspace is None or session_id != self._workspace.session_id:
            return None
        return self._workspace

    async def get_by_session_id_for_user(self, session_id: str, user_id: str):
        workspace = await self.get_by_session_id(session_id)
        if workspace is None or workspace.user_id != user_id:
            return None
        return workspace

    async def list_by_session_id(self, session_id: str):
        workspace = await self.get_by_session_id(session_id)
        return [workspace] if workspace is not None else []


class _StopSessionRepo(_SessionRepo):
    def __init__(self, session: Session, *, fail_on_update_status: bool = False) -> None:
        super().__init__(session, fail_on_update_status=fail_on_update_status)
        self.added_events: list[object] = []

    async def add_event_with_snapshot_if_absent(self, session_id: str, event, **_kwargs) -> None:
        self.added_events.append(event)


class _UoW:
    def __init__(
            self,
            session_repo: _SessionRepo,
            workflow_run_repo: _WorkflowRunRepo | None = None,
            workspace_repo: _WorkspaceRepo | None = None,
    ) -> None:
        self.session = session_repo
        self.workflow_run = workflow_run_repo or _WorkflowRunRepo()
        self.workspace = workspace_repo or _WorkspaceRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _UserFeedbackIngressService:
    def __init__(self) -> None:
        self.resume_calls: list[dict] = []
        self.cancel_calls: list[dict] = []
        self.continue_calls: list[dict] = []
        self.message_calls: list[dict] = []
        self.fail_message_record = False
        self.fail_cancel_record = False

    def capture_feedback_payload(self, payload):
        return SimpleNamespace(captured=payload is not None, intent=SimpleNamespace())

    async def record_feedback_from_message_event(self, **kwargs):
        self.message_calls.append(kwargs)
        if self.fail_message_record:
            raise FeedbackRequiredRecordError("feedback required write failed")
        return SimpleNamespace(success=True)

    async def record_feedback_from_wait_resume(self, **kwargs):
        self.resume_calls.append(kwargs)
        return SimpleNamespace(success=True)

    async def record_cancel_feedback(self, **kwargs):
        self.cancel_calls.append(kwargs)
        if self.fail_cancel_record:
            raise FeedbackRequiredRecordError("cancel feedback failed")
        return SimpleNamespace(success=True)

    async def record_continue_cancelled_feedback(self, **kwargs):
        self.continue_calls.append(kwargs)
        return SimpleNamespace(success=True)


def test_agent_service_chat_should_require_resume_when_session_waiting() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.WAITING,
    )

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                message="继续执行",
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_RESUME_REQUIRED


def test_agent_service_chat_should_reconcile_before_status_validation() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
    )
    session_repo = _SessionRepo(session)
    coordinator = _ReconcileOnlyCoordinator()

    async def _reconcile_current_run(session_id: str, *, reason: str):
        coordinator.reconcile_calls.append((session_id, reason))
        session.status = SessionStatus.WAITING

    coordinator.reconcile_current_run = _reconcile_current_run

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo)
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()
    service._runtime_state_coordinator = coordinator

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                message="继续执行",
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert coordinator.reconcile_calls == [("session-1", "before_chat")]
    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_RESUME_REQUIRED


def test_agent_service_chat_should_fail_closed_when_message_feedback_record_fails() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.PENDING,
        workspace_id="workspace-1",
    )
    workspace = Workspace(id="workspace-1", session_id="session-1", user_id="user-1")
    task = _IdleTask()
    feedback_ingress = _UserFeedbackIngressService()
    feedback_ingress.fail_message_record = True

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(
        _SessionRepo(session),
        _WorkflowRunRepo(WorkflowRun(id="run-1", session_id="session-1", user_id="user-1")),
        _WorkspaceRepo(workspace),
    )
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()
    service._user_feedback_ingress_service = feedback_ingress

    class _Coordinator:
        async def reconcile_current_run(self, session_id: str, *, reason: str):
            return SimpleNamespace(warnings=[], snapshot_after=None)

        async def accept_user_message(self, **kwargs):
            return None

    service._runtime_state_coordinator = _Coordinator()

    async def _create_task(_session: Session, *, reuse_current_run: bool = False):
        _session.current_run_id = "run-1"
        _session.status = SessionStatus.RUNNING
        workspace.current_run_id = "run-1"
        return task

    async def _ensure_periodic_sandbox_profile(*, user_id: str, session_id: str):
        return None

    service._create_task = _create_task
    service._ensure_periodic_sandbox_profile = _ensure_periodic_sandbox_profile

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                message="这次结果不对",
                feedback_intent={
                    "intent_kind": "correction",
                    "target_ref": {
                        "target_type": "message_event",
                        "target_id": "evt-final-1",
                        "target_run_id": "run-1",
                    },
                    "reason_code": "user_corrected_requirement",
                },
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error == "feedback required write failed"
    assert len(feedback_ingress.message_calls) == 1
    assert task.invoked is False


def test_agent_service_chat_should_reject_message_when_reconcile_finds_run_waiting() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        workspace_id="workspace-1",
        current_run_id="run-1",
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        status=WorkflowRunStatus.WAITING,
    )
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        current_run_id="run-1",
    )
    session_repo = _SessionRepo(session)
    workflow_run_repo = _WorkflowRunRepo(run)
    workspace_repo = _WorkspaceRepo(workspace)

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo, workflow_run_repo, workspace_repo)
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()
    service._runtime_state_coordinator = RuntimeStateCoordinator(uow_factory=service._uow_factory)

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                message="普通消息",
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_RESUME_REQUIRED
    assert session.status == SessionStatus.WAITING


def test_agent_service_chat_should_allow_resume_after_reconcile_syncs_run_waiting() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        workspace_id="workspace-1",
        current_run_id="run-1",
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        status=WorkflowRunStatus.WAITING,
    )
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        current_run_id="run-1",
    )
    task = _IdleTask()
    session_repo = _SessionRepo(session)
    workflow_run_repo = _WorkflowRunRepo(run)
    workspace_repo = _WorkspaceRepo(workspace)

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo, workflow_run_repo, workspace_repo)
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()
    service._runtime_state_coordinator = RuntimeStateCoordinator(uow_factory=service._uow_factory)

    async def _get_task(_session: Session):
        return task

    async def _inspect_resume_checkpoint(_session: Session):
        return SimpleNamespace(
            is_resumable=True,
            run_id="run-1",
            has_checkpoint=True,
            pending_interrupt={"kind": "input_text", "prompt": "请继续", "response_key": "message"},
        )

    service._get_task = _get_task
    service._inspect_resume_checkpoint = _inspect_resume_checkpoint

    async def _collect_events():
        events = []
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                resume={"message": "继续执行"},
        ):
            events.append(event)
        return events

    events = asyncio.run(_collect_events())

    assert not [event for event in events if isinstance(event, ErrorEvent)]
    assert task.input_stream.messages
    assert task.invoked is True
    assert session.status == SessionStatus.RUNNING
    assert run.status == WorkflowRunStatus.RUNNING


@pytest.mark.parametrize(
    ("pending_interrupt", "resume_value", "expected_reason_code"),
    [
        (
            {
                "kind": "confirm",
                "prompt": "确认继续执行？",
                "confirm_resume_value": True,
                "cancel_resume_value": False,
            },
            True,
            "user_confirmed",
        ),
        (
            {
                "kind": "select",
                "prompt": "请选择执行方式",
                "options": [
                    {"label": "方案A", "resume_value": "a"},
                    {"label": "方案B", "resume_value": "b"},
                ],
            },
            "b",
            "user_selected_option",
        ),
        (
            {
                "kind": "input_text",
                "prompt": "请补充约束",
                "response_key": "message",
            },
            {"message": "补充说明"},
            "user_provided_clarification",
        ),
    ],
)
def test_agent_service_chat_should_record_wait_resume_feedback(
    pending_interrupt,
    resume_value,
    expected_reason_code,
) -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.WAITING,
        workspace_id="workspace-1",
        current_run_id="run-1",
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        status=WorkflowRunStatus.WAITING,
    )
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
    )
    workflow_run_repo = _WorkflowRunRepo(run)
    task = _IdleTask()
    feedback_ingress = _UserFeedbackIngressService()

    async def _get_latest_event_record_by_session(session_id: str, *, event_type: str | None = None, run_id: str | None = None):
        assert session_id == "session-1"
        assert event_type == "wait"
        assert run_id == "run-1"
        return SimpleNamespace(
            session_id=session_id,
            run_id="run-1",
            event_id="wait-1",
            event_type="wait",
            event_payload=SimpleNamespace(type="wait"),
        )

    workflow_run_repo.get_latest_event_record_by_session = _get_latest_event_record_by_session

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session), workflow_run_repo, _WorkspaceRepo(workspace))
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()
    service._user_feedback_ingress_service = feedback_ingress

    async def _get_task(_session: Session):
        return task

    async def _inspect_resume_checkpoint(_session: Session):
        return SimpleNamespace(
            is_resumable=True,
            run_id="run-1",
            has_checkpoint=True,
            pending_interrupt=pending_interrupt,
        )

    service._get_task = _get_task
    service._inspect_resume_checkpoint = _inspect_resume_checkpoint

    async def _collect_events():
        return [event async for event in service.chat(
            session_id="session-1",
            user_id="user-1",
            resume=resume_value,
        )]

    asyncio.run(_collect_events())

    assert len(feedback_ingress.resume_calls) == 1
    resume_call = feedback_ingress.resume_calls[0]
    assert resume_call["wait_event_id"] == "wait-1"
    assert resume_call["wait_payload"] == pending_interrupt
    assert resume_call["resume_value"] == resume_value
    assert resume_call["access_scope"].run_id == "run-1"
    assert expected_reason_code in {"user_confirmed", "user_selected_option", "user_provided_clarification"}


def test_agent_service_unresolved_runtime_conflict_should_use_error_key() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
    )
    reconcile_result = SimpleNamespace(
        warnings=["session_status_mismatch_run_status"],
        snapshot_after=SimpleNamespace(session_status=SessionStatus.RUNNING),
    )

    with pytest.raises(BadRequestError) as exc_info:
        AgentService._reject_unresolved_runtime_conflict(
            session=session,
            reconcile_result=reconcile_result,
        )

    assert exc_info.value.error_key == error_keys.SESSION_RUNTIME_STATE_CONFLICT


def test_agent_service_chat_should_reject_resume_when_session_not_waiting() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
    )

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                resume={"message": "继续执行"},
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_NOT_WAITING


def test_agent_service_chat_should_rollback_resume_input_when_status_update_fails() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.WAITING,
        workspace_id="workspace-1",
        current_run_id="run-1",
    )
    workspace = Workspace(id="workspace-1", session_id="session-1", current_run_id="run-1")
    workflow_run = WorkflowRun(id="run-1", session_id="session-1", status=WorkflowRunStatus.WAITING)
    task = _Task()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(
        _SessionRepo(session, fail_on_update_status=True),
        _WorkflowRunRepo(workflow_run),
        _WorkspaceRepo(workspace),
    )
    service._task_cls = _TaskFactory

    async def _get_task(_session: Session):
        return task

    service._get_task = _get_task
    async def _inspect_resume_checkpoint(_session: Session):
        return SimpleNamespace(
            is_resumable=True,
            run_id="run-1",
            has_checkpoint=True,
            pending_interrupt={"kind": "input_text", "prompt": "请继续", "response_key": "message"},
        )
    service._inspect_resume_checkpoint = _inspect_resume_checkpoint

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                resume={"message": "继续执行"},
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error == "update status failed"
    assert task.input_stream.deleted == ["resume-msg-1"]
    assert task.invoked is False


def test_agent_service_chat_should_reject_resume_when_checkpoint_invalid() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.WAITING,
        current_run_id="run-1",
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        status=WorkflowRunStatus.WAITING,
    )

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session), _WorkflowRunRepo(run))
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()

    async def _inspect_resume_checkpoint(_session: Session):
        return SimpleNamespace(
            is_resumable=False,
            run_id="run-1",
            has_checkpoint=False,
            pending_interrupt={},
        )

    service._inspect_resume_checkpoint = _inspect_resume_checkpoint

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                resume={"approved": True},
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_RESUME_CHECKPOINT_INVALID
    assert session.status == SessionStatus.WAITING


def test_agent_service_inspect_resume_checkpoint_should_inject_runtime_context_service(monkeypatch) -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.WAITING,
    )

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
    service._task_cls = _TaskFactory

    async def _resolve_runtime_llm(_session: Session):
        return object()

    service._resolve_runtime_llm = _resolve_runtime_llm
    captured: dict[str, object] = {}

    class _FakeLangGraphRunEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def inspect_resume_checkpoint(self):
            return SimpleNamespace(
                is_resumable=True,
                run_id="run-1",
                has_checkpoint=True,
                pending_interrupt={},
            )

    monkeypatch.setattr(
        "app.application.service.agent_service.LangGraphRunEngine",
        _FakeLangGraphRunEngine,
    )
    monkeypatch.setattr(
        "app.application.service.agent_service.get_langgraph_checkpointer",
        lambda: SimpleNamespace(get_checkpointer=lambda: None),
    )

    inspection = asyncio.run(service._inspect_resume_checkpoint(session))

    assert inspection.is_resumable is True
    assert captured.get("runtime_context_service") is not None


def test_agent_service_chat_should_enqueue_continue_cancelled_task_command() -> None:
    cancelled_plan = Plan(
        title="被取消任务",
        goal="继续执行",
        steps=[
            Step(id="step-1", description="已完成步骤", status=ExecutionStatus.COMPLETED),
            Step(id="step-2", description="被取消步骤", status=ExecutionStatus.CANCELLED),
        ],
        status=ExecutionStatus.CANCELLED,
    )
    session = Session(
        id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
        current_run_id="run-1",
        status=SessionStatus.CANCELLED,
        events=[PlanEvent(plan=cancelled_plan, status=PlanEventStatus.CANCELLED)],
    )
    workspace = Workspace(id="workspace-1", session_id="session-1", current_run_id="run-1")
    workflow_run = WorkflowRun(id="run-1", session_id="session-1", user_id="user-1", status=WorkflowRunStatus.CANCELLED)
    new_workflow_run = WorkflowRun(
        id="run-2",
        session_id="session-1",
        user_id="user-1",
        status=WorkflowRunStatus.RUNNING,
    )
    wait_event = WaitEvent(id="wait-old-1", payload={"kind": "confirm", "prompt": "继续？"})
    workflow_run_repo = _WorkflowRunRepo(
        workflow_run,
        events=list(session.events),
        event_records=[
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                user_id="user-1",
                event_id=wait_event.id,
                event_type=wait_event.type,
                event_payload=wait_event,
            )
        ],
    )
    workspace_repo = _WorkspaceRepo(workspace)
    task = _IdleTask()
    feedback_ingress = _UserFeedbackIngressService()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session), workflow_run_repo, workspace_repo)
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()
    service._user_feedback_ingress_service = feedback_ingress

    async def _get_task(_session: Session):
        return None

    async def _create_task(_session: Session, *, reuse_current_run: bool = False):
        assert reuse_current_run is False
        _session.current_run_id = "run-2"
        _session.status = SessionStatus.RUNNING
        workspace.current_run_id = "run-2"
        workflow_run_repo._run = new_workflow_run
        return task

    service._get_task = _get_task
    service._create_task = _create_task

    async def _collect():
        return [event async for event in service.chat(
            session_id="session-1",
            user_id="user-1",
            command={"type": "continue_cancelled_task"},
        )]

    events = asyncio.run(_collect())

    assert events == []
    assert task.invoked is True
    assert session.status == SessionStatus.RUNNING
    runtime_input = RuntimeInput.model_validate_json(task.input_stream.messages[0])
    assert isinstance(runtime_input.payload, ContinueCancelledTaskInput)
    assert len(feedback_ingress.continue_calls) == 1
    assert feedback_ingress.continue_calls[0]["old_wait_event_id"] == "wait-old-1"
    assert feedback_ingress.continue_calls[0]["old_cancelled_run_id"] == "run-1"
    assert feedback_ingress.continue_calls[0]["access_scope"].run_id == "run-2"


def test_agent_service_chat_should_reject_continue_cancelled_task_when_session_not_cancelled() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.COMPLETED,
    )

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                command={"type": "continue_cancelled_task"},
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_NOT_CANCELLED


def test_agent_service_chat_should_reject_continue_cancelled_task_when_plan_unavailable() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.CANCELLED,
        events=[
            PlanEvent(
                plan=Plan(
                    title="无可继续计划",
                    goal="继续执行",
                    steps=[Step(id="step-1", description="已完成步骤", status=ExecutionStatus.COMPLETED)],
                    status=ExecutionStatus.CANCELLED,
                )
            )
        ],
    )

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                command={"type": "continue_cancelled_task"},
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_CANCELLED_CONTINUE_UNAVAILABLE


def test_agent_service_chat_should_reject_continue_cancelled_task_when_old_wait_event_missing() -> None:
    cancelled_plan = Plan(
        title="被取消任务",
        goal="继续执行",
        steps=[Step(id="step-1", description="被取消步骤", status=ExecutionStatus.CANCELLED)],
        status=ExecutionStatus.CANCELLED,
    )
    session = Session(
        id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
        current_run_id="run-1",
        status=SessionStatus.CANCELLED,
        events=[PlanEvent(plan=cancelled_plan, status=PlanEventStatus.CANCELLED)],
    )
    workflow_run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        status=WorkflowRunStatus.CANCELLED,
    )
    task = _IdleTask()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(
        _SessionRepo(session),
        _WorkflowRunRepo(workflow_run),
        _WorkspaceRepo(Workspace(id="workspace-1", session_id="session-1", current_run_id="run-1")),
    )
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()

    async def _get_task(_session: Session):
        return None

    async def _create_task(_session: Session, *, reuse_current_run: bool = False):
        return task

    service._get_task = _get_task
    service._create_task = _create_task

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                command={"type": "continue_cancelled_task"},
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_CANCELLED_CONTINUE_UNAVAILABLE
    assert task.invoked is False
    assert task.input_stream.messages == []


@pytest.mark.parametrize(
    ("pending_interrupt", "resume_value"),
    [
        (
            {
                "kind": "input_text",
                "prompt": "请输入目标网址",
                "response_key": "website",
                "allow_empty": False,
            },
            {"message": "https://example.com"},
        ),
        (
            {
                "kind": "confirm",
                "prompt": "确认继续执行？",
                "confirm_resume_value": True,
                "cancel_resume_value": False,
            },
            {"approved": True},
        ),
        (
            {
                "kind": "select",
                "prompt": "请选择执行方式",
                "options": [
                    {"label": "方案A", "resume_value": "a"},
                    {"label": "方案B", "resume_value": "b"},
                ],
            },
            "c",
        ),
    ],
)
def test_agent_service_chat_should_reject_resume_when_value_invalid(
    pending_interrupt,
    resume_value,
) -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.WAITING,
        current_run_id="run-1",
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        status=WorkflowRunStatus.WAITING,
    )
    task = _Task()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session), _WorkflowRunRepo(run))
    service._task_cls = _TaskFactory

    async def _get_task(_session: Session):
        return task

    async def _inspect_resume_checkpoint(_session: Session):
        return SimpleNamespace(
            is_resumable=True,
            run_id="run-1",
            has_checkpoint=True,
            pending_interrupt=pending_interrupt,
        )

    service._get_task = _get_task
    service._inspect_resume_checkpoint = _inspect_resume_checkpoint

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                resume=resume_value,
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_RESUME_VALUE_INVALID
    assert session.status == SessionStatus.WAITING
    assert task.input_stream.messages == []
    assert task.invoked is False


def test_agent_service_stop_session_should_mark_active_task_as_cancelled() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        current_run_id="run-1",
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        status=WorkflowRunStatus.RUNNING,
    )
    task = _CancellableTask(session)
    session_repo = _StopSessionRepo(session)

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo, _WorkflowRunRepo(run))
    service._task_cls = _TaskFactory

    async def _get_task(_session: Session):
        return task

    service._get_task = _get_task

    asyncio.run(service.stop_session("session-1", "user-1"))

    assert task.cancelled is True
    assert session.status == SessionStatus.CANCELLED


def test_agent_service_stop_session_should_not_fail_when_cancel_feedback_write_fails() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        current_run_id="run-1",
        workspace_id="workspace-1",
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        status=WorkflowRunStatus.RUNNING,
    )
    feedback_ingress = _UserFeedbackIngressService()
    feedback_ingress.fail_cancel_record = True
    task = _CancellableTask(session)
    workspace = Workspace(id="workspace-1", session_id="session-1", user_id="user-1", current_run_id="run-1")

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(
        _StopSessionRepo(session),
        _WorkflowRunRepo(run),
        _WorkspaceRepo(workspace),
    )
    service._task_cls = _TaskFactory
    service._user_feedback_ingress_service = feedback_ingress

    async def _get_task(_session: Session):
        return task

    service._get_task = _get_task

    asyncio.run(service.stop_session("session-1", "user-1"))

    assert task.cancelled is True
    assert session.status == SessionStatus.CANCELLED
    assert len(feedback_ingress.cancel_calls) == 1


def test_agent_service_stop_session_should_persist_cancelled_run_and_step_when_task_missing() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        workspace_id="workspace-1",
    )
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        current_run_id="run-1",
    )
    workflow_run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        status=WorkflowRunStatus.RUNNING,
        current_step_id="step-1",
        runtime_metadata={
            "graph_state_contract": {
                "graph_state": {
                    "plan": Plan(
                        title="任务",
                        goal="完成任务",
                        steps=[
                            Step(
                                id="step-1",
                                description="执行步骤1",
                                status=ExecutionStatus.RUNNING,
                            ),
                            Step(
                                id="step-2",
                                description="执行步骤2",
                                status=ExecutionStatus.PENDING,
                            ),
                        ],
                    ).model_dump(mode="json")
                }
            }
        },
    )
    session_repo = _StopSessionRepo(session)
    workflow_run_repo = _WorkflowRunRepo(workflow_run)
    workspace_repo = _WorkspaceRepo(workspace)

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo, workflow_run_repo, workspace_repo)
    service._task_cls = _TaskFactory

    async def _get_task(_session: Session):
        return None

    service._get_task = _get_task

    asyncio.run(service.stop_session("session-1", "user-1"))

    assert session.status == SessionStatus.CANCELLED
    assert workflow_run.status == WorkflowRunStatus.CANCELLED
    assert workflow_run_repo.cancelled_run_ids == []
    assert workflow_run_repo.upserted_step_count == 1
    assert workflow_run_repo.replaced_plan_count == 1
    assert workflow_run_repo.unfinished_steps_cancelled_count == 1
    inserted_events = workflow_run_repo._events
    assert len(inserted_events) == 2
    assert isinstance(inserted_events[0], StepEvent)
    assert inserted_events[0].status.value == "cancelled"
    assert isinstance(inserted_events[1], PlanEvent)
    assert inserted_events[1].status.value == "cancelled"


def test_agent_service_stop_session_should_build_cancelled_events_from_run_history_when_runtime_metadata_missing() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        workspace_id="workspace-1",
    )
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        current_run_id="run-1",
    )
    workflow_run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        status=WorkflowRunStatus.RUNNING,
        current_step_id="step-2",
        runtime_metadata={},
    )
    run_events = [
        MessageEvent(role="user", message="开始执行"),
        PlanEvent(
            plan=Plan(
                title="任务",
                goal="完成任务",
                steps=[
                    Step(
                        id="step-1",
                        description="执行步骤1",
                        status=ExecutionStatus.PENDING,
                    ),
                    Step(
                        id="step-2",
                        description="执行步骤2",
                        status=ExecutionStatus.PENDING,
                    ),
                ],
            )
        ),
        StepEvent(
            step=Step(
                id="step-1",
                description="执行步骤1",
                status=ExecutionStatus.COMPLETED,
            )
        ),
        StepEvent(
            step=Step(
                id="step-2",
                description="执行步骤2",
                status=ExecutionStatus.RUNNING,
            )
        ),
    ]
    session_repo = _StopSessionRepo(session)
    workflow_run_repo = _WorkflowRunRepo(workflow_run, events=run_events)
    workspace_repo = _WorkspaceRepo(workspace)

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo, workflow_run_repo, workspace_repo)
    service._task_cls = _TaskFactory

    async def _get_task(_session: Session):
        return None

    service._get_task = _get_task

    asyncio.run(service.stop_session("session-1", "user-1"))

    assert session.status == SessionStatus.CANCELLED
    assert workflow_run.status == WorkflowRunStatus.CANCELLED
    assert workflow_run_repo.cancelled_run_ids == []
    assert workflow_run_repo.unfinished_steps_cancelled_count == 1
    assert len(workflow_run_repo._events) == 6

    cancelled_step_event = workflow_run_repo._events[-2]
    assert isinstance(cancelled_step_event, StepEvent)
    assert cancelled_step_event.step.id == "step-2"
    assert cancelled_step_event.step.status == ExecutionStatus.CANCELLED
    assert cancelled_step_event.step.outcome is not None
    assert cancelled_step_event.step.outcome.summary == "任务已取消"

    cancelled_plan_event = workflow_run_repo._events[-1]
    assert isinstance(cancelled_plan_event, PlanEvent)
    assert cancelled_plan_event.plan.status == ExecutionStatus.CANCELLED
    assert [step.status for step in cancelled_plan_event.plan.steps] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.CANCELLED,
    ]


def _access_scope(run_id: str = "run-2") -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id=run_id,
    )


def test_user_feedback_capture_policy_should_force_continue_cancelled_session_scope() -> None:
    assert (
        UserFeedbackCapturePolicy.resolve_scope_kind(UserFeedbackIntentKind.CONTINUE_CANCELLED)
        == FeedbackScopeKind.SESSION
    )


@pytest.mark.parametrize(
    ("payload_kind", "resume_value", "expected_intent", "expected_reason"),
    [
        (
            "confirm",
            True,
            UserFeedbackIntentKind.CONFIRMATION,
            FeedbackReasonCode.USER_CONFIRMED,
        ),
        (
            "confirm",
            False,
            UserFeedbackIntentKind.CONFIRMATION,
            FeedbackReasonCode.USER_REJECTED,
        ),
        (
            "select",
            "b",
            UserFeedbackIntentKind.SELECTION,
            FeedbackReasonCode.USER_SELECTED_OPTION,
        ),
        (
            "input_text",
            {"message": "补充"},
            UserFeedbackIntentKind.CLARIFICATION,
            FeedbackReasonCode.USER_PROVIDED_CLARIFICATION,
        ),
    ],
)
def test_user_feedback_ingress_should_resolve_wait_resume_feedback_semantics(
    payload_kind,
    resume_value,
    expected_intent,
    expected_reason,
) -> None:
    payload = {
        "kind": payload_kind,
        "confirm_resume_value": True,
        "cancel_resume_value": False,
    }

    intent, reason = UserFeedbackIngressService._resolve_wait_resume_feedback_semantics(
        payload_kind=payload_kind,
        resume_value=resume_value,
        payload=payload,
    )

    assert intent == expected_intent
    assert reason == expected_reason


def test_user_feedback_ingress_should_record_continue_cancelled_with_new_feedback_input_source() -> None:
    old_wait_event = WaitEvent(id="wait-old-1", payload={"kind": "confirm", "prompt": "继续？"})
    old_wait_record = WorkflowRunEventRecord(
        run_id="run-1",
        session_id="session-1",
        user_id="user-1",
        event_id=old_wait_event.id,
        event_type=old_wait_event.type,
        event_payload=old_wait_event,
    )
    workflow_run_repo = _WorkflowRunRepo(
        WorkflowRun(id="run-1", session_id="session-1", user_id="user-1", status=WorkflowRunStatus.CANCELLED),
        event_records=[old_wait_record],
    )
    captured_commands = []

    class _FeedbackLedgerService:
        async def record_user_feedback(self, command):
            captured_commands.append(command)
            return SimpleNamespace(success=True)

    service = UserFeedbackIngressService(
        uow_factory=lambda: _UoW(
            _SessionRepo(Session(id="session-1", user_id="user-1")),
            workflow_run_repo,
            _WorkspaceRepo(Workspace(id="workspace-1", session_id="session-1", user_id="user-1")),
        ),
        feedback_service=_FeedbackLedgerService(),
    )

    result = asyncio.run(service.record_continue_cancelled_feedback(
        access_scope=_access_scope(run_id="run-2"),
        old_wait_event_id="wait-old-1",
        old_cancelled_run_id="run-1",
    ))

    assert result.success is True
    assert len(captured_commands) == 1
    command = captured_commands[0]
    assert command.requested_feedback_scope_kind == FeedbackScopeKind.SESSION
    assert command.source_ref.source_kind.value == "feedback_input"
    assert command.source_ref.source_run_id == "run-2"
    assert command.target_ref.target_type.value == "wait_event"
    assert command.target_ref.target_id == "wait-old-1"
    assert command.target_ref.target_run_id == "run-1"
    assert command.current_run_id_at_record_time == "run-2"
    feedback_input_events = [
        record for record in workflow_run_repo._event_records
        if record.event_type == "feedback_input"
    ]
    assert len(feedback_input_events) == 1
    assert feedback_input_events[0].run_id == "run-2"
    assert feedback_input_events[0].event_payload.payload.source_action == "continue_cancelled"


def test_user_feedback_ingress_should_reject_continue_cancelled_old_wait_event_cross_session() -> None:
    old_wait_event = WaitEvent(id="wait-old-1", payload={"kind": "confirm"})
    workflow_run_repo = _WorkflowRunRepo(
        WorkflowRun(id="run-1", session_id="other-session", user_id="user-1", status=WorkflowRunStatus.CANCELLED),
        event_records=[
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="other-session",
                user_id="user-1",
                event_id=old_wait_event.id,
                event_type=old_wait_event.type,
                event_payload=old_wait_event,
            )
        ],
    )
    captured_commands = []

    class _FeedbackLedgerService:
        async def record_user_feedback(self, command):
            captured_commands.append(command)
            return SimpleNamespace(success=True)

    service = UserFeedbackIngressService(
        uow_factory=lambda: _UoW(
            _SessionRepo(Session(id="session-1", user_id="user-1")),
            workflow_run_repo,
            _WorkspaceRepo(Workspace(id="workspace-1", session_id="session-1", user_id="user-1")),
        ),
        feedback_service=_FeedbackLedgerService(),
    )

    with pytest.raises(FeedbackRequiredRecordError):
        asyncio.run(service.record_continue_cancelled_feedback(
            access_scope=_access_scope(run_id="run-2"),
            old_wait_event_id="wait-old-1",
            old_cancelled_run_id="run-1",
        ))

    assert captured_commands == []
    assert all(record.event_type != "feedback_input" for record in workflow_run_repo._event_records)
