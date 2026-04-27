import asyncio
from types import SimpleNamespace

import pytest

from app.application.errors import BadRequestError, error_keys
from app.application.service.runtime_state_coordinator import RuntimeStateCoordinator
from app.application.service.agent_service import AgentService
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
    WorkflowRun,
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
    def __init__(self, run: WorkflowRun | None = None, *, events: list[object] | None = None) -> None:
        self._run = run
        self._events = list(events or [])
        self.cancelled_run_ids: list[str] = []
        self.replaced_plan_count = 0
        self.upserted_step_count = 0
        self.unfinished_steps_cancelled_count = 0

    async def get_by_id(self, run_id: str):
        if self._run is None:
            return None
        return self._run if self._run.id == run_id else None

    async def get_by_id_for_update(self, run_id: str):
        return await self.get_by_id(run_id)

    async def update_status(self, run_id: str, *, status: WorkflowRunStatus, **_kwargs) -> None:
        if self._run is not None and self._run.id == run_id:
            self._run.status = status

    async def add_event_record_if_absent(self, session_id: str, run_id: str, event) -> bool:
        self._events.append(event)
        return True

    async def replace_steps_from_plan(self, run_id: str, plan: Plan) -> None:
        self.replaced_plan_count += 1

    async def upsert_step_from_event(self, run_id: str, event) -> None:
        self.upserted_step_count += 1

    async def mark_unfinished_steps_cancelled(self, run_id: str) -> None:
        self.unfinished_steps_cancelled_count += 1

    async def list_event_records_by_session(self, session_id: str):
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

    async def get_by_id(self, workspace_id: str):
        if self._workspace is None or workspace_id != self._workspace.id:
            return None
        return self._workspace

    async def get_by_session_id(self, session_id: str):
        if self._workspace is None or session_id != self._workspace.session_id:
            return None
        return self._workspace


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

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
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
    workflow_run = WorkflowRun(id="run-1", session_id="session-1", status=WorkflowRunStatus.CANCELLED)
    new_workflow_run = WorkflowRun(id="run-2", session_id="session-1", status=WorkflowRunStatus.RUNNING)
    workflow_run_repo = _WorkflowRunRepo(workflow_run, events=list(session.events))
    workspace_repo = _WorkspaceRepo(workspace)
    task = _IdleTask()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session), workflow_run_repo, workspace_repo)
    service._task_cls = _TaskFactory
    service._graph_runtime = _NoTaskGraphRuntime()

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
    task = _Task()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
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
    task = _CancellableTask(session)
    session_repo = _StopSessionRepo(session)

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo)
    service._task_cls = _TaskFactory

    async def _get_task(_session: Session):
        return task

    service._get_task = _get_task

    asyncio.run(service.stop_session("session-1", "user-1"))

    assert task.cancelled is True
    assert session.status == SessionStatus.CANCELLED


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
