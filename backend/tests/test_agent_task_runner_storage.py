import asyncio
import io
from typing import Optional

import pytest

from app.domain.models import (
    ContinueCancelledTaskInput,
    DoneEvent,
    ExecutionStatus,
    File,
    MessageCommand,
    MessageEvent,
    Message,
    Plan,
    PlanEvent,
    RuntimeInput,
    Session,
    SessionStatus,
    Step,
    StepEvent,
    ToolEvent,
    ToolEventStatus,
    WorkflowRun,
    WorkflowRunStatus,
)
from app.domain.models.tool_result import ToolResult
from app.domain.services.agent_task_runner import AgentTaskRunner


class _DummySessionRepo:
    async def get_file_by_path(self, session_id: str, filepath: str):
        return None

    async def remove_file(self, session_id: str, file_id: str) -> None:
        return None

    async def add_file(self, session_id: str, file: File) -> None:
        return None


class _DummyUoW:
    def __init__(self) -> None:
        self.session = _DummySessionRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FailSandbox:
    async def check_file_exists(self, file_path: str):
        return ToolResult(success=True, data=True)

    async def download_file(self, file_path: str):
        raise RuntimeError("download failed")


class _DummyFileStorage:
    async def upload_file(self, upload_file, user_id=None):
        raise AssertionError("upload_file should not be called when sandbox download fails")


class _MissingFileSandbox:
    async def check_file_exists(self, file_path: str):
        return ToolResult(success=True, data=False)

    async def download_file(self, file_path: str):
        raise AssertionError("download_file should not be called when file does not exist")


class _MissingFileStorage:
    async def upload_file(self, upload_file, user_id=None):
        raise AssertionError("upload_file should not be called when file does not exist")


class _NoopSandbox:
    async def ensure_sandbox(self) -> None:
        return None

    async def destroy(self) -> None:
        return None


class _NoopTool:
    async def initialize(self, *_args, **_kwargs) -> None:
        return None

    async def cleanup(self) -> None:
        return None


class _NoopA2ATool:
    def __init__(self) -> None:
        self.manager = _NoopTool()

    async def initialize(self, *_args, **_kwargs) -> None:
        return None


class _InvokeSessionRepo:
    def __init__(self) -> None:
        self.updated_status: Optional[SessionStatus] = None

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        self.updated_status = status


class _InvokeUoW:
    def __init__(self, session_repo: _InvokeSessionRepo) -> None:
        self.session = session_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _InputStreamWithEmptyEvent:
    def __init__(self) -> None:
        self._is_empty_checks = 0

    async def is_empty(self) -> bool:
        self._is_empty_checks += 1
        return self._is_empty_checks > 1

    async def pop(self):
        return "evt-1", None


class _InvokeTask:
    def __init__(self) -> None:
        self.id = "task-1"
        self.input_stream = _InputStreamWithEmptyEvent()


def _build_runner_for_storage_sync() -> AgentTaskRunner:
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _DummyUoW()
    runner._sandbox = _FailSandbox()
    runner._file_storage = _DummyFileStorage()
    return runner


class _SandboxUploadFail:
    async def upload_file(self, *, file_data, file_path: str, filename: str):
        return ToolResult(success=False, message="sandbox rejected")


class _SandboxSyncFileStorage:
    async def download_file(self, file_id: str):
        return io.BytesIO(b"hello"), File(id=file_id, filename="a.txt")


class _SandboxSyncFileRepo:
    async def save(self, file: File) -> None:
        return None


class _SandboxSyncSessionRepo:
    async def add_file(self, session_id: str, file: File) -> None:
        return None


class _SandboxSyncUoW:
    def __init__(self) -> None:
        self.file = _SandboxSyncFileRepo()
        self.session = _SandboxSyncSessionRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_sync_file_to_storage_re_raises_exception() -> None:
    runner = _build_runner_for_storage_sync()

    with pytest.raises(RuntimeError, match="download failed"):
        asyncio.run(runner._sync_file_to_storage("/tmp/a.txt"))


def test_sync_file_to_sandbox_should_re_raise_when_upload_unsuccessful() -> None:
    runner = object.__new__(AgentTaskRunner)
    runner._sandbox = _SandboxUploadFail()
    runner._file_storage = _SandboxSyncFileStorage()
    runner._uow_factory = lambda: _SandboxSyncUoW()

    with pytest.raises(RuntimeError, match="sandbox rejected"):
        asyncio.run(runner._sync_file_to_sandbox("file-1"))


def test_sync_file_to_storage_should_skip_when_file_not_exists() -> None:
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _DummyUoW()
    runner._sandbox = _MissingFileSandbox()
    runner._file_storage = _MissingFileStorage()

    synced = asyncio.run(runner._sync_file_to_storage("/tmp/not_exists.txt"))

    assert synced is None


def test_sync_message_attachments_to_storage_re_raises_exception() -> None:
    runner = object.__new__(AgentTaskRunner)

    async def _raise_sync_error(filepath: str, _stage: str = "intermediate"):
        raise RuntimeError(f"sync failed: {filepath}")

    runner._user_id = "user-1"
    runner._sync_file_to_storage = _raise_sync_error
    event = MessageEvent(
        role="assistant",
        message="hello",
        attachments=[File(filename="a.txt", filepath="/tmp/a.txt")],
    )

    with pytest.raises(RuntimeError, match="sync failed: /tmp/a.txt"):
        asyncio.run(runner._sync_message_attachments_to_storage(event))


def test_sync_message_attachments_to_sandbox_should_re_raise_when_file_id_missing() -> None:
    runner = object.__new__(AgentTaskRunner)
    runner._uow_factory = lambda: _SandboxSyncUoW()
    event = MessageEvent(
        role="user",
        message="hello",
        attachments=[File(id="", filename="a.txt", filepath="/tmp/a.txt")],
    )

    with pytest.raises(RuntimeError, match="缺少 file_id"):
        asyncio.run(runner._sync_message_attachments_to_sandbox(event))


def test_invoke_skips_empty_input_stream_event() -> None:
    runner = object.__new__(AgentTaskRunner)
    session_repo = _InvokeSessionRepo()

    runner._session_id = "session-1"
    runner._sandbox = _NoopSandbox()
    runner._mcp_tool = _NoopTool()
    runner._a2a_tool = _NoopA2ATool()
    runner._mcp_config = None
    runner._a2a_config = None
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _InvokeUoW(session_repo)

    asyncio.run(runner.invoke(_InvokeTask()))

    assert session_repo.updated_status == SessionStatus.COMPLETED


class _QueuedInputStream:
    def __init__(self, events: list[MessageEvent]) -> None:
        self._events = list(events)

    async def is_empty(self) -> bool:
        return len(self._events) == 0

    async def pop_next(self) -> MessageEvent:
        return self._events.pop(0)


class _SerialInvokeTask:
    def __init__(self, events: list[MessageEvent]) -> None:
        self.id = "task-serial"
        self.input_stream = _QueuedInputStream(events)
        self.output_stream = _NoopOutputStream()


class _NoopOutputStream:
    def __init__(self) -> None:
        self.sequence = 0

    async def put(self, _message: str) -> str:
        self.sequence += 1
        return f"evt-{self.sequence}"


def test_invoke_should_finish_current_run_before_consuming_next_input() -> None:
    runner = object.__new__(AgentTaskRunner)
    session_repo = _InvokeSessionRepo()
    task = _SerialInvokeTask(
        [
            MessageEvent(role="user", message="first"),
            MessageEvent(role="user", message="second"),
        ]
    )

    runner._session_id = "session-1"
    runner._sandbox = _NoopSandbox()
    runner._mcp_tool = _NoopTool()
    runner._a2a_tool = _NoopA2ATool()
    runner._mcp_config = None
    runner._a2a_config = None
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _InvokeUoW(session_repo)

    emitted_pairs: list[tuple[str, Optional[SessionStatus]]] = []
    processed_messages: list[str] = []

    async def _pop_event(_task):
        message_event = await _task.input_stream.pop_next()
        return RuntimeInput(
            request_id=f"req-{message_event.message}",
            payload=message_event,
        )

    async def _run_flow(message):
        processed_messages.append(message.message)
        yield ToolEvent(
            tool_call_id=f"call-{message.message}",
            tool_name="dummy",
            function_name="dummy_tool",
            function_args={},
            status=ToolEventStatus.CALLING,
        )
        yield DoneEvent()

    async def _put_and_add_event(*, task, event, title=None, latest_message=None, latest_message_at=None, increment_unread=False, status=None):
        emitted_pairs.append((type(event).__name__, status))

    runner._pop_event = _pop_event
    runner._run_flow = _run_flow
    runner._put_and_add_event = _put_and_add_event

    asyncio.run(runner.invoke(task))

    assert processed_messages == ["first", "second"]
    assert emitted_pairs == [
        ("ToolEvent", None),
        ("DoneEvent", None),
        ("ToolEvent", None),
        ("DoneEvent", SessionStatus.COMPLETED),
    ]
    assert session_repo.updated_status == SessionStatus.COMPLETED


class _BrowserScreenshot:
    async def screenshot(self) -> bytes:
        return b"fake-image-bytes"


class _ScreenshotFileStorage:
    def __init__(self) -> None:
        self.upload_user_ids: list[str | None] = []
        self.upload_payloads: list[object] = []

    async def upload_file(self, upload_file, user_id=None):
        self.upload_user_ids.append(user_id)
        self.upload_payloads.append(upload_file)
        return File(id="file-screenshot", key="2026/03/19/s.png")

    def get_file_url(self, file: File) -> str:
        return f"https://cdn.example.com/{file.key}"


def test_get_browser_screenshot_should_use_storage_url_provider() -> None:
    runner = object.__new__(AgentTaskRunner)
    runner._browser = _BrowserScreenshot()
    runner._user_id = "user-1"
    runner._file_storage = _ScreenshotFileStorage()

    screenshot_url = asyncio.run(runner._get_browser_screenshot())

    assert screenshot_url == "https://cdn.example.com/2026/03/19/s.png"
    assert runner._file_storage.upload_user_ids == ["user-1"]
    payload = runner._file_storage.upload_payloads[0]
    assert getattr(payload, "filename").endswith(".png")
    assert getattr(payload, "size") == len(b"fake-image-bytes")
    assert getattr(payload, "file").read() == b"fake-image-bytes"


class _CancellationSessionRepo:
    def __init__(self, session: Session) -> None:
        self._session = session
        self.updated_status: Optional[SessionStatus] = None
        self.added_events: list[object] = []

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        return self._session if session_id == self._session.id else None

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        self.updated_status = status
        self._session.status = status

    async def add_event_with_snapshot_if_absent(self, session_id: str, event, **_kwargs) -> None:
        self.added_events.append(event)


class _CancellationWorkflowRunRepo:
    def __init__(self, run: WorkflowRun, *, events: list[object]) -> None:
        self._run = run
        self._events = list(events)
        self.cancelled_run_ids: list[str] = []

    async def get_by_id(self, run_id: str):
        return self._run if run_id == self._run.id else None

    async def list_events(self, run_id: str | None):
        if run_id != self._run.id:
            return []
        return list(self._events)

    async def cancel_run(self, run_id: str) -> None:
        self.cancelled_run_ids.append(run_id)
        self._run.status = WorkflowRunStatus.CANCELLED


class _CancellationUoW:
    def __init__(self, session_repo: _CancellationSessionRepo, workflow_run_repo: _CancellationWorkflowRunRepo) -> None:
        self.session = session_repo
        self.workflow_run = workflow_run_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_persist_cancellation_state_should_build_cancelled_events_from_run_history() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        current_run_id="run-1",
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        status=WorkflowRunStatus.RUNNING,
        current_step_id="step-2",
        runtime_metadata={},
    )
    run_events = [
        PlanEvent(
            plan=Plan(
                title="任务",
                steps=[
                    Step(id="step-1", description="执行步骤1", status=ExecutionStatus.PENDING),
                    Step(id="step-2", description="执行步骤2", status=ExecutionStatus.PENDING),
                ],
            )
        ),
        StepEvent(
            step=Step(id="step-1", description="执行步骤1", status=ExecutionStatus.COMPLETED)
        ),
        StepEvent(
            step=Step(id="step-2", description="执行步骤2", status=ExecutionStatus.RUNNING)
        ),
    ]
    session_repo = _CancellationSessionRepo(session)
    workflow_run_repo = _CancellationWorkflowRunRepo(run, events=run_events)
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._uow_factory = lambda: _CancellationUoW(session_repo, workflow_run_repo)

    asyncio.run(runner._persist_cancellation_state())

    assert session_repo.updated_status == SessionStatus.CANCELLED
    assert workflow_run_repo.cancelled_run_ids == ["run-1"]
    assert len(session_repo.added_events) == 2

    cancelled_step_event = session_repo.added_events[0]
    assert isinstance(cancelled_step_event, StepEvent)
    assert cancelled_step_event.step.id == "step-2"
    assert cancelled_step_event.step.status == ExecutionStatus.CANCELLED

    cancelled_plan_event = session_repo.added_events[1]
    assert isinstance(cancelled_plan_event, PlanEvent)
    assert [step.status for step in cancelled_plan_event.plan.steps] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.CANCELLED,
    ]


def test_run_flow_should_allow_empty_message_when_command_present() -> None:
    runner = object.__new__(AgentTaskRunner)

    class _RunEngine:
        async def invoke(self, message):
            assert isinstance(message, Message)
            assert message.message == ""
            assert message.command == MessageCommand(type="continue_cancelled_task")
            yield DoneEvent()

    runner._run_engine = _RunEngine()
    runner._handle_tool_event = None
    runner._sync_message_attachments_to_storage = None

    async def _collect():
        return [event async for event in runner._run_flow(
            Message(message="", command=MessageCommand(type="continue_cancelled_task"))
        )]

    events = asyncio.run(_collect())

    assert len(events) == 1
    assert isinstance(events[0], DoneEvent)


def test_pop_event_should_parse_continue_cancelled_task_input_without_touching_event_history() -> None:
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"

    class _InputStream:
        async def pop(self):
            return "evt-continue-1", RuntimeInput(
                request_id="req-1",
                payload=ContinueCancelledTaskInput(),
            ).model_dump_json()

    class _SessionRepo:
        def __init__(self) -> None:
            self.called = False

        async def add_event_if_absent(self, session_id: str, event) -> None:
            self.called = True

    session_repo = _SessionRepo()

    class _UoW:
        def __init__(self) -> None:
            self.session = session_repo

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    runner._uow_factory = lambda: _UoW()

    runtime_input = asyncio.run(runner._pop_event(type("Task", (), {"input_stream": _InputStream()})()))

    assert runtime_input is not None
    assert isinstance(runtime_input.payload, ContinueCancelledTaskInput)
    assert session_repo.called is False
