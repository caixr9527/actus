import asyncio
import io
from typing import Optional

import pytest

from app.domain.models import (
    ContinueCancelledTaskInput,
    DoneEvent,
    ErrorEvent,
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
    WaitEvent,
    Workspace,
    WorkflowRun,
    WorkflowRunStatus,
)
from app.domain.models.tool_result import ToolResult
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.tools import CapabilityRegistry, ToolRuntimeAdapter
from app.domain.services.workspace_runtime.projectors import (
    BrowserScreenshotArtifactService,
    MessageAttachmentProjector,
    ToolEventProjector,
    UserInputAttachmentProjector,
)


class _NoopProjector:
    async def project(self, _event) -> None:
        return None


def _new_runner() -> AgentTaskRunner:
    runner = object.__new__(AgentTaskRunner)
    runner._tool_runtime_adapter = ToolRuntimeAdapter(
        capability_registry=CapabilityRegistry.default_v1(),
    )
    runner._tool_event_projector = _NoopProjector()
    runner._message_attachment_projector = _NoopProjector()
    runner._user_input_attachment_projector = _NoopProjector()
    return runner


class _DummySessionRepo:
    async def get_file_by_path(self, session_id: str, filepath: str):
        return None

    async def remove_file(self, session_id: str, file_id: str) -> None:
        return None

    async def add_file(self, session_id: str, file: File) -> None:
        return None

    async def add_final_files(self, session_id: str, file: File) -> None:
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
        self.updated_statuses: list[SessionStatus] = []

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        self.updated_status = status
        self.updated_statuses.append(status)


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


def _build_attachment_projector(
        *,
        session_id: str = "session-1",
        user_id: str = "user-1",
        uow_factory=None,
        sandbox=None,
        file_storage=None,
        workspace_runtime_service=None,
) -> MessageAttachmentProjector:
    return MessageAttachmentProjector(
        session_id=session_id,
        user_id=user_id,
        uow_factory=uow_factory or (lambda: _DummyUoW()),
        sandbox=sandbox or _FailSandbox(),
        file_storage=file_storage or _DummyFileStorage(),
        workspace_runtime_service=workspace_runtime_service or _CaptureWorkspaceRuntimeService(),
    )


def _build_tool_event_projector(
        *,
        browser,
        file_storage,
        workspace_runtime_service,
        user_id: str = "user-1",
) -> ToolEventProjector:
    return ToolEventProjector(
        adapter=ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1()),
        browser=browser,
        file_storage=file_storage,
        workspace_runtime_service=workspace_runtime_service,
        user_id=user_id,
    )


def _build_user_input_attachment_projector(
        *,
        session_id: str = "session-1",
        sandbox=None,
        file_storage=None,
        uow_factory=None,
) -> UserInputAttachmentProjector:
    return UserInputAttachmentProjector(
        session_id=session_id,
        sandbox=sandbox or _SandboxUploadFail(),
        file_storage=file_storage or _SandboxSyncFileStorage(),
        uow_factory=uow_factory or (lambda: _SandboxSyncUoW()),
    )


class _SandboxUploadFail:
    async def upload_file(self, *, file_data, file_path: str, filename: str):
        return ToolResult(success=False, message="sandbox rejected")


class _SandboxUploadSuccess:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def upload_file(self, *, file_data, file_path: str, filename: str):
        self.calls.append(
            {
                "file_path": file_path,
                "filename": filename,
                "content": file_data.read(),
            }
        )
        return ToolResult(success=True, data={"file_path": file_path})


class _SandboxSyncFileStorage:
    async def download_file(self, file_id: str):
        return io.BytesIO(b"hello"), File(id=file_id, filename="a.txt")


class _SandboxSyncFileRepo:
    def __init__(self) -> None:
        self.saved_files: list[File] = []

    async def save(self, file: File) -> None:
        self.saved_files.append(file)
        return None


class _SandboxSyncSessionRepo:
    def __init__(self) -> None:
        self.added_files: list[tuple[str, File]] = []

    async def add_file(self, session_id: str, file: File) -> None:
        self.added_files.append((session_id, file))
        return None


class _SandboxSyncUoW:
    def __init__(self, file_repo: _SandboxSyncFileRepo | None = None, session_repo: _SandboxSyncSessionRepo | None = None) -> None:
        self.file = file_repo or _SandboxSyncFileRepo()
        self.session = session_repo or _SandboxSyncSessionRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_sync_file_to_storage_re_raises_exception() -> None:
    projector = _build_attachment_projector()

    with pytest.raises(RuntimeError, match="download failed"):
        asyncio.run(projector.sync_file_to_storage("/tmp/a.txt"))


def test_user_input_attachment_projector_should_re_raise_when_upload_unsuccessful() -> None:
    projector = _build_user_input_attachment_projector()

    with pytest.raises(RuntimeError, match="sandbox rejected"):
        asyncio.run(projector.sync_file_to_sandbox(file_id="file-1"))


def test_user_input_attachment_projector_should_sync_attachments_to_sandbox_and_session() -> None:
    sandbox = _SandboxUploadSuccess()
    file_repo = _SandboxSyncFileRepo()
    session_repo = _SandboxSyncSessionRepo()
    projector = _build_user_input_attachment_projector(
        sandbox=sandbox,
        uow_factory=lambda: _SandboxSyncUoW(file_repo=file_repo, session_repo=session_repo),
    )
    event = MessageEvent(
        role="user",
        message="hello",
        attachments=[File(id="file-1", filename="a.txt")],
    )

    asyncio.run(projector.project(event))

    assert sandbox.calls == [
        {
            "file_path": "/home/ubuntu/upload/file-1/a.txt",
            "filename": "a.txt",
            "content": b"hello",
        }
    ]
    assert [file.filepath for file in file_repo.saved_files] == ["/home/ubuntu/upload/file-1/a.txt"]
    assert [(session_id, file.filepath) for session_id, file in session_repo.added_files] == [
        ("session-1", "/home/ubuntu/upload/file-1/a.txt")
    ]
    assert [attachment.filepath for attachment in event.attachments] == ["/home/ubuntu/upload/file-1/a.txt"]


class _SameNameFileStorage:
    def __init__(self) -> None:
        self._files = {
            "file-1": File(id="file-1", filename="same.txt"),
            "file-2": File(id="file-2", filename="same.txt"),
        }

    async def download_file(self, file_id: str):
        file = self._files[file_id].model_copy(deep=True)
        return io.BytesIO(file_id.encode("utf-8")), file


def test_user_input_attachment_projector_should_not_override_same_name_attachments() -> None:
    sandbox = _SandboxUploadSuccess()
    file_repo = _SandboxSyncFileRepo()
    session_repo = _SandboxSyncSessionRepo()
    projector = _build_user_input_attachment_projector(
        sandbox=sandbox,
        file_storage=_SameNameFileStorage(),
        uow_factory=lambda: _SandboxSyncUoW(file_repo=file_repo, session_repo=session_repo),
    )
    event = MessageEvent(
        role="user",
        message="hello",
        attachments=[
            File(id="file-1", filename="same.txt"),
            File(id="file-2", filename="same.txt"),
        ],
    )

    asyncio.run(projector.project(event))

    assert sandbox.calls == [
        {
            "file_path": "/home/ubuntu/upload/file-1/same.txt",
            "filename": "same.txt",
            "content": b"file-1",
        },
        {
            "file_path": "/home/ubuntu/upload/file-2/same.txt",
            "filename": "same.txt",
            "content": b"file-2",
        },
    ]
    assert [file.filepath for file in file_repo.saved_files] == [
        "/home/ubuntu/upload/file-1/same.txt",
        "/home/ubuntu/upload/file-2/same.txt",
    ]
    assert [attachment.filepath for attachment in event.attachments] == [
        "/home/ubuntu/upload/file-1/same.txt",
        "/home/ubuntu/upload/file-2/same.txt",
    ]


def test_sync_file_to_storage_should_skip_when_file_not_exists() -> None:
    projector = _build_attachment_projector(
        sandbox=_MissingFileSandbox(),
        file_storage=_MissingFileStorage(),
    )

    synced = asyncio.run(projector.sync_file_to_storage("/tmp/not_exists.txt"))

    assert synced is None


class _ExistingFileSessionRepo:
    def __init__(self, file: File) -> None:
        self._file = file
        self.removed_file_ids: list[str] = []
        self.added_files: list[File] = []
        self.added_final_files: list[File] = []

    async def get_file_by_path(self, session_id: str, filepath: str):
        return self._file if filepath == self._file.filepath else None

    async def remove_file(self, session_id: str, file_id: str) -> None:
        self.removed_file_ids.append(file_id)

    async def add_file(self, session_id: str, file: File) -> None:
        self.added_files.append(file)

    async def add_final_files(self, session_id: str, file: File) -> None:
        self.added_final_files.append(file)


class _ExistingFileUoW:
    def __init__(self, session_repo: _ExistingFileSessionRepo) -> None:
        self.session = session_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ForbiddenFileStorage:
    async def upload_file(self, upload_file, user_id=None):
        raise AssertionError("存在会话文件时不应重新上传文件")


class _CaptureWorkspaceRuntimeService:
    def __init__(self, *, authoritative_paths: list[str] | None = None) -> None:
        self.delivery_calls: list[dict] = []
        self.authoritative_paths = list(authoritative_paths or [])
        self.artifact_calls: list[dict] = []

    async def mark_artifacts_delivery_state(self, *, paths: list[str], delivery_state: str):
        self.delivery_calls.append(
            {
                "paths": list(paths),
                "delivery_state": delivery_state,
            }
        )
        return []

    async def resolve_authoritative_artifact_paths(self, *, paths: list[str]) -> list[str]:
        if len(self.authoritative_paths) == 0:
            return list(paths)
        allowed = set(self.authoritative_paths)
        return [path for path in paths if path in allowed]

    async def upsert_artifact(self, **kwargs):
        self.artifact_calls.append(dict(kwargs))
        return kwargs

    async def get_latest_shell_tool_result(self):
        return ToolResult(success=False, data={"console_records": []})


class _FreshFinalSandbox:
    async def check_file_exists(self, file_path: str):
        return ToolResult(success=True, data=True)

    async def download_file(self, file_path: str):
        return io.BytesIO(b"final-version")


class _FreshFinalFileStorage:
    def __init__(self) -> None:
        self.upload_payloads: list[object] = []

    async def upload_file(self, upload_file, user_id=None):
        self.upload_payloads.append(upload_file)
        return File(id="file-final-2", filename="final.md", extension="md", size=13)


def test_sync_file_to_storage_should_refresh_final_attachment_from_sandbox_when_same_path_exists() -> None:
    existing_file = File(
        id="file-final-1",
        filename="final.md",
        filepath="/tmp/final.md",
        extension="md",
        size=12,
    )
    session_repo = _ExistingFileSessionRepo(existing_file)
    file_storage = _FreshFinalFileStorage()
    projector = _build_attachment_projector(
        uow_factory=lambda: _ExistingFileUoW(session_repo),
        sandbox=_FreshFinalSandbox(),
        file_storage=file_storage,
    )

    synced = asyncio.run(projector.sync_file_to_storage("/tmp/final.md", stage="final"))

    assert synced is not None
    assert synced.id == "file-final-2"
    assert synced.filepath == "/tmp/final.md"
    assert session_repo.removed_file_ids == ["file-final-1"]
    assert session_repo.added_files == [synced]
    assert session_repo.added_final_files == [synced]
    payload = file_storage.upload_payloads[0]
    assert getattr(payload, "filename") == "final.md"
    assert getattr(payload, "file").read() == b"final-version"


def test_sync_file_to_storage_should_fallback_to_existing_final_attachment_when_sandbox_file_missing() -> None:
    existing_file = File(
        id="file-final-1",
        filename="final.md",
        filepath="/tmp/final.md",
        extension="md",
        size=12,
    )
    session_repo = _ExistingFileSessionRepo(existing_file)
    projector = _build_attachment_projector(
        uow_factory=lambda: _ExistingFileUoW(session_repo),
        sandbox=_MissingFileSandbox(),
        file_storage=_ForbiddenFileStorage(),
    )

    synced = asyncio.run(projector.sync_file_to_storage("/tmp/final.md", stage="final"))

    assert synced == existing_file
    assert session_repo.removed_file_ids == []
    assert session_repo.added_files == []
    assert session_repo.added_final_files == [existing_file]


def test_sync_message_attachments_to_storage_re_raises_exception() -> None:
    projector = _build_attachment_projector()

    async def _raise_sync_error(filepath: str, _stage: str = "intermediate"):
        raise RuntimeError(f"sync failed: {filepath}")

    projector.sync_file_to_storage = _raise_sync_error
    event = MessageEvent(
        role="assistant",
        message="hello",
        attachments=[File(filename="a.txt", filepath="/tmp/a.txt")],
    )

    with pytest.raises(RuntimeError, match="sync failed: /tmp/a.txt"):
        asyncio.run(projector.project(event))


def test_sync_message_attachments_to_storage_should_mark_final_artifacts_delivered() -> None:
    workspace_runtime_service = _CaptureWorkspaceRuntimeService()
    projector = _build_attachment_projector(
        workspace_runtime_service=workspace_runtime_service,
    )

    async def _sync_file(filepath: str, _stage: str = "intermediate"):
        return File(id=f"file-{filepath}", filename="final.md", filepath=filepath)

    projector.sync_file_to_storage = _sync_file
    event = MessageEvent(
        role="assistant",
        message="最终结果",
        stage="final",
        attachments=[
            File(filename="final.md", filepath="/tmp/final.md"),
            File(filename="report.md", filepath="/tmp/report.md"),
        ],
    )

    asyncio.run(projector.project(event))

    assert [attachment.filepath for attachment in event.attachments] == [
        "/tmp/final.md",
        "/tmp/report.md",
    ]
    assert workspace_runtime_service.delivery_calls == [
        {
            "paths": ["/tmp/final.md", "/tmp/report.md"],
            "delivery_state": "final_delivered",
        }
    ]


def test_sync_message_attachments_to_storage_should_filter_non_workspace_final_attachments() -> None:
    workspace_runtime_service = _CaptureWorkspaceRuntimeService(
        authoritative_paths=["/tmp/final.md"],
    )
    projector = _build_attachment_projector(
        workspace_runtime_service=workspace_runtime_service,
    )

    synced_paths: list[str] = []

    async def _sync_file(filepath: str, _stage: str = "intermediate"):
        synced_paths.append(filepath)
        return File(id=f"file-{filepath}", filename="final.md", filepath=filepath)

    projector.sync_file_to_storage = _sync_file
    event = MessageEvent(
        role="assistant",
        message="最终结果",
        stage="final",
        attachments=[
            File(filename="final.md", filepath="/tmp/final.md"),
            File(filename="rogue.md", filepath="/tmp/rogue.md"),
        ],
    )

    asyncio.run(projector.project(event))

    assert synced_paths == ["/tmp/final.md"]
    assert [attachment.filepath for attachment in event.attachments] == ["/tmp/final.md"]
    assert workspace_runtime_service.delivery_calls == [
        {
            "paths": ["/tmp/final.md"],
            "delivery_state": "final_delivered",
        }
    ]


def test_user_input_attachment_projector_should_re_raise_when_file_id_missing() -> None:
    projector = _build_user_input_attachment_projector(
        uow_factory=lambda: _SandboxSyncUoW(),
    )
    event = MessageEvent(
        role="user",
        message="hello",
        attachments=[File(id="", filename="a.txt", filepath="/tmp/a.txt")],
    )

    with pytest.raises(RuntimeError, match="缺少 file_id"):
        asyncio.run(projector.project(event))


def test_invoke_skips_empty_input_stream_event() -> None:
    runner = _new_runner()
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
    runner = _new_runner()
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
    assert session_repo.updated_statuses == [SessionStatus.COMPLETED]


def test_invoke_should_keep_failed_status_after_error_event() -> None:
    runner = _new_runner()
    session_repo = _InvokeSessionRepo()
    task = _SerialInvokeTask([MessageEvent(role="user", message="first")])

    runner._session_id = "session-1"
    runner._sandbox = _NoopSandbox()
    runner._mcp_tool = _NoopTool()
    runner._a2a_tool = _NoopA2ATool()
    runner._mcp_config = None
    runner._a2a_config = None
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _InvokeUoW(session_repo)

    emitted_pairs: list[tuple[str, Optional[SessionStatus], Optional[str]]] = []

    async def _pop_event(_task):
        message_event = await _task.input_stream.pop_next()
        return RuntimeInput(
            request_id=f"req-{message_event.message}",
            payload=message_event,
        )

    async def _run_flow(_message):
        yield ErrorEvent(error="boom")

    async def _put_and_add_event(*, task, event, title=None, latest_message=None, latest_message_at=None, increment_unread=False, status=None):
        emitted_pairs.append((type(event).__name__, status, latest_message))
        if status is not None:
            session_repo.updated_status = status
            session_repo.updated_statuses.append(status)

    async def _emit_request_started(*, task, request_id):
        return None

    async def _emit_request_finished(*, task, request_id, terminal_event_type):
        return None

    async def _reject_pending_requests(*, task, message, error_key=None):
        return None

    runner._pop_event = _pop_event
    runner._run_flow = _run_flow
    runner._put_and_add_event = _put_and_add_event
    runner._emit_request_started = _emit_request_started
    runner._emit_request_finished = _emit_request_finished
    runner._reject_pending_requests = _reject_pending_requests

    asyncio.run(runner.invoke(task))

    assert emitted_pairs == [("ErrorEvent", SessionStatus.FAILED, "boom")]
    assert session_repo.updated_status == SessionStatus.FAILED
    assert session_repo.updated_statuses == [SessionStatus.FAILED]


def test_invoke_should_continue_consuming_stream_after_done_event() -> None:
    runner = _new_runner()
    session_repo = _InvokeSessionRepo()
    task = _SerialInvokeTask([MessageEvent(role="user", message="first")])

    runner._session_id = "session-1"
    runner._sandbox = _NoopSandbox()
    runner._mcp_tool = _NoopTool()
    runner._a2a_tool = _NoopA2ATool()
    runner._mcp_config = None
    runner._a2a_config = None
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _InvokeUoW(session_repo)

    continuation_markers: list[str] = []

    async def _pop_event(_task):
        message_event = await _task.input_stream.pop_next()
        return RuntimeInput(
            request_id=f"req-{message_event.message}",
            payload=message_event,
        )

    async def _run_flow(_message):
        yield DoneEvent()
        continuation_markers.append("done-stream-drained")

    async def _put_and_add_event(*, task, event, title=None, latest_message=None, latest_message_at=None, increment_unread=False, status=None):
        if status is not None:
            session_repo.updated_status = status
            session_repo.updated_statuses.append(status)

    async def _emit_request_started(*, task, request_id):
        return None

    async def _emit_request_finished(*, task, request_id, terminal_event_type):
        return None

    runner._pop_event = _pop_event
    runner._run_flow = _run_flow
    runner._put_and_add_event = _put_and_add_event
    runner._emit_request_started = _emit_request_started
    runner._emit_request_finished = _emit_request_finished

    asyncio.run(runner.invoke(task))

    assert continuation_markers == ["done-stream-drained"]
    assert session_repo.updated_status == SessionStatus.COMPLETED


def test_invoke_should_continue_consuming_stream_after_wait_event() -> None:
    runner = _new_runner()
    session_repo = _InvokeSessionRepo()
    task = _SerialInvokeTask([MessageEvent(role="user", message="first")])

    runner._session_id = "session-1"
    runner._sandbox = _NoopSandbox()
    runner._mcp_tool = _NoopTool()
    runner._a2a_tool = _NoopA2ATool()
    runner._mcp_config = None
    runner._a2a_config = None
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _InvokeUoW(session_repo)

    continuation_markers: list[str] = []

    async def _pop_event(_task):
        message_event = await _task.input_stream.pop_next()
        return RuntimeInput(
            request_id=f"req-{message_event.message}",
            payload=message_event,
        )

    async def _run_flow(_message):
        yield WaitEvent()
        continuation_markers.append("wait-stream-drained")

    async def _put_and_add_event(*, task, event, title=None, latest_message=None, latest_message_at=None, increment_unread=False, status=None):
        if status is not None:
            session_repo.updated_status = status
            session_repo.updated_statuses.append(status)

    async def _emit_request_started(*, task, request_id):
        return None

    async def _emit_request_finished(*, task, request_id, terminal_event_type):
        return None

    async def _reject_pending_requests(*, task, message, error_key=None):
        return None

    runner._pop_event = _pop_event
    runner._run_flow = _run_flow
    runner._put_and_add_event = _put_and_add_event
    runner._emit_request_started = _emit_request_started
    runner._emit_request_finished = _emit_request_finished
    runner._reject_pending_requests = _reject_pending_requests

    asyncio.run(runner.invoke(task))

    assert continuation_markers == ["wait-stream-drained"]
    assert session_repo.updated_status == SessionStatus.WAITING
    assert session_repo.updated_statuses == [SessionStatus.WAITING]


def test_invoke_should_project_latest_message_when_exception_falls_back_to_error_event() -> None:
    runner = _new_runner()
    session_repo = _InvokeSessionRepo()
    task = _SerialInvokeTask([MessageEvent(role="user", message="first")])

    runner._session_id = "session-1"
    runner._sandbox = _NoopSandbox()
    runner._mcp_tool = _NoopTool()
    runner._a2a_tool = _NoopA2ATool()
    runner._mcp_config = None
    runner._a2a_config = None
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _InvokeUoW(session_repo)

    emitted_pairs: list[tuple[str, Optional[SessionStatus], Optional[str]]] = []

    async def _pop_event(_task):
        message_event = await _task.input_stream.pop_next()
        return RuntimeInput(
            request_id=f"req-{message_event.message}",
            payload=message_event,
        )

    async def _run_flow(_message):
        raise RuntimeError("fatal boom")
        yield  # pragma: no cover

    async def _put_and_add_event(*, task, event, title=None, latest_message=None, latest_message_at=None, increment_unread=False, status=None):
        emitted_pairs.append((type(event).__name__, status, latest_message))
        if status is not None:
            session_repo.updated_status = status
            session_repo.updated_statuses.append(status)

    async def _emit_request_started(*, task, request_id):
        return None

    async def _emit_request_finished(*, task, request_id, terminal_event_type):
        return None

    async def _reject_pending_requests(*, task, message, error_key=None):
        return None

    runner._pop_event = _pop_event
    runner._run_flow = _run_flow
    runner._put_and_add_event = _put_and_add_event
    runner._emit_request_started = _emit_request_started
    runner._emit_request_finished = _emit_request_finished
    runner._reject_pending_requests = _reject_pending_requests

    asyncio.run(runner.invoke(task))

    assert emitted_pairs == [("ErrorEvent", SessionStatus.FAILED, "fatal boom")]
    assert session_repo.updated_status == SessionStatus.FAILED
    assert session_repo.updated_statuses == [SessionStatus.FAILED]


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


def test_browser_screenshot_artifact_service_should_upload_and_index_workspace_artifact() -> None:
    file_storage = _ScreenshotFileStorage()
    workspace_runtime_service = _CaptureWorkspaceRuntimeService()
    service = BrowserScreenshotArtifactService(
        browser=_BrowserScreenshot(),
        file_storage=file_storage,
        workspace_runtime_service=workspace_runtime_service,
        user_id="user-1",
    )

    screenshot_url = asyncio.run(service.capture(source_capability="browser_view"))

    assert screenshot_url == "https://cdn.example.com/2026/03/19/s.png"
    assert file_storage.upload_user_ids == ["user-1"]
    payload = file_storage.upload_payloads[0]
    assert getattr(payload, "filename").endswith(".png")
    assert getattr(payload, "content_type") == "image/png"
    assert getattr(payload, "size") == len(b"fake-image-bytes")
    assert getattr(payload, "file").read() == b"fake-image-bytes"
    assert len(workspace_runtime_service.artifact_calls) == 1
    artifact_call = workspace_runtime_service.artifact_calls[0]
    assert artifact_call["path"] == "/.workspace/browser-screenshots/2026/03/19/s.png"
    assert artifact_call["artifact_type"] == "browser_screenshot"
    assert artifact_call["summary"] == "浏览器截图: /.workspace/browser-screenshots/2026/03/19/s.png"
    assert artifact_call["source_capability"] == "browser_view"
    assert artifact_call["record_as_changed_file"] is False
    assert artifact_call["metadata"] == {
        "file_id": "file-screenshot",
        "filename": getattr(payload, "filename"),
        "filepath": "",
        "key": "2026/03/19/s.png",
        "mime_type": "image/png",
        "size": len(b"fake-image-bytes"),
        "url": "https://cdn.example.com/2026/03/19/s.png",
    }


class _ShellObservationWorkspaceRuntimeService:
    async def get_latest_shell_tool_result(self):
        return ToolResult(
            success=True,
            data={
                "output": "pytest -q\n24 passed",
                "console_records": [
                    {
                        "command": "pytest -q",
                        "output": "24 passed",
                    }
                ],
            },
        )


def test_tool_event_projector_should_use_workspace_shell_observation_without_sandbox() -> None:
    projector = _build_tool_event_projector(
        browser=_BrowserScreenshot(),
        file_storage=_ScreenshotFileStorage(),
        workspace_runtime_service=_ShellObservationWorkspaceRuntimeService(),
    )
    event = ToolEvent(
        tool_name="shell",
        function_name="read_shell_output",
        function_args={},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
    )

    asyncio.run(projector.project(event))

    assert event.tool_content is not None
    assert event.tool_content.console == [
        {
            "command": "pytest -q",
            "output": "24 passed",
        }
    ]


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


class _CancellationWorkspaceRepo:
    def __init__(self, workspace: Workspace | None) -> None:
        self._workspace = workspace

    async def get_by_id(self, workspace_id: str):
        if self._workspace is None or workspace_id != self._workspace.id:
            return None
        return self._workspace

    async def get_by_session_id(self, session_id: str):
        if self._workspace is None or session_id != self._workspace.session_id:
            return None
        return self._workspace


class _CancellationUoW:
    def __init__(
            self,
            session_repo: _CancellationSessionRepo,
            workflow_run_repo: _CancellationWorkflowRunRepo,
            workspace_repo: _CancellationWorkspaceRepo | None = None,
    ) -> None:
        self.session = session_repo
        self.workflow_run = workflow_run_repo
        self.workspace = workspace_repo or _CancellationWorkspaceRepo(None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_persist_cancellation_state_should_build_cancelled_events_from_run_history() -> None:
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
    workspace_repo = _CancellationWorkspaceRepo(workspace)
    runner = _new_runner()
    runner._session_id = "session-1"
    runner._uow_factory = lambda: _CancellationUoW(session_repo, workflow_run_repo, workspace_repo)

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
    runner = _new_runner()

    class _RunEngine:
        async def invoke(self, message):
            assert isinstance(message, Message)
            assert message.message == ""
            assert message.command == MessageCommand(type="continue_cancelled_task")
            yield DoneEvent()

    runner._run_engine = _RunEngine()

    async def _collect():
        return [event async for event in runner._run_flow(
            Message(message="", command=MessageCommand(type="continue_cancelled_task"))
        )]

    events = asyncio.run(_collect())

    assert len(events) == 1
    assert isinstance(events[0], DoneEvent)


def test_pop_event_should_parse_continue_cancelled_task_input_without_touching_event_history() -> None:
    runner = _new_runner()
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
