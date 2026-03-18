import asyncio
from typing import Optional

import pytest

from app.domain.models import File, MessageEvent, SessionStatus
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
    async def download_file(self, file_path: str):
        raise RuntimeError("download failed")


class _DummyFileStorage:
    async def upload_file(self, upload_file, user_id=None):
        raise AssertionError("upload_file should not be called when sandbox download fails")


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


def test_sync_file_to_storage_re_raises_exception() -> None:
    runner = _build_runner_for_storage_sync()

    with pytest.raises(RuntimeError, match="download failed"):
        asyncio.run(runner._sync_file_to_storage("/tmp/a.txt"))


def test_sync_message_attachments_to_storage_re_raises_exception() -> None:
    runner = object.__new__(AgentTaskRunner)

    async def _raise_sync_error(filepath: str):
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
