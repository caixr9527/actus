import asyncio

import pytest

from app.domain.models import File, MessageEvent
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
    async def upload_file(self, upload_file):
        raise AssertionError("upload_file should not be called when sandbox download fails")


def _build_runner_for_storage_sync() -> AgentTaskRunner:
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
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

    runner._sync_file_to_storage = _raise_sync_error
    event = MessageEvent(
        role="assistant",
        message="hello",
        attachments=[File(filename="a.txt", filepath="/tmp/a.txt")],
    )

    with pytest.raises(RuntimeError, match="sync failed: /tmp/a.txt"):
        asyncio.run(runner._sync_message_attachments_to_storage(event))
