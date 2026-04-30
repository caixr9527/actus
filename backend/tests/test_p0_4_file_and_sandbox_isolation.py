import asyncio
from types import SimpleNamespace

import pytest

from app.application.errors import NotFoundError, error_keys
from app.application.service.file_service import FileService
from app.application.service.session_service import SessionService
from app.domain.models import File, Session, Workspace
from app.domain.models.tool_result import ToolResult


class _SessionRepo:
    def __init__(self, session: Session | None) -> None:
        self.session = session

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        if self.session is None or self.session.id != session_id:
            return None
        if user_id is not None and self.session.user_id != user_id:
            return None
        return self.session


class _WorkspaceRepo:
    def __init__(self, workspace: Workspace | None) -> None:
        self.workspace = workspace

    async def get_by_id_for_user(self, workspace_id: str, user_id: str):
        if self.workspace is None:
            return None
        if self.workspace.id != workspace_id or self.workspace.user_id != user_id:
            return None
        return self.workspace

    async def list_by_session_id(self, session_id: str):
        if self.workspace is None or self.workspace.session_id != session_id:
            return []
        return [self.workspace]


class _WorkflowRunRepo:
    async def get_by_id_for_user(self, run_id: str, user_id: str):
        return None


class _FileRepo:
    def __init__(self, file: File | None) -> None:
        self.file = file

    async def get_by_id_and_user_id(self, file_id: str, user_id: str):
        if self.file is None or self.file.id != file_id or self.file.user_id != user_id:
            return None
        return self.file


class _WorkspaceArtifactRepo:
    async def list_by_user_workspace_id_and_paths(self, user_id: str, workspace_id: str, paths: list[str]):
        return []


class _UoW:
    def __init__(
            self,
            *,
            session: Session | None = None,
            workspace: Workspace | None = None,
            file: File | None = None,
    ) -> None:
        self.session = _SessionRepo(session)
        self.workspace = _WorkspaceRepo(workspace)
        self.workflow_run = _WorkflowRunRepo()
        self.file = _FileRepo(file)
        self.workspace_artifact = _WorkspaceArtifactRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SpySandbox:
    get_calls: list[str] = []
    read_file_calls: list[str] = []
    read_shell_calls: list[str] = []

    @classmethod
    async def get(cls, id: str):
        cls.get_calls.append(id)
        return cls()

    async def read_file(self, file_path: str):
        self.__class__.read_file_calls.append(file_path)
        return ToolResult(success=True, data={"filepath": file_path, "content": "ok"})

    async def read_shell_output(self, session_id: str, console: bool = False):
        self.__class__.read_shell_calls.append(session_id)
        return ToolResult(success=True, data={"output": "ok", "console_records": []})

    @property
    def vnc_url(self) -> str:
        return "ws://sandbox/vnc"


class _SpyStorage:
    def __init__(self) -> None:
        self.download_calls: list[tuple[str, str | None]] = []

    async def upload_file(self, upload_file, user_id=None):
        return None

    async def download_file(self, file_id: str, user_id=None):
        self.download_calls.append((file_id, user_id))
        return SimpleNamespace(), File(id=file_id, user_id=user_id, filename="a.txt")


def _reset_sandbox() -> None:
    _SpySandbox.get_calls = []
    _SpySandbox.read_file_calls = []
    _SpySandbox.read_shell_calls = []


def _session(*, user_id: str = "user-2") -> Session:
    return Session(id="session-1", user_id=user_id, workspace_id="workspace-1")


def _workspace(*, user_id: str = "user-2") -> Workspace:
    return Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id=user_id,
        sandbox_id="sandbox-1",
        shell_session_id="shell-1",
    )


def test_read_sandbox_file_should_reject_cross_user_before_sandbox_lookup() -> None:
    _reset_sandbox()
    service = SessionService(
        uow_factory=lambda: _UoW(session=_session(), workspace=_workspace()),
        sandbox_cls=_SpySandbox,
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.read_file("user-1", "session-1", "reports/a.md"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
    assert _SpySandbox.get_calls == []
    assert _SpySandbox.read_file_calls == []


def test_read_shell_output_should_reject_cross_workspace_before_sandbox_lookup() -> None:
    _reset_sandbox()
    service = SessionService(
        uow_factory=lambda: _UoW(session=_session(user_id="user-1"), workspace=_workspace(user_id="user-2")),
        sandbox_cls=_SpySandbox,
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.read_shell_output("user-1", "session-1"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
    assert _SpySandbox.get_calls == []
    assert _SpySandbox.read_shell_calls == []


def test_get_vnc_url_should_reject_cross_workspace_before_sandbox_lookup() -> None:
    _reset_sandbox()
    service = SessionService(
        uow_factory=lambda: _UoW(session=_session(user_id="user-1"), workspace=_workspace(user_id="user-2")),
        sandbox_cls=_SpySandbox,
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.get_vnc_url("user-1", "session-1"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
    assert _SpySandbox.get_calls == []


def test_get_file_info_should_hide_cross_user_file() -> None:
    storage = _SpyStorage()
    service = FileService(
        uow_factory=lambda: _UoW(file=File(id="file-1", user_id="user-2", filename="foreign.txt")),
        file_storage=storage,
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.get_file_info("user-1", "file-1"))

    assert exc.value.error_key == error_keys.FILE_NOT_FOUND
    assert storage.download_calls == []


def test_download_file_should_reject_cross_user_before_storage_download() -> None:
    storage = _SpyStorage()
    service = FileService(
        uow_factory=lambda: _UoW(file=File(id="file-1", user_id="user-2", filename="foreign.txt")),
        file_storage=storage,
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.download_file("user-1", "file-1"))

    assert exc.value.error_key == error_keys.FILE_NOT_FOUND
    assert storage.download_calls == []
