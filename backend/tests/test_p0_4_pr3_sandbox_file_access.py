import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.application.errors import NotFoundError, ValidationError, error_keys
from app.application.service.file_service import FileService
from app.application.service.session_service import SessionService
from app.domain.models import File, Session, User, Workspace, WorkspaceArtifact
from app.domain.models.tool_result import ToolResult
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.services import get_session_service
from app.interfaces.endpoints.session_routes import router as session_router
from app.interfaces.errors.exception_handlers import register_exception_handlers


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
        if self.file is None:
            return None
        if self.file.id != file_id or self.file.user_id != user_id:
            return None
        return self.file


class _WorkspaceArtifactRepo:
    def __init__(self, artifacts: list[WorkspaceArtifact] | None = None) -> None:
        self.artifacts = list(artifacts or [])

    async def list_by_user_workspace_id_and_paths(self, user_id: str, workspace_id: str, paths: list[str]):
        normalized_paths = {str(path or "").strip() for path in list(paths or [])}
        return [
            artifact
            for artifact in self.artifacts
            if artifact.user_id == user_id
            and artifact.workspace_id == workspace_id
            and artifact.path in normalized_paths
        ]


class _UoW:
    def __init__(
            self,
            *,
            session: Session | None = None,
            workspace: Workspace | None = None,
            file: File | None = None,
            artifacts: list[WorkspaceArtifact] | None = None,
    ) -> None:
        self.session = _SessionRepo(session)
        self.workspace = _WorkspaceRepo(workspace)
        self.workflow_run = _WorkflowRunRepo()
        self.file = _FileRepo(file)
        self.workspace_artifact = _WorkspaceArtifactRepo(artifacts)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Sandbox:
    read_paths: list[str] = []

    @classmethod
    async def get(cls, id: str):
        if id != "sandbox-1":
            return None
        return cls()

    async def read_file(self, file_path: str):
        self.__class__.read_paths.append(file_path)
        return ToolResult(success=True, data={"filepath": file_path, "content": "hello"})

    async def read_shell_output(self, session_id: str, console: bool = False):
        return ToolResult(success=True, data={"output": "ok", "console_records": []})

    @property
    def vnc_url(self) -> str:
        return "ws://sandbox/vnc"


class _FileStorage:
    def __init__(self) -> None:
        self.download_calls: list[tuple[str, str | None]] = []

    async def upload_file(self, upload_file, user_id=None):
        return None

    async def download_file(self, file_id: str, user_id=None):
        self.download_calls.append((file_id, user_id))
        return SimpleNamespace(), File(id=file_id, user_id=user_id, filename="a.txt")


def _build_session() -> Session:
    return Session(
        id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
    )


def _build_workspace(*, user_id: str = "user-1") -> Workspace:
    return Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id=user_id,
        sandbox_id="sandbox-1",
        shell_session_id="shell-1",
    )


def test_session_service_read_file_should_reject_invalid_sandbox_path(caplog) -> None:
    service = SessionService(
        uow_factory=lambda: _UoW(session=_build_session(), workspace=_build_workspace()),
        sandbox_cls=_Sandbox,
    )

    caplog.set_level(logging.WARNING)
    with pytest.raises(ValidationError) as exc:
        asyncio.run(service.read_file("user-1", "session-1", "../secret.txt"))

    assert exc.value.error_key == "error.session.sandbox_path_invalid"
    record = next(item for item in caplog.records if getattr(item, "reason_code", "") == "sandbox_path_invalid")
    assert record.event == "sandbox_path_access_denied"
    assert record.user_id == "user-1"
    assert record.session_id == "session-1"
    assert record.resource_kind == "sandbox_file"
    assert record.action == "read"
    assert not hasattr(record, "content")


def test_session_service_read_file_should_use_owned_workspace_and_normalized_path() -> None:
    _Sandbox.read_paths = []
    artifact = WorkspaceArtifact(
        workspace_id="workspace-1",
        user_id="user-1",
        session_id="session-1",
        path="/workspace/report.md",
        artifact_type="file",
    )
    service = SessionService(
        uow_factory=lambda: _UoW(
            session=_build_session(),
            workspace=_build_workspace(),
            artifacts=[artifact],
        ),
        sandbox_cls=_Sandbox,
    )

    result = asyncio.run(service.read_file("user-1", "session-1", "/workspace/./report.md"))

    assert result.filepath == "/workspace/report.md"
    assert _Sandbox.read_paths == ["/workspace/report.md"]


def test_session_service_read_file_should_resolve_relative_path_to_workspace(caplog) -> None:
    _Sandbox.read_paths = []
    service = SessionService(
        uow_factory=lambda: _UoW(
            session=_build_session(),
            workspace=_build_workspace(),
            artifacts=[],
        ),
        sandbox_cls=_Sandbox,
    )

    caplog.set_level(logging.INFO)
    result = asyncio.run(service.read_file("user-1", "session-1", "reports/a.md"))

    assert result.filepath == "/workspace/reports/a.md"
    assert _Sandbox.read_paths == ["/workspace/reports/a.md"]
    record = next(
        item for item in caplog.records
        if getattr(item, "reason_code", "") == "sandbox_unindexed_path_read"
    )
    assert record.event == "sandbox_path_access"
    assert record.user_id == "user-1"
    assert record.session_id == "session-1"
    assert record.workspace_id == "workspace-1"
    assert record.filepath == "/workspace/reports/a.md"
    assert not hasattr(record, "content")


def test_session_service_read_shell_should_reject_foreign_workspace() -> None:
    service = SessionService(
        uow_factory=lambda: _UoW(session=_build_session(), workspace=_build_workspace(user_id="user-2")),
        sandbox_cls=_Sandbox,
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.read_shell_output("user-1", "session-1"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_file_service_download_should_check_access_before_storage_download() -> None:
    storage = _FileStorage()
    service = FileService(
        uow_factory=lambda: _UoW(file=None),
        file_storage=storage,
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.download_file("user-1", "file-1"))

    assert exc.value.error_key == error_keys.FILE_NOT_FOUND
    assert storage.download_calls == []


def test_file_service_download_should_call_storage_after_owned_file_check() -> None:
    storage = _FileStorage()
    service = FileService(
        uow_factory=lambda: _UoW(file=File(id="file-1", user_id="user-1", filename="a.txt")),
        file_storage=storage,
    )

    asyncio.run(service.download_file("user-1", "file-1"))

    assert storage.download_calls == [("file-1", "user-1")]


def _build_current_user() -> User:
    return User(id="user-1", email="user@example.com", password="hashed")


class _RejectVncService:
    def __init__(self) -> None:
        self.accepted = False

    async def get_vnc_url(self, user_id: str, session_id: str):
        raise NotFoundError(
            msg="该会话不存在，请核实后重试",
            error_key=error_keys.SESSION_NOT_FOUND,
            error_params={"session_id": session_id},
        )


class _AllowVncService:
    def __init__(self, order: list[str]) -> None:
        self._order = order

    async def get_vnc_url(self, user_id: str, session_id: str):
        self._order.append("get_vnc_url")
        return "ws://sandbox/vnc"


class _FailingConnectContext:
    def __init__(self, order: list[str]) -> None:
        self._order = order

    async def __aenter__(self):
        self._order.append("connect")
        raise ConnectionError("sandbox unavailable")

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_vnc_websocket_should_reject_before_accept_when_scope_invalid(monkeypatch) -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(session_router, prefix="/api")
    app.dependency_overrides[get_current_user] = _build_current_user
    app.dependency_overrides[get_session_service] = lambda: _RejectVncService()

    connect_mock = AsyncMock()
    monkeypatch.setattr("app.interfaces.endpoints.session_routes.websockets.connect", connect_mock)

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/api/sessions/session-1/vnc?access_token=test"):
                pass

    assert exc.value.code == 1008
    connect_mock.assert_not_called()


def test_vnc_websocket_should_resolve_url_before_accept(monkeypatch) -> None:
    order: list[str] = []
    service = _AllowVncService(order)
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(session_router, prefix="/api")
    app.dependency_overrides[get_current_user] = _build_current_user
    app.dependency_overrides[get_session_service] = lambda: service

    original_accept = WebSocket.accept

    async def _accept_spy(self, *args, **kwargs):
        order.append("accept")
        return await original_accept(self, *args, **kwargs)

    def _connect_spy(*args, **kwargs):
        return _FailingConnectContext(order)

    monkeypatch.setattr(WebSocket, "accept", _accept_spy)
    monkeypatch.setattr("app.interfaces.endpoints.session_routes.websockets.connect", _connect_spy)

    with TestClient(app) as client:
        with client.websocket_connect("/api/sessions/session-1/vnc?access_token=test") as websocket:
            with pytest.raises(WebSocketDisconnect) as exc:
                websocket.receive_text()

    assert exc.value.code == 1011
    assert order == ["get_vnc_url", "accept", "connect"]
