import asyncio
import io

from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from app.application.errors import NotFoundError
from app.application.errors import error_keys
from app.application.service.agent_service import AgentService
from app.application.service.file_service import FileService
from app.application.service.session_service import SessionService
from app.domain.models import ErrorEvent, Session, User
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.services import get_file_service, get_session_service
from app.interfaces.endpoints.file_routes import router as file_router
from app.interfaces.endpoints.session_routes import router as session_router
from app.interfaces.errors.exception_handlers import register_exception_handlers


def _build_current_user() -> User:
    return User(
        id="user-a",
        email="user-a@example.com",
        password="hashed-password",
    )


class _CreateSessionRepo:
    def __init__(self) -> None:
        self.saved_sessions: list[Session] = []

    async def save(self, session: Session) -> None:
        self.saved_sessions.append(session)


class _CreateSessionUoW:
    def __init__(self, session_repo: _CreateSessionRepo) -> None:
        self.session = session_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_create_session_should_persist_current_user_id() -> None:
    session_repo = _CreateSessionRepo()
    service = SessionService(
        uow_factory=lambda: _CreateSessionUoW(session_repo),
        sandbox_cls=type("_DummySandbox", (), {}),
    )

    session = asyncio.run(service.create_session(user_id="user-a"))

    assert session.user_id == "user-a"
    assert session_repo.saved_sessions[0].user_id == "user-a"


class _UploadCaptureStorage:
    def __init__(self) -> None:
        self.user_ids: list[str | None] = []

    async def upload_file(self, upload_file: UploadFile, user_id: str | None = None):
        self.user_ids.append(user_id)
        return None

    async def download_file(self, file_id: str, user_id: str | None = None):
        raise AssertionError("download_file should not be called in this test")


def test_upload_file_should_forward_current_user_id_to_storage() -> None:
    storage = _UploadCaptureStorage()
    service = FileService(
        uow_factory=lambda: None,
        file_storage=storage,
    )
    upload = UploadFile(filename="a.txt", file=io.BytesIO(b"hello"))

    asyncio.run(service.upload_file(user_id="user-a", upload_file=upload))

    assert storage.user_ids == ["user-a"]


class _ForeignAttachmentFileRepo:
    async def get_by_id_and_user_id(self, file_id: str, user_id: str):
        return None


class _OwnedSessionRepo:
    def __init__(self) -> None:
        self.session = Session(
            id="session-a",
            user_id="user-a",
            task_id="task-a",
            status="running",
        )

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        if session_id == self.session.id and user_id == self.session.user_id:
            return self.session
        return None

    async def add_event(self, session_id: str, event) -> None:
        return None

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None


class _AgentUoW:
    def __init__(self) -> None:
        self.session = _OwnedSessionRepo()
        self.file = _ForeignAttachmentFileRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ExistingTask:
    done = True


class _TaskFactory:
    @classmethod
    def get(cls, task_id: str):
        return _ExistingTask()


def test_agent_service_chat_should_reject_foreign_attachment() -> None:
    service = object.__new__(AgentService)
    service._uow_factory = _AgentUoW
    service._task_cls = _TaskFactory

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-a",
                user_id="user-a",
                message="hello",
                attachments=["file-b"],
        ):
            return event
        return None

    event = asyncio.run(_collect_first_event())

    assert isinstance(event, ErrorEvent)
    assert event.error_key == error_keys.FILE_NOT_FOUND
    assert event.error_params == {"file_id": "file-b"}


class _SessionOwnershipService:
    async def get_session(self, user_id: str, session_id: str):
        if user_id == "user-a" and session_id == "session-a":
            return Session(id="session-a", user_id="user-a", title="我的会话")
        return None


class _FileOwnershipService:
    async def get_file_info(self, user_id: str, file_id: str):
        if user_id == "user-a" and file_id == "file-a":
            raise AssertionError("foreign access test should not hit owned branch")
        raise NotFoundError(
            msg=f"该文件[{file_id}]不存在",
            error_key=error_keys.FILE_NOT_FOUND,
            error_params={"file_id": file_id},
        )


def test_session_route_should_return_404_for_foreign_session() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(session_router, prefix="/api")
    app.dependency_overrides[get_current_user] = _build_current_user
    app.dependency_overrides[get_session_service] = lambda: _SessionOwnershipService()

    with TestClient(app) as client:
        response = client.get("/api/sessions/session-b")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_key"] == error_keys.SESSION_NOT_FOUND


def test_file_route_should_return_404_for_foreign_file() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(file_router, prefix="/api")
    app.dependency_overrides[get_current_user] = _build_current_user
    app.dependency_overrides[get_file_service] = lambda: _FileOwnershipService()

    with TestClient(app) as client:
        response = client.get("/api/files/file-b")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_key"] == error_keys.FILE_NOT_FOUND
