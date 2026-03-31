import asyncio
from types import SimpleNamespace

import pytest

from app.application.errors import error_keys
from app.application.service.agent_service import AgentService
from app.domain.models import ErrorEvent, Session, SessionStatus


class _TaskFactory:
    @classmethod
    def get(cls, task_id: str):
        return None


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


class _SessionRepo:
    def __init__(self, session: Session, *, fail_on_update_status: bool = False) -> None:
        self._session = session
        self._fail_on_update_status = fail_on_update_status

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        return self._session

    async def add_event(self, session_id: str, event) -> None:
        return None

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        if self._fail_on_update_status:
            raise RuntimeError("update status failed")
        self._session.status = status


class _UoW:
    def __init__(self, session_repo: _SessionRepo) -> None:
        self.session = session_repo

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


def test_agent_service_chat_should_reject_resume_when_session_not_waiting() -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
    )

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
    service._task_cls = _TaskFactory

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
        current_run_id="run-1",
    )
    task = _Task()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session, fail_on_update_status=True))
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
