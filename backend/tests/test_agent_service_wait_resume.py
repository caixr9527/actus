import asyncio
from datetime import datetime

from app.application.errors import error_keys
from app.application.service.agent_service import AgentService
from app.domain.models import ErrorEvent, Session, SessionStatus, WaitEvent


class _TaskFactory:
    @classmethod
    def get(cls, task_id: str):
        return None


class _SessionRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        return self._session

    async def add_event(self, session_id: str, event) -> None:
        return None

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None


class _UoW:
    def __init__(self, session_repo: _SessionRepo) -> None:
        self.session = session_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _build_waiting_session(wait_event: WaitEvent) -> Session:
    return Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.WAITING,
        events=[wait_event],
    )


def test_agent_service_chat_should_reject_invalid_resume_token() -> None:
    wait_event = WaitEvent.build_for_user_input(
        session_id="session-1",
        question="请确认是否继续",
        resume_token="resume-token-1",
    )
    session = _build_waiting_session(wait_event=wait_event)

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
    service._task_cls = _TaskFactory

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                message="继续执行",
                resume_token="resume-token-invalid",
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_RESUME_TOKEN_INVALID


def test_agent_service_chat_should_reject_timeout_waiting_task() -> None:
    wait_event = WaitEvent.build_for_user_input(
        session_id="session-1",
        question="请确认是否继续",
        resume_token="resume-token-timeout",
    )
    assert wait_event.human_task is not None
    wait_event.human_task.timeout.timeout_at = datetime(2000, 1, 1, 0, 0, 0)
    wait_event.human_task.timeout.timeout_seconds = 60
    session = _build_waiting_session(wait_event=wait_event)

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(_SessionRepo(session))
    service._task_cls = _TaskFactory

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                message="继续执行",
                resume_token="resume-token-timeout",
        ):
            return event
        return None

    first_event = asyncio.run(_collect_first_event())

    assert isinstance(first_event, ErrorEvent)
    assert first_event.error_key == error_keys.SESSION_WAIT_TASK_TIMEOUT
