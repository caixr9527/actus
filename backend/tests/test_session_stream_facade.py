import asyncio
import json

import pytest

from app.application.errors import NotFoundError
from app.domain.models import DoneEvent, MessageEvent, Session
from app.interfaces.facades.session_stream_facade import SessionStreamFacade
from app.interfaces.schemas import ChatRequest


class _SessionServiceForNotFound:
    async def get_session(self, user_id: str, session_id: str):
        return None


class _SessionServiceForChat:
    async def get_session(self, user_id: str, session_id: str):
        return Session(id=session_id, user_id=user_id, current_run_id="run-legacy")


class _AgentServiceForChat:
    def __init__(self) -> None:
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        yield MessageEvent(id="evt-1", role="assistant", message="hello")
        yield DoneEvent(id="evt-2")


def test_stream_chat_should_raise_not_found_when_session_missing() -> None:
    facade = SessionStreamFacade()
    session_service = _SessionServiceForNotFound()
    request = ChatRequest(message="hello")

    async def _consume():
        await facade.stream_chat(
            user_id="user-1",
            session_id="session-1",
            request=request,
            session_service=session_service,
            agent_service=_AgentServiceForChat(),
        )

    with pytest.raises(NotFoundError):
        asyncio.run(_consume())


def test_stream_chat_should_map_events() -> None:
    facade = SessionStreamFacade()
    session_service = _SessionServiceForChat()
    agent_service = _AgentServiceForChat()
    request = ChatRequest(
        message="hello",
        attachments=["file-1"],
        event_id="evt-cursor-1",
        timestamp=1711234567,
    )

    async def _collect():
        iterator = await facade.stream_chat(
            user_id="user-1",
            session_id="session-1",
            request=request,
            session_service=session_service,
            agent_service=agent_service,
        )
        events = []
        async for event in iterator:
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert [event.event for event in events] == ["message", "done"]
    first_payload = json.loads(events[0].data)
    assert first_payload["message"] == "hello"

    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["session_id"] == "session-1"
    assert agent_service.calls[0]["user_id"] == "user-1"
    assert agent_service.calls[0]["message"] == "hello"
    assert agent_service.calls[0]["attachments"] == ["file-1"]
    assert agent_service.calls[0]["resume"] is None
    assert agent_service.calls[0]["latest_event_id"] == "evt-cursor-1"


def test_stream_chat_should_forward_resume_payload() -> None:
    facade = SessionStreamFacade()
    session_service = _SessionServiceForChat()
    agent_service = _AgentServiceForChat()
    request = ChatRequest(
        resume=ChatRequest.ResumePayload(value={"approved": True}),
        event_id="evt-cursor-2",
    )

    async def _collect():
        iterator = await facade.stream_chat(
            user_id="user-1",
            session_id="session-1",
            request=request,
            session_service=session_service,
            agent_service=agent_service,
        )
        events = []
        async for event in iterator:
            events.append(event)
        return events

    asyncio.run(_collect())

    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["message"] is None
    assert agent_service.calls[0]["attachments"] == []
    assert agent_service.calls[0]["resume"] == {"approved": True}


def test_chat_request_should_allow_empty_stream_request() -> None:
    request = ChatRequest(event_id="evt-cursor-3")

    assert request.message is None
    assert request.resume is None
    assert request.event_id == "evt-cursor-3"
