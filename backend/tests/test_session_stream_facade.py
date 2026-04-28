import asyncio
import json

import pytest

from app.application.service.runtime_observation_service import (
    RuntimeCapabilityResult,
    RuntimeCursorResult,
    RuntimeEventMetaResult,
    RuntimeInteractionResult,
    RuntimeObservableEventResult,
    RuntimeObservationResult,
)
from app.application.errors import NotFoundError
from app.domain.models import DoneEvent, MessageEvent, Session, SessionStatus, TextStreamChannel, TextStreamDeltaEvent
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


class _AgentServiceForTextStream:
    async def chat(self, **kwargs):
        yield TextStreamDeltaEvent(
            id="evt-ts-1",
            stream_id="stream-1",
            channel=TextStreamChannel.FINAL_MESSAGE,
            text="draft",
            sequence=1,
        )


class _RuntimeObservationServiceForChat:
    async def build_session_observation(self, user_id: str, session_id: str):
        return RuntimeObservationResult(
            session_id=session_id,
            run_id="run-1",
            status=SessionStatus.RUNNING,
            cursor=RuntimeCursorResult(),
            capabilities=RuntimeCapabilityResult(can_send_message=False, can_cancel=True),
            interaction=RuntimeInteractionResult(),
        )

    async def build_observable_event(
            self,
            *,
            session_id: str,
            event,
            run_id: str | None,
            source_event_id: str | None,
            cursor_event_id: str | None,
            source: str,
    ):
        is_text_stream = str(event.type).startswith("text_stream_")
        return RuntimeObservableEventResult(
            event=event,
            runtime=RuntimeEventMetaResult(
                session_id=session_id,
                run_id=run_id,
                source_event_id=None if is_text_stream else source_event_id,
                cursor_event_id=None if is_text_stream else cursor_event_id,
                durability="live_only" if is_text_stream else "persistent",
                visibility="draft" if is_text_stream else "timeline",
            ),
        )


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
            runtime_observation_service=_RuntimeObservationServiceForChat(),
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
            runtime_observation_service=_RuntimeObservationServiceForChat(),
        )
        events = []
        async for event in iterator:
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert [event.event for event in events] == ["message", "done"]
    first_payload = json.loads(events[0].data)
    assert first_payload["message"] == "hello"
    assert first_payload["runtime"]["session_id"] == "session-1"
    assert first_payload["runtime"]["run_id"] == "run-1"
    assert first_payload["runtime"]["source_event_id"] == "evt-1"
    assert first_payload["runtime"]["cursor_event_id"] == "evt-1"

    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["session_id"] == "session-1"
    assert agent_service.calls[0]["user_id"] == "user-1"
    assert agent_service.calls[0]["message"] == "hello"
    assert agent_service.calls[0]["attachments"] == ["file-1"]
    assert agent_service.calls[0]["resume"] is None
    assert agent_service.calls[0]["command"] is None
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
            runtime_observation_service=_RuntimeObservationServiceForChat(),
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
    assert agent_service.calls[0]["command"] is None


def test_stream_chat_should_forward_command_payload() -> None:
    facade = SessionStreamFacade()
    session_service = _SessionServiceForChat()
    agent_service = _AgentServiceForChat()
    request = ChatRequest(
        command=ChatRequest.CommandPayload(type="continue_cancelled_task"),
        event_id="evt-cursor-4",
    )

    async def _collect():
        iterator = await facade.stream_chat(
            user_id="user-1",
            session_id="session-1",
            request=request,
            session_service=session_service,
            agent_service=agent_service,
            runtime_observation_service=_RuntimeObservationServiceForChat(),
        )
        events = []
        async for event in iterator:
            events.append(event)
        return events

    asyncio.run(_collect())

    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["message"] is None
    assert agent_service.calls[0]["resume"] is None
    assert agent_service.calls[0]["command"] == {"type": "continue_cancelled_task"}


def test_stream_chat_should_emit_text_stream_as_live_only_runtime_event() -> None:
    facade = SessionStreamFacade()
    session_service = _SessionServiceForChat()
    request = ChatRequest(message="hello")

    async def _collect():
        iterator = await facade.stream_chat(
            user_id="user-1",
            session_id="session-1",
            request=request,
            session_service=session_service,
            agent_service=_AgentServiceForTextStream(),
            runtime_observation_service=_RuntimeObservationServiceForChat(),
        )
        events = []
        async for event in iterator:
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert [event.event for event in events] == ["text_stream_delta"]
    payload = json.loads(events[0].data)
    assert payload["event_id"] is None
    assert payload["runtime"]["session_id"] == "session-1"
    assert payload["runtime"]["run_id"] == "run-1"
    assert payload["runtime"]["durability"] == "live_only"
    assert payload["runtime"]["visibility"] == "draft"
    assert payload["runtime"]["source_event_id"] is None
    assert payload["runtime"]["cursor_event_id"] is None


def test_chat_request_should_allow_empty_stream_request() -> None:
    request = ChatRequest(event_id="evt-cursor-3")

    assert request.message is None
    assert request.resume is None
    assert request.command is None
    assert request.event_id == "evt-cursor-3"


def test_chat_request_should_reject_command_with_attachments() -> None:
    with pytest.raises(ValueError, match="不允许携带 attachments"):
        ChatRequest(
            command=ChatRequest.CommandPayload(type="continue_cancelled_task"),
            attachments=["file-1"],
        )
