import asyncio
import json

import pytest

from app.application.service.runtime_observation_service import (
    RuntimeCapabilityResult,
    RuntimeCursorResult,
    RuntimeEventMetaResult,
    RuntimeInteractionResult,
    RuntimeObservationContextResult,
    RuntimeObservableEventResult,
    RuntimeObservationResult,
    RuntimeReplayResult,
)
from app.application.errors import NotFoundError
from app.domain.models import DoneEvent, MessageEvent, Session, SessionStatus, TextStreamChannel, TextStreamDeltaEvent
from app.domain.models import WaitEvent, WorkflowRunEventRecord
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


class _AgentServiceForEmptyStream:
    def __init__(self) -> None:
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return
        yield


class _AgentServiceForLiveAttachBoundary:
    def __init__(self, events: list[MessageEvent]) -> None:
        self.calls = []
        self.events = events

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        latest_event_id = kwargs["latest_event_id"]
        start_index = 0
        if latest_event_id is not None:
            for index, event in enumerate(self.events):
                if event.id == latest_event_id:
                    start_index = index + 1
                    break
        for event in self.events[start_index:]:
            yield event


class _RuntimeObservationServiceForChat:
    def __init__(self) -> None:
        self.session_observation_calls = 0
        self.event_context_calls = 0
        self.context_built_before_replay = False
        self.replay_calls = []
        self.replay_result = RuntimeReplayResult()

    async def build_session_observation(self, user_id: str, session_id: str):
        self.session_observation_calls += 1
        return RuntimeObservationResult(
            session_id=session_id,
            run_id="run-1",
            status=SessionStatus.RUNNING,
            cursor=RuntimeCursorResult(),
            capabilities=RuntimeCapabilityResult(can_send_message=False, can_cancel=True),
            interaction=RuntimeInteractionResult(),
        )

    async def build_event_context(
            self,
            user_id: str,
            session_id: str,
            reconcile_reason: str | None = None,
    ):
        self.event_context_calls += 1
        assert reconcile_reason in {None, "before_chat"}
        if reconcile_reason == "before_chat":
            self.context_built_before_replay = True
        return RuntimeObservationContextResult(
            session_id=session_id,
            run_id="run-1",
            status=SessionStatus.RUNNING,
            current_step_id="step-1",
        )

    async def list_persistent_events_after_cursor(
            self,
            *,
            user_id: str,
            session_id: str,
            cursor_event_id: str | None,
    ):
        assert self.context_built_before_replay
        self.replay_calls.append({
            "user_id": user_id,
            "session_id": session_id,
            "cursor_event_id": cursor_event_id,
        })
        return self.replay_result

    async def build_observable_event(
            self,
            *,
            session_id: str,
            event,
            run_id: str | None,
            source_event_id: str | None,
            cursor_event_id: str | None,
            source: str,
            context=None,
    ):
        is_text_stream = str(event.type).startswith("text_stream_")
        status_after_event = SessionStatus.COMPLETED if isinstance(event, DoneEvent) else None
        return RuntimeObservableEventResult(
            event=event,
            runtime=RuntimeEventMetaResult(
                session_id=session_id,
                run_id=run_id,
                status_after_event=status_after_event,
                current_step_id=getattr(context, "current_step_id", None),
                source_event_id=None if is_text_stream else source_event_id,
                cursor_event_id=None if is_text_stream else cursor_event_id,
                durability="live_only" if is_text_stream else "persistent",
                visibility="draft" if is_text_stream else "timeline",
            ),
        )

    def advance_event_context(self, context, event):
        if isinstance(event, DoneEvent):
            return context.model_copy(update={"status": SessionStatus.COMPLETED, "current_step_id": None})
        return context


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
    runtime_observation_service = _RuntimeObservationServiceForChat()
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
            runtime_observation_service=runtime_observation_service,
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
    assert first_payload["runtime"]["current_step_id"] == "step-1"
    assert first_payload["runtime"]["status_after_event"] is None
    assert first_payload["runtime"]["source_event_id"] == "evt-1"
    assert first_payload["runtime"]["cursor_event_id"] == "evt-1"
    second_payload = json.loads(events[1].data)
    assert second_payload["runtime"]["status_after_event"] == "completed"

    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["session_id"] == "session-1"
    assert agent_service.calls[0]["user_id"] == "user-1"
    assert agent_service.calls[0]["message"] == "hello"
    assert agent_service.calls[0]["attachments"] == ["file-1"]
    assert agent_service.calls[0]["resume"] is None
    assert agent_service.calls[0]["command"] is None
    assert agent_service.calls[0]["latest_event_id"] == "evt-cursor-1"
    assert runtime_observation_service.session_observation_calls == 0
    assert runtime_observation_service.event_context_calls == 1


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


def test_empty_stream_should_replay_db_records_after_persistent_cursor_without_redis_cursor() -> None:
    facade = SessionStreamFacade()
    session_service = _SessionServiceForChat()
    agent_service = _AgentServiceForEmptyStream()
    runtime_observation_service = _RuntimeObservationServiceForChat()
    runtime_observation_service.replay_result = RuntimeReplayResult(
        records=[
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                event_id="evt-2",
                event_type="wait",
                event_payload=WaitEvent(id="evt-2", interrupt_id="interrupt-1"),
            ),
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                event_id="evt-3",
                event_type="done",
                event_payload=DoneEvent(id="evt-3"),
            ),
        ]
    )
    request = ChatRequest(event_id="evt-1")

    async def _collect():
        iterator = await facade.stream_chat(
            user_id="user-1",
            session_id="session-1",
            request=request,
            session_service=session_service,
            agent_service=agent_service,
            runtime_observation_service=runtime_observation_service,
        )
        events = []
        async for event in iterator:
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert [event.event for event in events] == ["wait", "done"]
    assert runtime_observation_service.replay_calls == [{
        "user_id": "user-1",
        "session_id": "session-1",
        "cursor_event_id": "evt-1",
    }]
    assert runtime_observation_service.context_built_before_replay is True
    first_payload = json.loads(events[0].data)
    second_payload = json.loads(events[1].data)
    assert first_payload["runtime"]["cursor_event_id"] == "evt-2"
    assert second_payload["runtime"]["cursor_event_id"] == "evt-3"
    assert len(agent_service.calls) == 0


def test_empty_stream_should_attach_live_after_latest_replayed_record_without_duplicates() -> None:
    facade = SessionStreamFacade()
    session_service = _SessionServiceForChat()
    agent_service = _AgentServiceForLiveAttachBoundary(events=[
        MessageEvent(id="evt-2", role="assistant", message="old-2"),
        MessageEvent(id="evt-3", role="assistant", message="old-3"),
        MessageEvent(id="evt-4", role="assistant", message="live-4"),
    ])
    runtime_observation_service = _RuntimeObservationServiceForChat()
    runtime_observation_service.replay_result = RuntimeReplayResult(
        records=[
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                event_id="evt-2",
                event_type="message",
                event_payload=MessageEvent(id="evt-2", role="assistant", message="old-2"),
            ),
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                event_id="evt-3",
                event_type="message",
                event_payload=MessageEvent(id="evt-3", role="assistant", message="old-3"),
            ),
        ],
        live_attach_after_event_id="evt-3",
    )
    request = ChatRequest(event_id="evt-1")

    async def _collect():
        iterator = await facade.stream_chat(
            user_id="user-1",
            session_id="session-1",
            request=request,
            session_service=session_service,
            agent_service=agent_service,
            runtime_observation_service=runtime_observation_service,
        )
        events = []
        async for event in iterator:
            events.append(event)
        return events

    events = asyncio.run(_collect())

    messages = [json.loads(event.data)["message"] for event in events]
    assert messages == ["old-2", "old-3", "live-4"]
    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["latest_event_id"] == "evt-3"


def test_empty_stream_should_attach_live_after_invalid_cursor_replay_boundary() -> None:
    facade = SessionStreamFacade()
    session_service = _SessionServiceForChat()
    agent_service = _AgentServiceForLiveAttachBoundary(events=[
        MessageEvent(id="evt-1", role="assistant", message="old-1"),
        MessageEvent(id="evt-2", role="assistant", message="old-2"),
        MessageEvent(id="evt-3", role="assistant", message="old-3"),
        MessageEvent(id="evt-4", role="assistant", message="live-4"),
    ])
    runtime_observation_service = _RuntimeObservationServiceForChat()
    runtime_observation_service.replay_result = RuntimeReplayResult(
        records=[
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                event_id="evt-1",
                event_type="message",
                event_payload=MessageEvent(id="evt-1", role="assistant", message="old-1"),
            ),
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                event_id="evt-2",
                event_type="message",
                event_payload=MessageEvent(id="evt-2", role="assistant", message="old-2"),
            ),
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                event_id="evt-3",
                event_type="message",
                event_payload=MessageEvent(id="evt-3", role="assistant", message="old-3"),
            ),
        ],
        cursor_invalid=True,
        live_attach_after_event_id="evt-3",
    )
    request = ChatRequest(event_id="missing-event")

    async def _collect():
        iterator = await facade.stream_chat(
            user_id="user-1",
            session_id="session-1",
            request=request,
            session_service=session_service,
            agent_service=agent_service,
            runtime_observation_service=runtime_observation_service,
        )
        events = []
        async for event in iterator:
            events.append(event)
        return events

    events = asyncio.run(_collect())

    messages = [json.loads(event.data)["message"] for event in events]
    assert messages == ["old-1", "old-2", "old-3", "live-4"]
    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["latest_event_id"] == "evt-3"


def test_empty_stream_should_attach_live_with_none_only_when_no_db_history() -> None:
    facade = SessionStreamFacade()
    session_service = _SessionServiceForChat()
    agent_service = _AgentServiceForChat()
    runtime_observation_service = _RuntimeObservationServiceForChat()
    request = ChatRequest(event_id=None)

    async def _collect():
        iterator = await facade.stream_chat(
            user_id="user-1",
            session_id="session-1",
            request=request,
            session_service=session_service,
            agent_service=agent_service,
            runtime_observation_service=runtime_observation_service,
        )
        events = []
        async for event in iterator:
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert [event.event for event in events] == ["message", "done"]
    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["latest_event_id"] is None


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
