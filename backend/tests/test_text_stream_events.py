from app.domain.models import TextStreamChannel, TextStreamDeltaEvent, TextStreamEndEvent, TextStreamStartEvent
from app.infrastructure.runtime.langgraph.graphs.planner_react.streaming import (
    build_text_stream_events,
    split_text_for_stream,
)


def test_split_text_for_stream_should_prefer_natural_breakpoints() -> None:
    chunks = split_text_for_stream(
        "第一句。第二句需要继续说明，第三句结束。",
        chunk_size=6,
        max_chunk_size=12,
    )

    assert chunks[0] == "第一句。"
    assert "".join(chunks) == "第一句。第二句需要继续说明，第三句结束。"


def test_build_text_stream_events_should_build_start_delta_end_sequence() -> None:
    events = build_text_stream_events(
        channel=TextStreamChannel.PLANNER_MESSAGE,
        text="计划说明第一段。计划说明第二段。",
        state={
            "run_id": "run-1",
            "session_id": "session-1",
            "thread_id": "thread-1",
        },
        stage="planner",
    )

    assert isinstance(events[0], TextStreamStartEvent)
    assert isinstance(events[-1], TextStreamEndEvent)
    assert all(isinstance(event, TextStreamDeltaEvent) for event in events[1:-1])
    assert events[0].stream_id == "run-1:planner_message"
    assert events[0].run_id == "run-1"
    assert events[0].session_id == "session-1"
    assert events[0].stage == "planner"
    assert "".join(event.text for event in events[1:-1] if isinstance(event, TextStreamDeltaEvent)) == (
        "计划说明第一段。计划说明第二段。"
    )
    assert events[-1].full_text_length == len("计划说明第一段。计划说明第二段。")
