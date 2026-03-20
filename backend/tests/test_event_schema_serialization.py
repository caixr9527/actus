from datetime import datetime

from app.domain.models import ErrorEvent, MessageEvent
from app.interfaces.schemas.event import BaseEventData, CommonEventData, EventMapper


def test_base_event_data_created_at_serialized_as_timestamp() -> None:
    created_at = datetime(2026, 3, 11, 12, 0, 1)
    data = BaseEventData(event_id="evt-1", created_at=created_at)

    payload = data.model_dump(mode="json")

    assert payload["created_at"] == int(created_at.timestamp())


def test_common_event_data_keeps_extra_fields_and_timestamp_serializer() -> None:
    created_at = datetime(2026, 3, 11, 12, 0, 2)
    data = CommonEventData.model_validate(
        {
            "event_id": "evt-2",
            "created_at": created_at,
            "custom_key": "custom-value",
        }
    )

    payload = data.model_dump(mode="json")

    assert payload["created_at"] == int(created_at.timestamp())
    assert payload["custom_key"] == "custom-value"


def test_event_mapper_preserves_timestamp_json_shape() -> None:
    created_at = datetime(2026, 3, 11, 12, 0, 3)
    event = MessageEvent(
        id="evt-3",
        created_at=created_at,
        role="assistant",
        message="hello",
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["data"]["created_at"] == int(created_at.timestamp())


def test_event_mapper_should_serialize_error_event_key_and_params() -> None:
    event = ErrorEvent(
        id="evt-4",
        created_at=datetime(2026, 3, 11, 12, 0, 4),
        error="任务会话不存在",
        error_key="error.session.not_found",
        error_params={"session_id": "session-1"},
    )

    sse_event = EventMapper.event_to_sse_event(event)
    payload = sse_event.model_dump(mode="json")

    assert payload["event"] == "error"
    assert payload["data"]["error"] == "任务会话不存在"
    assert payload["data"]["error_key"] == "error.session.not_found"
    assert payload["data"]["error_params"] == {"session_id": "session-1"}
