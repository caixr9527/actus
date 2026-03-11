import asyncio
import json

from app.domain.models import Session
from app.interfaces.endpoints import session_routes


class _FakeSessionService:
    def __init__(self) -> None:
        self.calls = 0

    async def get_all_sessions(self):
        self.calls += 1
        if self.calls <= 2:
            return [Session(id="session-1", title="会话1")]
        return [
            Session(id="session-1", title="会话1"),
            Session(id="session-2", title="会话2"),
        ]


class _BrokenRedisClient:
    @property
    def client(self):
        raise RuntimeError("redis unavailable")


def test_stream_sessions_fallback_only_pushes_when_snapshot_changes(monkeypatch) -> None:
    service = _FakeSessionService()
    monkeypatch.setattr(session_routes, "SESSION_LIST_FALLBACK_REFRESH_SECONDS", 0.001)
    monkeypatch.setattr(session_routes, "get_redis_client", lambda: _BrokenRedisClient())

    async def _collect_two_events():
        response = await session_routes.stream_sessions(session_service=service)
        iterator = response.body_iterator
        first_event = await anext(iterator)
        second_event = await anext(iterator)
        await iterator.aclose()
        return first_event, second_event

    first_event, second_event = asyncio.run(_collect_two_events())
    first_payload = json.loads(first_event.data)
    second_payload = json.loads(second_event.data)

    assert len(first_payload["sessions"]) == 1
    assert len(second_payload["sessions"]) == 2
    # 第2次拉取与首帧相同且被跳过，第3次变化后才发送下一帧
    assert service.calls >= 3
