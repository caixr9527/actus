import asyncio

# 先初始化依赖模块，避免触发 app.application.service 的历史循环导入路径。
from app.interfaces.service_dependencies import get_agent_service  # noqa: F401
from app.application.service.agent_service import AgentService
from app.domain.models import DoneEvent, MessageEvent, SessionStatus


class _Session:
    def __init__(self) -> None:
        self.id = "session-1"
        self.task_id = "task-1"
        self.status = SessionStatus.RUNNING


class _SessionRepo:
    def __init__(self, session: _Session) -> None:
        self._session = session
        self.events_saved = []

    async def get_by_id(self, session_id: str):
        return self._session

    async def add_event_if_absent(self, session_id: str, event) -> bool:
        self.events_saved.append(event)
        return True

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None


class _UoW:
    def __init__(self, session_repo: _SessionRepo) -> None:
        self.session = session_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _parse_stream_id(stream_id: str) -> tuple[int, int]:
    major, minor = stream_id.split("-", 1)
    return int(major), int(minor)


class _RedisLikeOutputStream:
    """模拟 Redis Stream xread 的核心语义：返回严格大于 start_id 的下一条消息。"""

    def __init__(self, task: "_Task") -> None:
        self._task = task
        self.start_ids: list[str | None] = []
        self._empty_after_first = False
        first = MessageEvent(role="assistant", message="first reply").model_dump_json()
        done = DoneEvent().model_dump_json()
        self._events: list[tuple[str, str]] = [("1-0", first), ("2-0", done)]

    async def get(self, start_id=None, block_ms=0):
        self.start_ids.append(start_id)
        cursor = start_id or "0-0"

        # 插入一次空读，模拟真实阻塞读取超时场景。
        if cursor == "1-0" and not self._empty_after_first:
            self._empty_after_first = True
            return None, None

        cursor_tuple = _parse_stream_id(cursor)
        for event_id, payload in self._events:
            if _parse_stream_id(event_id) > cursor_tuple:
                if event_id == "2-0":
                    self._task.done = True
                return event_id, payload
        return None, None


class _Task:
    def __init__(self) -> None:
        self.done = False
        self.output_stream = _RedisLikeOutputStream(self)


class _TaskFactory:
    task = _Task()

    @classmethod
    def get(cls, task_id: str):
        return cls.task


def test_chat_stream_cursor_progression_matches_redis_semantics() -> None:
    session = _Session()
    session_repo = _SessionRepo(session=session)
    _TaskFactory.task = _Task()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo)
    service._task_cls = _TaskFactory

    async def _consume():
        events = []
        async for event in service.chat(session_id=session.id):
            events.append(event)
        return events

    events = asyncio.run(_consume())
    message_events = [event for event in events if getattr(event, "type", "") == "message"]
    done_events = [event for event in events if getattr(event, "type", "") == "done"]

    # 空读后游标应保持为上一条已消费ID（1-0），不能回退为 None/0-0。
    assert _TaskFactory.task.output_stream.start_ids[:3] == [None, "1-0", "1-0"]
    assert len(message_events) == 1
    assert message_events[0].message == "first reply"
    assert len(done_events) == 1
