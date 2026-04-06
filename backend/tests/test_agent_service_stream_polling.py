import asyncio

# 先初始化依赖模块，避免触发 app.application.service 的历史循环导入路径。
from app.interfaces.dependencies.services import get_agent_service  # noqa: F401
from app.application.service.agent_service import AgentService, OUTPUT_STREAM_BLOCK_MS
from app.domain.models import SessionStatus, MessageEvent, DoneEvent, TaskStreamEventRecord


class _Session:
    def __init__(self) -> None:
        self.id = "session-1"
        self.user_id = "user-1"
        self.task_id = "task-1"
        self.status = SessionStatus.RUNNING
        self.unread_message_count = 0


class _SessionRepo:
    def __init__(self, session: _Session) -> None:
        self._session = session
        self.unread_count_updated = False
        self.unread_update_calls = 0

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        return self._session

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        self.unread_count_updated = True
        self.unread_update_calls += 1

    async def add_event_if_absent(self, session_id: str, event) -> bool:
        return True


class _UoW:
    def __init__(self, session_repo: _SessionRepo) -> None:
        self.session = session_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _OutputStream:
    def __init__(self, task: "_Task") -> None:
        self._task = task
        self.block_ms_calls: list[int] = []

    async def get(self, start_id=None, block_ms=0):
        self.block_ms_calls.append(block_ms)
        # 首次空读后将任务置为完成，确保聊天循环可自然退出。
        self._task.done = True
        return None, None


class _Task:
    def __init__(self) -> None:
        self.done = False
        self.output_stream = _OutputStream(self)


class _TaskFactory:
    task = _Task()

    @classmethod
    def get(cls, task_id: str):
        return cls.task


def test_chat_uses_blocking_output_stream_polling() -> None:
    session = _Session()
    session_repo = _SessionRepo(session=session)
    _TaskFactory.task = _Task()
    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo)
    service._task_cls = _TaskFactory

    async def _consume() -> None:
        async for _ in service.chat(session_id=session.id, user_id=session.user_id):
            pass

    asyncio.run(_consume())

    assert _TaskFactory.task.output_stream.block_ms_calls
    assert _TaskFactory.task.output_stream.block_ms_calls[0] == OUTPUT_STREAM_BLOCK_MS


class _CursorOutputStream:
    def __init__(self, task: "_CursorTask") -> None:
        self._task = task
        self.start_ids: list[str | None] = []
        self._none_cursor_reads = 0
        self._after_first_reads = 0

    async def get(self, start_id=None, block_ms=0):
        self.start_ids.append(start_id)
        if start_id is None:
            self._none_cursor_reads += 1
            event = TaskStreamEventRecord(
                event=MessageEvent(role="assistant", message="first reply"),
            )
            if self._none_cursor_reads == 1:
                return "evt-1", event.model_dump_json()
            # 若游标被错误重置为 None，会重复返回首条消息。
            self._task.done = True
            return "evt-1", event.model_dump_json()

        if start_id == "evt-1":
            self._after_first_reads += 1
            if self._after_first_reads == 1:
                # 模拟一次空读
                return None, None
            self._task.done = True
            return "evt-2", TaskStreamEventRecord(event=DoneEvent()).model_dump_json()

        self._task.done = True
        return None, None


class _CursorTask:
    def __init__(self) -> None:
        self.done = False
        self.output_stream = _CursorOutputStream(self)


class _CursorTaskFactory:
    task = _CursorTask()

    @classmethod
    def get(cls, task_id: str):
        return cls.task


def test_chat_does_not_reset_cursor_after_empty_poll() -> None:
    session = _Session()
    session_repo = _SessionRepo(session=session)
    _CursorTaskFactory.task = _CursorTask()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo)
    service._task_cls = _CursorTaskFactory

    async def _consume():
        events = []
        async for event in service.chat(session_id=session.id, user_id=session.user_id):
            events.append(event)
        return events

    events = asyncio.run(_consume())
    assistant_messages = [event for event in events if getattr(event, "type", "") == "message"]

    # 空读后游标应保持在 evt-1，后续应读取 evt-2(done)，而不是回到 None 重放 evt-1。
    assert _CursorTaskFactory.task.output_stream.start_ids[:3] == [None, "evt-1", "evt-1"]
    assert len(assistant_messages) == 1
    assert any(getattr(event, "type", "") == "done" for event in events)


def test_chat_should_reset_unread_count_only_once_for_stream_events() -> None:
    session = _Session()
    session.unread_message_count = 3
    session_repo = _SessionRepo(session=session)
    _CursorTaskFactory.task = _CursorTask()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo)
    service._task_cls = _CursorTaskFactory

    async def _consume():
        async for _ in service.chat(session_id=session.id, user_id=session.user_id):
            pass

    asyncio.run(_consume())

    assert session_repo.unread_update_calls == 1
