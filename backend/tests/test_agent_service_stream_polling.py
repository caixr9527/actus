import asyncio

# 先初始化依赖模块，避免触发 app.application.service 的历史循环导入路径。
from app.interfaces.service_dependencies import get_agent_service  # noqa: F401
from app.application.service.agent_service import AgentService, OUTPUT_STREAM_BLOCK_MS
from app.domain.models import SessionStatus


class _Session:
    def __init__(self) -> None:
        self.id = "session-1"
        self.task_id = "task-1"
        self.status = SessionStatus.RUNNING


class _SessionRepo:
    def __init__(self, session: _Session) -> None:
        self._session = session
        self.unread_count_updated = False

    async def get_by_id(self, session_id: str):
        return self._session

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        self.unread_count_updated = True


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
    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo)
    service._task_cls = _TaskFactory

    async def _consume() -> None:
        async for _ in service.chat(session_id=session.id):
            pass

    asyncio.run(_consume())

    assert _TaskFactory.task.output_stream.block_ms_calls
    assert _TaskFactory.task.output_stream.block_ms_calls[0] == OUTPUT_STREAM_BLOCK_MS
