import asyncio

from pydantic import TypeAdapter

from app.application.service.agent_service import AgentService
from app.application.errors import error_keys
from app.domain.models import (
    DoneEvent,
    ErrorEvent,
    MessageEvent,
    RuntimeInput,
    SessionStatus,
    TaskRequestFinishedRecord,
    TaskRequestRejectedRecord,
    TaskRequestStartedRecord,
    TaskStreamEventRecord,
    WaitEvent,
)


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

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        return self._session

    async def add_event_if_absent(self, session_id: str, event) -> bool:
        return True

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None

    async def add_event_with_snapshot_if_absent(self, session_id: str, event, **kwargs) -> None:
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


class _QueuedInputStream:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.deleted: list[str] = []
        self._queue: list[str] = []
        self._second_enqueued = asyncio.Event()

    async def put(self, message: str) -> str:
        self.messages.append(message)
        self._queue.append(message)
        if len(self.messages) >= 2:
            self._second_enqueued.set()
        return f"in-{len(self.messages)}"

    async def delete_message(self, event_id: str) -> None:
        self.deleted.append(event_id)

    async def pop_next(self) -> str:
        return self._queue.pop(0)

    def has_pending(self) -> bool:
        return len(self._queue) > 0

    async def wait_for_second_request(self) -> None:
        await self._second_enqueued.wait()


class _QueuedOutputStream:
    def __init__(self, task: "_QueuedTask") -> None:
        self._task = task
        self._records: list[tuple[str, str]] = []
        self._condition = asyncio.Condition()

    async def put_record(self, payload: str) -> str:
        async with self._condition:
            event_id = f"{len(self._records) + 1}-0"
            self._records.append((event_id, payload))
            self._condition.notify_all()
            return event_id

    async def get(self, start_id=None, block_ms=0):
        cursor = start_id or "0-0"
        timeout_seconds = max(block_ms, 0) / 1000
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds

        while True:
            cursor_tuple = _parse_stream_id(cursor)
            for event_id, payload in self._records:
                if _parse_stream_id(event_id) > cursor_tuple:
                    if event_id == "1-0":
                        self._task.first_boundary_consumed.set()
                    return event_id, payload

            if self._task.done:
                return None, None

            remaining = deadline - loop.time()
            if remaining <= 0:
                return None, None

            async with self._condition:
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None, None


class _QueuedTask:
    def __init__(self) -> None:
        self.input_stream = _QueuedInputStream()
        self.output_stream = _QueuedOutputStream(self)
        self._execution_task: asyncio.Task | None = None
        self.allow_processing = asyncio.Event()
        self.first_boundary_consumed = asyncio.Event()

    @property
    def done(self) -> bool:
        return self._execution_task is None or self._execution_task.done()

    async def invoke(self) -> None:
        if self.done:
            self._execution_task = asyncio.create_task(self._run())

    async def _emit(self, record) -> None:
        await self.output_stream.put_record(record.model_dump_json())

    async def _run(self) -> None:
        processed_count = 0
        while self.input_stream.has_pending():
            envelope = TypeAdapter(RuntimeInput).validate_json(await self.input_stream.pop_next())
            payload = envelope.payload
            processed_count += 1

            await self._emit(TaskRequestStartedRecord(request_id=envelope.request_id))
            if processed_count == 1:
                # 第二个请求必须在第一个 run 未结束时入队，才能复现 service 层提前收口问题。
                await self.input_stream.wait_for_second_request()
                await self.first_boundary_consumed.wait()
                await self.allow_processing.wait()

            message = payload.message if isinstance(payload, MessageEvent) else "resume"
            await self._emit(
                TaskStreamEventRecord(
                    event=MessageEvent(role="assistant", message=f"reply:{message}"),
                )
            )
            await self._emit(TaskStreamEventRecord(event=DoneEvent()))
            await self._emit(
                TaskRequestFinishedRecord(
                    request_id=envelope.request_id,
                    terminal_event_type="done",
                )
            )


class _WaitingQueuedTask(_QueuedTask):
    async def _run(self) -> None:
        processed_count = 0
        while self.input_stream.has_pending():
            envelope = TypeAdapter(RuntimeInput).validate_json(await self.input_stream.pop_next())
            payload = envelope.payload
            processed_count += 1

            await self._emit(TaskRequestStartedRecord(request_id=envelope.request_id))
            if processed_count == 1:
                await self.input_stream.wait_for_second_request()
                await self.first_boundary_consumed.wait()
                await self.allow_processing.wait()

                message = payload.message if isinstance(payload, MessageEvent) else "resume"
                await self._emit(
                    TaskStreamEventRecord(
                        event=MessageEvent(role="assistant", message=f"reply:{message}"),
                    )
                )
                await self._emit(TaskStreamEventRecord(event=WaitEvent()))
                await self._emit(
                    TaskRequestFinishedRecord(
                        request_id=envelope.request_id,
                        terminal_event_type="wait",
                    )
                )

                while self.input_stream.has_pending():
                    pending = TypeAdapter(RuntimeInput).validate_json(await self.input_stream.pop_next())
                    await self._emit(
                        TaskRequestRejectedRecord(
                            request_id=pending.request_id,
                            message="当前任务进入等待状态，请使用 resume 恢复执行",
                            error_key=error_keys.SESSION_RESUME_REQUIRED,
                        )
                    )
                return


def test_chat_should_wait_for_its_own_queued_request_boundary() -> None:
    session = _Session()
    session_repo = _SessionRepo(session=session)
    task = _QueuedTask()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo)

    async def _get_task(_session):
        return task

    service._get_task = _get_task

    async def _collect_remaining(iterator, initial_events):
        events = list(initial_events)
        async for event in iterator:
            events.append(event)
        return events

    async def _run_collectors():
        first_iterator = service.chat(session_id=session.id, user_id=session.user_id, message="first")
        second_iterator = service.chat(session_id=session.id, user_id=session.user_id, message="second")

        first_events = [await anext(first_iterator)]
        second_events = [await anext(second_iterator)]

        first_task = asyncio.create_task(_collect_remaining(first_iterator, first_events))
        second_task = asyncio.create_task(_collect_remaining(second_iterator, second_events))

        task.allow_processing.set()
        return await asyncio.gather(first_task, second_task)

    first_events, second_events = asyncio.run(_run_collectors())

    assert [event.type for event in first_events] == ["message", "message", "done"]
    assert [event.type for event in second_events] == ["message", "message", "done"]
    assert [event.message for event in first_events if event.type == "message"] == ["first", "reply:first"]
    assert [event.message for event in second_events if event.type == "message"] == ["second", "reply:second"]


def test_chat_should_reject_queued_request_when_current_run_enters_wait() -> None:
    session = _Session()
    session_repo = _SessionRepo(session=session)
    task = _WaitingQueuedTask()

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(session_repo)

    async def _get_task(_session):
        return task

    service._get_task = _get_task

    async def _collect_remaining(iterator, initial_events):
        events = list(initial_events)
        async for event in iterator:
            events.append(event)
        return events

    async def _run_collectors():
        first_iterator = service.chat(session_id=session.id, user_id=session.user_id, message="first")
        second_iterator = service.chat(session_id=session.id, user_id=session.user_id, message="second")

        first_events = [await anext(first_iterator)]
        second_events = [await anext(second_iterator)]

        first_task = asyncio.create_task(_collect_remaining(first_iterator, first_events))
        second_task = asyncio.create_task(_collect_remaining(second_iterator, second_events))

        task.allow_processing.set()
        return await asyncio.gather(first_task, second_task)

    first_events, second_events = asyncio.run(_run_collectors())

    assert [event.type for event in first_events] == ["message", "message", "wait"]
    assert [event.type for event in second_events] == ["message", "error"]
    assert [event.message for event in first_events if event.type == "message"] == ["first", "reply:first"]
    assert isinstance(second_events[-1], ErrorEvent)
    assert second_events[-1].error_key == error_keys.SESSION_RESUME_REQUIRED
