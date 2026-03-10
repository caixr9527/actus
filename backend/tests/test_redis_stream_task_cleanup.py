import asyncio

from app.infrastructure.external.task.redis_stream_task import RedisStreamTask


class _FakeStream:
    def __init__(self) -> None:
        self.delete_calls = 0

    async def delete_stream(self) -> bool:
        self.delete_calls += 1
        return True


class _FakeRunner:
    def __init__(self) -> None:
        self.destroy_calls = 0

    async def destroy(self) -> None:
        self.destroy_calls += 1


def _build_fake_task(task_id: str = "task-1") -> RedisStreamTask:
    task = object.__new__(RedisStreamTask)
    task._id = task_id
    task._input_stream = _FakeStream()
    task._output_stream = _FakeStream()
    task._task_runner = _FakeRunner()
    task._execution_task = None
    task._streams_cleaned = False
    return task


def test_cleanup_streams_is_idempotent() -> None:
    task = _build_fake_task()

    asyncio.run(task._cleanup_streams())
    asyncio.run(task._cleanup_streams())

    assert task._input_stream.delete_calls == 1
    assert task._output_stream.delete_calls == 1


def test_destroy_cleans_streams_and_runner() -> None:
    backup_registry = dict(RedisStreamTask._task_registry)
    task = _build_fake_task(task_id="task-destroy-1")
    RedisStreamTask._task_registry = {task.id: task}
    try:
        asyncio.run(RedisStreamTask.destroy())
    finally:
        RedisStreamTask._task_registry = backup_registry

    assert task._input_stream.delete_calls == 1
    assert task._output_stream.delete_calls == 1
    assert task._task_runner.destroy_calls == 1
