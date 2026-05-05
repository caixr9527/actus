import asyncio

import pytest

from app.infrastructure.external.task.redis_stream_task import RedisStreamTask


class _Runner:
    def __init__(self) -> None:
        self.cancelled = False
        self.on_done_called = False

    async def invoke(self, task) -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def destroy(self) -> None:
        return None

    async def on_done(self, task) -> None:
        await asyncio.sleep(0)
        self.on_done_called = True


def test_redis_stream_task_cancel_should_wait_for_runner_finalization() -> None:
    async def _run() -> None:
        runner = _Runner()
        task = RedisStreamTask(task_runner=runner)

        async def _noop_cleanup_streams() -> None:
            return None

        task._cleanup_streams = _noop_cleanup_streams  # type: ignore[method-assign]

        await task.invoke()
        await asyncio.sleep(0)
        cancelled = await task.cancel()

        assert cancelled is True
        assert runner.cancelled is True
        assert runner.on_done_called is True
        assert RedisStreamTask.get(task.id) is None

    asyncio.run(_run())


def test_redis_stream_task_allocate_should_not_be_executable_before_bind() -> None:
    async def _run() -> None:
        task = RedisStreamTask.allocate(task_id="allocated-task")

        async def _noop_cleanup_streams() -> None:
            return None

        task._cleanup_streams = _noop_cleanup_streams  # type: ignore[attr-defined, method-assign]
        try:
            assert task.id == "allocated-task"
            assert task.is_bound is False
            assert RedisStreamTask.get("allocated-task") is task
            with pytest.raises(RuntimeError, match="任务未绑定 runner"):
                await task.invoke()
        finally:
            await task.cancel()

        assert RedisStreamTask.get("allocated-task") is None

    asyncio.run(_run())


def test_redis_stream_task_bind_runner_should_only_succeed_once() -> None:
    async def _run() -> None:
        task = RedisStreamTask.allocate(task_id="bind-task")

        async def _noop_cleanup_streams() -> None:
            return None

        task._cleanup_streams = _noop_cleanup_streams  # type: ignore[attr-defined, method-assign]
        try:
            first_runner = _Runner()
            task.bind_runner(first_runner)

            assert task.is_bound is True
            with pytest.raises(RuntimeError, match="runner 已绑定"):
                task.bind_runner(_Runner())
        finally:
            await task.cancel()

    asyncio.run(_run())


def test_redis_stream_task_create_should_allocate_then_bind() -> None:
    async def _run() -> None:
        task = RedisStreamTask.create(_Runner())

        async def _noop_cleanup_streams() -> None:
            return None

        task._cleanup_streams = _noop_cleanup_streams  # type: ignore[attr-defined, method-assign]
        try:
            assert task.id
            assert task.is_bound is True
            assert RedisStreamTask.get(task.id) is task
        finally:
            await task.cancel()

    asyncio.run(_run())
