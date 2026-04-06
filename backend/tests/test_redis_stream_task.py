import asyncio

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
