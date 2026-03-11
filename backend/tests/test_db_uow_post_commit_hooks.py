import asyncio

# 先初始化依赖模块，避免触发 app.infrastructure.repositories 的历史循环导入路径。
from app.interfaces.service_dependencies import get_agent_service  # noqa: F401
from app.infrastructure.repositories.db_uow import DBUnitOfWork


class _FakeSession:
    def __init__(self) -> None:
        self.info = {}
        self.commit_called = 0
        self.rollback_called = 0
        self.close_called = 0

    async def commit(self) -> None:
        self.commit_called += 1

    async def rollback(self) -> None:
        self.rollback_called += 1

    async def close(self) -> None:
        self.close_called += 1


def test_uow_runs_post_commit_hooks_after_success() -> None:
    session = _FakeSession()
    uow = DBUnitOfWork(session_factory=lambda: session)
    hook_calls = {"count": 0}

    async def _hook() -> None:
        hook_calls["count"] += 1

    async def _run() -> None:
        async with uow:
            uow.db_session.info["post_commit_hooks"].append(_hook)

    asyncio.run(_run())

    assert session.commit_called == 1
    assert session.rollback_called == 0
    assert session.close_called == 1
    assert hook_calls["count"] == 1


def test_uow_does_not_run_post_commit_hooks_on_rollback() -> None:
    session = _FakeSession()
    uow = DBUnitOfWork(session_factory=lambda: session)
    hook_calls = {"count": 0}

    async def _hook() -> None:
        hook_calls["count"] += 1

    async def _run() -> None:
        try:
            async with uow:
                uow.db_session.info["post_commit_hooks"].append(_hook)
                raise RuntimeError("boom")
        except RuntimeError:
            return

    asyncio.run(_run())

    assert session.commit_called == 0
    assert session.rollback_called == 1
    assert session.close_called == 1
    assert hook_calls["count"] == 0
