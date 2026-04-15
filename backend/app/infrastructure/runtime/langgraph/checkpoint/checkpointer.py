#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangGraph 持久化 checkpointer 生命周期管理。"""

import asyncio
import inspect
import logging
from functools import lru_cache
from typing import Any, Optional

from core.config import get_settings

logger = logging.getLogger(__name__)

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from langgraph.checkpoint.postgres import PostgresSaver


class LangGraphCheckpointer:
    """管理 LangGraph 持久化 checkpointer 的初始化与关闭。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._checkpointer: Optional[Any] = None
        self._context_manager: Optional[Any] = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_connection_string(connection_string: str) -> str:
        normalized = str(connection_string or "").strip()
        if normalized.startswith("postgresql+"):
            scheme, suffix = normalized.split("://", 1)
            driverless_scheme = scheme.split("+", 1)[0]
            return f"{driverless_scheme}://{suffix}"
        return normalized

    async def _run_maybe_awaitable(self, result: Any) -> Any:
        if inspect.isawaitable(result):
            return await result
        return result

    async def _open_async_postgres_saver(self, connection_string: str) -> Any:
        context_manager = AsyncPostgresSaver.from_conn_string(connection_string)
        if hasattr(context_manager, "__aenter__") and hasattr(context_manager, "__aexit__"):
            self._context_manager = context_manager
            return await context_manager.__aenter__()
        return context_manager

    async def _open_sync_postgres_saver(self, connection_string: str) -> Any:
        context_manager = PostgresSaver.from_conn_string(connection_string)
        if hasattr(context_manager, "__enter__") and hasattr(context_manager, "__exit__"):
            self._context_manager = context_manager
            return context_manager.__enter__()
        return context_manager

    async def init(self) -> None:
        """初始化生产级 Postgres checkpointer 连接，不执行建表/迁移。"""
        async with self._lock:
            if self._checkpointer is not None:
                logger.warning("LangGraph checkpointer 已初始化")
                return

            connection_string = self._normalize_connection_string(
                self._settings.sqlalchemy_database_uri
            )
            if not connection_string:
                raise RuntimeError("缺少数据库连接串，无法初始化 LangGraph checkpointer")

            if AsyncPostgresSaver is not None:
                logger.info("初始化 LangGraph AsyncPostgresSaver")
                self._checkpointer = await self._open_async_postgres_saver(connection_string)
            elif PostgresSaver is not None:
                logger.info("初始化 LangGraph PostgresSaver")
                self._checkpointer = await self._open_sync_postgres_saver(connection_string)
            else:
                raise RuntimeError(
                    "缺少 LangGraph PostgreSQL checkpointer 依赖，无法初始化持久化短期记忆"
                )

    async def ensure_schema(self) -> None:
        """显式执行 LangGraph checkpoint schema 初始化。"""
        async with self._lock:
            if self._checkpointer is None:
                raise RuntimeError("LangGraph checkpointer 尚未初始化，请先执行 init()")

            setup = getattr(self._checkpointer, "setup", None)
            if callable(setup):
                await self._run_maybe_awaitable(setup())

    def get_checkpointer(self) -> Any:
        """返回已初始化的持久化 checkpointer。"""
        if self._checkpointer is None:
            raise RuntimeError("LangGraph checkpointer 尚未初始化，请先在应用生命周期中执行 init()")
        return self._checkpointer

    async def close(self) -> None:
        """关闭持久化 checkpointer。"""
        async with self._lock:
            if self._checkpointer is None:
                _get_langgraph_checkpointer.cache_clear()
                return

            try:
                if self._context_manager is not None:
                    if hasattr(self._context_manager, "__aexit__"):
                        await self._context_manager.__aexit__(None, None, None)
                    elif hasattr(self._context_manager, "__exit__"):
                        self._context_manager.__exit__(None, None, None)
                else:
                    close = getattr(self._checkpointer, "close", None)
                    if callable(close):
                        await self._run_maybe_awaitable(close())
            finally:
                self._checkpointer = None
                self._context_manager = None
                _get_langgraph_checkpointer.cache_clear()


def get_langgraph_checkpointer() -> LangGraphCheckpointer:
    """获取 LangGraph checkpointer 单例。"""
    return _get_langgraph_checkpointer()


@lru_cache()
def _get_langgraph_checkpointer() -> LangGraphCheckpointer:
    return LangGraphCheckpointer()
