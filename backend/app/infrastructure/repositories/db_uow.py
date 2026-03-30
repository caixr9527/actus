#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/2/14 21:02
@Author : caixiaorong01@outlook.com
@File   : db_uow.py
"""
import logging
from typing import Optional, Awaitable, Callable

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.domain.repositories import IUnitOfWork
from .db_file_repository import DBFileRepository
from .db_llm_model_config_repository import DBLLMModelConfigRepository
from .db_long_term_memory_repository import DBLongTermMemoryRepository
from .db_session_repository import DBSessionRepository
from .db_user_repository import DBUserRepository
from .db_workflow_run_repository import DBWorkflowRunRepository

logger = logging.getLogger(__name__)


class DBUnitOfWork(IUnitOfWork):
    """基于Postgres数据库的UoW实例"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """构造函数，完成UoW类初始化"""
        self.session_factory = session_factory
        self.db_session: Optional[AsyncSession] = None
        # 同一UoW实例只允许进入一次上下文，防止重入导致会话串线。
        self._entered = False

    async def commit(self):
        """提交数据库持久化"""
        if self.db_session is None:
            raise RuntimeError("UoW尚未进入上下文，无法提交事务")
        await self.db_session.commit()

    async def rollback(self):
        """数据库回退操作"""
        if self.db_session is None:
            raise RuntimeError("UoW尚未进入上下文，无法回滚事务")
        await self.db_session.rollback()

    async def __aenter__(self) -> "DBUnitOfWork":
        # fail-fast：同一实例不允许重入/并发复用。
        if self._entered:
            raise RuntimeError("DBUnitOfWork不支持重入/并发复用，请为每次操作创建新实例")
        if self.db_session is not None:
            raise RuntimeError("DBUnitOfWork会话状态异常，请勿复用已污染实例")

        # 为每个上下文开启一个新的会话。
        self.db_session = self.session_factory()
        self._entered = True
        self.db_session.info["post_commit_hooks"] = []
        self.db_session.info["session_list_changed_ids"] = set()

        # 初始化所有数据库仓库。
        self.file = DBFileRepository(db_session=self.db_session)
        self.session = DBSessionRepository(db_session=self.db_session)
        self.user = DBUserRepository(db_session=self.db_session)
        self.llm_model_config = DBLLMModelConfigRepository(db_session=self.db_session)
        self.long_term_memory = DBLongTermMemoryRepository(db_session=self.db_session)
        self.workflow_run = DBWorkflowRunRepository(db_session=self.db_session)

        return self

    async def _run_post_commit_hooks(self) -> None:
        """在事务提交成功后执行注册的回调（best-effort）"""
        if self.db_session is None:
            return

        hooks: list[Callable[[], Awaitable[None]]] = list(
            self.db_session.info.get("post_commit_hooks", [])
        )
        for hook in hooks:
            try:
                await hook()
            except Exception as e:
                # 提交已成功，回调失败不应回滚业务结果；按告警记录并依赖兜底机制恢复。
                logger.warning(f"执行post-commit回调失败: {e}", exc_info=True)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：出现异常回滚，否则提交。

        策略说明：
        1. 原始业务异常优先，不被回滚/关闭异常覆盖。
        2. 正常路径下提交失败必须向上抛出，避免“写失败但接口成功”。
        3. 无论成功失败，都复位内部状态，避免脏实例被再次复用。
        """
        tx_error: Optional[BaseException] = None
        close_error: Optional[BaseException] = None

        try:
            if self.db_session is None:
                # 正常流程不应命中，仅记录告警便于排查异常调用路径。
                logger.warning("UoW退出时发现db_session为空，跳过事务结束流程")
            elif exc_type:
                # 业务异常路径仅尝试回滚，保留原始异常传播语义。
                try:
                    await self.rollback()
                except Exception as e:
                    tx_error = e
                    logger.warning(f"UoW回滚失败: {e}", exc_info=True)
            else:
                # 正常路径提交失败不可吞掉，必须反馈给上层调用方。
                try:
                    await self.commit()
                    await self._run_post_commit_hooks()
                except Exception as e:
                    tx_error = e
                    logger.warning(f"UoW提交失败: {e}", exc_info=True)
        finally:
            # close放到finally里，确保连接被尽最大努力归还连接池。
            try:
                if self.db_session is not None:
                    await self.db_session.close()
            except Exception as e:
                close_error = e
                logger.warning(f"UoW关闭数据库会话失败: {e}", exc_info=True)
            finally:
                # 无条件复位UoW状态，避免后续误复用当前实例。
                if self.db_session is not None:
                    self.db_session.info.pop("post_commit_hooks", None)
                    self.db_session.info.pop("session_list_changed_ids", None)
                self.db_session = None
                self._entered = False

        # 有原始业务异常时，保持上下文管理器默认行为：继续抛原异常。
        if exc_type is not None:
            if tx_error is not None:
                logger.warning(f"业务异常触发回滚时发生附加错误: {tx_error}")
            if close_error is not None:
                logger.warning(f"业务异常触发退出时发生关闭错误: {close_error}")
            return False

        # 无业务异常时，优先抛事务错误，其次抛关闭错误。
        if tx_error is not None:
            raise tx_error
        if close_error is not None:
            raise close_error

        return False
