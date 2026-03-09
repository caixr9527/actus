#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/12 17:02
@Author : caixiaorong01@outlook.com
@File   : postgres.py
"""
import logging
from functools import lru_cache
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.domain.repositories import IUnitOfWork
from app.infrastructure.repositories import DBUnitOfWork
from core.config import get_settings

logger = logging.getLogger(__name__)


class Postgres:
    """Postgres数据库基础类，用于完成数据库连接等配置操作"""

    def __init__(self):
        """构造函数，完成postgres数据库引擎、会话工厂的创建"""
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._settings = get_settings()

    async def init(self) -> None:
        """初始化postgres连接"""
        # 判断是否已经创建好引擎，如果连上了则中断程序
        if self._engine is not None:
            logger.warning(f"Postgres引擎已初始化")
            return

        try:
            # 创建异步引擎
            logger.info("正在初始化Postgres连接...")
            # SQLAlchemy Engine 的 echo 会直接触发 SQL 输出，按统一日志模式控制：
            # - all: 始终开启
            # - 白名单前缀模式: 仅当包含 sqlalchemy.engine.Engine 时开启
            enable_engine_echo = (
                self._settings.is_log_output_all
                or self._settings.is_logger_allowed_by_output_mode("sqlalchemy.engine.Engine")
            )
            self._engine = create_async_engine(
                self._settings.sqlalchemy_database_uri,
                echo=enable_engine_echo,
                pool_pre_ping=True,  # 启用连接池预检，检查数据库连接是否正常
            )

            # 创建会话工厂
            self._session_factory = async_sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine,
            )
            logger.info("Postgres会话工厂创建完毕")

            # 连接Postgres并执行预操作
            async with self._engine.begin() as async_conn:
                # 检查是否安装了uuid扩展，如果没有的话则安装
                await async_conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
                logger.info("成功连接Postgres并安装uuid-ossp扩展")
        except Exception as e:
            logger.error(f"连接Postgres失败: {e}")
            raise

    async def close(self) -> None:
        """关闭Postgres连接"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("成功关闭Postgres连接")

        get_postgres.cache_clear()

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """只读属性，返回已初始化的会话工厂"""
        if self._session_factory is None:
            raise RuntimeError("Postgres未初始化，请先调用init()函数初始化")
        return self._session_factory


@lru_cache()
def get_postgres() -> Postgres:
    """使用lru_cache实现单例模式，获取Postgres实例"""
    return Postgres()


async def get_db_session() -> AsyncSession:
    """FastAPI依赖项，用于在每个请求中异步获取数据库会话实例，确保会话在正确使用后被关闭"""
    # 获取引擎和会话工厂
    db = get_postgres()
    session_factory = db.session_factory

    # 创建会话上下文，在上下文内完成数据提交
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as _:
            await session.rollback()
            raise


def get_session_factory():
    db = get_postgres()
    session_factory = db.session_factory
    return session_factory


def get_uow() -> IUnitOfWork:
    return DBUnitOfWork(session_factory=get_session_factory())
