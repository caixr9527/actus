#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/2/8 15:43
@Author : caixiaorong01@outlook.com
@File   : session_service.py
"""
import logging
from typing import List, Callable, Type

from app.application.errors import NotFoundError, ServerError
from app.domain.external import Sandbox
from app.domain.models import Session, File
from app.domain.repositories import IUnitOfWork
from app.interfaces.schemas import FileReadResponse, ShellReadResponse

logger = logging.getLogger(__name__)


class SessionService:
    """会话服务"""

    def __init__(self, uow_factory: Callable[[], IUnitOfWork], sandbox_cls: Type[Sandbox]) -> None:
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._sandbox_cls = sandbox_cls

    async def create_session(self) -> Session:
        logger.info("创建任务会话")
        session = Session(title="新会话")
        async with self._uow:
            await self._uow.session.save(session)
        logger.info(f"创建任务会话成功: {session.id}")
        return session

    async def get_all_sessions(self) -> List[Session]:
        async with self._uow:
            return await self._uow.session.get_all()

    async def clear_unread_message_count(self, session_id: str) -> None:
        logger.info(f"清除任务会话未读消息数: {session_id}")
        async with self._uow:
            await self._uow.session.update_unread_message_count(session_id=session_id, count=0)

    async def delete_session(self, session_id: str) -> None:
        logger.info(f"删除任务会话: {session_id}")
        async with self._uow:
            session = await self._uow.session.get_by_id(session_id=session_id)
        if not session:
            logger.error(f"任务会话不存在: {session_id}")
            raise NotFoundError(msg=f"任务会话不存在: {session_id}")
        async with self._uow:
            await self._uow.session.delete_by_id(session_id=session_id)
        logger.info(f"删除任务会话成功: {session_id}")

    async def get_session(self, session_id: str) -> Session:
        async with self._uow:
            return await self._uow.session.get_by_id(session_id=session_id)

    async def get_session_files(self, session_id: str) -> List[File]:
        logger.info(f"获取任务会话文件列表: {session_id}")
        async with self._uow:
            session = await self._uow.session.get_by_id(session_id=session_id)
        if not session:
            logger.error(f"任务会话不存在: {session_id}")
            raise RuntimeError(f"任务会话不存在: {session_id}")
        return session.files

    async def read_file(self, session_id: str, filepath: str) -> FileReadResponse:
        logger.info(f"获取会话：{session_id} 中文件路径：{filepath} 的内容")
        # 获取指定会话信息
        async with self._uow:
            session = await self._uow.session.get_by_id(session_id=session_id)
        if not session:
            logger.error(f"任务会话不存在: {session_id}")
            raise RuntimeError(f"任务会话不存在: {session_id}")

        # 检查会话是否关联了沙盒
        if not session.sandbox_id:
            raise NotFoundError(msg="任务会话未关联沙盒")

        # 根据沙盒ID获取沙盒实例
        sandbox = await self._sandbox_cls.get(id=session.sandbox_id)
        if not sandbox:
            raise NotFoundError(msg="任务会话未关联沙盒或已销毁")

        # 调用沙盒读取文件方法
        result = await sandbox.read_file(file_path=filepath)
        if result.success:
            # 返回文件读取结果
            return FileReadResponse(**result.data)

        # 文件读取失败，抛出服务器错误
        raise ServerError(msg=result.message)

    async def read_shell_output(self, session_id: str, shell_session_id: str) -> ShellReadResponse:
        logger.info(f"获取会话：{session_id} 中Shell会话ID：{shell_session_id} 的输出")
        # 获取指定会话信息
        async with self._uow:
            session = await self._uow.session.get_by_id(session_id=session_id)
        if not session:
            logger.error(f"任务会话不存在: {session_id}")
            raise RuntimeError(f"任务会话不存在: {session_id}")

        # 检查会话是否关联了沙盒
        if not session.sandbox_id:
            raise NotFoundError(msg="任务会话未关联沙盒")

        # 根据沙盒ID获取沙盒实例
        sandbox = await self._sandbox_cls.get(id=session.sandbox_id)
        if not sandbox:
            raise NotFoundError(msg="任务会话未关联沙盒或已销毁")
        result = await sandbox.read_shell_output(session_id=shell_session_id, console=True)
        if result.success:
            # 读取成功，返回结果
            return ShellReadResponse(**result.data)
        raise ServerError(msg=result.message)

    async def get_vnc_url(self, session_id: str) -> str:
        logger.info(f"获取会话：{session_id} 的VNC地址")
        # 获取指定会话信息
        async with self._uow:
            session = await self._uow.session.get_by_id(session_id=session_id)
        if not session:
            logger.error(f"任务会话不存在: {session_id}")
            raise RuntimeError(f"任务会话不存在: {session_id}")

        # 检查会话是否关联了沙盒
        if not session.sandbox_id:
            raise NotFoundError(msg="任务会话未关联沙盒")

        # 根据沙盒ID获取沙盒实例
        sandbox = await self._sandbox_cls.get(id=session.sandbox_id)
        if not sandbox:
            raise NotFoundError(msg="任务会话未关联沙盒或已销毁")

        return sandbox.vnc_url
