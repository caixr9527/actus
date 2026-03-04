#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/2/8 16:16
@Author : caixiaorong01@outlook.com
@File   : agent_service.py
"""
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, List, Type, Callable

from pydantic import TypeAdapter

from app.domain.external import Task, Sandbox, LLM, JSONParser, SearchEngine, FileStorage
from app.domain.models import (
    BaseEvent,
    ErrorEvent,
    SessionStatus,
    AgentConfig,
    MCPConfig,
    A2AConfig,
    Session,
    MessageEvent,
    Event,
    DoneEvent,
    WaitEvent
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.agent_task_runner import AgentTaskRunner

logger = logging.getLogger(__name__)


class AgentService:

    def __init__(
            self,
            llm: LLM,
            agent_config: AgentConfig,
            mcp_config: MCPConfig,
            a2a_config: A2AConfig,
            sandbox_cls: Type[Sandbox],
            task_cls: Type[Task],
            json_parser: JSONParser,
            search_engine: SearchEngine,
            file_storage: FileStorage,
            uow_factory: Callable[[], IUnitOfWork]
    ) -> None:
        self._sandbox_cls = sandbox_cls
        self._task_cls = task_cls
        self._json_parser = json_parser
        self._search_engine = search_engine
        self._file_storage = file_storage
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._mcp_config = mcp_config
        self._llm = llm
        self._agent_config = agent_config
        self._a2a_config = a2a_config
        logger.info(f"初始化会话服务: {self.__class__.__name__}")

    async def _get_task(self, session) -> Optional[Task]:
        task_id = session.task_id
        if not task_id:
            return None

        return self._task_cls.get(task_id=task_id)

    async def _create_task(self, session: Session) -> Task:
        # 获取沙箱实例
        sandbox = None
        sandbox_id = session.sandbox_id
        if sandbox_id:
            sandbox = await self._sandbox_cls.get(id=sandbox_id)

        # 如果没有沙箱，则创建新的沙箱并保存到会话中
        if not sandbox:
            sandbox = await self._sandbox_cls.create()
            session.sandbox_id = sandbox.id
            async with self._uow:
                await self._uow.session.save(session=session)

        # 获取沙箱中的浏览器实例
        browser = await sandbox.get_browser()
        if not browser:
            logger.error(f"会话{session.id}的聊天请求失败: 沙箱{sandbox_id},创建浏览器失败")
            raise RuntimeError(f"会话{session.id}的聊天请求失败: 沙箱{sandbox_id},创建浏览器失败")

        # 创建任务运行器
        task_runner = AgentTaskRunner(
            llm=self._llm,
            agent_config=self._agent_config,
            mcp_config=self._mcp_config,
            a2a_config=self._a2a_config,
            session_id=session.id,
            uow_factory=self._uow_factory,
            file_storage=self._file_storage,
            json_parser=self._json_parser,
            browser=browser,
            search_engine=self._search_engine,
            sandbox=sandbox,
        )

        # 创建任务并关联到会话
        task = self._task_cls.create(task_runner=task_runner)

        session.task_id = task.id
        async with self._uow:
            await self._uow.session.save(session=session)
        return task

    async def _safe_update_unread_count(self, session_id: str) -> None:
        """在独立的后台任务中安全地更新未读消息计数

        该方法通过asyncio.create_task()调用，运行在一个全新的asyncio Task中，
        因此不受sse_starlette的anyio cancel scope影响，数据库操作可以正常完成。
        使用uow_factory创建全新的UoW实例，避免与被取消的上下文共享数据库连接。
        """
        try:
            uow = self._uow_factory()
            async with uow:
                await uow.session.update_unread_message_count(session_id, 0)
        except Exception as e:
            logger.warning(f"会话[{session_id}]后台更新未读消息计数失败: {e}")

    async def chat(
            self,
            session_id: str,
            message: Optional[str] = None,
            attachments: Optional[List[str]] = None,
            latest_event_id: Optional[str] = None,
            timestamp: Optional[datetime] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        try:
            # 获取会话信息
            async with self._uow:
                session = await self._uow.session.get_by_id(session_id=session_id)
            if not session:
                logger.error(f"会话{session_id}不存在")
                raise RuntimeError(f"会话{session_id}不存在")

            # 获取当前会话的任务实例
            task = await self._get_task(session)

            # 处理用户发送的消息
            if message:
                # 如果会话未处于运行状态，或者没有任务，则创建新任务
                if session.status != SessionStatus.RUNNING or task is None:
                    task = await self._create_task(session)
                    if not task:
                        logger.error(f"会话{session_id}的聊天请求失败: 创建任务失败")
                        raise RuntimeError(f"会话{session_id}的聊天请求失败: 创建任务失败")

                # 更新会话的最新消息
                async with self._uow:
                    await self._uow.session.update_latest_message(
                        session_id=session_id,
                        message=message,
                        timestamp=timestamp or datetime.now(),
                    )

                # 从文件数据库中查询数据并更新attachments实际内容, 并返回人类消息事件
                async with self._uow:
                    db_attachments = [await self._uow.file.get_by_id(id) for id in attachments]

                # 创建用户消息事件
                message_event = MessageEvent(
                    role="user",
                    message=message,
                    attachments=[attachment for attachment in db_attachments if attachment is not None],
                )

                # 将消息事件放入任务输入流
                event_id = await task.input_stream.put(message_event.model_dump_json())
                message_event.id = event_id

                yield message_event
                # 将消息事件保存到会话历史中
                async with self._uow:
                    await self._uow.session.add_event(session_id=session_id, event=message_event)

                # 启动任务执行
                await task.invoke()

                logger.info(f"会话{session_id},输入消息队列写入消息: {message[:50]}...")

            logger.info(f"会话{session_id},已启动,任务实例: {task}")

            # 持续监听任务输出流，直到任务完成
            while task and not task.done:
                # 从输出流获取下一个事件
                event_id, event_str = await task.output_stream.get(start_id=latest_event_id, block_ms=0)
                latest_event_id = event_id
                if event_str is None:
                    logger.debug(f"会话{session_id},输出队列中未发现事件内容")
                    continue

                # 解析事件数据
                event = TypeAdapter(Event).validate_json(event_str)
                event.id = event_id
                logger.debug(f"会话{session_id},输出队列中已发现事件: {type(event).__name__}")

                # 重置未读消息计数
                async with self._uow:
                    await self._uow.session.update_unread_message_count(session_id=session_id, count=0)

                # 返回事件给调用方
                yield event

                # 如果遇到终止类型的事件，则退出循环
                if isinstance(event, (DoneEvent, ErrorEvent, WaitEvent)):
                    break

            logger.info(f"会话{session_id},任务本轮运行结束")
        except Exception as e:
            # 处理异常情况，记录错误日志并返回错误事件
            logger.error(f"会话{session_id}的聊天请求失败: {e}")
            event = ErrorEvent(error=str(e))
            try:
                async with self._uow:
                    await self._uow.session.add_event(session_id, event)
            except (asyncio.CancelledError, Exception) as add_err:
                logger.error(f"会话{session_id}的聊天请求失败,添加错误事件失败(可能是客户端断开连接): {add_err}")
            yield event
        finally:
            # 确保最终重置未读消息计数
            # 会话完整传递给前端后，表示至少用户肯定收到了这些消息，所以不应该有未读消息数
            # 注意：当SSE客户端断开连接时，sse_starlette使用anyio cancel scope取消当前Task中
            # 所有的await操作（asyncio.shield也无法对抗anyio的cancel scope）。
            # 如果在finally块中直接执行数据库操作，该操作会被立即取消，并且SQLAlchemy在尝试
            # 终止被中断的连接时也会被取消，从而产生ERROR日志并可能污染连接池。
            # 解决方案：将数据库更新操作放到独立的asyncio Task中执行，新Task不受当前
            # cancel scope的影响，可以正常完成数据库操作。
            try:
                await asyncio.create_task(self._safe_update_unread_count(session_id))
            except RuntimeError:
                # 事件循环已关闭（如应用正在关闭），无法创建后台任务
                logger.warning(f"会话[{session_id}]无法创建后台任务更新未读消息计数")

    async def stop_session(self, session_id: str) -> None:
        # 获取指定会话的信息
        async with self._uow:
            session = await self._uow.session.get_by_id(session_id=session_id)
        # 如果会话不存在，记录错误日志并抛出异常
        if not session:
            logger.error(f"会话{session_id}不存在")
            raise RuntimeError(f"会话{session_id}不存在")

        # 获取与会话关联的任务实例
        task = await self._get_task(session)
        # 如果任务存在，则取消该任务
        if task:
            task.cancel()

        # 更新会话状态为已完成
        async with self._uow:
            await self._uow.session.update_status(session_id=session_id, status=SessionStatus.COMPLETED)

    async def shutdown(self) -> None:
        """关闭会话服务"""
        logger.info("关闭会话服务并释放资源")
        await self._task_cls.destroy()
        logger.info("会话服务已关闭")
