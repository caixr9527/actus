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

from app.application.errors import AppException, NotFoundError
from app.application.errors import error_keys
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

# 输出流读取阻塞时间（毫秒）。使用阻塞读取避免空闲时 busy polling 占满 CPU。
OUTPUT_STREAM_BLOCK_MS = 1000


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
        # 获取沙箱实例。
        # 注意：这里不再在“创建沙箱后立即单独落库”，而是统一在任务创建完成后一次性写回。
        # 这样可以把sandbox_id/task_id收敛为同一事务边界，减少跨阶段窗口。
        sandbox = None
        sandbox_id = session.sandbox_id
        created_new_sandbox = False
        if sandbox_id:
            sandbox = await self._sandbox_cls.get(id=sandbox_id)

        # 如果没有沙箱，则创建新的沙箱。
        # 这里先只在内存中持有，等任务也创建成功后再统一写回会话。
        if not sandbox:
            sandbox = await self._sandbox_cls.create()
            created_new_sandbox = True

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
            user_id=session.user_id,
            uow_factory=self._uow_factory,
            file_storage=self._file_storage,
            json_parser=self._json_parser,
            browser=browser,
            search_engine=self._search_engine,
            sandbox=sandbox,
        )

        # 创建任务并关联到会话
        task = self._task_cls.create(task_runner=task_runner)

        # 记录旧值，用于异常时回滚内存态与数据库态。
        previous_sandbox_id = session.sandbox_id
        previous_task_id = session.task_id

        # 统一写回两个跨资源关联字段，确保会话视图一次提交完成。
        session.sandbox_id = sandbox.id
        session.task_id = task.id
        try:
            async with self._uow_factory() as uow:
                await uow.session.save(session=session)
            return task
        except Exception as save_err:
            logger.error(f"会话{session.id}写回sandbox/task关联失败，开始补偿: {save_err}")

            # 补偿1：先从任务注册表移除刚创建的任务，避免无主任务残留。
            try:
                task.cancel()
            except Exception as cancel_err:
                logger.error(f"会话{session.id}补偿取消任务失败: {cancel_err}")

            # 补偿2：若本次新建了沙箱，写回失败时立即销毁，避免外部资源泄露。
            if created_new_sandbox:
                try:
                    await sandbox.destroy()
                except Exception as destroy_err:
                    logger.error(f"会话{session.id}补偿销毁沙箱失败: {destroy_err}")

            # 补偿3：回滚会话对象状态，并尽力写回数据库，避免出现悬挂关联字段。
            session.sandbox_id = previous_sandbox_id
            session.task_id = previous_task_id
            try:
                async with self._uow_factory() as uow:
                    await uow.session.save(session=session)
            except Exception as rollback_err:
                logger.error(f"会话{session.id}补偿回滚关联字段失败: {rollback_err}")

            raise

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

    async def _repair_output_event_history(self, session_id: str, event: Event) -> None:
        """基于输出流事件对会话历史做幂等修复

        设计目的：
        1. 覆盖“Redis已写成功，但DB写入/补偿未完成”的崩溃窗口。
        2. 使用事件ID做幂等保护，避免重复消费导致事件历史重复。
        """
        try:
            async with self._uow_factory() as uow:
                await uow.session.add_event_if_absent(session_id=session_id, event=event)
        except Exception as e:
            # 修复失败不影响当前SSE消息继续下发，避免用户流式体验被阻断。
            logger.warning(f"会话{session_id}输出流事件历史修复失败: {e}")

    async def chat(
            self,
            session_id: str,
            user_id: str,
            message: Optional[str] = None,
            attachments: Optional[List[str]] = None,
            latest_event_id: Optional[str] = None,
            timestamp: Optional[datetime] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        try:
            # 获取会话信息
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id=session_id, user_id=user_id)
            if not session:
                logger.error(f"会话{session_id}不存在")
                raise NotFoundError(
                    msg=f"会话{session_id}不存在",
                    error_key=error_keys.SESSION_NOT_FOUND,
                    error_params={"session_id": session_id},
                )

            # 获取当前会话的任务实例
            task = await self._get_task(session)

            # 处理用户发送的消息
            if message:
                # 统一归一化可选参数，避免后续列表处理触发None错误。
                attachments = attachments or []

                # 如果会话未处于运行状态，或者没有任务，则创建新任务
                if session.status != SessionStatus.RUNNING or task is None:
                    task = await self._create_task(session)
                    if not task:
                        logger.error(f"会话{session_id}的聊天请求失败: 创建任务失败")
                        raise RuntimeError(f"会话{session_id}的聊天请求失败: 创建任务失败")

                # 先查询附件元数据，构建完整消息事件。
                # 该步骤为只读，不涉及会话投影写入，因此不会产生“先写latest_message”的窗口。
                async with self._uow_factory() as uow:
                    db_attachments = []
                    for file_id in attachments:
                        attachment = await uow.file.get_by_id_and_user_id(file_id=file_id, user_id=user_id)
                        if attachment is not None:
                            db_attachments.append(attachment)

                # 创建用户消息事件
                message_event = MessageEvent(
                    role="user",
                    message=message,
                    attachments=[attachment for attachment in db_attachments if attachment is not None],
                )

                # 阶段2：输入流投递后，在单事务中同步“事件历史 + latest_message投影”。
                # 这里将两项数据库写操作合并，避免“最新消息已更新但事件未落库”的拆分问题。
                event_id = await task.input_stream.put(message_event.model_dump_json())
                message_event.id = event_id
                try:
                    async with self._uow_factory() as uow:
                        await uow.session.add_event_with_snapshot_if_absent(
                            session_id=session_id,
                            event=message_event,
                            latest_message=message,
                            latest_message_at=timestamp or datetime.now(),
                        )
                except Exception as add_err:
                    logger.error(f"会话{session_id}保存用户事件失败，开始补偿输入流消息: {add_err}")
                    try:
                        await task.input_stream.delete_message(event_id)
                    except Exception as compensate_err:
                        logger.error(f"会话{session_id}输入流补偿删除失败: {compensate_err}")
                    raise

                yield message_event

                # 启动任务执行
                await task.invoke()

                logger.info(f"会话{session_id},输入消息队列写入消息: {message[:50]}...")

            logger.info(f"会话{session_id},已启动,任务实例: {task}")

            # 持续监听任务输出流，直到任务完成
            while task and not task.done:
                # 从输出流获取下一个事件
                event_id, event_str = await task.output_stream.get(
                    start_id=latest_event_id,
                    block_ms=OUTPUT_STREAM_BLOCK_MS,
                )
                # 仅在成功读取到事件ID时推进游标，避免空读将游标重置为None导致重复回放首条消息。
                if event_id is not None:
                    latest_event_id = event_id
                if event_str is None:
                    logger.debug(f"会话{session_id},输出队列中未发现事件内容")
                    continue

                # 解析事件数据
                event = TypeAdapter(Event).validate_json(event_str)
                event.id = event_id
                logger.debug(f"会话{session_id},输出队列中已发现事件: {type(event).__name__}")

                # 消费输出流时做一次幂等补写，用于修复“消息入流成功但DB历史缺失”的情况。
                await self._repair_output_event_history(session_id=session_id, event=event)

                # 重置未读消息计数
                async with self._uow_factory() as uow:
                    await uow.session.update_unread_message_count(session_id=session_id, count=0)

                # 返回事件给调用方
                yield event

                # 如果遇到终止类型的事件，则退出循环
                if isinstance(event, (DoneEvent, ErrorEvent, WaitEvent)):
                    break

            logger.info(f"会话{session_id},任务本轮运行结束")
        except Exception as e:
            # 处理异常情况，记录错误日志并返回错误事件
            logger.error(f"会话{session_id}的聊天请求失败: {e}")
            event = ErrorEvent(
                error=str(e),
                error_key=e.error_key if isinstance(e, AppException) else None,
                error_params=e.error_params if isinstance(e, AppException) else None,
            )
            try:
                async with self._uow_factory() as uow:
                    await uow.session.add_event(session_id, event)
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

    async def stop_session(self, session_id: str, user_id: str) -> None:
        # 获取指定会话的信息
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id=session_id, user_id=user_id)
        # 如果会话不存在，记录错误日志并抛出异常
        if not session:
            logger.error(f"会话{session_id}不存在")
            raise NotFoundError(
                msg=f"会话{session_id}不存在",
                error_key=error_keys.SESSION_NOT_FOUND,
                error_params={"session_id": session_id},
            )

        # 获取与会话关联的任务实例
        task = await self._get_task(session)
        # 如果任务存在，则取消该任务
        if task:
            task.cancel()

        # 更新会话状态为已完成
        async with self._uow_factory() as uow:
            await uow.session.update_status(session_id=session_id, status=SessionStatus.COMPLETED)

    async def shutdown(self) -> None:
        """关闭会话服务"""
        logger.info("关闭会话服务并释放资源")
        await self._task_cls.destroy()
        logger.info("会话服务已关闭")
