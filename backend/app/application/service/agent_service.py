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
from typing import AsyncGenerator, Optional, List, Type, Callable, Any

from pydantic import TypeAdapter

from app.application.errors import AppException, BadRequestError, NotFoundError
from app.application.errors import error_keys
from app.domain.external import Task, Sandbox, LLM, JSONParser, SearchEngine, FileStorage, Browser
from app.domain.models import (
    BaseEvent,
    ErrorEvent,
    SessionStatus,
    AgentConfig,
    MCPConfig,
    A2AConfig,
    Session,
    MessageEvent,
    ResumeInput,
    Event,
    DoneEvent,
    WaitEvent,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.runtime import RunEngine, GraphRuntime, DefaultGraphRuntime
from app.infrastructure.runtime import LangGraphRunEngine, get_langgraph_checkpointer

logger = logging.getLogger(__name__)

# 输出流读取阻塞时间（毫秒）。使用阻塞读取避免空闲时 busy polling 占满 CPU。
OUTPUT_STREAM_BLOCK_MS = 1000


class AgentService:

    def __init__(
            self,
            agent_config: AgentConfig,
            mcp_config: MCPConfig,
            a2a_config: A2AConfig,
            sandbox_cls: Type[Sandbox],
            task_cls: Type[Task],
            json_parser: JSONParser,
            search_engine: SearchEngine,
            file_storage: FileStorage,
            uow_factory: Callable[[], IUnitOfWork],
            model_runtime_resolver=None,
            llm_factory=None,
            run_engine_factory: Optional[Callable[..., RunEngine]] = None,
            graph_runtime: Optional[GraphRuntime] = None,
    ) -> None:
        self._sandbox_cls = sandbox_cls
        self._task_cls = task_cls
        self._json_parser = json_parser
        self._search_engine = search_engine
        self._file_storage = file_storage
        self._uow_factory = uow_factory
        self._mcp_config = mcp_config
        self._model_runtime_resolver = model_runtime_resolver
        self._llm_factory = llm_factory
        self._agent_config = agent_config
        self._a2a_config = a2a_config
        self._run_engine_factory = run_engine_factory
        # BE-LG-07：将任务实例生命周期访问点统一收口到 GraphRuntime。
        # 这样 AgentService 只保留 facade 职责，不再直接操作 task registry。
        self._graph_runtime = graph_runtime or DefaultGraphRuntime(
            sandbox_cls=self._sandbox_cls,
            task_cls=self._task_cls,
            uow_factory=self._uow_factory,
            task_runner_factory=self._build_task_runner,
        )
        logger.info(f"初始化会话服务: {self.__class__.__name__}")

    def _build_task_runner(
            self,
            session: Session,
            llm: LLM,
            sandbox: Sandbox,
            browser: Browser,
    ) -> AgentTaskRunner:
        """构建任务执行器，供 GraphRuntime 在创建任务时回调。"""
        return AgentTaskRunner(
            llm=llm,
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
            run_engine_factory=self._run_engine_factory,
        )

    async def _get_task(self, session: Session) -> Optional[Task]:
        """读取会话任务实例，优先走 GraphRuntime，保留历史回退以兼容旧测试。"""
        runtime = getattr(self, "_graph_runtime", None)
        if runtime is not None:
            return await runtime.get_task(session=session)

        task_id = session.task_id
        if not task_id:
            return None
        task_cls = getattr(self, "_task_cls", None)
        if task_cls is None:
            return None
        return task_cls.get(task_id=task_id)

    async def _resolve_runtime_llm(self, session: Session) -> LLM:
        """根据会话当前模型解析运行时 LLM。"""
        model_runtime_resolver = getattr(self, "_model_runtime_resolver", None)
        llm_factory = getattr(self, "_llm_factory", None)
        if model_runtime_resolver is not None and llm_factory is not None:
            resolved_model_id, llm_config = await model_runtime_resolver.resolve(session)
            logger.info(
                f"会话{session.id}运行时模型解析完成: requested={session.current_model_id}, resolved={resolved_model_id}")
            return llm_factory.create(llm_config)

        raise RuntimeError("未配置运行时LLM解析能力")

    async def _create_task(
            self,
            session: Session,
            *,
            reuse_current_run: bool = False,
    ) -> Task:
        """创建或恢复会话任务。"""
        llm = await self._resolve_runtime_llm(session)
        runtime = getattr(self, "_graph_runtime", None)
        if runtime is None:
            raise RuntimeError("未配置GraphRuntime，无法创建会话任务")
        if reuse_current_run:
            return await runtime.resume_task(session=session, llm=llm)
        return await runtime.create_task(session=session, llm=llm)

    async def _safe_update_unread_count(self, session_id: str) -> None:
        """在独立的后台任务中安全地更新未读消息计数

        该方法通过asyncio.create_task()调用，运行在一个全新的asyncio Task中，
        因此不受sse_starlette的anyio cancel scope影响，数据库操作可以正常完成。
        使用uow_factory创建全新的UoW实例，避免与被取消的上下文共享数据库连接。
        """
        try:
            uow = self._uow_factory()
            async with uow:
                session = await uow.session.get_by_id(session_id=session_id)
                if session is None:
                    return
                # 已经是0时直接短路，避免无效写入。
                if int(getattr(session, "unread_message_count", 0) or 0) <= 0:
                    return
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

    async def _inspect_resume_checkpoint(self, session: Session) -> Any:
        """读取当前等待态对应的 checkpoint，确认恢复点仍然有效。"""
        llm = await self._resolve_runtime_llm(session)
        inspector = LangGraphRunEngine(
            session_id=session.id,
            llm=llm,
            user_id=session.user_id,
            uow_factory=self._uow_factory,
            checkpointer=get_langgraph_checkpointer().get_checkpointer(),
        )
        return await inspector.inspect_resume_checkpoint()

    async def _ensure_resume_checkpoint_available(self, session: Session) -> None:
        """在接受 resume 前，先确认等待态对应的 checkpoint 仍可恢复。"""
        try:
            inspection = await self._inspect_resume_checkpoint(session)
        except Exception as exc:
            logger.warning("会话[%s]读取恢复点失败，拒绝 resume: %s", session.id, exc)
            raise BadRequestError(
                msg="当前等待点已失效或无法读取，请重新发起任务",
                error_key=error_keys.SESSION_RESUME_CHECKPOINT_INVALID,
                error_params={"session_id": session.id},
            ) from exc

        if inspection.is_resumable:
            return

        logger.warning(
            "会话[%s]恢复点不可用: run_id=%s has_checkpoint=%s pending_interrupt=%s",
            session.id,
            inspection.run_id,
            inspection.has_checkpoint,
            bool(inspection.pending_interrupt),
        )
        raise BadRequestError(
            msg="当前等待点已失效或无法读取，请重新发起任务",
            error_key=error_keys.SESSION_RESUME_CHECKPOINT_INVALID,
            error_params={"session_id": session.id},
        )

    async def chat(
            self,
            session_id: str,
            user_id: str,
            message: Optional[str] = None,
            attachments: Optional[List[str]] = None,
            resume: Any = None,
            latest_event_id: Optional[str] = None,
            timestamp: Optional[datetime] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        unread_reset_pending = False
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
            # Context: 高频事件流会导致“每条事件重置未读数”的写放大。
            # Decision: 未读数只在首条输出事件重置一次，finally 仅在 pending 时兜底。
            # Trade-off: 状态收敛略依赖流程标志位，但显著降低无效写入。
            # Removal Plan: 若后续引入独立未读服务，可迁移为事件驱动批量收敛。
            unread_reset_pending = int(getattr(session, "unread_message_count", 0) or 0) > 0

            # 获取当前会话的任务实例
            task = await self._get_task(session)
            is_resume_request = resume is not None
            is_message_request = bool(str(message or "").strip())

            if session.status == SessionStatus.WAITING and is_message_request and not is_resume_request:
                raise BadRequestError(
                    msg="当前会话处于等待状态，请使用 resume 恢复执行",
                    error_key=error_keys.SESSION_RESUME_REQUIRED,
                    error_params={"session_id": session.id},
                )
            if is_resume_request and session.status != SessionStatus.WAITING:
                raise BadRequestError(
                    msg="当前会话不处于等待状态，无法恢复执行",
                    error_key=error_keys.SESSION_NOT_WAITING,
                    error_params={"session_id": session.id},
                )
            if is_resume_request:
                await self._ensure_resume_checkpoint_available(session)

            # 处理用户发送的消息
            if is_message_request:
                # 统一归一化可选参数，避免后续列表处理触发None错误。
                attachments = attachments or []

                # 如果会话未处于运行状态，或者没有任务，则创建/恢复任务。
                if task is None:
                    task = await self._create_task(
                        session,
                        reuse_current_run=(
                                session.status == SessionStatus.RUNNING
                                and bool(str(session.current_run_id or "").strip())
                        ),
                    )
                    if not task:
                        logger.error(f"会话{session_id}的聊天请求失败: 创建任务失败")
                        raise RuntimeError(f"会话{session_id}的聊天请求失败: 创建任务失败")
                elif session.status != SessionStatus.RUNNING:
                    task = await self._create_task(session, reuse_current_run=False)
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
                        else:
                            raise NotFoundError(
                                msg=f"该文件[{file_id}]不存在",
                                error_key=error_keys.FILE_NOT_FOUND,
                                error_params={"file_id": file_id},
                            )

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
                            # 用户发起新一轮输入后立即置为 RUNNING，避免前端切换会话后丢失运行态。
                            status=SessionStatus.RUNNING,
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
            elif is_resume_request:
                if task is None:
                    task = await self._create_task(session, reuse_current_run=True)
                    if not task:
                        logger.error(f"会话{session_id}的恢复请求失败: 创建恢复任务失败")
                        raise RuntimeError(f"会话{session_id}的恢复请求失败: 创建恢复任务失败")

                event_id = await task.input_stream.put(ResumeInput(value=resume).model_dump_json())
                try:
                    async with self._uow_factory() as uow:
                        await uow.session.update_status(session_id=session_id, status=SessionStatus.RUNNING)
                except Exception as update_err:
                    logger.error(f"会话{session_id}更新恢复状态失败，开始补偿输入流消息: {update_err}")
                    try:
                        await task.input_stream.delete_message(event_id)
                    except Exception as compensate_err:
                        logger.error(f"会话{session_id}恢复输入流补偿删除失败: {compensate_err}")
                    raise

                await task.invoke()
                logger.info("会话%s, 已写入恢复请求并启动执行", session_id)

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

                # 首条事件到达时重置一次未读数，后续事件不再重复写入。
                if unread_reset_pending:
                    async with self._uow_factory() as uow:
                        await uow.session.update_unread_message_count(session_id=session_id, count=0)
                    unread_reset_pending = False

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
            if unread_reset_pending:
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
        runtime = getattr(self, "_graph_runtime", None)
        if runtime is not None:
            await runtime.destroy()
        else:
            # 兼容历史测试中 object.__new__ 未注入 GraphRuntime 的构造方式。
            await self._task_cls.destroy()
        logger.info("会话服务已关闭")
