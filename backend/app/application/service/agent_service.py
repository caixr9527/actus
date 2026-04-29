#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/2/8 16:16
@Author : caixiaorong01@outlook.com
@File   : agent_service.py
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional, List, Type, Callable, Any

from pydantic import TypeAdapter

from app.application.errors import AppException, BadRequestError, NotFoundError
from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.application.service.runtime_state_coordinator import RuntimeStateCoordinator
from app.application.errors import error_keys
from app.domain.external import Task, Sandbox, LLM, JSONParser, SearchEngine, FileStorage, Browser
from app.domain.models import (
    BaseEvent,
    ErrorEvent,
    ExecutionStatus,
    SessionStatus,
    AgentConfig,
    MCPConfig,
    A2AConfig,
    Session,
    ContinueCancelledTaskInput,
    MessageEvent,
    ResumeInput,
    Event,
    DoneEvent,
    WaitEvent,
    validate_wait_resume_value,
    RuntimeInput,
    TaskStreamRecord,
    TaskStreamEventRecord,
    TaskRequestStartedRecord,
    TaskRequestFinishedRecord,
    TaskRequestRejectedRecord,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.runtime import RunEngine, GraphRuntime, DefaultGraphRuntime
from app.domain.services.runtime.contracts.event_delivery_policy import should_persist_event
from app.domain.services.runtime.contracts.data_access_contract import DataAccessAction
from app.domain.services.runtime.stage_llm import build_uniform_stage_llms
from app.domain.services.tools import CapabilityRegistry, ToolRuntimeAdapter
from app.domain.services.workspace_runtime import WorkspaceManager, WorkspaceRuntimeService
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph import LangGraphRunEngine, get_langgraph_checkpointer

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
            access_control_service: RuntimeAccessControlService | None = None,
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
        self._tool_runtime_adapter = ToolRuntimeAdapter(
            capability_registry=CapabilityRegistry.default_v1(),
        )
        self._workspace_manager = WorkspaceManager(uow_factory=self._uow_factory)
        self._runtime_state_coordinator = RuntimeStateCoordinator(uow_factory=self._uow_factory)
        self._access_control_service = access_control_service or RuntimeAccessControlService(
            uow_factory=self._uow_factory,
        )
        # BE-LG-07：将任务实例生命周期访问点统一收口到 GraphRuntime。
        # 这样 AgentService 只保留 facade 职责，不再直接操作 task registry。
        self._graph_runtime = graph_runtime or DefaultGraphRuntime(
            sandbox_cls=self._sandbox_cls,
            task_cls=self._task_cls,
            uow_factory=self._uow_factory,
            task_runner_factory=self._build_task_runner,
            runtime_state_coordinator=self._runtime_state_coordinator,
        )
        logger.info(f"初始化会话服务: {self.__class__.__name__}")

    def _get_workspace_manager(self) -> WorkspaceManager:
        manager = getattr(self, "_workspace_manager", None)
        if manager is None:
            manager = WorkspaceManager(uow_factory=self._uow_factory)
            self._workspace_manager = manager
        return manager

    def _get_runtime_state_coordinator(self) -> RuntimeStateCoordinator:
        coordinator = getattr(self, "_runtime_state_coordinator", None)
        if coordinator is None:
            coordinator = RuntimeStateCoordinator(uow_factory=self._uow_factory)
            self._runtime_state_coordinator = coordinator
        return coordinator

    def _get_access_control_service(self) -> RuntimeAccessControlService:
        access_control_service = getattr(self, "_access_control_service", None)
        if access_control_service is None:
            access_control_service = RuntimeAccessControlService(uow_factory=self._uow_factory)
            self._access_control_service = access_control_service
        return access_control_service

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
            tool_runtime_adapter=self._tool_runtime_adapter,
            runtime_state_coordinator=self._get_runtime_state_coordinator(),
        )

    async def _get_task(self, session: Session) -> Optional[Task]:
        """读取会话任务实例。"""
        runtime = getattr(self, "_graph_runtime", None)
        if runtime is None:
            # 历史单元测试会用 object.__new__ 构造服务，只注入 task_cls。
            return self._task_cls.get(session.id)
        return await runtime.get_task(session=session)

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
        if not should_persist_event(event):
            return
        try:
            await self._get_runtime_state_coordinator().persist_runtime_event(
                session_id=session_id,
                event=event,
            )
        except Exception as e:
            # 修复失败不影响当前SSE消息继续下发，避免用户流式体验被阻断。
            logger.warning(f"会话{session_id}输出流事件历史修复失败: {e}")

    async def _inspect_resume_checkpoint(self, session: Session) -> Any:
        """读取当前等待态对应的 checkpoint，确认恢复点仍然有效。"""
        llm = await self._resolve_runtime_llm(session)
        workspace_runtime_service = WorkspaceRuntimeService(
            session_id=session.id,
            uow_factory=self._uow_factory,
        )
        inspector = LangGraphRunEngine(
            session_id=session.id,
            stage_llms=build_uniform_stage_llms(llm),
            user_id=session.user_id,
            uow_factory=self._uow_factory,
            runtime_context_service=RuntimeContextService(
                workspace_runtime_service=workspace_runtime_service,
            ),
            checkpointer=get_langgraph_checkpointer().get_checkpointer(),
        )
        return await inspector.inspect_resume_checkpoint()

    async def _ensure_resume_checkpoint_available(self, session: Session) -> Any:
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
            return inspection

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

    @staticmethod
    def _ensure_resume_value_valid(*, session: Session, pending_interrupt: Any, resume_value: Any) -> None:
        """resume 值必须符合等待态契约，避免非法恢复值清掉等待点后继续执行。"""
        if validate_wait_resume_value(pending_interrupt, resume_value):
            return

        raise BadRequestError(
            msg="当前恢复输入与等待态要求不匹配，请按界面提示重新提交",
            error_key=error_keys.SESSION_RESUME_VALUE_INVALID,
            error_params={"session_id": session.id},
        )

    @staticmethod
    def _is_continue_cancelled_request(command: Optional[dict[str, Any]]) -> bool:
        if not isinstance(command, dict):
            return False
        return str(command.get("type") or "").strip() == "continue_cancelled_task"

    @staticmethod
    def _get_reopenable_cancelled_plan(session: Session) -> Optional[Any]:
        latest_plan = session.get_latest_plan()
        if latest_plan is None or latest_plan.status != ExecutionStatus.CANCELLED:
            return None
        if not any(step.status == ExecutionStatus.CANCELLED for step in list(latest_plan.steps or [])):
            return None
        return latest_plan

    @classmethod
    def _ensure_continue_cancelled_available(cls, session: Session) -> None:
        if session.status != SessionStatus.CANCELLED:
            raise BadRequestError(
                msg="当前会话不处于已取消状态，无法继续已取消任务",
                error_key=error_keys.SESSION_NOT_CANCELLED,
                error_params={"session_id": session.id},
            )

        if cls._get_reopenable_cancelled_plan(session) is not None:
            return

        raise BadRequestError(
            msg="当前已取消任务没有可继续的执行计划，请重新发起任务",
            error_key=error_keys.SESSION_CANCELLED_CONTINUE_UNAVAILABLE,
            error_params={"session_id": session.id},
        )

    @staticmethod
    def _reject_unresolved_runtime_conflict(*, session: Session, reconcile_result: Any) -> None:
        warnings = set(getattr(reconcile_result, "warnings", []) or [])
        snapshot = getattr(reconcile_result, "snapshot_after", None)
        if "session_status_mismatch_run_status" not in warnings:
            return

        session_status = getattr(snapshot, "session_status", session.status)
        if session_status == SessionStatus.WAITING:
            return

        raise BadRequestError(
            msg="当前运行状态不一致，请稍后重试",
            error_key=error_keys.SESSION_RUNTIME_STATE_CONFLICT,
            error_params={"session_id": session.id},
        )

    async def chat(
            self,
            session_id: str,
            user_id: str,
            message: Optional[str] = None,
            attachments: Optional[List[str]] = None,
            resume: Any = None,
            command: Optional[dict[str, Any]] = None,
            latest_event_id: Optional[str] = None,
            timestamp: Optional[datetime] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        unread_reset_pending = False
        request_id: Optional[str] = None
        try:
            await self._get_access_control_service().resolve_session_scope(
                user_id=user_id,
                session_id=session_id,
            )
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
            reconcile_result = await self._get_runtime_state_coordinator().reconcile_current_run(
                session_id=session_id,
                reason="before_chat",
            )
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id=session_id, user_id=user_id)
            if not session:
                logger.error(f"会话{session_id}不存在")
                raise NotFoundError(
                    msg=f"会话{session_id}不存在",
                    error_key=error_keys.SESSION_NOT_FOUND,
                    error_params={"session_id": session_id},
                )
            self._reject_unresolved_runtime_conflict(
                session=session,
                reconcile_result=reconcile_result,
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
            is_continue_cancelled_request = self._is_continue_cancelled_request(command)

            if session.status == SessionStatus.WAITING and is_message_request:
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
                inspection = await self._ensure_resume_checkpoint_available(session)
                self._ensure_resume_value_valid(
                    session=session,
                    pending_interrupt=inspection.pending_interrupt,
                    resume_value=resume,
                )
            if is_continue_cancelled_request:
                self._ensure_continue_cancelled_available(session)

            # 处理用户发送的消息
            if is_message_request:
                # 统一归一化可选参数，避免后续列表处理触发None错误。
                attachments = attachments or []

                # 如果会话未处于运行状态，或者没有任务，则创建/恢复任务。
                if task is None:
                    current_run_id = await self._get_workspace_manager().resolve_current_run_id(session=session)
                    task = await self._create_task(
                        session,
                        reuse_current_run=(
                                session.status == SessionStatus.RUNNING
                                and bool(current_run_id)
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
                        await self._get_access_control_service().assert_file_access(
                            user_id=user_id,
                            file_id=file_id,
                            action=DataAccessAction.READ,
                        )
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
                request_id = str(uuid.uuid4())
                runtime_input = RuntimeInput(
                    request_id=request_id,
                    payload=message_event,
                )

                # 阶段2：输入流投递后，在单事务中同步“事件历史 + latest_message投影”。
                # 这里将两项数据库写操作合并，避免“最新消息已更新但事件未落库”的拆分问题。
                event_id = await task.input_stream.put(runtime_input.model_dump_json())
                message_event.id = event_id
                try:
                    await self._get_runtime_state_coordinator().accept_user_message(
                        session_id=session_id,
                        event=message_event,
                        latest_message_at=timestamp or datetime.now(),
                        stream_event_id=event_id,
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

                request_id = str(uuid.uuid4())
                event_id = await task.input_stream.put(
                    RuntimeInput(
                        request_id=request_id,
                        payload=ResumeInput(value=resume),
                    ).model_dump_json()
                )
                try:
                    await self._get_runtime_state_coordinator().mark_resume_requested(
                        session_id=session_id,
                        request_id=request_id,
                        pending_interrupt=inspection.pending_interrupt,
                    )
                except Exception as update_err:
                    logger.error(f"会话{session_id}更新恢复状态失败，开始补偿输入流消息: {update_err}")
                    try:
                        await task.input_stream.delete_message(event_id)
                    except Exception as compensate_err:
                        logger.error(f"会话{session_id}恢复输入流补偿删除失败: {compensate_err}")
                    raise

                await task.invoke()
                logger.info("会话%s, 已写入恢复请求并启动执行", session_id)
            elif is_continue_cancelled_request:
                if task is None or session.status != SessionStatus.RUNNING:
                    task = await self._create_task(session, reuse_current_run=False)
                    if not task:
                        logger.error(f"会话{session_id}的继续取消任务请求失败: 创建任务失败")
                        raise RuntimeError(f"会话{session_id}的继续取消任务请求失败: 创建任务失败")

                request_id = str(uuid.uuid4())
                event_id = await task.input_stream.put(
                    RuntimeInput(
                        request_id=request_id,
                        payload=ContinueCancelledTaskInput(),
                    ).model_dump_json()
                )
                try:
                    await self._get_runtime_state_coordinator().mark_continue_cancelled_requested(
                        session_id=session_id,
                        request_id=request_id,
                    )
                except Exception as update_err:
                    logger.error(f"会话{session_id}更新继续取消任务状态失败，开始补偿输入流消息: {update_err}")
                    try:
                        await task.input_stream.delete_message(event_id)
                    except Exception as compensate_err:
                        logger.error(f"会话{session_id}继续取消任务输入流补偿删除失败: {compensate_err}")
                    raise

                await task.invoke()
                logger.info("会话%s, 已写入继续已取消任务命令并启动执行", session_id)

            logger.info(f"会话{session_id},已启动,任务实例: {task}")

            # 持续监听任务输出流，直到任务完成
            request_started = request_id is None
            while task:
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
                    if task.done:
                        break
                    continue

                # 解析任务内部输出流记录。
                record = TypeAdapter(TaskStreamRecord).validate_json(event_str)

                if isinstance(record, TaskRequestStartedRecord):
                    if request_id is not None and record.request_id == request_id:
                        request_started = True
                    continue

                if isinstance(record, TaskRequestFinishedRecord):
                    if request_id is not None and record.request_id == request_id:
                        break
                    continue

                if isinstance(record, TaskRequestRejectedRecord):
                    if request_id is not None and record.request_id == request_id:
                        yield ErrorEvent(
                            error=record.message,
                            error_key=record.error_key,
                        )
                        break
                    continue

                if not isinstance(record, TaskStreamEventRecord):
                    continue

                if request_id is not None and not request_started:
                    # 当前请求尚未进入自己的 run 边界前，共享流上的历史事件一律丢弃。
                    continue

                event = record.event
                event.id = event_id
                logger.debug(f"会话{session_id},输出队列中已发现事件: {type(event).__name__}")

                # 消费输出流时做一次幂等补写，用于修复“消息入流成功但DB历史缺失”的情况。
                if should_persist_event(event):
                    await self._repair_output_event_history(session_id=session_id, event=event)

                # 首条事件到达时重置一次未读数，后续事件不再重复写入。
                if unread_reset_pending:
                    async with self._uow_factory() as uow:
                        await uow.session.update_unread_message_count(session_id=session_id, count=0)
                    unread_reset_pending = False

                # 返回事件给调用方
                yield event

                # 空请求只是在附着已有运行流，仍按历史语义在首个终态处收口。
                if request_id is None and isinstance(event, (DoneEvent, ErrorEvent, WaitEvent)):
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
        await self._get_access_control_service().assert_session_access(
            user_id=user_id,
            session_id=session_id,
            action=DataAccessAction.UPDATE,
        )
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

        runtime = getattr(self, "_graph_runtime", None)
        if runtime is not None:
            # stop 接口必须等 cancelled 状态与事件真正收敛后再返回，
            # 避免前端 refresh 时只看到 session.cancelled、却看不到 step/plan.cancelled。
            cancelled = await runtime.cancel_task(session=session)
            if cancelled:
                return
        else:
            task = await self._get_task(session)
            if task and await task.cancel():
                return

        await self._persist_cancelled_session_state(session_id=session_id)

    async def _persist_cancelled_session_state(self, session_id: str) -> None:
        """在没有活跃 task 实例时，通过 coordinator 收敛 cancelled 状态。"""
        await self._get_runtime_state_coordinator().cancel_current_run(
            session_id=session_id,
            reason="stop_session",
        )

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
