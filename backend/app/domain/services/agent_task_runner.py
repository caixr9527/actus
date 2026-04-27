#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/29 09:01
@Author : caixiaorong01@outlook.com
@File   : agent_task_runner.py
"""
import asyncio
import logging
from datetime import datetime
from typing import List, AsyncGenerator, Callable, Optional, Any, Literal

from pydantic import TypeAdapter

from app.application.errors import error_keys
from app.domain.external import (
    TaskRunner,
    Task,
    LLM,
    FileStorage,
    JSONParser,
    Browser,
    SearchEngine,
    Sandbox,
)
from app.domain.models import (
    AgentConfig,
    MCPConfig,
    A2AConfig,
    ErrorEvent,
    SessionStatus,
    Event,
    MessageEvent,
    Message,
    MessageCommand,
    BaseEvent,
    ToolEvent,
    DoneEvent,
    TitleEvent,
    WaitEvent,
    ContinueCancelledTaskInput,
    ResumeInput,
    RuntimeInput,
    TaskStreamEventRecord,
    TaskRequestStartedRecord,
    TaskRequestFinishedRecord,
    TaskRequestRejectedRecord,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine
from app.domain.services.runtime.contracts.event_delivery_policy import should_persist_event
from app.domain.models import RuntimeEventProjection
from app.domain.services.workspace_runtime import WorkspaceManager, WorkspaceRuntimeService
from app.domain.services.workspace_runtime.projectors import (
    MessageAttachmentProjector,
    ToolEventProjector,
    UserInputAttachmentProjector,
)
from app.domain.services.tools import MCPTool, A2ATool, ToolRuntimeAdapter

logger = logging.getLogger(__name__)


class AgentTaskRunner(TaskRunner):
    """
    任务执行者，用于执行任务，并返回结果
    """

    def __init__(
            self,
            llm: LLM,
            agent_config: AgentConfig,
            mcp_config: MCPConfig,
            a2a_config: A2AConfig,
            session_id: str,
            user_id: Optional[str],
            file_storage: FileStorage,
            uow_factory: Callable[[], IUnitOfWork],
            json_parser: JSONParser,
            browser: Browser,
            search_engine: SearchEngine,
            sandbox: Sandbox,
            run_engine_factory: Optional[Callable[..., RunEngine]] = None,
            tool_runtime_adapter: Optional[ToolRuntimeAdapter] = None,
            runtime_state_coordinator: Any = None,
    ) -> None:
        if tool_runtime_adapter is None:
            raise ValueError("tool_runtime_adapter 不能为空")
        self._session_id = session_id
        self._user_id = user_id
        self._sandbox = sandbox
        self._mcp_config = mcp_config
        self._mcp_tool = MCPTool()
        self._a2a_config = a2a_config
        self._a2a_tool = A2ATool()
        self._file_storage = file_storage
        self._uow_factory = uow_factory
        self._workspace_runtime_service = WorkspaceRuntimeService(
            session_id=session_id,
            uow_factory=uow_factory,
        )
        self._workspace_manager = WorkspaceManager(uow_factory=uow_factory)
        self._runtime_state_coordinator = runtime_state_coordinator
        self._tool_runtime_adapter = tool_runtime_adapter
        self._tool_event_projector = ToolEventProjector(
            adapter=tool_runtime_adapter,
            browser=browser,
            file_storage=file_storage,
            workspace_runtime_service=self._workspace_runtime_service,
            user_id=user_id,
        )
        self._message_attachment_projector = MessageAttachmentProjector(
            session_id=session_id,
            user_id=user_id,
            sandbox=sandbox,
            file_storage=file_storage,
            uow_factory=uow_factory,
            workspace_runtime_service=self._workspace_runtime_service,
        )
        self._user_input_attachment_projector = UserInputAttachmentProjector(
            session_id=session_id,
            sandbox=sandbox,
            file_storage=file_storage,
            uow_factory=uow_factory,
        )
        self._run_engine = self._build_run_engine(
            llm=llm,
            agent_config=agent_config,
            session_id=session_id,
            uow_factory=uow_factory,
            json_parser=json_parser,
            browser=browser,
            sandbox=sandbox,
            search_engine=search_engine,
            run_engine_factory=run_engine_factory,
        )

    def _build_run_engine(
            self,
            llm: LLM,
            agent_config: AgentConfig,
            session_id: str,
            uow_factory: Callable[[], IUnitOfWork],
            json_parser: JSONParser,
            browser: Browser,
            sandbox: Sandbox,
            search_engine: SearchEngine,
            run_engine_factory: Optional[Callable[..., RunEngine]],
    ) -> RunEngine:
        if run_engine_factory is None:
            raise RuntimeError(
                "未配置 run_engine_factory，当前仅支持 LangGraph 运行时，禁止回退 legacy planner-react"
            )

        return run_engine_factory(
            llm=llm,
            agent_config=agent_config,
            session_id=session_id,
            file_storage=self._file_storage,
            uow_factory=uow_factory,
            json_parser=json_parser,
            browser=browser,
            sandbox=sandbox,
            search_engine=search_engine,
            mcp_tool=self._mcp_tool,
            a2a_tool=self._a2a_tool,
            workspace_runtime_service=self._workspace_runtime_service,
            mcp_config=self._mcp_config,
            user_id=self._user_id,
            tool_runtime_adapter=self._tool_runtime_adapter,
        )

    def _get_workspace_manager(self) -> WorkspaceManager:
        manager = getattr(self, "_workspace_manager", None)
        if manager is None:
            manager = WorkspaceManager(uow_factory=self._uow_factory)
            self._workspace_manager = manager
        return manager

    def _get_runtime_state_coordinator(self):
        coordinator = getattr(self, "_runtime_state_coordinator", None)
        if coordinator is not None:
            return coordinator
        # 测试可通过 object.__new__ 构造 runner；此处延迟导入避免领域层顶层依赖应用层。
        from app.application.service.runtime_state_coordinator import RuntimeStateCoordinator

        coordinator = RuntimeStateCoordinator(uow_factory=self._uow_factory)
        self._runtime_state_coordinator = coordinator
        return coordinator

    async def _put_stream_record(
            self,
            task: Task,
            record: TaskStreamEventRecord
                    | TaskRequestStartedRecord
                    | TaskRequestFinishedRecord
                    | TaskRequestRejectedRecord,
    ) -> str:
        """写入任务内部输出流。"""
        return await task.output_stream.put(record.model_dump_json())

    async def _put_and_add_event(
            self,
            task: Task,
            event: Event,
            title: Optional[str] = None,
            latest_message: Optional[str] = None,
            latest_message_at: Optional[datetime] = None,
            increment_unread: bool = False,
            allow_status_transition: bool = True,
    ) -> None:
        event_id = None
        try:
            # 先把事件写入输出流，拿到流事件ID用于幂等与补偿。
            event_id = await self._put_stream_record(
                task=task,
                record=TaskStreamEventRecord(event=event),
            )
            if not should_persist_event(event):
                return
            # 注意：不要原地修改传入 event.id。
            # LangGraphRunEngine 会对同一事件对象做 live + final replay 去重，
            # 若这里直接改写 id，可能导致 replay 阶段去重失效并重复落库。
            persisted_event = event.model_copy(deep=True)
            persisted_event.id = event_id

            projection = RuntimeEventProjection(
                title=title,
                latest_message=latest_message,
                latest_message_at=latest_message_at,
                increment_unread=increment_unread,
            )
            await self._get_runtime_state_coordinator().persist_runtime_event(
                session_id=self._session_id,
                event=persisted_event,
                projection=projection,
                allow_status_transition=allow_status_transition,
            )
        except Exception as e:
            # 落库失败时补偿删除输出流，尽量降低Redis/DB分叉概率。
            logger.error(f"写入输出流后保存事件历史失败: {e}")
            if event_id is not None:
                try:
                    await task.output_stream.delete_message(event_id)
                except Exception as rollback_err:
                    logger.error(f"输出流补偿删除失败: {rollback_err}")
            raise

    async def _emit_request_started(self, task: Task, request_id: str) -> None:
        await self._put_stream_record(
            task=task,
            record=TaskRequestStartedRecord(request_id=request_id),
        )

    async def _emit_request_finished(
            self,
            task: Task,
            *,
            request_id: str,
            terminal_event_type: Literal["wait", "done", "error"],
    ) -> None:
        await self._put_stream_record(
            task=task,
            record=TaskRequestFinishedRecord(
                request_id=request_id,
                terminal_event_type=terminal_event_type,
            ),
        )

    async def _emit_request_rejected(
            self,
            task: Task,
            *,
            request_id: str,
            message: str,
            error_key: Optional[str] = None,
    ) -> None:
        await self._put_stream_record(
            task=task,
            record=TaskRequestRejectedRecord(
                request_id=request_id,
                message=message,
                error_key=error_key,
            ),
        )

    async def _reject_pending_requests(
            self,
            task: Task,
            *,
            message: str,
            error_key: Optional[str] = None,
    ) -> None:
        """拒绝当前终态之后尚未开始执行的排队请求。"""
        while not await task.input_stream.is_empty():
            pending_input = await self._pop_event(task)
            if pending_input is None:
                continue
            await self._emit_request_rejected(
                task=task,
                request_id=pending_input.request_id,
                message=message,
                error_key=error_key,
            )

    async def _pop_event(self, task: Task) -> Optional[RuntimeInput]:
        # 从输入流中取出事件数据
        event_id, event_str = await task.input_stream.pop()
        if event_str is None:
            logger.warning(f"接收到空消息")
            return None

        # 解析JSON字符串为运行时输入对象
        runtime_input = TypeAdapter(RuntimeInput).validate_json(event_str)
        event = runtime_input.payload

        if isinstance(event, MessageEvent):
            # 仅消息事件拥有领域事件ID；恢复输入是运行时内部控制消息，不进入事件历史。
            event.id = event_id
            # 输入流消费阶段补写事件历史：
            # 如果之前发生“入流成功但写库失败/进程中断”，这里会按event_id幂等补齐。
            try:
                async with self._uow_factory() as uow:
                    if should_persist_event(event):
                        await uow.session.add_event_if_absent(self._session_id, event)
            except Exception as e:
                # 补写失败只记录，不阻断任务主流程，避免出现“可处理消息被丢弃”。
                logger.warning(f"会话[{self._session_id}]输入事件历史补写失败: {e}")

        return runtime_input

    async def _run_flow(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        # 检查消息是否为空，如果为空则记录警告并返回错误事件
        if not message.message and message.command is None:
            logger.warning(f"接收了一条空消息")
            yield ErrorEvent(error=f"空消息错误")
            return

        # 遍历流程执行过程中产生的事件
        async for event in self._run_engine.invoke(message):
            # 处理工具事件，根据工具类型进行相应的内容填充
            if isinstance(event, ToolEvent):
                await self._tool_event_projector.project(event)
            # 处理消息事件，同步附件到存储
            elif isinstance(event, MessageEvent):
                await self._message_attachment_projector.project(event)

            # 产出事件
            yield event

    async def _continue_cancelled_flow(self) -> AsyncGenerator[BaseEvent, None]:
        """显式继续已取消任务，不伪造用户自然语言输入。"""
        async for event in self._run_flow(
                Message(
                    message="",
                    attachments=[],
                    command=MessageCommand(type="continue_cancelled_task"),
                )
        ):
            yield event

    async def _resume_flow(self, value: Any) -> AsyncGenerator[BaseEvent, None]:
        async for event in self._run_engine.resume(value):
            if isinstance(event, ToolEvent):
                await self._tool_event_projector.project(event)
            elif isinstance(event, MessageEvent):
                await self._message_attachment_projector.project(event)
            yield event

    async def _cleanup_tools(self) -> None:
        """清理MCP和A2A工具资源，确保在同一任务上下文中释放

        注意：该方法必须在初始化MCP/A2A的同一个asyncio Task中调用，
        否则anyio的cancel scope会检测到任务上下文切换并抛出RuntimeError。
        """
        await self._tool_runtime_adapter.cleanup_remote_tools(
            mcp_tool=self._mcp_tool,
            a2a_tool=self._a2a_tool,
        )

    async def _mark_session_completed_fallback(self, scene: str) -> None:
        """在事件写入失败时兜底更新会话状态为 completed。"""
        try:
            async with self._uow_factory() as uow:
                await uow.session.update_status(session_id=self._session_id, status=SessionStatus.COMPLETED)
        except Exception as fallback_err:
            # 兜底再失败时保留错误日志，便于后续排障或离线修复。
            logger.error(f"会话[{self._session_id}]在{scene}状态兜底失败: {fallback_err}")

    async def _mark_session_failed_fallback(self, scene: str) -> None:
        """在错误收敛失败时至少落地 session 的 failed 状态。"""
        try:
            async with self._uow_factory() as uow:
                await uow.session.update_status(session_id=self._session_id, status=SessionStatus.FAILED)
        except Exception as fallback_err:
            logger.error(f"会话[{self._session_id}]在{scene}失败兜底失败: {fallback_err}")

    async def _mark_session_cancelled_fallback(self, scene: str) -> None:
        """在取消收敛失败时至少落地 session/run 的 cancelled 状态。"""
        try:
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id=self._session_id)
                if session is None:
                    return
                await uow.session.update_status(
                    session_id=self._session_id,
                    status=SessionStatus.CANCELLED,
                )
                current_run_id = await self._get_workspace_manager().resolve_current_run_id(
                    session=session,
                    uow=uow,
                )
                if current_run_id:
                    await uow.workflow_run.cancel_run(current_run_id)
        except Exception as fallback_err:
            logger.error(f"会话[{self._session_id}]在{scene}取消兜底失败: {fallback_err}")

    async def _persist_cancellation_state(self) -> None:
        """将取消状态收敛委托给 RuntimeStateCoordinator。"""
        await self._get_runtime_state_coordinator().cancel_current_run(
            session_id=self._session_id,
            reason="task_cancelled",
        )

    async def invoke(self, task: Task) -> None:
        active_request_id: Optional[str] = None
        terminal_session_status: Optional[SessionStatus] = None
        pending_requests_rejected = False
        try:
            # 记录任务开始执行的日志
            logger.info(f"开始执行任务: {task.id}")

            # 确保沙箱环境就绪并初始化各种工具
            await self._sandbox.ensure_sandbox()
            await self._tool_runtime_adapter.initialize_remote_tools(
                mcp_tool=self._mcp_tool,
                mcp_config=self._mcp_config,
                a2a_tool=self._a2a_tool,
                a2a_config=self._a2a_config,
            )

            # 循环处理输入流中的事件，直到输入流为空
            while not await task.input_stream.is_empty():
                # 从输入流中取出事件
                runtime_input = await self._pop_event(task)
                if runtime_input is None:
                    # 空读场景直接跳过，避免后续访问空对象字段导致任务中断。
                    continue
                active_request_id = runtime_input.request_id
                event = runtime_input.payload
                await self._emit_request_started(task=task, request_id=active_request_id)

                # 初始化消息变量
                message = ""

                # 如果事件是消息事件，处理消息内容和附件
                if isinstance(event, MessageEvent):
                    message = event.message or ""

                    # 同步消息附件到沙箱环境
                    await self._user_input_attachment_projector.project(event)

                    # 记录接收到的消息日志
                    logger.info(f"收到消息: {message[:50]}...")

                if isinstance(event, MessageEvent):
                    # 创建消息对象，包含消息内容和附件路径列表
                    message_obj = Message(
                        message=message,
                        attachments=[attachment.filepath for attachment in event.attachments],
                    )
                    event_stream = self._run_flow(message_obj)
                elif isinstance(event, ResumeInput):
                    event_stream = self._resume_flow(event.value)
                elif isinstance(event, ContinueCancelledTaskInput):
                    logger.info("收到继续已取消任务命令")
                    event_stream = self._continue_cancelled_flow()
                else:
                    logger.warning("收到不支持的运行时输入类型: %s", type(event).__name__)
                    continue

                # 运行流程并处理每个产生的事件
                async for event in event_stream:
                    # 将事件写入输出流+数据库，并在同一事务中更新会话投影字段。
                    # 说明：每种事件对应的投影更新在这里统一声明，便于审计事务边界。
                    if isinstance(event, TitleEvent):
                        # 标题事件：事件历史 + 标题投影。
                        await self._put_and_add_event(
                            task=task,
                            event=event,
                            title=event.title,
                        )
                    elif isinstance(event, MessageEvent):
                        # 消息事件：事件历史 + latest_message + 未读数递增。
                        await self._put_and_add_event(
                            task=task,
                            event=event,
                            latest_message=event.message,
                            latest_message_at=event.created_at,
                            increment_unread=True,
                        )
                    elif isinstance(event, WaitEvent):
                        # 等待事件：先投影 WAITING，但必须继续消费到 run_engine 自然结束，
                        # 否则会打断 graph state / summary / snapshot 的收尾写入。
                        await self._put_and_add_event(
                            task=task,
                            event=event,
                        )
                        terminal_session_status = SessionStatus.WAITING
                        if active_request_id is not None:
                            await self._emit_request_finished(
                                task=task,
                                request_id=active_request_id,
                                terminal_event_type="wait",
                            )
                            active_request_id = None
                        if not pending_requests_rejected:
                            await self._reject_pending_requests(
                                task=task,
                                message="当前任务进入等待状态，请使用 resume 恢复执行",
                                error_key=error_keys.SESSION_RESUME_REQUIRED,
                            )
                            pending_requests_rejected = True
                    elif isinstance(event, DoneEvent):
                        # Done 事件先完成本轮终态投影，但不能立即 break；
                        # run_engine 还需要继续完成 finally 中的状态合同与上下文投影收尾。
                        has_more_input = not await task.input_stream.is_empty()
                        await self._put_and_add_event(
                            task=task,
                            event=event,
                            allow_status_transition=not has_more_input,
                        )
                        terminal_session_status = None if has_more_input else SessionStatus.COMPLETED
                        if active_request_id is not None:
                            await self._emit_request_finished(
                                task=task,
                                request_id=active_request_id,
                                terminal_event_type="done",
                            )
                            active_request_id = None
                    elif isinstance(event, ErrorEvent):
                        # 错误事件必须收敛 failed，但同样不能提前 break，
                        # 否则 run_engine 的失败收尾与状态回写会被截断。
                        await self._put_and_add_event(
                            task=task,
                            event=event,
                            latest_message=event.error,
                            latest_message_at=event.created_at,
                        )
                        terminal_session_status = SessionStatus.FAILED
                        if active_request_id is not None:
                            await self._emit_request_finished(
                                task=task,
                                request_id=active_request_id,
                                terminal_event_type="error",
                            )
                            active_request_id = None
                        if not pending_requests_rejected:
                            await self._reject_pending_requests(
                                task=task,
                                message="当前任务执行失败，排队请求已终止，请重新发起",
                            )
                            pending_requests_rejected = True
                    else:
                        # 其他事件：仅记录事件历史。
                        await self._put_and_add_event(task=task, event=event)

            # 所有事件处理完成后，执行一次状态兜底：
            # 这里只兜底正常完成路径，不能覆盖 wait/failed 等已真实收敛的终态。
            if terminal_session_status in {None, SessionStatus.COMPLETED}:
                async with self._uow_factory() as uow:
                    await uow.session.update_status(session_id=self._session_id, status=SessionStatus.COMPLETED)
        except asyncio.CancelledError:
            # 处理任务被取消的情况
            logger.info(f"AgentTaskRunner任务运行取消")
            try:
                await self._reject_pending_requests(
                    task=task,
                    message="当前任务已取消，排队请求已终止，请重新发起",
                )
                await self._persist_cancellation_state()
            except Exception as cancel_err:
                logger.error(f"任务取消分支收敛 cancelled 状态失败: {cancel_err}")
                await self._mark_session_cancelled_fallback(scene="取消分支")
            raise
        except Exception as e:
            # 处理其他异常情况
            logger.exception(f"AgentTaskRunner运行出错: {str(e)}")
            # 异常时优先提交“错误事件 + failed状态”，失败时降级为 failed 兜底更新。
            try:
                error_event = ErrorEvent(error=f"{str(e)}")
                await self._put_and_add_event(
                    task=task,
                    event=error_event,
                    latest_message=error_event.error,
                    latest_message_at=error_event.created_at,
                )
                if active_request_id is not None:
                    await self._emit_request_finished(
                        task=task,
                        request_id=active_request_id,
                        terminal_event_type="error",
                    )
                    active_request_id = None
                await self._reject_pending_requests(
                    task=task,
                    message="当前任务执行失败，排队请求已终止，请重新发起",
                )
            except Exception as error_event_err:
                logger.error(f"异常分支写入Error事件失败: {error_event_err}")
                await self._mark_session_failed_fallback(scene="异常分支")
        finally:
            # 在同一个asyncio Task上下文中清理MCP/A2A工具资源
            # 这是关键：streamablehttp_client内部使用anyio.create_task_group()，
            # 要求在同一个Task中进入和退出cancel scope，
            # 所以必须在invoke()的finally块（即初始化MCP的同一个Task）中清理
            await self._cleanup_tools()

    async def destroy(self) -> None:
        logger.info(f"开始清除销毁AgentTaskRunner资源")
        if self._sandbox:
            logger.info("销毁AgentTaskRunner中的沙箱环境")
            await self._sandbox.destroy()

        # 清除mcp和a2a工具（幂等操作，如果invoke()中已清理则不会重复执行）
        await self._cleanup_tools()

    async def on_done(self, task: Task) -> None:
        logger.info(f"任务完成: {task.id}")
