#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/29 09:01
@Author : caixiaorong01@outlook.com
@File   : agent_task_runner.py
"""
import asyncio
import io
import logging
import uuid
from datetime import datetime
from typing import List, AsyncGenerator, Callable, BinaryIO, Optional, Any, Literal

from pydantic import TypeAdapter

from app.domain.external import (
    TaskRunner,
    Task,
    LLM,
    FileStorage,
    JSONParser,
    Browser,
    SearchEngine,
    Sandbox,
    FileUploadPayload,
)
from app.domain.models import (
    AgentConfig,
    MCPConfig,
    A2AConfig,
    ErrorEvent,
    SessionStatus,
    Event,
    MessageEvent,
    File,
    Message,
    BaseEvent,
    ToolEvent,
    ToolResult,
    DoneEvent,
    TitleEvent,
    WaitEvent,
    ResumeInput,
    RuntimeInput,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine
from app.domain.services.tools import MCPTool, A2ATool, ToolRuntimeAdapter, ToolRuntimeEventHooks

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
    ) -> None:
        self._session_id = session_id
        self._user_id = user_id
        self._sandbox = sandbox
        self._mcp_config = mcp_config
        self._mcp_tool = MCPTool()
        self._a2a_config = a2a_config
        self._a2a_tool = A2ATool()
        self._file_storage = file_storage
        self._browser = browser
        self._uow_factory = uow_factory
        # BE-LG-05：统一工具运行时适配入口。
        # 若外部未显式注入，则使用默认实现（包含 CapabilityRegistry.default_v1）。
        self._tool_runtime_adapter = tool_runtime_adapter or ToolRuntimeAdapter()
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

    def _get_tool_runtime_adapter(self) -> ToolRuntimeAdapter:
        """懒加载 tool runtime adapter，兼容测试中 object.__new__ 的构造方式。"""
        adapter = getattr(self, "_tool_runtime_adapter", None)
        if adapter is None:
            adapter = ToolRuntimeAdapter()
            self._tool_runtime_adapter = adapter
        return adapter

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
            mcp_config=self._mcp_config,
            user_id=self._user_id,
            tool_runtime_adapter=self._get_tool_runtime_adapter(),
        )

    async def _put_and_add_event(
            self,
            task: Task,
            event: Event,
            title: Optional[str] = None,
            latest_message: Optional[str] = None,
            latest_message_at: Optional[datetime] = None,
            increment_unread: bool = False,
            status: Optional[SessionStatus] = None,
    ) -> None:
        event_id = None
        try:
            # 先把事件写入输出流，拿到流事件ID用于幂等与补偿。
            event_id = await task.output_stream.put(event.model_dump_json())
            # 注意：不要原地修改传入 event.id。
            # LangGraphRunEngine 会对同一事件对象做 live + final replay 去重，
            # 若这里直接改写 id，可能导致 replay 阶段去重失效并重复落库。
            persisted_event = event.model_copy(deep=True)
            persisted_event.id = event_id

            # 再把同一事件和会话投影在单事务内落库。
            # 这样可以保证“事件历史”与“会话列表投影(latest/status/title/unread)”一致提交。
            async with self._uow_factory() as uow:
                await uow.session.add_event_with_snapshot_if_absent(
                    session_id=self._session_id,
                    event=persisted_event,
                    title=title,
                    latest_message=latest_message,
                    latest_message_at=latest_message_at,
                    increment_unread=increment_unread,
                    status=status,
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

    async def _pop_event(self, task: Task) -> Optional[RuntimeInput]:
        # 从输入流中取出事件数据
        event_id, event_str = await task.input_stream.pop()
        if event_str is None:
            logger.warning(f"接收到空消息")
            return None

        # 解析JSON字符串为运行时输入对象
        event = TypeAdapter(RuntimeInput).validate_json(event_str)

        if isinstance(event, MessageEvent):
            # 仅消息事件拥有领域事件ID；恢复输入是运行时内部控制消息，不进入事件历史。
            event.id = event_id
            # 输入流消费阶段补写事件历史：
            # 如果之前发生“入流成功但写库失败/进程中断”，这里会按event_id幂等补齐。
            try:
                async with self._uow_factory() as uow:
                    await uow.session.add_event_if_absent(self._session_id, event)
            except Exception as e:
                # 补写失败只记录，不阻断任务主流程，避免出现“可处理消息被丢弃”。
                logger.warning(f"会话[{self._session_id}]输入事件历史补写失败: {e}")

        return event

    async def _sync_file_to_sandbox(self, file_id: str) -> File:

        try:
            # 从文件存储中下载文件数据和文件元信息
            file_data, file = await self._file_storage.download_file(file_id=file_id)
            # 构建沙箱中的文件路径
            filepath = f"/home/ubuntu/upload/{file.filename}"

            # 将文件上传到沙箱环境中
            tool_result = await self._sandbox.upload_file(
                file_data=file_data,
                file_path=filepath,
                filename=file.filename,
            )

            # 如果上传成功，则更新文件的存储路径并保存到文件仓库
            if tool_result.success:
                file.filepath = filepath
                async with self._uow_factory() as uow:
                    await uow.file.save(file=file)
                return file
            raise RuntimeError(
                f"同步文件[{file_id}]到沙箱失败: {tool_result.message or 'upload_file returned unsuccessful result'}"
            )
        except Exception as e:
            # 记录同步文件到沙箱时出现的异常
            logger.exception(f"同步文件 [{file_id}] 到沙箱失败: {e}")
            raise

    async def _sync_message_attachments_to_sandbox(self, event: MessageEvent) -> None:
        # 初始化附件列表
        attachments: List[File] = []
        # 检查事件是否包含附件
        if not event.attachments:
            return
        try:
            # 遍历所有附件，同步到沙箱环境
            for attachment in event.attachments:
                file_id = str(attachment.id or "").strip()
                if not file_id:
                    raise RuntimeError("消息附件缺少 file_id，无法同步到沙箱")
                # 将附件同步到沙箱
                file = await self._sync_file_to_sandbox(file_id=file_id)
                # 添加到附件列表
                attachments.append(file)
                # 将文件添加到会话存储中
                async with self._uow_factory() as uow:
                    await uow.session.add_file(session_id=self._session_id, file=file)

            # 更新事件中的附件列表为已同步的文件
            event.attachments = attachments
        except Exception as e:
            # 记录同步附件到沙箱时发生的异常
            logger.exception(f"同步消息附件到沙箱失败: {e}")
            raise

    @classmethod
    def _get_stream_size(cls, f: BinaryIO) -> int:
        # 获取当前文件指针位置
        current_pos = f.tell()
        # 将文件指针移动到文件末尾
        f.seek(0, 2)
        # 获取文件大小（当前位置即为文件大小）
        size = f.tell()
        # 将文件指针恢复到原来的位置
        f.seek(current_pos)
        # 返回文件大小
        return size

    @staticmethod
    def _parse_sandbox_file_exists_payload(payload: Any) -> Optional[bool]:
        """解析沙箱 check_file_exists 的 data 字段，兼容不同返回结构。"""
        if isinstance(payload, bool):
            return payload
        if isinstance(payload, dict):
            for key in ("exists", "is_exists", "file_exists", "found"):
                if key in payload:
                    return bool(payload[key])
        if payload is None:
            return None
        return bool(payload)

    async def _check_sandbox_file_exists(self, filepath: str) -> bool:
        """判断沙箱文件是否存在；检查失败时按“不存在”处理，避免误下载。"""
        try:
            result = await self._sandbox.check_file_exists(file_path=filepath)
        except Exception as e:
            logger.warning(f"检查沙箱文件[{filepath}]是否存在失败，跳过附件同步: {e}")
            return False

        if not result.success:
            logger.info(f"沙箱文件[{filepath}]不存在或不可访问，跳过附件同步: {result.message}")
            return False

        parsed = self._parse_sandbox_file_exists_payload(result.data)
        if parsed is None:
            # 兼容历史返回：仅返回 success，不带 data。
            return True
        return parsed

    async def _sync_file_to_storage(self,
                                    filepath: str,
                                    stage: Optional[Literal["intermediate", "final"]] = "intermediate") -> Optional[
        File]:
        try:
            if not filepath.strip():
                logger.info("接收到空附件路径，跳过附件同步")
                return None

            # 先校验沙箱路径是否存在，避免模型幻觉附件触发不必要的沙箱下载。
            if not await self._check_sandbox_file_exists(filepath=filepath):
                return None

            # 根据文件路径从会话存储中获取旧文件信息（如存在）
            async with self._uow_factory() as uow:
                old_file = await uow.session.get_file_by_path(session_id=self._session_id, filepath=filepath)

            # 从沙箱环境中下载文件数据
            file_data = await self._sandbox.download_file(file_path=filepath)

            # 从路径中提取文件名
            filename = filepath.split("/")[-1]

            # 创建 UploadFile 对象用于上传
            upload_file = FileUploadPayload(
                file=file_data,
                filename=filename,
                size=self._get_stream_size(file_data),
            )

            # 上传并接收新文件对象，后续会话映射以该对象为准
            new_file = await self._file_storage.upload_file(upload_file=upload_file, user_id=self._user_id)
            new_file.filepath = filepath

            # 原子更新会话文件索引：删除旧引用（若存在）并新增新引用
            async with self._uow_factory() as uow:
                if old_file:
                    await uow.session.remove_file(session_id=self._session_id, file_id=old_file.id)
                await uow.session.add_file(session_id=self._session_id, file=new_file)
                if stage == "final":
                    await uow.session.add_final_files(session_id=self._session_id, file=new_file)

            return new_file
        except Exception as e:
            # 记录同步文件到存储时发生的异常
            logger.exception(f"同步文件到存储失败: {e}")
            raise

    async def _sync_message_attachments_to_storage(self, event: MessageEvent) -> None:
        attachments: List[File] = []

        try:
            if event.attachments:
                for attachment in event.attachments:
                    # 将附件上传到存储
                    file = await self._sync_file_to_storage(attachment.filepath, event.stage)
                    if file:
                        attachments.append(file)

                # 更新事件中的附件列表为已上传的文件
                event.attachments = attachments
        except Exception as e:
            # 记录同步附件到存储时发生的异常
            logger.exception(f"同步消息附件到存储失败: {e}")
            raise

    async def _get_browser_screenshot(self) -> str:
        # 获取浏览器截图
        screenshot = await self._browser.screenshot()
        screenshot_stream = io.BytesIO(screenshot)
        file = await self._file_storage.upload_file(
            upload_file=FileUploadPayload(
                file=screenshot_stream,
                filename=f"{str(uuid.uuid4())}.png",
                size=self._get_stream_size(screenshot_stream),
            ),
            user_id=self._user_id,
        )
        return self._file_storage.get_file_url(file)

    async def _read_shell_output_for_tool(self, session_id: str) -> ToolResult:
        """读取 shell 控制台输出，供 ToolRuntimeAdapter 富化 shell 工具结果。"""
        return await self._sandbox.read_shell_output(session_id=session_id, console=True)

    async def _read_file_content_for_tool(self, filepath: str) -> ToolResult:
        """读取文件内容，供 ToolRuntimeAdapter 富化 file 工具结果。"""
        return await self._sandbox.read_file(file_path=filepath)

    async def _handle_tool_event(self, event: ToolEvent) -> None:
        try:
            hooks = ToolRuntimeEventHooks(
                get_browser_screenshot=self._get_browser_screenshot,
                read_shell_output=self._read_shell_output_for_tool,
                read_file_content=self._read_file_content_for_tool,
                sync_file_to_storage=self._sync_file_to_storage,
            )
            await self._get_tool_runtime_adapter().enrich_tool_event(event=event, hooks=hooks)
        except Exception as e:
            # 记录处理工具事件时发生的异常
            logger.exception(f"处理工具事件失败: {e}")

    async def _run_flow(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        # 检查消息是否为空，如果为空则记录警告并返回错误事件
        if not message.message:
            logger.warning(f"接收了一条空消息")
            yield ErrorEvent(error=f"空消息错误")
            return

        # 遍历流程执行过程中产生的事件
        async for event in self._run_engine.invoke(message):
            # 处理工具事件，根据工具类型进行相应的内容填充
            if isinstance(event, ToolEvent):
                await self._handle_tool_event(event)
            # 处理消息事件，同步附件到存储
            elif isinstance(event, MessageEvent):
                await self._sync_message_attachments_to_storage(event)

            # 产出事件
            yield event

    async def _resume_flow(self, value: Any) -> AsyncGenerator[BaseEvent, None]:
        async for event in self._run_engine.resume(value):
            if isinstance(event, ToolEvent):
                await self._handle_tool_event(event)
            elif isinstance(event, MessageEvent):
                await self._sync_message_attachments_to_storage(event)
            yield event

    async def _cleanup_tools(self) -> None:
        """清理MCP和A2A工具资源，确保在同一任务上下文中释放

        注意：该方法必须在初始化MCP/A2A的同一个asyncio Task中调用，
        否则anyio的cancel scope会检测到任务上下文切换并抛出RuntimeError。
        """
        await self._get_tool_runtime_adapter().cleanup_remote_tools(
            mcp_tool=self._mcp_tool,
            a2a_tool=self._a2a_tool,
        )

    async def _mark_session_completed_fallback(self, scene: str) -> None:
        """在事件写入失败时兜底更新会话状态为COMPLETED"""
        try:
            async with self._uow_factory() as uow:
                await uow.session.update_status(session_id=self._session_id, status=SessionStatus.COMPLETED)
        except Exception as fallback_err:
            # 兜底再失败时保留错误日志，便于后续排障或离线修复。
            logger.error(f"会话[{self._session_id}]在{scene}状态兜底失败: {fallback_err}")

    async def invoke(self, task: Task) -> None:
        try:
            # 记录任务开始执行的日志
            logger.info(f"开始执行任务: {task.id}")

            # 确保沙箱环境就绪并初始化各种工具
            await self._sandbox.ensure_sandbox()
            await self._get_tool_runtime_adapter().initialize_remote_tools(
                mcp_tool=self._mcp_tool,
                mcp_config=self._mcp_config,
                a2a_tool=self._a2a_tool,
                a2a_config=self._a2a_config,
            )

            # 循环处理输入流中的事件，直到输入流为空
            while not await task.input_stream.is_empty():
                # 从输入流中取出事件
                event = await self._pop_event(task)
                if event is None:
                    # 空读场景直接跳过，避免后续访问空对象字段导致任务中断。
                    continue

                # 初始化消息变量
                message = ""

                # 如果事件是消息事件，处理消息内容和附件
                if isinstance(event, MessageEvent):
                    message = event.message or ""

                    # 同步消息附件到沙箱环境
                    await self._sync_message_attachments_to_sandbox(event)

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
                        # 等待事件：事件历史 + 状态切换为WAITING，并立即结束本轮消费。
                        await self._put_and_add_event(
                            task=task,
                            event=event,
                            status=SessionStatus.WAITING,
                        )
                        return
                    elif isinstance(event, DoneEvent):
                        # Done事件优先走原子提交：
                        # 当输入队列已空时，和COMPLETED状态一并落库，避免事件/状态分叉。
                        has_more_input = not await task.input_stream.is_empty()
                        await self._put_and_add_event(
                            task=task,
                            event=event,
                            status=None if has_more_input else SessionStatus.COMPLETED,
                        )
                    else:
                        # 其他事件：仅记录事件历史。
                        await self._put_and_add_event(task=task, event=event)

                    # 检查输入流是否还有更多事件，如果没有则退出循环
                    if not await task.input_stream.is_empty():
                        break

            # 所有事件处理完成后，执行一次状态兜底：
            # 正常情况下Done事件已将状态置为COMPLETED，这里仅作为防御性保障。
            async with self._uow_factory() as uow:
                await uow.session.update_status(session_id=self._session_id, status=SessionStatus.COMPLETED)
        except asyncio.CancelledError:
            # 处理任务被取消的情况
            logger.info(f"AgentTaskRunner任务运行取消")
            # 取消时优先写入Done事件+状态；如果该路径失败，再走独立状态兜底。
            try:
                await self._put_and_add_event(
                    task=task,
                    event=DoneEvent(),
                    status=SessionStatus.COMPLETED,
                )
            except Exception as done_err:
                logger.error(f"任务取消分支写入Done事件失败: {done_err}")
                await self._mark_session_completed_fallback(scene="取消分支")
            # 抛出异常
            raise
        except Exception as e:
            # 处理其他异常情况
            logger.exception(f"AgentTaskRunner运行出错: {str(e)}")
            # 异常时优先提交“错误事件 + 状态完成”，失败时降级为状态兜底更新。
            try:
                await self._put_and_add_event(
                    task=task,
                    event=ErrorEvent(error=f"{str(e)}"),
                    status=SessionStatus.COMPLETED,
                )
            except Exception as error_event_err:
                logger.error(f"异常分支写入Error事件失败: {error_event_err}")
                await self._mark_session_completed_fallback(scene="异常分支")
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
