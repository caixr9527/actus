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
from typing import List, AsyncGenerator, Callable, BinaryIO

from fastapi import UploadFile
from pydantic import TypeAdapter

from app.domain.external import (
    TaskRunner,
    Task,
    LLM,
    FileStorage,
    JSONParser,
    Browser,
    SearchEngine,
    Sandbox
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
    ToolEventStatus,
    ToolResult,
    SearchResults,
    DoneEvent,
    TitleEvent,
    WaitEvent,
    BrowserToolContent,
    SearchToolContent,
    ShellToolContent,
    FileToolContent,
    MCPToolContent,
    A2AToolContent,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.flows import PlannerReActFlow
from app.domain.services.tools import MCPTool, A2ATool
from core.config import get_settings

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
            file_storage: FileStorage,
            uow_factory: Callable[[], IUnitOfWork],
            json_parser: JSONParser,
            browser: Browser,
            search_engine: SearchEngine,
            sandbox: Sandbox,
    ) -> None:
        self._session_id = session_id
        self._sandbox = sandbox
        self._mcp_config = mcp_config
        self._mcp_tool = MCPTool()
        self._a2a_config = a2a_config
        self._a2a_tool = A2ATool()
        self._file_storage = file_storage
        self._browser = browser
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._flow = PlannerReActFlow(
            llm=llm,
            agent_config=agent_config,
            session_id=session_id,
            uow_factory=self._uow_factory,
            json_parser=json_parser,
            browser=browser,
            sandbox=sandbox,
            search_engine=search_engine,
            mcp_tool=self._mcp_tool,
            a2a_tool=self._a2a_tool,
        )

    async def _put_and_add_event(self, task: Task, event: Event) -> None:
        # 将事件数据放入输出流并获取事件ID
        event_id = await task.output_stream.put(event.model_dump_json())
        # 设置事件ID
        event.id = event_id
        # 将事件添加到会话存储中
        async with self._uow:
            await self._uow.session.add_event(self._session_id, event)

    @classmethod
    async def _pop_event(cls, task: Task) -> Event:
        # 从输入流中取出事件数据
        event_id, event_str = await task.input_stream.pop()
        if event_str is None:
            logger.warning(f"接收到空消息")
            return

        # 解析JSON字符串为Event对象
        event = TypeAdapter(Event).validate_json(event_str)
        # 设置事件ID
        event.id = event_id
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
                async with self._uow:
                    await self._uow.file.save(file=file)
                return file
        except Exception as e:
            # 记录同步文件到沙箱时出现的异常
            logger.exception(f"同步文件 [{file_id}] 到沙箱失败: {e}")

    async def _sync_message_attachments_to_sandbox(self, event: MessageEvent) -> None:
        # 初始化附件列表
        attachments: List[File] = []
        try:
            # 检查事件是否包含附件
            if event.attachments:
                # 遍历所有附件，同步到沙箱环境
                for attachment in event.attachments:
                    # 将附件同步到沙箱
                    file = await self._sync_file_to_sandbox(file_id=attachment.id)
                    if file:
                        # 添加到附件列表
                        attachments.append(file)
                        # 将文件添加到会话存储中
                        async with self._uow:
                            await self._uow.session.add_file(session_id=self._session_id, file=file)

                # 更新事件中的附件列表为已同步的文件
                event.attachments = attachments
        except Exception as e:
            # 记录同步附件到沙箱时发生的异常
            logger.exception(f"同步消息附件到沙箱失败: {e}")

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

    async def _sync_file_to_storage(self, filepath: str) -> File:

        try:
            # 根据文件路径从会话存储中获取文件信息
            async with self._uow:
                file = await self._uow.session.get_file_by_path(session_id=self._session_id, filepath=filepath)

            # 从沙箱环境中下载文件数据
            file_data = await self._sandbox.download_file(file_path=filepath)

            # 如果文件存在，则从会话存储中移除该文件
            if file:
                async with self._uow:
                    await self._uow.session.remove_file(session_id=self._session_id, file_id=file.filepath)

            # 从路径中提取文件名
            filename = filepath.split("/")[-1]

            # 创建UploadFile对象用于上传
            upload_file = UploadFile(file=file_data, filename=filename, size=self._get_stream_size(file_data))

            # 将文件上传到文件存储系统
            await self._file_storage.upload_file(upload_file=upload_file)

            # 更新文件的存储路径
            file.filepath = filepath

            # 将文件重新添加到会话存储中
            async with self._uow:
                await self._uow.session.add_file(session_id=self._session_id, file=file)

            # 返回同步后的文件对象
            return file
        except Exception as e:
            # 记录同步文件到存储时发生的异常
            logger.exception(f"同步文件到存储失败: {e}")

    async def _sync_message_attachments_to_storage(self, event: MessageEvent) -> None:
        attachments: List[File] = []

        try:
            if event.attachments:
                for attachment in event.attachments:
                    # 将附件上传到存储
                    file = await self._sync_file_to_storage(attachment.filepath)
                    if file:
                        attachments.append(file)

                # 更新事件中的附件列表为已上传的文件
                event.attachments = attachments
        except Exception as e:
            # 记录同步附件到存储时发生的异常
            logger.exception(f"同步消息附件到存储失败: {e}")

    async def _get_browser_screenshot(self) -> str:
        # 获取浏览器截图
        screenshot = await self._browser.screenshot()
        file = await self._file_storage.upload_file(
            upload_file=UploadFile(
                file=io.BytesIO(screenshot),
                filename=f"{str(uuid.uuid4())}.png",
                size=self._get_stream_size(io.BytesIO(screenshot))
            )
        )
        settings = get_settings()
        # todo 修改为配置
        return f"https://{settings.cos_bucket}.cos.{settings.cos_region}.myqcloud.com/{file.key}"

    async def _handle_tool_event(self, event: ToolEvent) -> None:
        try:
            # 检查工具事件的状态是否为已调用
            if event.status == ToolEventStatus.CALLED:
                # 处理浏览器工具事件 - 生成屏幕截图
                if event.tool_name == "browser":
                    event.tool_content = BrowserToolContent(
                        screenshot=await self._get_browser_screenshot(),
                    )
                # 处理搜索工具事件 - 提取搜索结果
                elif event.tool_name == "search":
                    search_results: ToolResult[SearchResults] = event.function_result
                    logger.info(f"搜索工具结果: {search_results}")
                    event.tool_content = SearchToolContent(results=search_results.data.results)
                # 处理Shell工具事件 - 读取Shell命令输出
                elif event.tool_name == "shell":
                    if "session_id" in event.function_args:
                        # 如果提供了session_id参数，读取对应会话的shell输出
                        shell_result = await self._sandbox.read_shell_output(
                            session_id=event.function_args["session_id"],
                            console=True
                        )
                        event.tool_content = ShellToolContent(
                            console=(shell_result.data or {}).get("console_records", [])
                        )
                    else:
                        # 如果没有session_id参数，设置默认值
                        event.tool_content = ShellToolContent(console="(No console)")
                # 处理文件工具事件 - 读取文件内容并同步到存储
                elif event.tool_name == "file":
                    if "filepath" in event.function_args:
                        # 从函数参数中获取文件路径
                        filepath = event.function_args["filepath"]
                        # 从沙箱中读取文件内容
                        file_read_result = await self._sandbox.read_file(file_path=filepath)
                        file_content: str = (file_read_result.data or {}).get("content", "")
                        event.tool_content = FileToolContent(content=file_content)
                        # 将文件同步到沙存储
                        await self._sync_file_to_storage(filepath=filepath)
                    else:
                        # 如果没有提供文件路径参数，设置默认内容
                        event.tool_content = FileToolContent(content="(No Content)")
                # 处理MCP和A2A工具事件 - 处理外部工具调用结果
                elif event.tool_name in ["mcp", "a2a"]:
                    logger.info(f"处理MCP/A2A工具事件, function_result: {event.function_result}")
                    if event.function_result:
                        # 检查函数结果是否有数据属性
                        if hasattr(event.function_result, "data") and event.function_result.data:
                            logger.info(f"MCP/A2A工具调用结果: {event.function_result.data}")
                            # 根据工具名称设置相应的工具内容
                            event.tool_content = MCPToolContent(result=event.function_result.data) \
                                if event.tool_name == "mcp" \
                                else A2AToolContent(a2a_result=event.function_result.data)
                        # 检查函数结果是否成功但无数据
                        elif hasattr(event.function_result, "success") and event.function_result.success:
                            logger.info(f"MCP/A2A工具调用成功返回，但无结果: {event.function_result}")
                            # 获取结果数据，优先使用model_dump方法
                            result_data = event.function_result.model_dump() \
                                if hasattr(event.function_result, "model_dump") \
                                else str(event.function_result)
                            event.tool_content = MCPToolContent(result=result_data) \
                                if event.tool_name == "mcp" \
                                else A2AToolContent(a2a_result=result_data)
                        # 其他情况，直接转换为字符串
                        else:
                            logger.info(f"MCP/A2A工具结果: {event.function_result}")
                            event.tool_content = MCPToolContent(result=str(event.function_result)) \
                                if event.tool_name == "mcp" \
                                else A2AToolContent(a2a_result=str(event.function_result))
                    else:
                        # 没有找到函数结果时的默认处理
                        logger.warning("MCP/A2A工具调用结果未发现")
                        event.tool_content = MCPToolContent(result="(MCP工具无可用结果)") \
                            if event.tool_name == "mcp" \
                            else A2AToolContent(a2a_result="(A2A智能体无可用结果)")
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
        async for event in self._flow.invoke(message):
            # 处理工具事件，根据工具类型进行相应的内容填充
            if isinstance(event, ToolEvent):
                await self._handle_tool_event(event)
            # 处理消息事件，同步附件到存储
            elif isinstance(event, MessageEvent):
                await self._sync_message_attachments_to_storage(event)

            # 产出事件
            yield event

    async def _cleanup_tools(self) -> None:
        """清理MCP和A2A工具资源，确保在同一任务上下文中释放

        注意：该方法必须在初始化MCP/A2A的同一个asyncio Task中调用，
        否则anyio的cancel scope会检测到任务上下文切换并抛出RuntimeError。
        """
        try:
            if self._mcp_tool:
                await self._mcp_tool.cleanup()
        except Exception as e:
            logger.warning(f"清理MCP工具资源时出错: {e}")
        try:
            if self._a2a_tool:
                await self._a2a_tool.manager.cleanup()
        except Exception as e:
            logger.warning(f"清理A2A工具资源时出错: {e}")

    async def invoke(self, task: Task) -> None:
        try:
            # 记录任务开始执行的日志
            logger.info(f"开始执行任务: {task.id}")

            # 确保沙箱环境就绪并初始化各种工具
            await self._sandbox.ensure_sandbox()
            await self._mcp_tool.initialize(self._mcp_config)
            await self._a2a_tool.initialize(self._a2a_config)

            # 循环处理输入流中的事件，直到输入流为空
            while not await task.input_stream.is_empty():
                # 从输入流中取出事件
                event = await self._pop_event(task)

                # 初始化消息变量
                message = ""

                # 如果事件是消息事件，处理消息内容和附件
                if isinstance(event, MessageEvent):
                    message = event.message or ""

                    # 同步消息附件到沙箱环境
                    await self._sync_message_attachments_to_sandbox(event)

                    # 记录接收到的消息日志
                    logger.info(f"收到消息: {message[:50]}...")

                # 创建消息对象，包含消息内容和附件路径列表
                message_obj = Message(
                    message=message,
                    attachments=[attachment.filepath for attachment in event.attachments]
                )

                # 运行流程并处理每个产生的事件
                async for event in self._run_flow(message_obj):
                    # 将事件添加到输出流和会话存储
                    await self._put_and_add_event(task, event)

                    # 根据事件类型更新会话状态或信息
                    if isinstance(event, TitleEvent):
                        # 更新会话标题
                        async with self._uow:
                            await self._uow.session.update_title(session_id=self._session_id, title=event.title)
                    elif isinstance(event, MessageEvent):
                        # 更新会话最新消息和未读消息计数
                        async with self._uow:
                            await self._uow.session.update_latest_message(
                                session_id=self._session_id,
                                message=event.message,
                                timestamp=event.created_at,
                            )
                            await self._uow.session.increment_unread_message_count(session_id=self._session_id)
                    elif isinstance(event, WaitEvent):
                        # 如果是等待事件，将会话状态设置为等待并返回
                        async with self._uow:
                            await self._uow.session.update_status(
                                session_id=self._session_id,
                                status=SessionStatus.WAITING
                            )
                        return

                    # 检查输入流是否还有更多事件，如果没有则退出循环
                    if not await task.input_stream.is_empty():
                        break

            # 所有事件处理完成后，将会话状态更新为已完成
            async with self._uow:
                await self._uow.session.update_status(session_id=self._session_id, status=SessionStatus.COMPLETED)
        except asyncio.CancelledError:
            # 处理任务被取消的情况
            logger.info(f"AgentTaskRunner任务运行取消")
            await self._put_and_add_event(task=task, event=DoneEvent())
            async with self._uow:
                await self._uow.session.update_status(session_id=self._session_id, status=SessionStatus.COMPLETED)
            # 抛出异常
            raise
        except Exception as e:
            # 处理其他异常情况
            logger.exception(f"AgentTaskRunner运行出错: {str(e)}")
            await self._put_and_add_event(task=task, event=ErrorEvent(error=f"AgentTaskRunner出错: {str(e)}"))
            async with self._uow:
                await self._uow.session.update_status(session_id=self._session_id, status=SessionStatus.COMPLETED)
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
