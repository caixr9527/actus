#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/4 10:13
@Author : caixiaorong01@outlook.com
@File   : mcp.py
"""
import logging
import os
from contextlib import AsyncExitStack
from typing import Optional, Dict, List, Any

from mcp import ClientSession, Tool, StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from app.application.errors import NotFoundError
from app.domain.models import MCPConfig, MCPServerConfig, MCPTransport, ToolResult
from .base import BaseTool

logger = logging.getLogger(__name__)


class MCPClientManager:

    def __init__(self, mcp_config: Optional[MCPConfig] = None) -> None:
        self._mcp_config: MCPConfig = mcp_config
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._clients: Dict[str, ClientSession] = {}
        self._tools: Dict[str, List[Tool]] = {}
        self._initialized: bool = False

    @property
    def tools(self) -> Dict[str, List[Tool]]:
        """
        获取所有工具列表
        """
        return self._tools

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            logger.info(f"初始化MCP客户端, 加载了{len(self._mcp_config.mcpServers)}个MCP服务器")
            await self._connect_mcp_servers()
            self._initialized = True
            logger.info("MCP客户端初始化完成")
        except Exception as e:
            logger.error(f"初始化MCP客户端失败: {e}")
            raise

    async def _connect_mcp_servers(self) -> None:
        # 遍历所有配置的MCP服务器并尝试连接
        for server_name, server_config in self._mcp_config.mcpServers.items():
            try:
                # 尝试连接单个MCP服务器
                await self._connect_mcp_server(server_name, server_config)
            except Exception as e:
                # 记录连接失败的错误信息并继续连接其他服务器
                logger.error(f"连接MCP服务器{server_name}失败: {e}")
                continue

    async def _connect_mcp_server(self, server_name: str, server_config: MCPServerConfig) -> None:
        """
        连接MCP服务器
        """
        try:
            # 获取服务器配置中的传输方式
            transport = server_config.transport
            # 根据不同的传输方式调用对应的连接方法
            if transport == MCPTransport.STDIO:
                # 连接标准输入输出(STDIO)类型的MCP服务器
                await self._connect_stdio_server(server_name, server_config)
            elif transport == MCPTransport.SSE:
                # 连接服务器发送事件(SSE)类型的MCP服务器
                await self._connect_sse_server(server_name, server_config)
            elif transport == MCPTransport.STREAMABLE_HTTP:
                # 连接可流式HTTP(Streamable HTTP)类型的MCP服务器
                await self._connect_streamable_http_server(server_name, server_config)
            else:
                # 不支持的传输方式，抛出异常
                raise ValueError(f"MCP服务器：{server_name}, 使用不支持的传输方式: {transport}")
        except Exception as e:
            # 记录连接MCP服务器失败的错误日志并重新抛出异常
            logger.error(f"连接MCP服务器{server_name}失败: {e}")
            raise

    async def _connect_stdio_server(self, server_name: str, server_config: MCPServerConfig) -> None:
        """
        连接stdio服务器
        """
        # 获取stdio服务器配置参数
        command = server_config.command
        args = server_config.args
        env = server_config.env

        # 检查command是否为空，为空则抛出异常
        if not command:
            raise ValueError("当传输方式为 stdio 时，command 是必需的")

        # 构造StdioServerParameters对象，用于启动stdio服务器
        server_parameters = StdioServerParameters(
            command=command,
            args=args,
            env={**os.environ, **env},
        )

        try:
            # 使用_exit_stack管理stdio_client上下文，确保资源正确释放
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_parameters)
            )
            # 分别获取读取流和写入流
            read_stream, write_stream = stdio_transport

            # 创建ClientSession并与服务器建立连接
            session: ClientSession = await self._exit_stack.enter_async_context(
                ClientSession(read_stream=read_stream, write_stream=write_stream)
            )
            # 初始化session
            await session.initialize()
            # 将已连接的session存储至_clients字典中
            self._clients[server_name] = session
            # 缓存该服务器提供的工具列表
            await self._cache_mcp_server_tools(server_name, session)
            # 记录连接成功的日志
            logger.info(f"连接STDIO-MCP服务器 {server_name} 成功")
        except Exception as e:
            # 记录连接失败的日志并重新抛出异常
            logger.error(f"连接STDIO-MCP服务器 {server_name} 失败: {e}")
            raise

    async def _connect_sse_server(self, server_name: str, server_config: MCPServerConfig) -> None:
        """
        连接sse服务器
        """
        """
        连接sse服务器
        """
        # 获取sse服务器配置参数
        url = server_config.url
        headers = server_config.headers
        # 检查url是否为空，为空则抛出异常
        if not url:
            raise ValueError("当传输方式为 sse 时，url 是必需的")
        try:
            # 使用_exit_stack管理sse_client上下文，确保资源正确释放
            sse_transport = await self._exit_stack.enter_async_context(
                sse_client(url=url, headers=headers)
            )
            # 分别获取读取流和写入流
            read_stream, write_stream = sse_transport
            # 创建ClientSession并与服务器建立连接
            session: ClientSession = await self._exit_stack.enter_async_context(
                ClientSession(read_stream=read_stream, write_stream=write_stream)
            )
            # 初始化session
            await session.initialize()
            # 将已连接的session存储至_clients字典中
            self._clients[server_name] = session
            # 缓存该服务器提供的工具列表
            await self._cache_mcp_server_tools(server_name, session)
            # 记录连接成功的日志
            logger.info(f"连接SSE-MCP服务器 {server_name} 成功")
        except Exception as e:
            # 记录连接失败的日志并重新抛出异常
            logger.error(f"连接SSE-MCP服务器 {server_name} 失败: {e}")
            raise

    async def _connect_streamable_http_server(self, server_name: str, server_config: MCPServerConfig) -> None:
        """
        连接streamable_http服务器
        """
        """
        连接streamable_http服务器
        """
        # 获取streamable_http服务器配置参数
        url = server_config.url
        headers = server_config.headers
        # 检查url是否为空，为空则抛出异常
        if not url:
            raise ValueError("当传输方式为 streamable_http 时，url 是必需的")
        try:
            # 使用_exit_stack管理streamablehttp_client上下文，确保资源正确释放
            streamable_http_transport = await self._exit_stack.enter_async_context(
                streamablehttp_client(url=url, headers=headers)
            )
            # 根据返回的transport元组长度分别获取读取流和写入流
            if len(streamable_http_transport) == 3:
                # 当返回三个元素时，第三个元素通常为额外的控制流(此处未使用)
                read_stream, write_stream, _ = streamable_http_transport
            else:
                # 当返回两个元素时，直接解包获取读取流和写入流
                read_stream, write_stream = streamable_http_transport

            # 创建ClientSession并与服务器建立连接
            session: ClientSession = await self._exit_stack.enter_async_context(
                ClientSession(read_stream=read_stream, write_stream=write_stream)
            )
            # 初始化session
            await session.initialize()
            # 将已连接的session存储至_clients字典中
            self._clients[server_name] = session
            # 缓存该服务器提供的工具列表
            await self._cache_mcp_server_tools(server_name, session)
            # 记录连接成功的日志
            logger.info(f"连接STREAMABLE_HTTP-MCP服务器 {server_name} 成功")
        except Exception as e:
            # 记录连接失败的日志并重新抛出异常
            logger.error(f"连接STREAMABLE_HTTP-MCP服务器 {server_name} 失败: {e}")
            raise

    async def _cache_mcp_server_tools(self, server_name: str, session: ClientSession) -> None:
        """
        从MCP服务器获取并缓存工具列表
        :param server_name: MCP服务器名称
        :param session: 已建立连接的ClientSession实例
        """
        try:
            # 调用MCP协议的list_tools方法获取服务器提供的工具列表
            tools_response = await session.list_tools()
            # 将获取到的工具列表缓存到_tools字典中，如果响应为空则缓存空列表
            self._tools[server_name] = tools_response.tools if tools_response else []
            # 记录成功缓存工具的日志，包含工具数量
            logger.info(f"缓存MCP服务器 {server_name} 的工具成功, 提供了{len(self._tools[server_name])} 个工具")
        except Exception as e:
            # 记录缓存工具失败的错误日志
            logger.error(f"缓存MCP服务器 {server_name} 的工具失败: {e}")
            # 缓存失败时，为该服务器设置空的工具列表
            self._tools[server_name] = []

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有工具
        """
        # 收集所有MCP服务器提供的工具
        all_tools = []
        # 遍历所有已连接的MCP服务器及其工具列表
        for server_name, tools in self._tools.items():
            # 遍历每个服务器上的工具
            for tool in tools:
                # 根据服务器名称前缀规则构建统一的工具名称
                if server_name.startswith("mcp_"):
                    # 如果服务器名称已包含mcp_前缀，则直接拼接
                    tool_name = f"{server_name}_{tool.name}"
                else:
                    # 否则添加mcp_前缀后再拼接
                    tool_name = f"mcp_{server_name}_{tool.name}"

                # 构建符合函数调用规范的工具schema
                tool_schema = {
                    "type": "function",
                    "function": {
                        # 工具的唯一标识名称
                        "name": tool_name,
                        # 工具描述信息，包含服务器来源和工具描述
                        "description": f"[{server_name}] {tool.description or tool.name}",
                        # 工具输入参数的JSON Schema定义
                        "parameters": tool.inputSchema,
                    }
                }

                # 将构建好的工具schema添加到结果列表中
                all_tools.append(tool_schema)

        # 返回所有工具的schema列表
        return all_tools

    async def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        调用指定的MCP工具
        
        :param tool_name: 工具名称，格式为 mcp_{server_name}_{tool_name}
        :param arguments: 工具调用参数
        :return: ToolResult对象，包含调用结果
        """
        try:
            # 解析工具名称，提取原始服务器名和工具名
            original_server_name = None
            original_tool_name = None
            for server_name in self._mcp_config.mcpServers.keys():
                # 构造期望的前缀格式
                expected_prefix = server_name if server_name.startswith("mcp_") else f"mcp_{server_name}"
                # 检查工具名称是否以期望前缀开头
                if tool_name.startswith(f"{expected_prefix}_"):
                    original_server_name = server_name
                    original_tool_name = tool_name[len(expected_prefix) + 1:]
                    break

            # 如果无法解析出服务器名或工具名，则抛出异常
            if not original_tool_name or not original_server_name:
                raise NotFoundError(f"无法解析工具名称 {tool_name}")

            # 获取对应服务器的客户端会话
            session = self._clients.get(original_server_name)
            if not session:
                return ToolResult(success=False, message=f"无法链接到MCP服务器 {original_server_name}")

            # 调用工具
            result = await session.call_tool(original_tool_name, arguments)
            if result:
                # 处理返回结果，提取文本内容
                content = []
                if hasattr(result, "content") and result.content:
                    for item in result.content:
                        if hasattr(item, "text"):
                            content.append(item.text)
                        else:
                            content.append(str(item))

                # 返回成功结果
                return ToolResult(
                    success=True,
                    data="\n".join(content) if content else "工具执行成功"
                )
            else:
                # 工具执行成功但无返回内容
                return ToolResult(success=True, data="工具执行成功")
        except Exception as e:
            # 记录错误日志并返回失败结果
            logger.error(f"调用MCP工具 {tool_name} 失败: {e}")
            return ToolResult(success=False, message=f"调用MCP工具 {tool_name} 失败: {e}")

    async def cleanup(self) -> None:
        """当退出MCP服务时，清除对应资源

        该方法是幂等的，多次调用不会产生副作用。
        注意：必须在初始化MCP的同一个asyncio Task中调用此方法，
        否则anyio会因cancel scope上下文不匹配而抛出RuntimeError。
        """
        # 幂等检查：如果未初始化则跳过清理
        if not self._initialized:
            return

        try:
            await self._exit_stack.aclose()
            logger.info(f"清除MCP客户端管理器成功")
        except RuntimeError as e:
            # 防御性处理：anyio.create_task_group() 在不同任务中退出的已知问题
            if "Attempted to exit cancel scope in a different task" in str(e):
                logger.warning(f"清理MCP客户端管理器时遇到任务上下文切换警告（可忽略）: {str(e)}")
            else:
                logger.error(f"清理MCP客户端管理器失败: {str(e)}")
        except Exception as e:
            logger.error(f"清理MCP客户端管理器失败: {str(e)}")
        finally:
            # 无论aclose()是否成功，都必须清除缓存并重置状态
            self._clients.clear()
            self._tools.clear()
            self._initialized = False


class MCPTool(BaseTool):
    """
    MCP工具类，继承自BaseTool
    """
    name: str = "mcp"

    def __init__(self) -> None:
        super().__init__()
        self._initialized: bool = False
        self._tools = []
        self._manager: MCPClientManager | None = None

    async def initialize(self, mcp_config: Optional[MCPConfig] = None) -> None:
        """
        初始化MCP工具
        """
        # 检查是否已经初始化，避免重复初始化
        if not self._initialized:
            # 创建MCP客户端管理器实例并传入配置
            self._manager = MCPClientManager(mcp_config=mcp_config)
            # 初始化MCP客户端管理器，建立与各MCP服务器的连接
            await self._manager.initialize()

            # 获取所有MCP服务器提供的工具列表
            self._tools = await self._manager.get_all_tools()
            # 标记初始化完成状态
            self._initialized = True

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有MCP工具
        """
        return self._tools

    def has_tool(self, tool_name: str) -> bool:
        """
        检查MCP工具集合中是否存在指定工具
        """
        for tool in self._tools:
            if tool["function"]["name"] == tool_name:
                return True
        return False

    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        """
        调用指定的MCP工具
        """
        return await self._manager.invoke(tool_name, kwargs)

    async def cleanup(self) -> None:
        """
        清理MCP工具
        """
        if self._manager:
            await self._manager.cleanup()
