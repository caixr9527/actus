#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/24 14:35
@Author : caixiaorong01@outlook.com
@File   : base.py
"""
import asyncio
import logging
import uuid
from abc import ABC
from typing import Optional, List, AsyncGenerator, Dict, Any, Callable

from app.domain.external import LLM, JSONParser
from app.domain.models import (
    AgentConfig,
    Event,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
    ErrorEvent,
    Message,
    MessageEvent,
    Memory,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.tools import BaseTool

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """基础智能体"""
    name: str = ""  # 智能体名称
    _system_prompt: str = ""  # 系统提示
    _format: Optional[str] = None  # 输出格式
    _retry_interval: float = 1.0  # 重试间隔
    _tool_choice: Optional[str] = None  # 工具选择策略

    def __init__(self,
                 session_id: str,
                 uow_factory: Callable[[], IUnitOfWork],
                 agent_config: AgentConfig,
                 llm: LLM,
                 json_parser: JSONParser,
                 tools: List[BaseTool]) -> None:
        """
        :param agent_config: 智能体配置信息
        :param llm: 大语言模型实例
        :param json_parser: JSON解析器实例
        :param tools: 工具列表
        """
        self._session_id = session_id
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._agent_config = agent_config
        self._llm = llm
        self._memory: Optional[Memory] = None
        self._json_parser = json_parser
        self._tools = tools

    async def _ensure_memory(self) -> None:
        if self._memory is None:
            async with self._uow:
                self._memory = await self._uow.session.get_memory(self._session_id, self.name)

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用的工具列表"""
        available_tools = []
        for tool in self._tools:
            available_tools.extend(tool.get_tools())
        return available_tools

    def _get_tool(self, tool_name: str) -> BaseTool:

        for tool in self._tools:
            if tool.has_tool(tool_name):
                return tool
        raise ValueError(f"无效工具: {tool_name}")

    async def _invoke_llm(self, messages: List[Dict[str, Any]], format: Optional[str] = None) -> Dict[str, Any]:
        """
        调用大语言模型
        :param messages: 消息列表
        :param format: 输出格式
        :return: 模型响应结果
        """
        # 将输入消息添加到记忆存储中
        await self._add_to_memery(messages)

        # 构造响应格式参数
        response_format = {"type": format} if format else None

        error = "调用语言模型发生错误"
        # 最多重试max_retries次
        for _ in range(self._agent_config.max_retries):
            try:
                # 调用大语言模型
                message = await self._llm.invoke(
                    messages=self._memory.get_messages(),
                    tools=self._get_available_tools(),
                    response_format=response_format,
                    tool_choice=self._tool_choice,
                )

                # 如果模型返回的是assistant角色的消息
                if message.get("role") == "assistant":
                    # 如果既没有内容也没有工具调用，则记录警告并重试
                    if not message.get("content") and not message.get("tool_calls"):
                        logger.warning("LLM返回空结果, 重试")
                        # 添加空响应和提示消息到记忆存储中
                        await self._add_to_memery([
                            {"role": "assistant", "content": ""},
                            {"role": "user", "content": "AI无响应内容,请继续。"}
                        ])
                        # 等待重试间隔后继续重试
                        await asyncio.sleep(self._retry_interval)
                        continue

                    # 过滤消息内容，只保留必要信息
                    filtered_message = {"role": "assistant", "content": message.get("content")}
                    # 兼容Deepseek思考模型写法
                    if message.get("reasoning_content"):
                        filtered_message["reasoning_content"] = message.get("reasoning_content")
                    # 如果存在工具调用，只保留第一个工具调用
                    if message.get("tool_calls"):
                        filtered_message["tool_calls"] = message["tool_calls"][:1]
                else:
                    # 如果不是assistant角色的消息，记录警告并直接使用原消息
                    logger.warning(f"LLM返回非assistant消息: {message.get('role')}")
                    filtered_message = message

                # 将过滤后的消息添加到记忆存储中
                await self._add_to_memery([filtered_message])
                return filtered_message
            except Exception as e:
                # 捕获异常，记录错误日志并等待重试间隔后继续重试
                logger.error(f"调用大语言模型失败: {e}")
                error = str(e)
                await asyncio.sleep(self._retry_interval)
                continue
        raise RuntimeError(f"调用语言模型失败, 已达到最大重试次数({self._agent_config.max_retries}): {error}")


    async def _invoke_tool(self, tool: BaseTool, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        # 初始化错误信息为空字符串
        err = ""
        # 根据最大重试次数进行循环重试
        for _ in range(self._agent_config.max_retries):
            try:
                # 尝试调用工具的invoke方法并返回结果
                return await tool.invoke(tool_name, **arguments)
            except Exception as e:
                # 捕获异常，记录错误日志，并保存错误信息
                err = str(e)
                logger.exception(f"调用工具[{tool_name}]出错, 错误: {str(e)}")
                # 等待重试间隔后继续重试
                await asyncio.sleep(self._retry_interval)
                continue

        # 重试次数用尽后，返回失败的工具结果
        return ToolResult(
            success=False,
            message=err,
        )

    async def _add_to_memery(self, messages: List[Dict[str, Any]]) -> None:
        """
        将消息添加到记忆存储中
        :param messages: 需要添加到记忆存储的消息列表
        :return: None
        """
        await self._ensure_memory()
        # 如果记忆存储为空，则先添加系统提示消息
        if self._memory.empty:
            self._memory.add_message({
                "role": "system",
                "content": self._system_prompt,
            })

        # 将传入的消息列表添加到记忆存储中
        self._memory.add_messages(messages)
        async with self._uow:
            await self._uow.session.save_memory(self._session_id, self.name, self._memory)

    async def compact_memory(self) -> None:
        await self._ensure_memory()
        self._memory.compact()
        async with self._uow:
            await self._uow.session.save_memory(self._session_id, self.name, self._memory)

    async def roll_back(self, message: Message) -> None:
        """状态回滚，确保Agent消息列表状态是正确的，用于发送新消息、暂停/停止任务、通知用户"""
        # 获取记忆存储中的最后一条消息
        await self._ensure_memory()
        last_message = self._memory.get_last_message()
        # 检查最后一条消息是否存在且包含工具调用
        if (
                not last_message or
                not last_message.get("tool_calls") or
                len(last_message.get("tool_calls")) == 0
        ):
            return
        # 获取第一个工具调用信息
        tool_call = last_message.get("tool_calls")[0]

        # 提取函数名称和工具调用ID
        function_name = tool_call.get("function", {}).get("name")
        tool_call_id = tool_call.get("id")
        # 如果是询问用户的工具调用，则添加工具响应消息到记忆存储中
        if function_name == "message_ask_user":
            self._memory.add_message({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "function_name": function_name,
                "content": message.model_dump_json(),
            })
        else:
            # 否则执行记忆存储回滚操作
            self._memory.roll_back()
        async with self._uow:
            await self._uow.session.save_memory(self._session_id, self.name, self._memory)

    async def invoke(self, query: str, format: Optional[str] = None) -> AsyncGenerator[Event, None]:
        """
        调用智能体
        :param query: 查询内容
        :param format: 输出格式
        :return: 智能体事件生成器
        """
        # 设置输出格式，如果未指定则使用默认格式
        format = format if format else self._format
        # 调用大语言模型处理用户查询
        message = await self._invoke_llm(
            [{"role": "user", "content": query}, ],
            format,
        )
        # 根据最大迭代次数进行循环处理
        for _ in range(self._agent_config.max_iterations):
            # 如果没有工具调用需求，则跳出循环
            if not message.get("tool_calls"):
                break

            # 存储工具调用相关消息
            tool_messages = []
            # 遍历所有工具调用
            for tool_call in message["tool_calls"]:
                # 如果工具调用没有函数信息，则跳过
                if not tool_call.get("function"):
                    continue

                # 获取或生成工具调用ID
                tool_call_id = tool_call["id"] or str(uuid.uuid4())
                # 获取函数名称和参数
                function_name = tool_call["function"]["name"]
                function_args = await self._json_parser.invoke(tool_call["function"]["arguments"])

                # 根据函数名称获取对应工具
                tool = self._get_tool(function_name)

                # 发送工具调用事件
                yield ToolEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    function_name=function_name,
                    function_args=function_args,
                    status=ToolEventStatus.CALLING,
                )

                # 执行工具调用
                result = await self._invoke_tool(tool, function_name, function_args)
                # 发送工具调用完成事件
                yield ToolEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    function_name=function_name,
                    function_args=function_args,
                    function_result=result,
                    status=ToolEventStatus.CALLED,
                )

                # 将工具执行结果添加到工具消息列表中
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "function_name": function_name,
                    "content": result.model_dump_json(),
                })

            # 使用工具执行结果再次调用大语言模型
            message = await self._invoke_llm(tool_messages)
        else:
            # 如果达到最大迭代次数仍未完成，则发送错误事件
            yield ErrorEvent(error=f"迭代次数超出限制:{self._agent_config.max_iterations},任务处理失败")

        # 在指定步骤内完成了迭代则返回消息事件
        if message and message.get("content") is not None:
            yield MessageEvent(message=message["content"])
        else:
            yield ErrorEvent(error="Agent未能生成有效回复内容")
