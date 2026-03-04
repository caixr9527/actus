#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/12 14:17
@Author : caixiaorong01@outlook.com
@File   : a2a.py
"""
import logging
import uuid
from contextlib import AsyncExitStack
from typing import Optional, Any, Dict

import httpx

from app.application.errors import BadRequestError
from app.domain.models import ToolResult
from app.domain.models.app_config import A2AConfig
from .base import BaseTool, tool

logger = logging.getLogger(__name__)


class A2AClientManager:

    def __init__(self, a2a_config: Optional[A2AConfig] = None):
        self._a2a_config = a2a_config
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._httpx_clients: Optional[httpx.AsyncClient] = None
        self._agent_cards: Dict[str, Any] = {}
        self._initialized: bool = False

    @property
    def agent_cards(self) -> Dict[str, Any]:
        return self._agent_cards

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            self._httpx_clients = await self._exit_stack.enter_async_context(
                httpx.AsyncClient(timeout=600),
            )

            logger.info(f"加载{len(self._a2a_config.a2a_servers)}个A2A服务")
            await self._get_a2a_agent_cards()
            self._initialized = True
            logger.info(f"A2A客户端初始化成功")
        except Exception as e:
            logger.error(f"A2A客户端初始化失败: {e}")
            raise BadRequestError("A2A客户端初始化失败")

    async def _get_a2a_agent_cards(self) -> None:
        for a2a_server_config in self._a2a_config.a2a_servers:
            try:
                agent_card_response = await self._httpx_client.get(
                    f"{a2a_server_config.base_url}/.well-known/agent-card.json"
                )
                agent_card_response.raise_for_status()
                agent_card = agent_card_response.json()

                # 存储到agent_cards
                agent_card["enabled"] = a2a_server_config.enabled
                self._agent_cards[a2a_server_config.id] = agent_card
            except Exception as e:
                logger.error(f"加载A2A服务 {a2a_server_config.id} 失败: {e}")
                continue

    async def invoke(self, agent_id: str, query: str) -> ToolResult:
        if agent_id not in self._agent_cards:
            return ToolResult(
                success=False,
                message="A2A服务不存在",
            )

        agent_card = self._agent_cards.get(agent_id, {})
        url = agent_card.get("url", "")
        if url == "":
            return ToolResult(
                success=False,
                message="A2A服务地址不存在",
            )

        try:
            response = await self._httpx_clients.post(
                url=url,
                json={
                    "id": str(uuid.uuid4()),
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": str(uuid.uuid4()),
                            "role": "user",
                            "parts": [
                                {"kind": "text", "text": query},
                            ],
                        },
                    },
                },
            )
            response.raise_for_status()

            result = response.json()
            return ToolResult(
                success=True,
                message="调用成功",
                data=result,
            )
        except Exception as e:
            logger.error(f"调用A2A服务 {agent_id} 失败: {e}")
            return ToolResult(
                success=False,
                message=f"调用A2A服务 {agent_id} : {url} 失败: {e}",
            )

    async def cleanup(self) -> None:
        """当退出A2A客户端管理器时，清除对应资源"""
        try:
            await self._exit_stack.aclose()
            self._agent_cards.clear()
            self._initialized = False
            logger.info(f"清除A2A客户端管理器成功")
        except Exception as e:
            logger.error(f"清理A2A客户端管理器失败: {str(e)}")


class A2ATool(BaseTool):
    """A2A工具包，根据传递的完成A2A工具包的初始化"""
    name: str = "a2a"

    def __init__(self) -> None:
        """构造函数，完成工具包初始化"""
        super().__init__()
        self._initialized: bool = False
        self.manager: Optional[A2AClientManager] = None

    async def initialize(self, a2a_config: Optional[A2AConfig] = None) -> None:
        """初始化A2A工具包"""
        if not self._initialized:
            self.manager = A2AClientManager(a2a_config)
            await self.manager.initialize()
            self._initialized = True

    @tool(
        name="get_remote_agent_cards",
        description="获取可远程调用的Agent卡片信息, 包含Agent id、名称、描述、技能、请求端点等。",
        parameters={},
        required=[]
    )
    async def get_remote_agent_cards(self) -> ToolResult:
        """获取远程Agent卡片信息列表"""
        agent_cards = []
        for id, agent_card in self.manager.agent_cards.items():
            agent_cards.append({
                "id": id,
                **agent_card,
            })

        return ToolResult(
            success=True,
            message="获取Agent卡片信息列表成功",
            data=agent_cards,
        )

    @tool(
        name="call_remote_agent",
        description="根据传递的id+query(分配给远程Agent完成的任务query)调用远程Agent完成对应需求",
        parameters={
            "id": {
                "type": "string",
                "description": "需要调用远程agent的id, 格式参考get_remote_agent_cards()返回的数据结构",
            },
            "query": {
                "type": "string",
                "description": "需要分配给该远程Agent实现的任务/需求query",
            },
        },
        required=["id", "query"],
    )
    async def call_remote_agent(self, id: str, query: str) -> ToolResult:
        """调用远程Agent并完成对应需求"""
        return await self.manager.invoke(agent_id=id, query=query)
