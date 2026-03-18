#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/17 10:23
@Author : caixiaorong01@outlook.com
@File   : app_config_service.py
"""
import uuid
from typing import List

from app.application.errors import NotFoundError
from app.application.errors import error_keys
from app.domain.models import AppConfig, AgentConfig, MCPConfig
from app.domain.models.app_config import A2AConfig, A2AServerConfig
from app.domain.repositories import AppConfigRepository
from app.domain.services.tools import MCPClientManager
from app.domain.services.tools.a2a import A2AClientManager
from app.interfaces.schemas import ListMCPServerItem, ListA2AServerItem


class AppConfigService:
    """应用配置服务"""

    def __init__(self, app_config_repository: AppConfigRepository):
        self.app_config_repository = app_config_repository

    async def _load_app_config(self) -> AppConfig:
        """
        加载应用配置
        
        Returns:
            AppConfig: 应用配置对象
        """
        return self.app_config_repository.load()

    async def get_agent_config(self) -> AgentConfig:
        """
        获取Agent配置
        
        Returns:
            AgentConfig: Agent配置对象
        """
        app_config = await self._load_app_config()
        return app_config.agent_config

    async def update_agent_config(self, agent_config: AgentConfig) -> AgentConfig:
        """
        更新Agent配置
        
        Args:
            agent_config (AgentConfig): 新的Agent配置对象
            
        Returns:
            AgentConfig: 更新后的Agent配置对象
        """
        app_config = await self._load_app_config()
        app_config.agent_config = agent_config
        self.app_config_repository.save(app_config)
        return app_config.agent_config

    async def get_mcp_servers(self) -> List[ListMCPServerItem]:
        """
        获取MCP服务器配置

        Returns:
            List[ListMCPServerItem]: MCP服务器配置列表
        """
        # 加载应用配置
        app_config = await self._load_app_config()
        mcp_servers = []
        # 创建MCP客户端管理器实例
        mcp_client_manager = MCPClientManager(mcp_config=app_config.mcp_config)
        try:
            # 初始化MCP客户端管理器
            await mcp_client_manager.initialize()
            # 获取所有已注册的工具
            tools = mcp_client_manager.tools
            # 遍历所有MCP服务器配置
            for server_name, server_config in app_config.mcp_config.mcpServers.items():
                # 构造每个服务器的响应数据，包括服务器名称、启用状态、传输方式和工具列表
                mcp_servers.append(ListMCPServerItem(
                    server_name=server_name,
                    enabled=server_config.enabled,
                    transport=server_config.transport,
                    tools=[tool.name for tool in tools.get(server_name, [])]
                ))
        finally:
            # 清理MCP客户端管理器资源
            await mcp_client_manager.cleanup()
        # 返回MCP服务器列表
        return mcp_servers

    async def update_and_create_mcp_servers(self, mcp_config: MCPConfig) -> MCPConfig:
        """
        更新并创建MCP服务器配置
        
        Args:
            mcp_config (MCPConfig): 新的MCP配置对象
            
        Returns:
            MCPConfig: 更新后的MCP配置对象
        """
        app_config = await self._load_app_config()
        # 更新MCP服务器配置信息
        app_config.mcp_config.mcpServers.update(mcp_config.mcpServers)
        # 保存更新后的应用配置
        self.app_config_repository.save(app_config)
        # 返回更新后的MCP配置
        return app_config.mcp_config

    async def delete_mcp_server(self, server_name: str) -> MCPConfig:
        """
        删除MCP服务器配置

        Args:
            server_name (str): 要删除的MCP服务器名称

        Returns:
            MCPConfig: 删除后的MCP配置对象
        """
        # 加载应用配置
        app_config = await self._load_app_config()
        # 检查要删除的MCP服务器是否存在
        if server_name not in app_config.mcp_config.mcpServers:
            raise NotFoundError(
                msg=f"MCP服务器 {server_name} 不存在",
                error_key=error_keys.APP_CONFIG_MCP_SERVER_NOT_FOUND,
                error_params={"server_name": server_name},
            )

        # 从配置中删除指定的MCP服务器
        del app_config.mcp_config.mcpServers[server_name]
        # 保存更新后的应用配置
        self.app_config_repository.save(app_config)
        # 返回更新后的MCP配置
        return app_config.mcp_config

    async def set_mcp_server_enabled(self, server_name: str, enabled: bool) -> MCPConfig:
        # 加载应用配置
        app_config = await self._load_app_config()
        # 检查要删除的MCP服务器是否存在
        if server_name not in app_config.mcp_config.mcpServers:
            raise NotFoundError(
                msg=f"MCP服务器 {server_name} 不存在",
                error_key=error_keys.APP_CONFIG_MCP_SERVER_NOT_FOUND,
                error_params={"server_name": server_name},
            )

        # 设置指定MCP服务器的启用状态
        app_config.mcp_config.mcpServers[server_name].enabled = enabled
        # 保存更新后的应用配置
        self.app_config_repository.save(app_config)
        # 返回更新后的MCP配置
        return app_config.mcp_config

    async def create_a2a_server(self, base_url: str) -> A2AConfig:
        """根据传递的配置新增a2a服务器"""
        app_config = await self._load_app_config()

        # 往数据中新增a2a服务(在新增之前其实可以检测下当前Agent是否存在)
        a2a_server_config = A2AServerConfig(
            id=str(uuid.uuid4()),
            base_url=base_url,
            enabled=True,
        )
        app_config.a2a_config.a2a_servers.append(a2a_server_config)

        self.app_config_repository.save(app_config)
        return app_config.a2a_config

    async def get_a2a_servers(self) -> List[ListA2AServerItem]:
        """获取A2A服务列表"""
        app_config = await self._load_app_config()

        a2a_servers = []
        a2a_client_manager = A2AClientManager(app_config.a2a_config)

        try:
            await a2a_client_manager.initialize()

            agent_cards = a2a_client_manager.agent_cards

            for id, agent_card in agent_cards.items():
                a2a_servers.append(ListA2AServerItem(
                    id=id,
                    name=agent_card.get("name", ""),
                    description=agent_card.get("description", ""),
                    input_modes=agent_card.get("defaultInputModes", []),
                    output_modes=agent_card.get("defaultOutputModes", []),
                    streaming=agent_card.get("capabilities", {}).get("streaming", False),
                    push_notifications=agent_card.get("capabilities", {}).get("push_notifications", False),
                    enabled=agent_card.get("enabled", False),
                ))
        finally:
            await a2a_client_manager.cleanup()

        return a2a_servers

    async def set_a2a_server_enabled(self, a2a_id: str, enabled: bool) -> A2AConfig:
        """根据传递的id+enabled更新服务启用状态"""

        # 加载应用配置
        app_config = await self._load_app_config()

        # 查找指定ID的A2A服务器索引
        idx = None
        for item_idx, item in enumerate(app_config.a2a_config.a2a_servers):
            if item.id == a2a_id:
                idx = item_idx
                break

        # 如果未找到对应的A2A服务器，抛出NotFoundError异常
        if idx is None:
            raise NotFoundError(
                msg=f"该A2A服务[{a2a_id}]不存在，请核实后重试",
                error_key=error_keys.APP_CONFIG_A2A_SERVER_NOT_FOUND,
                error_params={"a2a_id": a2a_id},
            )

        # 更新A2A服务器的启用状态
        app_config.a2a_config.a2a_servers[idx].enabled = enabled

        # 保存更新后的应用配置
        self.app_config_repository.save(app_config)

        # 返回更新后的A2A配置
        return app_config.a2a_config

    async def delete_a2a_server(self, a2a_id: str) -> A2AConfig:
        """根据传递的id删除指定的a2a服务"""
        app_config = await self._load_app_config()

        # 查找指定ID的A2A服务器索引
        idx = None
        for item_idx, item in enumerate(app_config.a2a_config.a2a_servers):
            if item.id == a2a_id:
                idx = item_idx
                break

        # 如果未找到对应的A2A服务器，抛出NotFoundError异常
        if idx is None:
            raise NotFoundError(
                msg=f"该A2A服务[{a2a_id}]不存在，请核实后重试",
                error_key=error_keys.APP_CONFIG_A2A_SERVER_NOT_FOUND,
                error_params={"a2a_id": a2a_id},
            )

        # 根据索引删除A2A服务器配置
        del app_config.a2a_config.a2a_servers[idx]

        # 保存更新后的应用配置
        self.app_config_repository.save(app_config)

        # 返回更新后的A2A配置
        return app_config.a2a_config
