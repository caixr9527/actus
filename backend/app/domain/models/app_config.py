#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/17 10:18
@Author : caixiaorong01@outlook.com
@File   : app_config.py
"""
import uuid
from enum import Enum
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentConfig(BaseModel):
    """Agent配置信息"""
    max_iterations: int = Field(default=100, gt=0, lt=1000)  # Agent执行的最大迭代次数，有效范围(0, 1000)
    max_retries: int = Field(default=3, gt=1, lt=10)  # Agent执行失败后的最大重试次数，有效范围(1, 10)
    max_search_results: int = Field(default=10, gt=1, lt=30)  # Agent搜索结果的最大返回数量，有效范围(1, 30)


class MCPTransport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


class MCPServerConfig(BaseModel):
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP
    enabled: bool = True
    description: Optional[str] = None
    env: Optional[Dict[str, Any]] = None

    command: Optional[str] = None
    args: Optional[List[str]] = None

    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def validate_mcp_server_config(self):
        if self.transport in [MCPTransport.SSE, MCPTransport.SSE]:
            if self.url is None:
                raise ValueError("当传输方式为 sse 或 streamable_http 时，url 是必需的")

        if self.transport == MCPTransport.STDIO:
            if self.command is None:
                raise ValueError("当传输方式为 stdio 时，command 是必需的")

        return self


class MCPConfig(BaseModel):
    """MCP配置信息"""
    mcpServers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class A2AServerConfig(BaseModel):
    """A2A服务器配置信息"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    base_url: str
    enabled: bool = True


class A2AConfig(BaseModel):
    """A2A网络配置信息"""
    a2a_servers: List[A2AServerConfig] = Field(default_factory=list)


class AppConfig(BaseModel):
    """应用配置信息，包含 Agent、A2A 网络与 MCP 服务器配置"""
    agent_config: AgentConfig
    mcp_config: MCPConfig
    a2a_config: A2AConfig

    # 允许传递额外的字段初始化
    model_config = ConfigDict(extra="allow")
