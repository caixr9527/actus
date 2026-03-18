#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/4 14:59
@Author : caixiaorong01@outlook.com
@File   : app_config.py
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from app.domain.models import MCPTransport


class ListMCPServerItem(BaseModel):
    server_name: str = ""
    enabled: bool = True
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP
    tools: List[str] = Field(default_factory=list)


class ListMCPServerResponse(BaseModel):
    mcp_servers: List[ListMCPServerItem] = Field(default_factory=list)


class ListA2AServerItem(BaseModel):
    """A2A服务列表条目选项"""
    id: str = ""  # id
    name: str = ""  # 名字
    description: str = ""  # 描述信息
    input_modes: List[str] = Field(default_factory=list)  # 输入模态
    output_modes: List[str] = Field(default_factory=list)  # 输出模态
    streaming: bool = False  # 是否支持流式
    push_notifications: bool = False  # 是否支持推送通知
    enabled: bool = True  # 启用状态


class ListA2AServerResponse(BaseModel):
    """获取A2A服务列表响应结构"""
    a2a_servers: List[ListA2AServerItem] = Field(default_factory=list)  # A2A服务列表


class PublicModelConfig(BaseModel):
    """模型公开配置字段"""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    description: Optional[str] = None
    badge: Optional[str] = None


class ListModelItem(BaseModel):
    """模型列表条目"""
    id: str
    display_name: str
    provider: str
    enabled: bool
    sort_order: int
    config: PublicModelConfig = Field(default_factory=PublicModelConfig)


class ListModelResponse(BaseModel):
    """模型列表响应结构"""
    default_model_id: str
    models: List[ListModelItem] = Field(default_factory=list)
