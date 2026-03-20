#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
应用配置相关 DTO
"""

from typing import List

from pydantic import BaseModel, Field

from app.domain.models import MCPTransport


class MCPServerItemResult(BaseModel):
    server_name: str = ""
    enabled: bool = True
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP
    tools: List[str] = Field(default_factory=list)


class A2AServerItemResult(BaseModel):
    id: str = ""
    name: str = ""
    description: str = ""
    input_modes: List[str] = Field(default_factory=list)
    output_modes: List[str] = Field(default_factory=list)
    streaming: bool = False
    push_notifications: bool = False
    enabled: bool = True
