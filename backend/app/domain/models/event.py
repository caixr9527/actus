#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/19 19:08
@Author : caixiaorong01@outlook.com
@File   : event.py
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Literal, List, Any, Union, Optional, Dict, Annotated

from pydantic import BaseModel, Field

from .file import File
from .plan import Plan, Step
from .search import SearchResultItem
from .tool_result import ToolResult
from .wait import normalize_wait_payload


class PlanEventStatus(str, Enum):
    """规划事件状态"""
    CREATED = "created"
    UPDATED = "updated"
    COMPLETED = "completed"


class StepEventStatus(str, Enum):
    """步骤事件状态"""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolEventStatus(str, Enum):
    """工具事件状态"""
    CALLING = "calling"
    CALLED = "called"


class BaseEvent(BaseModel):
    """基础事件模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal[""] = ""  # 事件类型
    created_at: datetime = Field(default_factory=datetime.now)


class PlanEvent(BaseEvent):
    """计划事件模型"""
    type: Literal["plan"] = "plan"
    plan: Plan  # 规划
    status: PlanEventStatus = PlanEventStatus.CREATED


class TitleEvent(BaseEvent):
    """标题事件模型"""
    type: Literal["title"] = "title"
    title: str = ""


class StepEvent(BaseEvent):
    """步骤事件模型"""
    type: Literal["step"] = "step"
    step: Step
    status: StepEventStatus = StepEventStatus.STARTED


class MessageEvent(BaseEvent):
    """消息事件模型"""
    type: Literal["message"] = "message"  # 事件类型为消息
    role: Literal["user", "assistant"] = "assistant"  # 消息发送者角色，用户或助手
    message: str = ""  # 消息内容
    attachments: List[File] = Field(default_factory=list)  # 消息附件列表


class BrowserToolContent(BaseModel):
    """浏览器工具扩展内容"""
    screenshot: str  # 浏览器快照截图


class SearchToolContent(BaseModel):
    results: List[SearchResultItem]


class ShellToolContent(BaseModel):
    """Shell工具扩展内容"""
    console: Any


class FileToolContent(BaseModel):
    """文件工具扩展内容"""
    content: str


class MCPToolContent(BaseModel):
    """MCPT工具扩展内容"""
    result: Any


class A2AToolContent(BaseModel):
    """A2A工具扩展内容"""
    a2a_result: Any


ToolContent = Union[
    BrowserToolContent,
    SearchToolContent,
    ShellToolContent,
    MCPToolContent,
    FileToolContent,
    A2AToolContent,
]


class ToolEvent(BaseEvent):
    """工具事件模型"""
    type: Literal["tool"] = "tool"
    tool_call_id: str = ""  # 工具调用ID
    tool_name: str = ""  # 工具名称
    tool_content: Optional[ToolContent] = None  # 工具扩展内容
    function_name: str  # 工具调用的函数名称
    function_args: Dict[str, Any]  # 工具调用的函数参数
    function_result: Optional[ToolResult] = None  # 工具调用结果
    status: ToolEventStatus = ToolEventStatus.CALLING  # 工具事件状态


class WaitEvent(BaseEvent):
    """等待事件模型"""
    type: Literal["wait"] = "wait"
    interrupt_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_interrupt(
            cls,
            *,
            interrupt_id: Optional[str],
            payload: Optional[Any] = None,
    ) -> "WaitEvent":
        return cls(
            interrupt_id=str(interrupt_id) if interrupt_id is not None else None,
            payload=normalize_wait_payload(payload),
        )


class ErrorEvent(BaseEvent):
    """错误事件模型"""
    type: Literal["error"] = "error"
    error: str = ""
    error_key: Optional[str] = None
    error_params: Optional[Dict[str, Any]] = None


class DoneEvent(BaseEvent):
    """完成事件模型"""
    type: Literal["done"] = "done"


Event = Annotated[
    Union[
        PlanEvent,
        TitleEvent,
        StepEvent,
        MessageEvent,
        ToolEvent,
        WaitEvent,
        ErrorEvent,
        DoneEvent
    ],
    Field(discriminator="type")
]
