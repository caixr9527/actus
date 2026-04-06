#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/2/8 15:41
@Author : caixiaorong01@outlook.com
@File   : session.py
"""
from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.domain.models import SessionStatus, File
from app.interfaces.schemas import AgentSSEEvent


class CreateSessionResponse(BaseModel):
    """创建会话响应结构"""
    session_id: str  # 会话id


class ListSessionItem(BaseModel):
    """会话列表条目基础信息"""
    session_id: str = ""
    title: str = ""
    latest_message: str = ""
    latest_message_at: Optional[datetime] = Field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.PENDING
    unread_message_count: int = 0


class ListSessionResponse(BaseModel):
    """获取会话列表基础信息响应结构"""
    sessions: List[ListSessionItem]


class ChatRequest(BaseModel):
    class ResumePayload(BaseModel):
        """恢复 LangGraph interrupt 的请求体。"""

        value: Any = None

    class CommandPayload(BaseModel):
        """显式结构化命令请求体。"""

        type: Literal["continue_cancelled_task"] = "continue_cancelled_task"

    """聊天请求结构"""
    message: Optional[str] = None  # 人类消息
    attachments: Optional[List[str]] = Field(default_factory=list)  # 附件列表(传递的是文件id列表)
    resume: Optional[ResumePayload] = None  # 恢复 LangGraph interrupt 的载荷
    command: Optional[CommandPayload] = None  # 显式结构化命令
    event_id: Optional[str] = None  # 最新事件id
    timestamp: Optional[int] = None  # 当前时间戳

    @model_validator(mode="after")
    def validate_request_shape(self) -> "ChatRequest":
        has_message = bool(str(self.message or "").strip())
        has_resume = self.resume is not None
        has_command = self.command is not None
        request_shape_count = int(has_message) + int(has_resume) + int(has_command)
        if request_shape_count > 1:
            raise ValueError("chat 请求不能同时携带 message、resume 和 command")
        if request_shape_count == 0 and len(self.attachments or []) > 0:
            raise ValueError("纯监听请求不允许携带 attachments")
        if (has_resume or has_command) and len(self.attachments or []) > 0:
            raise ValueError("resume 或 command 请求不允许携带 attachments")
        return self


class GetSessionResponse(BaseModel):
    """获取会话详情响应结构"""
    session_id: str
    title: Optional[str] = None
    status: SessionStatus
    current_model_id: Optional[str] = None
    events: List[AgentSSEEvent] = Field(default_factory=list)


class UpdateSessionModelRequest(BaseModel):
    """更新会话模型请求结构"""
    model_id: str


class UpdateSessionModelResponse(BaseModel):
    """更新会话模型响应结构"""
    session_id: str
    current_model_id: str


class GetSessionFilesResponse(BaseModel):
    """获取会话文件列表响应结构"""
    files: List[File] = Field(default_factory=list)


class FileReadRequest(BaseModel):
    """需要读取的沙箱文件请求结构"""
    filepath: str


class FileReadResponse(BaseModel):
    """需要读取的沙箱文件响应结构体"""
    filepath: str
    content: str


class ShellReadRequest(BaseModel):
    """需要读取的沙箱shell请求结构体"""
    session_id: str  # Shell会话id


class ConsoleRecord(BaseModel):
    """控制台记录模型，包含ps1、command、output"""
    ps1: str
    command: str
    output: str


class ShellReadResponse(BaseModel):
    """需要读取的沙箱shell响应结构体"""
    session_id: str
    output: str
    console_records: List[ConsoleRecord] = Field(default_factory=list)
