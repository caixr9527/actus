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
from app.application.service.runtime_observation_service import (
    RuntimeObservationResult,
)
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


class RuntimeCursorResponse(BaseModel):
    """会话详情 runtime 持久游标。"""

    latest_event_id: Optional[str] = None
    has_more: bool = False


class RuntimeCapabilityResponse(BaseModel):
    """会话详情 runtime 用户动作能力。"""

    can_send_message: bool = True
    can_resume: bool = False
    can_cancel: bool = False
    can_continue_cancelled: bool = False
    disabled_reasons: dict[str, str] = Field(default_factory=dict)


class RuntimeInteractionResponse(BaseModel):
    """会话详情 runtime 当前交互上下文。"""

    kind: Literal["none", "wait"] = "none"
    interrupt_id: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RuntimeObservationResponse(BaseModel):
    """会话详情 runtime 观察快照。"""

    session_id: str
    run_id: Optional[str] = None
    status: SessionStatus
    current_step_id: Optional[str] = None
    cursor: RuntimeCursorResponse
    capabilities: RuntimeCapabilityResponse
    interaction: RuntimeInteractionResponse

    @classmethod
    def from_result(cls, result: RuntimeObservationResult) -> "RuntimeObservationResponse":
        return cls(
            session_id=result.session_id,
            run_id=result.run_id,
            status=result.status,
            current_step_id=result.current_step_id,
            cursor=RuntimeCursorResponse(
                latest_event_id=result.cursor.latest_event_id,
                has_more=result.cursor.has_more,
            ),
            capabilities=RuntimeCapabilityResponse(
                can_send_message=result.capabilities.can_send_message,
                can_resume=result.capabilities.can_resume,
                can_cancel=result.capabilities.can_cancel,
                can_continue_cancelled=result.capabilities.can_continue_cancelled,
                disabled_reasons={
                    (action.value if hasattr(action, "value") else str(action)): reason
                    for action, reason in result.capabilities.disabled_reasons.items()
                },
            ),
            interaction=RuntimeInteractionResponse(
                kind=result.interaction.kind,
                interrupt_id=result.interaction.interrupt_id,
                payload=dict(result.interaction.payload or {}),
            ),
        )


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
    runtime: RuntimeObservationResponse
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


class ConsoleRecord(BaseModel):
    """控制台记录模型，包含ps1、command、output"""
    ps1: str
    command: str
    output: str


class ShellReadResponse(BaseModel):
    """需要读取的沙箱shell响应结构体"""
    output: str
    console_records: List[ConsoleRecord] = Field(default_factory=list)
