#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/20 10:59
@Author : caixiaorong01@outlook.com
@File   : session.py
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field

from .event import Event, PlanEvent
from .file import File
from .plan import Plan


class SessionStatus(str, Enum):
    """会话状态类型枚举"""
    PENDING = "pending"  # 等待任务
    RUNNING = "running"  # 运行中
    WAITING = "waiting"  # 等待人类响应
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 已失败
    CANCELLED = "cancelled"  # 已取消


class Session(BaseModel):
    """会话领域模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # 会话id
    user_id: Optional[str] = None  # 所属用户id，历史未认领会话允许为空
    current_model_id: Optional[str] = None  # 当前会话显式选择的模型id，None表示走默认模型
    workspace_id: Optional[str] = None  # 关联工作区id
    current_run_id: Optional[str] = None  # 当前运行id（WorkflowRun）
    title: str = ""  # 标题
    unread_message_count: int = 0  # 未读消息数
    latest_message: str = ""  # 最新消息
    latest_message_at: Optional[datetime] = None  # 最新消息时间
    events: List[Event] = Field(default_factory=list)  # 事件列表
    files: List[File] = Field(default_factory=list)  # 文件列表
    final_files: List[File] = Field(default_factory=list)  # 最终文件列表
    status: SessionStatus = SessionStatus.PENDING  # 状态
    updated_at: datetime = Field(default_factory=datetime.now)  # 更新时间
    created_at: datetime = Field(default_factory=datetime.now)  # 创建时间

    def get_latest_plan(self) -> Optional[Plan]:
        """获取会话中的最新计划"""
        # 倒序遍历会话中所有事件消息
        for event in reversed(self.events):
            # 判断事件的类型是否为PlanEvent，如果是则提取计划后返回
            if isinstance(event, PlanEvent):
                return event.plan

        return None
