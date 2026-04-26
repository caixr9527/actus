#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行时消息模型。"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MessageCommand(BaseModel):
    """消息驱动之外的结构化运行命令。"""

    type: Literal["continue_cancelled_task"] = "continue_cancelled_task"


class Message(BaseModel):
    """运行时输入消息。"""

    message: str = ""
    attachments: List[str] = Field(default_factory=list)  # 同步到沙箱到路径
    command: Optional[MessageCommand] = None
