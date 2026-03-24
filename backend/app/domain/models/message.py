#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/24 16:23
@Author : caixiaorong01@outlook.com
@File   : message.py
"""
from typing import List

from pydantic import BaseModel, Field

from .content_part import MessageInputEnvelope


class Message(BaseModel):
    """消息模型"""
    message: str = ""
    attachments: List[str] = Field(default_factory=list)
    input_envelope: MessageInputEnvelope = Field(default_factory=MessageInputEnvelope)
