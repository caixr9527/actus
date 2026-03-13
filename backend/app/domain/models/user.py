#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 19:26
@Author : caixiaorong01@outlook.com
@File   : user.py
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UserStatus(str, Enum):
    """用户状态枚举"""
    ACTIVE = "active"
    DISABLED = "disabled"
    PENDING_VERIFICATION = "pending_verification"


class User(BaseModel):
    """用户领域模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    password: str
    password_salt: str
    auth_provider: str = "email"
    external_id: Optional[str] = None
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_login_at: Optional[datetime] = None
    last_login_ip: Optional[str] = None


class UserProfile(BaseModel):
    """用户资料领域模型"""
    user_id: str
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: str = "Asia/Shanghai"
    locale: str = "zh-CN"
