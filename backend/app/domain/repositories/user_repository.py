#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 19:27
@Author : caixiaorong01@outlook.com
@File   : user_repository.py
"""
from typing import Optional, Protocol

from app.domain.models import User, UserProfile


class UserRepository(Protocol):
    """用户仓储协议"""

    async def save(self, user: User) -> None:
        """新增或更新用户"""
        ...

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """根据用户id查询用户"""
        ...

    async def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱查询用户"""
        ...

    async def save_profile(self, profile: UserProfile) -> None:
        """新增或更新用户资料"""
        ...

    async def get_profile_by_user_id(self, user_id: str) -> Optional[UserProfile]:
        """根据用户id查询用户资料"""
        ...
