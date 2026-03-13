#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 19:30
@Author : caixiaorong01@outlook.com
@File   : db_user_repository.py
"""
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import User, UserProfile
from app.domain.repositories import UserRepository
from app.infrastructure.models import UserModel, UserProfileModel


class DBUserRepository(UserRepository):
    """基于数据库的用户仓储"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def save(self, user: User) -> None:
        stmt = select(UserModel).where(UserModel.id == user.id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            self.db_session.add(UserModel.from_domain(user))
            return
        record.update_from_domain(user)

    async def get_by_id(self, user_id: str) -> Optional[User]:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def get_by_email(self, email: str) -> Optional[User]:
        stmt = select(UserModel).where(UserModel.email == email)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def save_profile(self, profile: UserProfile) -> None:
        stmt = select(UserProfileModel).where(UserProfileModel.user_id == profile.user_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            self.db_session.add(UserProfileModel.from_domain(profile))
            return
        record.update_from_domain(profile)

    async def get_profile_by_user_id(self, user_id: str) -> Optional[UserProfile]:
        stmt = select(UserProfileModel).where(UserProfileModel.user_id == user_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None
