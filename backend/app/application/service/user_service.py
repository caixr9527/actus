#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/16 12:10
@Author : caixiaorong01@outlook.com
@File   : user_service.py
"""
import secrets
from datetime import datetime
from typing import Callable

from app.application.errors import BadRequestError
from app.application.utils import PasswordHasher
from app.domain.models import User, UserProfile
from app.domain.repositories import IUnitOfWork


class UserService:
    """用户资料服务"""

    def __init__(self, uow_factory: Callable[[], IUnitOfWork]) -> None:
        self._uow_factory = uow_factory

    async def get_current_user_profile(self, user_id: str) -> tuple[User, UserProfile]:
        """获取当前用户资料（若资料不存在则补齐默认资料）。"""
        async with self._uow_factory() as uow:
            user = await uow.user.get_by_id(user_id)
            if user is None:
                raise BadRequestError(msg="用户不存在，请重新登录")

            profile = await uow.user.get_profile_by_user_id(user_id)
            if profile is None:
                profile = UserProfile(user_id=user.id)
                await uow.user.save_profile(profile)

            return user, profile

    async def update_current_user_profile(
            self,
            user_id: str,
            updates: dict[str, str | None],
    ) -> tuple[User, UserProfile]:
        """部分更新当前用户资料。"""
        allowed_fields = {"nickname", "avatar_url", "timezone", "locale"}
        normalized_updates = {
            field: value
            for field, value in updates.items()
            if field in allowed_fields
        }
        if not normalized_updates:
            raise BadRequestError(msg="至少需要更新一个字段")

        async with self._uow_factory() as uow:
            user = await uow.user.get_by_id(user_id)
            if user is None:
                raise BadRequestError(msg="用户不存在，请重新登录")

            profile = await uow.user.get_profile_by_user_id(user_id)
            if profile is None:
                profile = UserProfile(user_id=user.id)

            for field, value in normalized_updates.items():
                setattr(profile, field, value)

            user.updated_at = datetime.now()
            await uow.user.save_profile(profile)
            await uow.user.save(user)
            return user, profile

    async def update_current_user_password(
            self,
            user_id: str,
            old_password: str,
            new_password: str,
            confirm_password: str,
    ) -> None:
        """更新当前用户密码（校验旧密码 + 新密码确认）。"""
        if new_password != confirm_password:
            raise BadRequestError(msg="两次输入的新密码不一致")

        async with self._uow_factory() as uow:
            user = await uow.user.get_by_id(user_id)
            if user is None:
                raise BadRequestError(msg="用户不存在，请重新登录")

            current_password = PasswordHasher.hash_password_with_salt(
                old_password,
                user.password_salt,
            )
            if not secrets.compare_digest(current_password, user.password):
                raise BadRequestError(msg="旧密码错误")

            next_salt = PasswordHasher.generate_password_salt()
            next_password = PasswordHasher.hash_password_with_salt(new_password, next_salt)

            user.password_salt = next_salt
            user.password = next_password
            user.updated_at = datetime.now()
            await uow.user.save(user)
