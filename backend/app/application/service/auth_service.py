#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 19:34
@Author : caixiaorong01@outlook.com
@File   : auth_service.py
"""
import secrets
from datetime import datetime
from typing import Callable, Optional

from app.application.errors import BadRequestError, ServerError
from app.application.utils import PasswordHasher, AuthTokenManager, VerificationCodeManager
from app.domain.external import (
    RefreshTokenStore,
    RegisterVerificationCodeStore,
    EmailSender,
)
from app.domain.models import User, UserProfile, UserStatus
from app.domain.repositories import IUnitOfWork
from app.interfaces.schemas.auth import LoginResult, RegisterVerificationCodeResult
from core.config import get_settings


class AuthService:
    """认证服务"""

    def __init__(
            self,
            uow_factory: Callable[[], IUnitOfWork],
            refresh_token_store: RefreshTokenStore,
            register_verification_code_store: Optional[RegisterVerificationCodeStore] = None,
            email_sender: Optional[EmailSender] = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._refresh_token_store = refresh_token_store
        self._register_verification_code_store = register_verification_code_store
        self._email_sender = email_sender
        self._setting = get_settings()

    async def register(
            self,
            email: str,
            password: str,
            verification_code: Optional[str] = None,
    ) -> User:
        """邮箱注册"""
        email = email.strip().lower()

        async with self._uow_factory() as uow:
            existing_user = await uow.user.get_by_email(email)
            if existing_user is not None:
                raise BadRequestError(msg="该邮箱已注册，请直接登录")

            if self._setting.auth_register_verification_enabled:
                if verification_code is None:
                    raise BadRequestError(msg="请输入邮箱验证码")
                register_verification_code_store = self._ensure_register_verification_code_store()
                is_verified = await register_verification_code_store.verify_and_consume_verification_code(
                    email=email,
                    verification_code=verification_code,
                )
                if not is_verified:
                    raise BadRequestError(msg="邮箱验证码错误或已过期")

            password_salt = PasswordHasher.generate_password_salt()
            password = PasswordHasher.hash_password_with_salt(password, password_salt)

            user = User(
                email=email,
                password=password,
                password_salt=password_salt,
            )
            profile = UserProfile(user_id=user.id)

            await uow.user.save(user)
            await uow.user.save_profile(profile)

            return user

    async def send_register_verification_code(self, email: str) -> RegisterVerificationCodeResult:
        """发送注册验证码"""
        normalized_email = email.strip().lower()

        async with self._uow_factory() as uow:
            existing_user = await uow.user.get_by_email(normalized_email)
            if existing_user is not None:
                raise BadRequestError(msg="该邮箱已注册，请直接登录")

        if not self._setting.auth_register_verification_enabled:
            return RegisterVerificationCodeResult(
                verification_required=False,
                expires_in_seconds=self._setting.auth_register_code_expires_in,
            )

        register_verification_code_store = self._ensure_register_verification_code_store()
        email_sender = self._ensure_email_sender()

        verification_code = VerificationCodeManager.generate_numeric_code(
            length=self._setting.auth_register_code_length
        )

        await register_verification_code_store.save_verification_code(
            email=normalized_email,
            verification_code=verification_code,
            expires_in_seconds=self._setting.auth_register_code_expires_in,
        )
        try:
            await email_sender.send_register_verification_code(
                to_email=normalized_email,
                verification_code=verification_code,
                expires_in_seconds=self._setting.auth_register_code_expires_in,
            )
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise ServerError(msg=str(e)) from e
            raise ServerError(msg="验证码发送失败，请稍后重试") from e

        return RegisterVerificationCodeResult(
            verification_required=True,
            expires_in_seconds=self._setting.auth_register_code_expires_in,
        )

    async def login(
            self,
            email: str,
            password: str,
            client_ip: Optional[str] = None,
    ) -> LoginResult:
        """邮箱登录"""
        email = email.strip().lower()

        async with self._uow_factory() as uow:
            user = await uow.user.get_by_email(email)
            if user is None:
                raise BadRequestError(msg="邮箱或密码错误")
            if user.status != UserStatus.ACTIVE:
                raise BadRequestError(msg="账号状态异常，暂不可登录")

            expected_password = PasswordHasher.hash_password_with_salt(
                password,
                user.password_salt,
            )
            if not secrets.compare_digest(expected_password, user.password):
                raise BadRequestError(msg="邮箱或密码错误")

            profile = await uow.user.get_profile_by_user_id(user.id)
            if profile is None:
                profile = UserProfile(user_id=user.id)

            user.last_login_at = datetime.now()
            user.last_login_ip = client_ip
            await uow.user.save(user)

            access_token = AuthTokenManager.generate_access_token(
                user_id=user.id,
                email=user.email,
                secret_key=self._setting.auth_jwt_secret,
                algorithm=self._setting.auth_jwt_algorithm,
                expires_in_seconds=self._setting.auth_access_token_expires_in,
            )
            refresh_token = AuthTokenManager.generate_refresh_token()

            try:
                await self._refresh_token_store.save_refresh_token(
                    refresh_token=refresh_token,
                    user_id=user.id,
                    email=user.email,
                    expires_in_seconds=self._setting.auth_refresh_token_expires_in,
                )
            except Exception as e:
                raise ServerError(msg="登录失败，请稍后重试") from e

            return LoginResult(
                user=user,
                profile=profile,
                access_token=access_token,
                refresh_token=refresh_token,
                access_token_expires_in=self._setting.auth_access_token_expires_in,
                refresh_token_expires_in=self._setting.auth_refresh_token_expires_in,
            )

    def _ensure_register_verification_code_store(self) -> RegisterVerificationCodeStore:
        if self._register_verification_code_store is None:
            raise ServerError(msg="验证码服务未配置，请联系管理员")
        return self._register_verification_code_store

    def _ensure_email_sender(self) -> EmailSender:
        if self._email_sender is None:
            raise ServerError(msg="邮件服务未配置，请联系管理员")
        return self._email_sender
