#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 19:34
@Author : caixiaorong01@outlook.com
@File   : auth_service.py
"""
import logging
from datetime import datetime
from typing import Callable, Optional

from app.application.errors import BadRequestError, ServerError, TooManyRequestsError
from app.application.errors import error_keys
from app.application.utils import PasswordHasher, AuthTokenManager, VerificationCodeManager
from app.domain.external import (
    RefreshTokenStore,
    AccessTokenBlacklistStore,
    AuthRateLimitStore,
    RefreshTokenConsumeStatus,
    RegisterVerificationCodeStore,
    EmailSender,
)
from app.domain.models import User, UserProfile, UserStatus
from app.domain.repositories import IUnitOfWork
from app.interfaces.schemas.auth import (
    LoginResult,
    RegisterVerificationCodeResult,
    RefreshResult,
)
from core.config import get_settings

logger = logging.getLogger(__name__)


class AuthService:
    """认证服务"""

    def __init__(
            self,
            uow_factory: Callable[[], IUnitOfWork],
            refresh_token_store: RefreshTokenStore,
            access_token_blacklist_store: Optional[AccessTokenBlacklistStore] = None,
            auth_rate_limit_store: Optional[AuthRateLimitStore] = None,
            register_verification_code_store: Optional[RegisterVerificationCodeStore] = None,
            email_sender: Optional[EmailSender] = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._refresh_token_store = refresh_token_store
        self._access_token_blacklist_store = access_token_blacklist_store
        self._auth_rate_limit_store = auth_rate_limit_store
        self._register_verification_code_store = register_verification_code_store
        self._email_sender = email_sender
        self._setting = get_settings()

    async def register(
            self,
            email: str,
            password: str,
            confirm_password: str,
            verification_code: Optional[str] = None,
    ) -> User:
        """邮箱注册"""
        email = email.strip().lower()
        if password != confirm_password:
            raise BadRequestError(
                msg="密码不一致",
                error_key=error_keys.AUTH_PASSWORD_MISMATCH,
            )

        async with self._uow_factory() as uow:
            existing_user = await uow.user.get_by_email(email)
            if existing_user is not None:
                raise BadRequestError(
                    msg="该邮箱已注册，请直接登录",
                    error_key=error_keys.AUTH_EMAIL_ALREADY_REGISTERED,
                    error_params={"email": email},
                )

            if self._setting.auth_register_verification_enabled:
                if verification_code is None:
                    raise BadRequestError(
                        msg="请输入邮箱验证码",
                        error_key=error_keys.AUTH_REGISTER_CODE_REQUIRED,
                    )
                register_verification_code_store = self._ensure_register_verification_code_store()
                is_verified = await register_verification_code_store.verify_and_consume_verification_code(
                    email=email,
                    verification_code=verification_code,
                )
                if not is_verified:
                    raise BadRequestError(
                        msg="邮箱验证码错误或已过期",
                        error_key=error_keys.AUTH_REGISTER_CODE_INVALID,
                    )

            password_hash = PasswordHasher.hash_password(password)

            user = User(
                email=email,
                password=password_hash,
            )
            profile = UserProfile(user_id=user.id)

            await uow.user.save(user)
            await uow.user.save_profile(profile)

            return user

    async def send_register_verification_code(
            self,
            email: str,
            client_ip: Optional[str] = None,
    ) -> RegisterVerificationCodeResult:
        """发送注册验证码"""
        normalized_email = email.strip().lower()
        normalized_ip = self._normalize_ip(client_ip)
        await self._ensure_register_send_code_rate_limit(normalized_ip)

        async with self._uow_factory() as uow:
            existing_user = await uow.user.get_by_email(normalized_email)
            if existing_user is not None:
                raise BadRequestError(
                    msg="该邮箱已注册，请直接登录",
                    error_key=error_keys.AUTH_EMAIL_ALREADY_REGISTERED,
                    error_params={"email": normalized_email},
                )

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
                raise ServerError(
                    msg=str(e),
                    error_key=error_keys.AUTH_SEND_CODE_FAILED,
                    error_params={"email": normalized_email},
                ) from e
            raise ServerError(
                msg="验证码发送失败，请稍后重试",
                error_key=error_keys.AUTH_SEND_CODE_FAILED,
                error_params={"email": normalized_email},
            ) from e

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
        normalized_email = email.strip().lower()
        normalized_ip = self._normalize_ip(client_ip)
        await self._ensure_login_rate_limit(normalized_email, normalized_ip)

        async with self._uow_factory() as uow:
            user = await uow.user.get_by_email(normalized_email)
            if user is None:
                await self._record_login_failure(normalized_email, normalized_ip)
                self._log_login_failure(
                    email=normalized_email,
                    client_ip=normalized_ip,
                    reason="user_not_found",
                )
                raise BadRequestError(
                    msg="邮箱或密码错误",
                    error_key=error_keys.AUTH_LOGIN_INVALID_CREDENTIALS,
                )
            if user.status != UserStatus.ACTIVE:
                await self._record_login_failure(normalized_email, normalized_ip)
                self._log_login_failure(
                    email=normalized_email,
                    client_ip=normalized_ip,
                    reason=f"user_status_{user.status.value}",
                )
                raise BadRequestError(
                    msg="账号状态异常，暂不可登录",
                    error_key=error_keys.AUTH_USER_STATUS_INVALID,
                    error_params={"user_id": user.id},
                )

            if not PasswordHasher.verify_password(password, user.password):
                await self._record_login_failure(normalized_email, normalized_ip)
                self._log_login_failure(
                    email=normalized_email,
                    client_ip=normalized_ip,
                    reason="password_mismatch",
                )
                raise BadRequestError(
                    msg="邮箱或密码错误",
                    error_key=error_keys.AUTH_LOGIN_INVALID_CREDENTIALS,
                )

            profile = await uow.user.get_profile_by_user_id(user.id)
            if profile is None:
                profile = UserProfile(user_id=user.id)

            user.last_login_at = datetime.now()
            user.last_login_ip = normalized_ip
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
                raise ServerError(
                    msg="登录失败，请稍后重试",
                    error_key=error_keys.AUTH_LOGIN_FAILED,
                    error_params={"user_id": user.id},
                ) from e

            await self._clear_login_failure(normalized_email, normalized_ip)

            return LoginResult(
                user=user,
                profile=profile,
                access_token=access_token,
                refresh_token=refresh_token,
                access_token_expires_in=self._setting.auth_access_token_expires_in,
                refresh_token_expires_in=self._setting.auth_refresh_token_expires_in,
            )

    async def refresh_tokens(self, refresh_token: str) -> RefreshResult:
        """刷新 Token（Refresh Token 轮转）。"""
        normalized_refresh_token = refresh_token.strip()
        if not normalized_refresh_token:
            raise BadRequestError(
                msg="Refresh Token 不能为空",
                error_key=error_keys.AUTH_REFRESH_TOKEN_REQUIRED,
            )

        try:
            consume_result = await self._refresh_token_store.consume_refresh_token(normalized_refresh_token)
        except Exception as e:
            raise ServerError(
                msg="刷新失败，请稍后重试",
                error_key=error_keys.AUTH_REFRESH_FAILED,
            ) from e

        if consume_result.status == RefreshTokenConsumeStatus.NOT_FOUND:
            raise BadRequestError(
                msg="Refresh Token 无效或已过期",
                error_key=error_keys.AUTH_REFRESH_TOKEN_INVALID,
            )

        if consume_result.status == RefreshTokenConsumeStatus.REPLAYED:
            if consume_result.user_id:
                await self._refresh_token_store.revoke_user_refresh_tokens(consume_result.user_id)
            raise BadRequestError(
                msg="检测到登录状态异常，请重新登录",
                error_key=error_keys.AUTH_REFRESH_REPLAYED,
                error_params={"user_id": consume_result.user_id},
            )

        if consume_result.user_id is None:
            raise ServerError(
                msg="刷新失败，请稍后重试",
                error_key=error_keys.AUTH_REFRESH_FAILED,
            )

        async with self._uow_factory() as uow:
            user = await uow.user.get_by_id(consume_result.user_id)
            if user is None:
                await self._refresh_token_store.revoke_user_refresh_tokens(consume_result.user_id)
                raise BadRequestError(
                    msg="用户不存在，请重新登录",
                    error_key=error_keys.AUTH_USER_NOT_FOUND,
                    error_params={"user_id": consume_result.user_id},
                )
            if user.status != UserStatus.ACTIVE:
                await self._refresh_token_store.revoke_user_refresh_tokens(user.id)
                raise BadRequestError(
                    msg="账号状态异常，暂不可登录",
                    error_key=error_keys.AUTH_USER_STATUS_INVALID,
                    error_params={"user_id": user.id},
                )

        access_token = AuthTokenManager.generate_access_token(
            user_id=user.id,
            email=user.email,
            secret_key=self._setting.auth_jwt_secret,
            algorithm=self._setting.auth_jwt_algorithm,
            expires_in_seconds=self._setting.auth_access_token_expires_in,
        )
        next_refresh_token = AuthTokenManager.generate_refresh_token()

        try:
            await self._refresh_token_store.save_refresh_token(
                refresh_token=next_refresh_token,
                user_id=user.id,
                email=user.email,
                expires_in_seconds=self._setting.auth_refresh_token_expires_in,
            )
        except Exception as e:
            raise ServerError(
                msg="刷新失败，请稍后重试",
                error_key=error_keys.AUTH_REFRESH_FAILED,
                error_params={"user_id": user.id},
            ) from e

        return RefreshResult(
            access_token=access_token,
            refresh_token=next_refresh_token,
            access_token_expires_in=self._setting.auth_access_token_expires_in,
            refresh_token_expires_in=self._setting.auth_refresh_token_expires_in,
        )

    async def logout(
            self,
            refresh_token: str,
            access_token: str,
            access_token_expires_in_seconds: int,
    ) -> None:
        """退出登录（删除 Refresh Token + 拉黑当前 Access Token）。"""
        normalized_refresh_token = refresh_token.strip()
        if not normalized_refresh_token:
            raise BadRequestError(
                msg="Refresh Token 不能为空",
                error_key=error_keys.AUTH_REFRESH_TOKEN_REQUIRED,
            )
        normalized_access_token = access_token.strip()
        if not normalized_access_token:
            raise BadRequestError(
                msg="Access Token 不能为空",
                error_key=error_keys.AUTH_ACCESS_TOKEN_REQUIRED,
            )
        try:
            await self._refresh_token_store.delete_refresh_token(normalized_refresh_token)
            if self._access_token_blacklist_store is None:
                raise ServerError(
                    msg="认证服务未配置，请联系管理员",
                    error_key=error_keys.AUTH_SERVICE_NOT_CONFIGURED,
                )
            await self._access_token_blacklist_store.add_access_token_to_blacklist(
                access_token=normalized_access_token,
                expires_in_seconds=access_token_expires_in_seconds,
            )
        except ServerError:
            raise
        except Exception as e:
            raise ServerError(
                msg="退出失败，请稍后重试",
                error_key=error_keys.AUTH_LOGOUT_FAILED,
            ) from e

    def _ensure_register_verification_code_store(self) -> RegisterVerificationCodeStore:
        if self._register_verification_code_store is None:
            raise ServerError(
                msg="验证码服务未配置，请联系管理员",
                error_key=error_keys.AUTH_REGISTER_CODE_SERVICE_NOT_CONFIGURED,
            )
        return self._register_verification_code_store

    def _ensure_email_sender(self) -> EmailSender:
        if self._email_sender is None:
            raise ServerError(
                msg="邮件服务未配置，请联系管理员",
                error_key=error_keys.AUTH_EMAIL_SERVICE_NOT_CONFIGURED,
            )
        return self._email_sender

    @staticmethod
    def _normalize_ip(client_ip: Optional[str]) -> Optional[str]:
        if client_ip is None:
            return None
        normalized = client_ip.strip()
        return normalized or None

    @staticmethod
    def _mask_email(email: str) -> str:
        normalized_email = email.strip().lower()
        if "@" not in normalized_email:
            return "***"
        local_part, domain = normalized_email.split("@", 1)
        if len(local_part) <= 1:
            masked_local_part = "*"
        elif len(local_part) == 2:
            masked_local_part = f"{local_part[0]}*"
        else:
            masked_local_part = f"{local_part[0]}***{local_part[-1]}"
        return f"{masked_local_part}@{domain}"

    def _log_login_failure(self, email: str, client_ip: Optional[str], reason: str) -> None:
        logger.warning(
            "认证失败 event=login email=%s ip=%s reason=%s",
            self._mask_email(email),
            client_ip or "-",
            reason,
        )

    async def _ensure_login_rate_limit(self, email: str, client_ip: Optional[str]) -> None:
        if self._auth_rate_limit_store is None:
            return

        max_attempts = max(1, int(self._setting.auth_login_rate_limit_max_attempts))
        ip_attempts = 0
        if client_ip is not None:
            ip_attempts = await self._auth_rate_limit_store.get_login_attempt_count_by_ip(client_ip)
        email_attempts = await self._auth_rate_limit_store.get_login_attempt_count_by_email(email)

        if ip_attempts >= max_attempts or email_attempts >= max_attempts:
            self._log_login_failure(
                email=email,
                client_ip=client_ip,
                reason="rate_limited",
            )
            raise TooManyRequestsError(
                msg="登录尝试过于频繁，请稍后重试",
                error_key=error_keys.AUTH_LOGIN_RATE_LIMITED,
            )

    async def _record_login_failure(self, email: str, client_ip: Optional[str]) -> None:
        if self._auth_rate_limit_store is None:
            return
        window_seconds = max(1, int(self._setting.auth_login_rate_limit_window_seconds))
        await self._auth_rate_limit_store.increase_login_attempt_count(
            ip=client_ip,
            email=email,
            expires_in_seconds=window_seconds,
        )

    async def _clear_login_failure(self, email: str, client_ip: Optional[str]) -> None:
        if self._auth_rate_limit_store is None:
            return
        await self._auth_rate_limit_store.clear_login_attempt_count(
            ip=client_ip,
            email=email,
        )

    async def _ensure_register_send_code_rate_limit(self, client_ip: Optional[str]) -> None:
        if self._auth_rate_limit_store is None or client_ip is None:
            return

        window_seconds = max(1, int(self._setting.auth_send_code_rate_limit_window_seconds))
        max_attempts = max(1, int(self._setting.auth_send_code_rate_limit_max_attempts))
        current_attempts = await self._auth_rate_limit_store.increase_register_send_code_attempt_count_by_ip(
            ip=client_ip,
            expires_in_seconds=window_seconds,
        )
        if current_attempts > max_attempts:
            logger.warning(
                "认证失败 event=send_register_code ip=%s reason=rate_limited",
                client_ip,
            )
            raise TooManyRequestsError(
                msg="验证码发送过于频繁，请稍后重试",
                error_key=error_keys.AUTH_SEND_CODE_RATE_LIMITED,
                error_params={"client_ip": client_ip},
            )
