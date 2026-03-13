#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 18:58
@Author : caixiaorong01@outlook.com
@File   : auth.py
"""
import re
from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator

from app.domain.models import UserStatus, User, UserProfile

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
PASSWORD_ALLOWED_REGEX = re.compile(r"^[A-Za-z0-9!@#$%^&*._\-]+$")
VERIFICATION_CODE_REGEX = re.compile(r"^\d{6}$")


def validate_password_strength(value: str) -> str:
    """密码强度校验：8位以上，含字母+数字，且只允许常见符号"""
    if len(value) < 8:
        raise ValueError("密码长度不能少于8位")
    if not re.search(r"[A-Za-z]", value):
        raise ValueError("密码必须包含字母")
    if not re.search(r"\d", value):
        raise ValueError("密码必须包含数字")
    if not PASSWORD_ALLOWED_REGEX.fullmatch(value):
        raise ValueError("密码仅允许字母、数字和常见符号 !@#$%^&*._-")
    return value


class RegisterRequest(BaseModel):
    """邮箱注册请求结构"""
    email: str
    password: str = Field(min_length=8)
    verification_code: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """邮箱格式基础校验"""
        if not EMAIL_REGEX.match(value):
            raise ValueError("邮箱格式不正确")
        return value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        """注册密码强度校验"""
        return validate_password_strength(value)

    @field_validator("verification_code")
    @classmethod
    def validate_verification_code(cls, value: Optional[str]) -> Optional[str]:
        """验证码格式校验（仅在传值时校验）"""
        if value is None:
            return None
        if not VERIFICATION_CODE_REGEX.match(value):
            raise ValueError("验证码格式不正确，应为6位数字")
        return value


class RegisterResponse(BaseModel):
    """邮箱注册响应结构"""
    user_id: str
    email: str
    auth_provider: str = "email"
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime


class SendRegisterCodeRequest(BaseModel):
    """发送注册验证码请求结构"""
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """邮箱格式基础校验"""
        if not EMAIL_REGEX.match(value):
            raise ValueError("邮箱格式不正确")
        return value


class SendRegisterCodeResponse(BaseModel):
    """发送注册验证码响应结构"""
    email: str
    verification_required: bool
    expires_in_seconds: int


class LoginRequest(BaseModel):
    """邮箱登录请求结构"""
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """邮箱格式基础校验"""
        if not EMAIL_REGEX.match(value):
            raise ValueError("邮箱格式不正确")
        return value


class TokenPairResponse(BaseModel):
    """Token对响应结构"""
    access_token: str
    refresh_token: str
    token_type: Literal["Bearer"] = "Bearer"
    access_token_expires_in: int = 1800
    refresh_token_expires_in: int = 7 * 24 * 60 * 60


class CurrentUserResponse(BaseModel):
    """当前用户信息响应结构"""
    user_id: str
    email: str
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: str = "Asia/Shanghai"
    locale: str = "zh-CN"
    auth_provider: str = "email"
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    last_login_ip: Optional[str] = None


class LoginResponse(BaseModel):
    """邮箱登录响应结构"""
    tokens: TokenPairResponse
    user: CurrentUserResponse


class RefreshTokenRequest(BaseModel):
    """刷新Token请求结构"""
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """刷新Token响应结构"""
    tokens: TokenPairResponse


class UpdateCurrentUserRequest(BaseModel):
    """更新个人资料请求结构"""
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: Optional[str] = None
    locale: Optional[str] = None


class UpdateCurrentUserResponse(BaseModel):
    """更新个人资料响应结构"""
    user: CurrentUserResponse


class UpdatePasswordRequest(BaseModel):
    """更新密码请求结构"""
    old_password: str = Field(min_length=8)
    new_password: str = Field(min_length=8)

    @field_validator("old_password", "new_password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        """修改密码强度校验"""
        return validate_password_strength(value)


class LoginResult(BaseModel):
    """登录结果数据"""
    user: User
    profile: UserProfile
    access_token: str
    refresh_token: str
    access_token_expires_in: int
    refresh_token_expires_in: int


class RegisterVerificationCodeResult(BaseModel):
    """发送注册验证码结果"""

    verification_required: bool
    expires_in_seconds: int
