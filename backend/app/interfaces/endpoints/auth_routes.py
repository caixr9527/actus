#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 19:36
@Author : caixiaorong01@outlook.com
@File   : auth_routes.py
"""
from typing import Optional

from fastapi import APIRouter, Depends, Request

from app.application.service import AuthService
from app.interfaces.dependencies.auth import (
    AuthContext,
    get_current_auth_context,
    get_access_token_ttl_seconds,
)
from app.interfaces.dependencies.services import get_auth_service
from app.interfaces.schemas import Response
from app.interfaces.schemas.auth import (
    SendRegisterCodeRequest,
    SendRegisterCodeResponse,
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    LogoutRequest,
    LogoutResponse,
    TokenPairResponse,
    CurrentUserResponse,
)

router = APIRouter(prefix="/auth", tags=["认证模块"])


def _get_client_ip(http_request: Request) -> Optional[str]:
    """提取客户端IP（优先代理头）"""
    x_forwarded_for = http_request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    x_real_ip = http_request.headers.get("x-real-ip")
    if x_real_ip:
        return x_real_ip.strip()
    if http_request.client is not None:
        return http_request.client.host
    return None


@router.post(
    path="/register/send-code",
    response_model=Response[SendRegisterCodeResponse],
    summary="发送注册验证码",
    description="按邮箱发送注册验证码（开发环境默认不强制验证码校验，生产环境默认强制）",
)
async def send_register_verification_code(
        payload: SendRegisterCodeRequest,
        auth_service: AuthService = Depends(get_auth_service),
) -> Response[SendRegisterCodeResponse]:
    """发送注册验证码接口"""
    result = await auth_service.send_register_verification_code(payload.email)
    return Response.success(
        msg="验证码发送成功" if result.verification_required else "当前环境无需验证码校验",
        data=SendRegisterCodeResponse(
            email=payload.email.strip().lower(),
            verification_required=result.verification_required,
            expires_in_seconds=result.expires_in_seconds,
        ),
    )


@router.post(
    path="/register",
    response_model=Response[RegisterResponse],
    summary="邮箱注册",
    description="通过邮箱与密码完成新用户注册，创建用户与默认用户资料",
)
async def register(
        request: RegisterRequest,
        auth_service: AuthService = Depends(get_auth_service),
) -> Response[RegisterResponse]:
    """邮箱注册接口"""
    user = await auth_service.register(
        email=request.email,
        password=request.password,
        confirm_password=request.confirm_password,
        verification_code=request.verification_code,
    )
    return Response.success(
        msg="注册成功",
        data=RegisterResponse(
            user_id=user.id,
            email=user.email,
            auth_provider=user.auth_provider,
            status=user.status,
            created_at=user.created_at,
        )
    )


@router.post(
    path="/login",
    response_model=Response[LoginResponse],
    summary="邮箱登录",
    description="通过邮箱与密码登录，签发Access Token与Refresh Token",
)
async def login(
        payload: LoginRequest,
        http_request: Request,
        auth_service: AuthService = Depends(get_auth_service),
) -> Response[LoginResponse]:
    """邮箱登录接口"""
    result = await auth_service.login(
        email=payload.email,
        password=payload.password,
        client_ip=_get_client_ip(http_request),
    )
    return Response.success(
        msg="登录成功",
        data=LoginResponse(
            tokens=TokenPairResponse(
                access_token=result.access_token,
                refresh_token=result.refresh_token,
                access_token_expires_in=result.access_token_expires_in,
                refresh_token_expires_in=result.refresh_token_expires_in,
            ),
            user=CurrentUserResponse(
                user_id=result.user.id,
                email=result.user.email,
                nickname=result.profile.nickname,
                avatar_url=result.profile.avatar_url,
                timezone=result.profile.timezone,
                locale=result.profile.locale,
                auth_provider=result.user.auth_provider,
                status=result.user.status,
                created_at=result.user.created_at,
                updated_at=result.user.updated_at,
                last_login_at=result.user.last_login_at,
                last_login_ip=result.user.last_login_ip,
            ),
        ),
    )


@router.post(
    path="/refresh",
    response_model=Response[RefreshTokenResponse],
    summary="刷新登录 Token",
    description="通过 Refresh Token 轮转签发新的 Access Token 与 Refresh Token",
)
async def refresh_tokens(
        payload: RefreshTokenRequest,
        auth_service: AuthService = Depends(get_auth_service),
) -> Response[RefreshTokenResponse]:
    """刷新 Token 接口"""
    result = await auth_service.refresh_tokens(refresh_token=payload.refresh_token)
    return Response.success(
        msg="刷新成功",
        data=RefreshTokenResponse(
            tokens=TokenPairResponse(
                access_token=result.access_token,
                refresh_token=result.refresh_token,
                access_token_expires_in=result.access_token_expires_in,
                refresh_token_expires_in=result.refresh_token_expires_in,
            )
        ),
    )


@router.post(
    path="/logout",
    response_model=Response[LogoutResponse],
    summary="退出登录",
    description="删除当前会话对应的 Refresh Token，使登录态失效",
)
async def logout(
        payload: LogoutRequest,
        auth_context: AuthContext = Depends(get_current_auth_context),
        auth_service: AuthService = Depends(get_auth_service),
) -> Response[LogoutResponse]:
    """退出登录接口"""
    await auth_service.logout(
        refresh_token=payload.refresh_token,
        access_token=auth_context.access_token,
        access_token_expires_in_seconds=get_access_token_ttl_seconds(auth_context.token_payload),
    )
    return Response.success(
        msg="退出成功",
        data=LogoutResponse(success=True),
    )
