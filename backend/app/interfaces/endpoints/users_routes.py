#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/16 12:15
@Author : caixiaorong01@outlook.com
@File   : users_routes.py
"""
from fastapi import APIRouter, Depends

from app.application.service import UserService
from app.domain.external import AccessTokenBlacklistStore, RefreshTokenStore
from app.domain.models import User, UserProfile
from app.interfaces.dependencies.auth import (
    AuthContext,
    get_current_auth_context,
    get_access_token_ttl_seconds,
    get_current_user,
)
from app.interfaces.dependencies.services import (
    get_access_token_blacklist_store,
    get_refresh_token_store,
    get_user_service,
)
from app.interfaces.schemas import Response
from app.interfaces.schemas.auth import (
    CurrentUserResponse,
    UpdateCurrentUserRequest,
    UpdateCurrentUserResponse,
    UpdatePasswordRequest,
    UpdatePasswordResponse,
)

router = APIRouter(prefix="/users", tags=["用户模块"])


def _build_current_user_response(user: User, profile: UserProfile) -> CurrentUserResponse:
    return CurrentUserResponse(
        user_id=user.id,
        email=user.email,
        nickname=profile.nickname,
        avatar_url=profile.avatar_url,
        timezone=profile.timezone,
        locale=profile.locale,
        auth_provider=user.auth_provider,
        status=user.status,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login_at=user.last_login_at,
        last_login_ip=user.last_login_ip,
    )


@router.get(
    path="/me",
    response_model=Response[CurrentUserResponse],
    summary="获取当前用户信息",
    description="返回当前登录用户的资料信息，用于个人中心初始化与登录后资料刷新",
)
async def get_current_user_profile(
        current_user: User = Depends(get_current_user),
        user_service: UserService = Depends(get_user_service),
) -> Response[CurrentUserResponse]:
    """获取当前用户信息接口"""
    user, profile = await user_service.get_current_user_profile(current_user.id)
    return Response.success(
        msg="获取成功",
        data=_build_current_user_response(user=user, profile=profile),
    )


@router.patch(
    path="/me",
    response_model=Response[UpdateCurrentUserResponse],
    summary="更新当前用户资料",
    description="支持部分更新 nickname、avatar_url、timezone、locale",
)
async def update_current_user_profile(
        payload: UpdateCurrentUserRequest,
        current_user: User = Depends(get_current_user),
        user_service: UserService = Depends(get_user_service),
) -> Response[UpdateCurrentUserResponse]:
    """更新当前用户资料接口"""
    user, profile = await user_service.update_current_user_profile(
        user_id=current_user.id,
        updates=payload.model_dump(exclude_unset=True),
    )
    return Response.success(
        msg="更新成功",
        data=UpdateCurrentUserResponse(
            user=_build_current_user_response(user=user, profile=profile),
        ),
    )


@router.patch(
    path="/me/password",
    response_model=Response[UpdatePasswordResponse],
    summary="更新当前用户密码",
    description="接收 old_password/new_password/confirm_password，校验后更新加盐密码",
)
async def update_current_user_password(
        payload: UpdatePasswordRequest,
        auth_context: AuthContext = Depends(get_current_auth_context),
        user_service: UserService = Depends(get_user_service),
        access_token_blacklist_store: AccessTokenBlacklistStore = Depends(get_access_token_blacklist_store),
        refresh_token_store: RefreshTokenStore = Depends(get_refresh_token_store),
) -> Response[UpdatePasswordResponse]:
    """更新当前用户密码接口"""
    await user_service.update_current_user_password(
        user_id=auth_context.user.id,
        old_password=payload.old_password,
        new_password=payload.new_password,
        confirm_password=payload.confirm_password,
        refresh_token_store=refresh_token_store,
        access_token_blacklist_store=access_token_blacklist_store,
        current_access_token=auth_context.access_token,
        access_token_expires_in_seconds=get_access_token_ttl_seconds(auth_context.token_payload),
    )
    return Response.success(
        msg="密码更新成功",
        data=UpdatePasswordResponse(success=True),
    )
