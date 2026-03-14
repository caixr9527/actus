#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/14 11:28
@Author : caixiaorong01@outlook.com
@File   : auth_dependencies.py
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import jwt
from fastapi import Depends, WebSocketException, status
from jwt import ExpiredSignatureError, InvalidTokenError
from starlette.requests import HTTPConnection

from app.application.errors import UnauthorizedError, ServerError
from app.domain.models import User, UserStatus
from app.interfaces.dependencies.services import get_access_token_blacklist_store
from app.infrastructure.storage import get_uow
from core.config import get_settings


@dataclass(frozen=True)
class AuthContext:
    """认证上下文"""

    user: User
    access_token: str
    token_payload: dict[str, Any]


def _is_websocket_connection(connection: HTTPConnection) -> bool:
    return connection.scope.get("type") == "websocket"


def _extract_bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise UnauthorizedError(msg="缺少认证信息，请先登录")
    auth_value = authorization.strip()
    prefix = "Bearer "
    if not auth_value.startswith(prefix):
        raise UnauthorizedError(msg="认证格式错误，请使用 Bearer Token")
    access_token = auth_value[len(prefix):].strip()
    if not access_token:
        raise UnauthorizedError(msg="Access Token 不能为空")
    return access_token


def _decode_access_token(access_token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(
            access_token,
            settings.auth_jwt_secret,
            algorithms=[settings.auth_jwt_algorithm],
        )
    except ExpiredSignatureError as e:
        raise UnauthorizedError(msg="登录已过期，请重新登录") from e
    except InvalidTokenError as e:
        raise UnauthorizedError(msg="Access Token 无效，请重新登录") from e

    if payload.get("type") != "access":
        raise UnauthorizedError(msg="Token 类型错误，请重新登录")
    user_id = payload.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        raise UnauthorizedError(msg="Token 用户信息缺失，请重新登录")
    return payload


def get_access_token_ttl_seconds(token_payload: dict[str, Any]) -> int:
    """根据 token payload 计算剩余有效期（秒）"""
    exp = token_payload.get("exp")
    if not isinstance(exp, int):
        return 1
    now_timestamp = int(datetime.now(timezone.utc).timestamp())
    return max(1, exp - now_timestamp)


async def get_current_auth_context(connection: HTTPConnection) -> AuthContext:
    """解析并校验当前请求的认证上下文（支持 HTTP + WebSocket）"""
    try:
        access_token = _extract_bearer_token(connection.headers.get("Authorization"))
        blacklist_store = get_access_token_blacklist_store()
        try:
            if await blacklist_store.is_access_token_blacklisted(access_token):
                raise UnauthorizedError(msg="登录状态已失效，请重新登录")
        except UnauthorizedError:
            raise
        except Exception as e:
            raise ServerError(msg="认证服务异常，请稍后重试") from e

        token_payload = _decode_access_token(access_token)
        user_id = str(token_payload["user_id"])

        async with get_uow() as uow:
            user = await uow.user.get_by_id(user_id)
        if user is None:
            raise UnauthorizedError(msg="用户不存在，请重新登录")
        if user.status != UserStatus.ACTIVE:
            raise UnauthorizedError(msg="账号状态异常，暂不可访问")

        connection.state.current_user = user
        connection.state.current_access_token = access_token
        connection.state.current_access_token_payload = token_payload
        return AuthContext(user=user, access_token=access_token, token_payload=token_payload)
    except UnauthorizedError as e:
        if _is_websocket_connection(connection):
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=e.msg,
            ) from e
        raise


async def get_current_user(auth_context: AuthContext = Depends(get_current_auth_context)) -> User:
    """获取当前登录用户"""
    return auth_context.user
