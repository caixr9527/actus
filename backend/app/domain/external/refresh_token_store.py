#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 21:26
@Author : caixiaorong01@outlook.com
@File   : refresh_token_store.py
"""
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class RefreshTokenConsumeStatus(str, Enum):
    """Refresh Token 消费结果状态"""

    CONSUMED = "consumed"
    REPLAYED = "replayed"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class RefreshTokenConsumeResult:
    """Refresh Token 消费结果"""

    status: RefreshTokenConsumeStatus
    user_id: str | None = None
    email: str | None = None


class RefreshTokenStore(Protocol):
    """Refresh Token 存储协议"""

    async def save_refresh_token(
            self,
            refresh_token: str,
            user_id: str,
            email: str,
            expires_in_seconds: int,
    ) -> None:
        """保存 Refresh Token"""
        ...

    async def consume_refresh_token(
            self,
            refresh_token: str,
    ) -> RefreshTokenConsumeResult:
        """消费 Refresh Token（单次有效）"""
        ...

    async def revoke_user_refresh_tokens(self, user_id: str) -> None:
        """使指定用户的所有 Refresh Token 失效"""
        ...

    async def delete_refresh_token(self, refresh_token: str) -> None:
        """删除指定 Refresh Token（用于退出）"""
        ...
