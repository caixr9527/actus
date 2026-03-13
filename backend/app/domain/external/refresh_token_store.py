#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 21:26
@Author : caixiaorong01@outlook.com
@File   : refresh_token_store.py
"""
from typing import Protocol


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
