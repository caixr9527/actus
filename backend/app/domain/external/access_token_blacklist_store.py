#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/14 11:22
@Author : caixiaorong01@outlook.com
@File   : access_token_blacklist_store.py
"""
from typing import Protocol


class AccessTokenBlacklistStore(Protocol):
    """Access Token 黑名单存储协议"""

    async def add_access_token_to_blacklist(
            self,
            access_token: str,
            expires_in_seconds: int,
    ) -> None:
        """将 Access Token 加入黑名单"""
        ...

    async def is_access_token_blacklisted(self, access_token: str) -> bool:
        """判断 Access Token 是否在黑名单中"""
        ...
