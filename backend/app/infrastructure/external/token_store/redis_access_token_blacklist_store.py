#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/14 11:23
@Author : caixiaorong01@outlook.com
@File   : redis_access_token_blacklist_store.py
"""
import hashlib

from app.domain.external import AccessTokenBlacklistStore
from app.infrastructure.storage import RedisClient


class RedisAccessTokenBlacklistStore(AccessTokenBlacklistStore):
    """基于 Redis 的 Access Token 黑名单实现"""

    _ACCESS_TOKEN_BLACKLIST_KEY_PREFIX = "auth:access_token_blacklist"

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis_client = redis_client

    def _build_blacklist_key(self, access_token: str) -> str:
        # 使用哈希作为 key，避免在 Redis 中直接暴露明文 token。
        token_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
        return f"{self._ACCESS_TOKEN_BLACKLIST_KEY_PREFIX}:{token_hash}"

    async def add_access_token_to_blacklist(
            self,
            access_token: str,
            expires_in_seconds: int,
    ) -> None:
        ttl_seconds = max(1, int(expires_in_seconds))
        key = self._build_blacklist_key(access_token)
        await self._redis_client.client.set(key, "1", ex=ttl_seconds)

    async def is_access_token_blacklisted(self, access_token: str) -> bool:
        key = self._build_blacklist_key(access_token)
        return bool(await self._redis_client.client.exists(key))
