#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/16 15:25
@Author : caixiaorong01@outlook.com
@File   : redis_auth_rate_limit_store.py
"""
import hashlib
from typing import Any, cast

from app.domain.external import AuthRateLimitStore
from app.infrastructure.storage import RedisClient


class RedisAuthRateLimitStore(AuthRateLimitStore):
    """基于 Redis 的认证限流实现"""

    _LOGIN_IP_KEY_PREFIX = "auth:rate_limit:login:ip"
    _LOGIN_EMAIL_KEY_PREFIX = "auth:rate_limit:login:email"
    _REGISTER_SEND_CODE_IP_KEY_PREFIX = "auth:rate_limit:register_send_code:ip"

    _INCREASE_WITH_EXPIRE_SCRIPT = """
    local current = redis.call("INCR", KEYS[1])
    if current == 1 then
        redis.call("EXPIRE", KEYS[1], ARGV[1])
    end
    return current
    """

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis_client = redis_client

    def _build_login_ip_key(self, ip: str) -> str:
        return f"{self._LOGIN_IP_KEY_PREFIX}:{ip.strip()}"

    def _build_login_email_key(self, email: str) -> str:
        normalized_email = email.strip().lower()
        email_hash = hashlib.sha256(normalized_email.encode("utf-8")).hexdigest()
        return f"{self._LOGIN_EMAIL_KEY_PREFIX}:{email_hash}"

    def _build_register_send_code_ip_key(self, ip: str) -> str:
        return f"{self._REGISTER_SEND_CODE_IP_KEY_PREFIX}:{ip.strip()}"

    async def _get_count(self, key: str) -> int:
        raw = await self._redis_client.client.get(key)
        if raw is None:
            return 0
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    async def _increase_count(self, key: str, expires_in_seconds: int) -> int:
        ttl_seconds = max(1, int(expires_in_seconds))
        result = await self._redis_client.client.eval(
            self._INCREASE_WITH_EXPIRE_SCRIPT,
            1,
            key,
            ttl_seconds,
        )
        if isinstance(result, int):
            return max(0, result)
        try:
            return max(0, int(result))
        except (TypeError, ValueError):
            return 0

    async def get_login_attempt_count_by_ip(self, ip: str) -> int:
        return await self._get_count(self._build_login_ip_key(ip))

    async def get_login_attempt_count_by_email(self, email: str) -> int:
        return await self._get_count(self._build_login_email_key(email))

    async def increase_login_attempt_count(
            self,
            ip: str | None,
            email: str,
            expires_in_seconds: int,
    ) -> None:
        if ip is not None and ip.strip():
            await self._increase_count(
                self._build_login_ip_key(ip),
                expires_in_seconds=expires_in_seconds,
            )
        await self._increase_count(
            self._build_login_email_key(email),
            expires_in_seconds=expires_in_seconds,
        )

    async def clear_login_attempt_count(self, ip: str | None, email: str) -> None:
        pipeline = self._redis_client.client.pipeline(transaction=True)
        pipeline_commands = cast(Any, pipeline)
        if ip is not None and ip.strip():
            pipeline_commands.delete(self._build_login_ip_key(ip))
        pipeline_commands.delete(self._build_login_email_key(email))
        await pipeline.execute()

    async def increase_register_send_code_attempt_count_by_ip(
            self,
            ip: str,
            expires_in_seconds: int,
    ) -> int:
        return await self._increase_count(
            self._build_register_send_code_ip_key(ip),
            expires_in_seconds=expires_in_seconds,
        )
