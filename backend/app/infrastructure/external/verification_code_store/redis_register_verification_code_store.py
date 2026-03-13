#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 23:06
@Author : caixiaorong01@outlook.com
@File   : redis_register_verification_code_store.py
"""
from app.domain.external import RegisterVerificationCodeStore
from app.infrastructure.storage import RedisClient


class RedisRegisterVerificationCodeStore(RegisterVerificationCodeStore):
    """基于 Redis 的注册验证码存储实现"""

    _KEY_PREFIX = "auth:register_verification_code"

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis_client = redis_client

    def _build_key(self, email: str) -> str:
        return f"{self._KEY_PREFIX}:{email}"

    async def save_verification_code(
            self,
            email: str,
            verification_code: str,
            expires_in_seconds: int,
    ) -> None:
        key = self._build_key(email=email)
        await self._redis_client.client.set(key, verification_code, ex=expires_in_seconds)

    async def verify_and_consume_verification_code(
            self,
            email: str,
            verification_code: str,
    ) -> bool:
        """原子校验并消费验证码，防止重复使用。"""
        key = self._build_key(email=email)
        script = """
        local current = redis.call("GET", KEYS[1])
        if not current then
            return 0
        end
        if current ~= ARGV[1] then
            return -1
        end
        redis.call("DEL", KEYS[1])
        return 1
        """
        result = await self._redis_client.client.eval(script, 1, key, verification_code)
        return int(result) == 1
