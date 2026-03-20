#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 21:29
@Author : caixiaorong01@outlook.com
@File   : redis_refresh_token_store.py
"""
import json
from datetime import datetime
from typing import Any, cast

from app.domain.external.refresh_token_store import (
    RefreshTokenStore,
    RefreshTokenConsumeResult,
    RefreshTokenConsumeStatus,
)
from app.infrastructure.storage import RedisClient


class RedisRefreshTokenStore(RefreshTokenStore):
    """基于 Redis 的 Refresh Token 存储实现"""

    _REFRESH_TOKEN_KEY_PREFIX = "auth:refresh_token"
    _USER_REFRESH_TOKEN_KEY_PREFIX = "auth:user_refresh_tokens"

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis_client = redis_client

    def _build_token_key(self, refresh_token: str) -> str:
        return f"{self._REFRESH_TOKEN_KEY_PREFIX}:{refresh_token}"

    def _build_user_tokens_key(self, user_id: str) -> str:
        return f"{self._USER_REFRESH_TOKEN_KEY_PREFIX}:{user_id}"

    async def save_refresh_token(
            self,
            refresh_token: str,
            user_id: str,
            email: str,
            expires_in_seconds: int,
    ) -> None:
        """保存 Refresh Token 及用户维度索引"""
        token_key = self._build_token_key(refresh_token)
        user_tokens_key = self._build_user_tokens_key(user_id)
        payload = json.dumps(
            {
                "token": refresh_token,
                "user_id": user_id,
                "email": email,
                "is_used": False,
                "issued_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
        )

        pipeline = self._redis_client.client.pipeline(transaction=True)
        # Pipeline 的 set/sadd/expire 仅入队命令；实际 I/O 在 execute() 执行。
        # 这里通过类型收窄规避静态检查器把它们误判为“未 await 协程”。
        pipeline_commands = cast(Any, pipeline)
        pipeline_commands.set(token_key, payload, ex=expires_in_seconds)
        pipeline_commands.sadd(user_tokens_key, refresh_token)
        pipeline_commands.expire(user_tokens_key, expires_in_seconds)
        await pipeline.execute()

    async def consume_refresh_token(self, refresh_token: str) -> RefreshTokenConsumeResult:
        """消费 Refresh Token（原子校验并标记已使用）。"""
        token_key = self._build_token_key(refresh_token)
        script = """
        local raw = redis.call("GET", KEYS[1])
        if not raw then
            return {0}
        end

        local payload = cjson.decode(raw)
        if payload["is_used"] == true then
            return {2, payload["user_id"] or "", payload["email"] or ""}
        end

        payload["is_used"] = true
        payload["used_at"] = ARGV[1]
        redis.call("SET", KEYS[1], cjson.encode(payload), "KEEPTTL")
        return {1, payload["user_id"] or "", payload["email"] or ""}
        """
        script_result = await self._redis_client.client.eval(
            script,
            1,
            token_key,
            datetime.now().isoformat(),
        )

        result = cast(list[Any] | int, script_result)
        if isinstance(result, int):
            status_code = int(result)
            user_id = None
            email = None
        else:
            status_code = int(result[0]) if len(result) >= 1 else 0
            user_id = str(result[1]) if len(result) >= 2 and result[1] else None
            email = str(result[2]) if len(result) >= 3 and result[2] else None

        if status_code == 1:
            return RefreshTokenConsumeResult(
                status=RefreshTokenConsumeStatus.CONSUMED,
                user_id=user_id,
                email=email,
            )
        if status_code == 2:
            return RefreshTokenConsumeResult(
                status=RefreshTokenConsumeStatus.REPLAYED,
                user_id=user_id,
                email=email,
            )
        return RefreshTokenConsumeResult(status=RefreshTokenConsumeStatus.NOT_FOUND)

    async def revoke_user_refresh_tokens(self, user_id: str) -> None:
        """删除用户全部 Refresh Token。"""
        user_tokens_key = self._build_user_tokens_key(user_id)
        user_tokens = await self._redis_client.client.smembers(user_tokens_key)

        pipeline = self._redis_client.client.pipeline(transaction=True)
        pipeline_commands = cast(Any, pipeline)
        for refresh_token in user_tokens:
            pipeline_commands.delete(self._build_token_key(refresh_token))
        pipeline_commands.delete(user_tokens_key)
        await pipeline.execute()

    async def delete_refresh_token(self, refresh_token: str) -> None:
        """删除指定 Refresh Token。"""
        token_key = self._build_token_key(refresh_token)
        raw_payload = await self._redis_client.client.get(token_key)

        user_id: str | None = None
        if raw_payload:
            try:
                payload = json.loads(raw_payload)
                user_id_value = payload.get("user_id")
                if isinstance(user_id_value, str) and user_id_value:
                    user_id = user_id_value
            except json.JSONDecodeError:
                user_id = None

        pipeline = self._redis_client.client.pipeline(transaction=True)
        pipeline_commands = cast(Any, pipeline)
        pipeline_commands.delete(token_key)
        if user_id is not None:
            pipeline_commands.srem(self._build_user_tokens_key(user_id), refresh_token)
        await pipeline.execute()
