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

from app.domain.external.refresh_token_store import RefreshTokenStore
from app.infrastructure.storage import RedisClient


class RedisRefreshTokenStore(RefreshTokenStore):
    """基于 Redis 的 Refresh Token 存储实现"""

    _REFRESH_TOKEN_KEY_PREFIX = "auth:refresh_token"
    _USER_REFRESH_TOKEN_KEY_PREFIX = "auth:user_refresh_tokens"

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis_client = redis_client

    async def save_refresh_token(
            self,
            refresh_token: str,
            user_id: str,
            email: str,
            expires_in_seconds: int,
    ) -> None:
        """保存 Refresh Token 及用户维度索引"""
        token_key = f"{self._REFRESH_TOKEN_KEY_PREFIX}:{refresh_token}"
        user_tokens_key = f"{self._USER_REFRESH_TOKEN_KEY_PREFIX}:{user_id}"
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
