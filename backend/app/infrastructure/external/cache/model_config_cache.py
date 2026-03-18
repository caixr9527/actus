#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 16:15
@Author : caixiaorong01@outlook.com
@File   : model_config_cache.py
"""
import logging

from pydantic import TypeAdapter

from app.domain.external import ModelConfigCacheStore
from app.domain.models import LLMModelConfig
from app.infrastructure.storage.redis import RedisClient

logger = logging.getLogger(__name__)

MODELS_ADAPTER = TypeAdapter(list[LLMModelConfig])


class ModelConfigCache(ModelConfigCacheStore):
    """模型配置 Redis 缓存"""

    MODELS_CACHE_KEY = "llm:model_configs:all"
    DEFAULT_MODEL_CACHE_KEY = "llm:model_configs:default"

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis_client = redis_client

    async def get_models(self) -> tuple[list[LLMModelConfig], str | None] | None:
        """获取缓存中的启用模型列表与默认模型 ID。"""
        try:
            cached_models, cached_default_model_id = await self._redis_client.client.mget(
                [self.MODELS_CACHE_KEY, self.DEFAULT_MODEL_CACHE_KEY]
            )
            if cached_models is None or cached_default_model_id is None:
                return None
            return MODELS_ADAPTER.validate_json(cached_models), cached_default_model_id or None
        except Exception as e:
            logger.warning(f"读取模型配置 Redis 缓存失败，已忽略: {e}")
            return None

    async def save_models(
            self,
            models: list[LLMModelConfig],
            default_model_id: str | None,
            expires_in_seconds: int,
    ) -> None:
        try:
            pipeline = self._redis_client.client.pipeline()
            pipeline.set(
                self.MODELS_CACHE_KEY,
                MODELS_ADAPTER.dump_json(models).decode("utf-8"),
                ex=expires_in_seconds,
            )
            pipeline.set(
                self.DEFAULT_MODEL_CACHE_KEY,
                default_model_id or "",
                ex=expires_in_seconds,
            )
            await pipeline.execute()
        except Exception as e:
            logger.warning(f"写入模型配置 Redis 缓存失败，已忽略: {e}")
