#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 21:20
@Author : caixiaorong01@outlook.com
@File   : model_config_cache_store.py
"""
from abc import ABC, abstractmethod

from app.domain.models import LLMModelConfig


class ModelConfigCacheStore(ABC):
    """模型配置缓存存储接口"""

    @abstractmethod
    async def get_models(self) -> tuple[list[LLMModelConfig], str | None] | None:
        """读取缓存中的模型列表与默认模型 ID。缓存未命中时返回 None。"""
        ...

    @abstractmethod
    async def save_models(
            self,
            models: list[LLMModelConfig],
            default_model_id: str | None,
            expires_in_seconds: int,
    ) -> None:
        """写入模型列表与默认模型 ID 到缓存。"""
        ...
