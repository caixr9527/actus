#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 16:18
@Author : caixiaorong01@outlook.com
@File   : model_config_service.py
"""
from typing import Callable

from app.application.errors import ServerError
from app.application.errors import error_keys
from app.domain.external import ModelConfigCacheStore
from app.domain.models import LLMModelConfig
from app.domain.repositories import IUnitOfWork


class ModelConfigService:
    """模型配置服务"""

    _CACHE_TTL_SECONDS = 60

    def __init__(
            self,
            uow_factory: Callable[[], IUnitOfWork],
            model_config_cache_store: ModelConfigCacheStore | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._model_config_cache_store = model_config_cache_store

    async def get_enabled_models(self) -> tuple[list[LLMModelConfig], str | None]:
        """获取启用模型列表与默认模型 ID。"""
        cached_models = await self._get_cached_models()
        if cached_models is not None:
            return cached_models

        return await self._load_models_from_db()

    async def get_enabled_model_by_id(self, model_id: str) -> LLMModelConfig | None:
        """根据模型 ID 获取启用模型。"""
        models, _ = await self.get_enabled_models()
        for model in models:
            if model.id == model_id:
                return model
        return None

    async def get_public_models(self) -> tuple[str, list[LLMModelConfig]]:
        """获取供前端展示的模型列表与默认模型 ID。"""
        models, default_model_id = await self.get_enabled_models()
        if not models:
            raise ServerError(
                msg="当前没有可用的模型配置",
                error_key=error_keys.APP_CONFIG_DEFAULT_MODEL_UNAVAILABLE,
            )
        if not default_model_id:
            raise ServerError(
                msg="默认模型不可用，请检查模型配置",
                error_key=error_keys.APP_CONFIG_DEFAULT_MODEL_UNAVAILABLE,
            )

        if not any(model.id == default_model_id for model in models):
            raise ServerError(
                msg="默认模型不可用，请检查模型配置",
                error_key=error_keys.APP_CONFIG_DEFAULT_MODEL_UNAVAILABLE,
                error_params={"default_model_id": default_model_id},
            )

        return default_model_id, models

    async def _get_cached_models(self) -> tuple[list[LLMModelConfig], str | None] | None:
        if self._model_config_cache_store is None:
            return None
        return await self._model_config_cache_store.get_models()

    async def _load_models_from_db(self) -> tuple[list[LLMModelConfig], str | None]:
        async with self._uow_factory() as uow:
            models = await uow.llm_model_config.list_enabled()
            default_model = await uow.llm_model_config.get_default()

        default_model_id = default_model.id if default_model is not None else None

        if self._model_config_cache_store is not None:
            await self._model_config_cache_store.save_models(
                models=models,
                default_model_id=default_model_id,
                expires_in_seconds=self._CACHE_TTL_SECONDS,
            )

        return models, default_model_id
