#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 16:30
@Author : caixiaorong01@outlook.com
@File   : model_runtime_resolver.py
"""
from app.application.errors import ServerError
from app.application.errors import error_keys
from app.application.service.model_config_service import ModelConfigService
from app.application.service.model_input_policy import normalize_supported_input_types
from app.domain.models import Session, RuntimeLLMConfig, LLMModelConfig


class ModelRuntimeResolver:
    """会话运行时模型解析器"""

    def __init__(self, model_config_service: ModelConfigService) -> None:
        self._model_config_service = model_config_service

    async def resolve(self, session: Session) -> tuple[str, RuntimeLLMConfig]:
        """根据会话当前模型解析运行时 LLM 配置。"""
        models, default_model_id = await self._model_config_service.get_enabled_models()
        if not default_model_id:
            raise ServerError(
                msg="默认模型不可用，请检查模型配置",
                error_key=error_keys.APP_CONFIG_DEFAULT_MODEL_UNAVAILABLE,
            )

        enabled_models = {model.id: model for model in models}
        default_model = enabled_models.get(default_model_id)
        if default_model is None:
            raise ServerError(
                msg="默认模型不可用，请检查模型配置",
                error_key=error_keys.APP_CONFIG_DEFAULT_MODEL_UNAVAILABLE,
                error_params={"default_model_id": default_model_id},
            )

        requested_model_id = session.current_model_id
        if requested_model_id in (None, "auto"):
            resolved_model = default_model
        else:
            resolved_model = enabled_models.get(requested_model_id, default_model)

        return resolved_model.id, self._build_runtime_llm_config(resolved_model)

    @staticmethod
    def _build_runtime_llm_config(model: LLMModelConfig) -> RuntimeLLMConfig:
        """把模型目录记录转换为运行时 LLM 配置。"""
        config = model.config or {}
        capabilities = config.get("capabilities", {})
        try:
            return RuntimeLLMConfig(
                base_url=model.base_url,
                api_key=model.api_key,
                model_name=model.model_name,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 8192),
                multimodal=capabilities.get("multimodal", False),
                supported=normalize_supported_input_types(
                    capabilities.get("supported", [])
                ),
            )
        except Exception as e:
            raise ServerError(
                msg=f"模型[{model.id}]配置无效",
                error_key=error_keys.APP_CONFIG_MODEL_INVALID,
                error_params={"model_id": model.id, "reason": str(e)},
            ) from e
