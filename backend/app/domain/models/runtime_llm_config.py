#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 22:06
@Author : caixiaorong01@outlook.com
@File   : runtime_llm_config.py
"""
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.domain.services.runtime.contracts.document_input_contract import normalize_document_supported_input_types


class RuntimeLLMConfig(BaseModel):
    """运行时使用的 LLM 配置"""

    base_url: HttpUrl
    api_key: str
    model_name: str
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=8192, ge=0)
    # 显式声明当前模型接入使用的 API 风格，避免把 OpenAI 专有协议误压给兼容网关。
    api_style: Literal["chat_completions", "responses"] = Field(default="chat_completions")
    # BE-LG-12：模型输入能力声明。
    multimodal: bool = Field(default=False)
    supported: list[str] = Field(default_factory=list)

    @field_validator("supported", mode="before")
    @classmethod
    def _normalize_supported(cls, value: object) -> list[str]:
        """运行时模型能力只保留 P0-5 文档输入类型。"""
        return normalize_document_supported_input_types(value)
