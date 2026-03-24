#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 22:06
@Author : caixiaorong01@outlook.com
@File   : runtime_llm_config.py
"""
from pydantic import BaseModel, Field, HttpUrl


class RuntimeLLMConfig(BaseModel):
    """运行时使用的 LLM 配置"""

    base_url: HttpUrl
    api_key: str
    model_name: str
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=8192, ge=0)
    # BE-LG-12：模型输入能力声明。
    multimodal: bool = Field(default=False)
    supported: list[str] = Field(default_factory=lambda: ["text"])
