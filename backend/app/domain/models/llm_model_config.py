#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 14:30
@Author : caixiaorong01@outlook.com
@File   : llm_model_config.py
"""
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field


class LLMModelConfig(BaseModel):
    """LLM 模型配置领域模型"""

    id: str  # 业务模型ID，如 deepseek / gpt-5.4
    provider: str  # 供应商标识，如 openai / deepseek
    display_name: str  # 前端展示名
    base_url: str  # OpenAI Compatible 接口基地址
    api_key: str  # 当前阶段明文存储的访问密钥
    model_name: str  # 真实调用模型名
    enabled: bool = True  # 是否启用
    sort_order: int = 0  # 前端展示排序
    is_default: bool = False  # 是否默认模型
    config: Dict[str, Any] = Field(default_factory=dict)  # 运行参数与展示辅助字段
    created_at: datetime = Field(default_factory=datetime.now)  # 创建时间
    updated_at: datetime = Field(default_factory=datetime.now)  # 更新时间
