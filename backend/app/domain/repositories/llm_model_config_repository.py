#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 16:05
@Author : caixiaorong01@outlook.com
@File   : llm_model_config_repository.py
"""
from typing import Protocol, Optional, List

from app.domain.models import LLMModelConfig


class LLMModelConfigRepository(Protocol):
    """LLM 模型配置仓库协议"""

    async def list_enabled(self) -> List[LLMModelConfig]:
        """获取所有启用状态的模型配置，按展示顺序返回。"""
        ...

    async def get_by_id(self, model_id: str) -> Optional[LLMModelConfig]:
        """根据模型 ID 获取模型配置。"""
        ...

    async def get_default(self) -> Optional[LLMModelConfig]:
        """获取当前默认模型配置。"""
        ...
