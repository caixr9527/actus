#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ollama 记忆沉淀 provider 适配器。

本模块只实现 MemoryConsolidationProvider 端口，将记忆沉淀输入转换为 prompt，
实际模型调用统一委托给通用 OllamaLLM。
"""

from __future__ import annotations

from app.domain.services.memory_consolidation import (
    MemoryConsolidationInput,
    MemoryConsolidationProvider,
    MemoryConsolidationResult,
)
from app.domain.services.memory_consolidation.prompts import MEMORY_CONSOLIDATION_PROMPT
from app.infrastructure.external.llm.ollama_llm import OllamaLLM


class OllamaMemoryConsolidationProvider(MemoryConsolidationProvider):
    """使用通用 OllamaLLM 生成结构化记忆沉淀结果。"""

    def __init__(self, llm: OllamaLLM) -> None:
        if llm is None:
            raise ValueError("OllamaLLM 不能为空")
        self._llm = llm

    async def consolidate(self, payload: MemoryConsolidationInput) -> MemoryConsolidationResult:
        """调用通用结构化输出接口，返回记忆沉淀领域结果。"""
        prompt = MEMORY_CONSOLIDATION_PROMPT.format(
            payload_json=payload.model_dump_json(indent=2),
        )
        return await self._llm.generate_structured(
            prompt=prompt,
            output_model=MemoryConsolidationResult,
        )
