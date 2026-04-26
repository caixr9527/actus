#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 16:31
@Author : caixiaorong01@outlook.com
@File   : factory.py
"""
from app.domain.external import LLM
from app.domain.models import RuntimeLLMConfig

from .ollama_llm import OllamaLLM
from .ollama_memory_consolidation_provider import OllamaMemoryConsolidationProvider
from .openai_llm import OpenAILLM


class OpenAILLMFactory:
    """OpenAI Compatible LLM 工厂"""

    def create(self, llm_config: RuntimeLLMConfig) -> LLM:
        return OpenAILLM(llm_config=llm_config)


class OllamaLLMFactory:
    """Ollama 本地 LLM 工厂。"""

    def create(
            self,
            *,
            base_url: str,
            model: str,
            timeout_seconds: float,
            temperature: float = 0.0,
            max_tokens: int = 1024,
    ) -> OllamaLLM:
        return OllamaLLM(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def create_memory_consolidation_provider(
            self,
            *,
            base_url: str,
            model: str,
            timeout_seconds: float,
            temperature: float = 0.0,
            max_tokens: int = 1024,
    ) -> OllamaMemoryConsolidationProvider:
        """创建 Memory Consolidation 场景使用的 Ollama provider。"""
        llm = self.create(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return OllamaMemoryConsolidationProvider(llm=llm)
