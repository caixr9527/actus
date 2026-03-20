#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 16:31
@Author : caixiaorong01@outlook.com
@File   : factory.py
"""
from app.domain.external import LLM
from app.domain.models import RuntimeLLMConfig

from .openai_llm import OpenAILLM


class OpenAILLMFactory:
    """OpenAI Compatible LLM 工厂"""

    def create(self, llm_config: RuntimeLLMConfig) -> LLM:
        return OpenAILLM(llm_config=llm_config)
