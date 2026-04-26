#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆沉淀外部能力端口。"""

from __future__ import annotations

from typing import Protocol

from app.domain.services.memory_consolidation.contracts import (
    MemoryConsolidationInput,
    MemoryConsolidationResult,
)


class MemoryConsolidationProvider(Protocol):
    """可插拔沉淀模型端口。

    infrastructure 层负责实现该协议，例如本地 Ollama；domain 层不感知具体厂商。
    """

    async def consolidate(self, payload: MemoryConsolidationInput) -> MemoryConsolidationResult:
        """根据沉淀输入返回结构化沉淀结果。"""
        ...
