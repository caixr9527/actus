#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆仓储协议。"""
from typing import List, Optional, Protocol

from app.domain.models import LongTermMemory


class LongTermMemoryRepository(Protocol):
    """长期记忆仓储协议定义。"""

    async def search(
            self,
            namespace_prefixes: List[str],
            query: str = "",
            limit: int = 10,
            memory_types: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
    ) -> List[LongTermMemory]:
        """按命名空间前缀与简单过滤条件检索长期记忆。"""
        ...

    async def upsert(self, memory: LongTermMemory) -> LongTermMemory:
        """按 id 或 dedupe_key 幂等写入长期记忆。"""
        ...
