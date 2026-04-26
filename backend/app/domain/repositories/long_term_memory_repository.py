#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆仓储协议。"""
from typing import List, Optional, Protocol

from app.domain.models import LongTermMemory, LongTermMemorySearchQuery


class LongTermMemoryRepository(Protocol):
    """长期记忆仓储协议定义。"""

    async def search(
            self,
            query: LongTermMemorySearchQuery,
    ) -> List[LongTermMemory]:
        """按结构化检索请求召回长期记忆。"""
        ...

    async def upsert(self, memory: LongTermMemory) -> LongTermMemory:
        """按 id 或 dedupe_key 幂等写入长期记忆。"""
        ...
