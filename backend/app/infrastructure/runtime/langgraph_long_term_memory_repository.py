#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangGraph 运行时长期记忆仓储适配器。"""
from typing import Callable, List, Optional

from app.domain.models import LongTermMemory
from app.domain.repositories import IUnitOfWork, LongTermMemoryRepository


class LangGraphLongTermMemoryRepository(LongTermMemoryRepository):
    """基于 UoW 工厂的 LangGraph 长期记忆仓储适配器。"""

    def __init__(self, uow_factory: Callable[[], IUnitOfWork]) -> None:
        self._uow_factory = uow_factory

    async def search(
            self,
            namespace_prefixes: List[str],
            query: str = "",
            limit: int = 10,
            memory_types: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
    ) -> List[LongTermMemory]:
        async with self._uow_factory() as uow:
            return await uow.long_term_memory.search(
                namespace_prefixes=namespace_prefixes,
                query=query,
                limit=limit,
                memory_types=memory_types,
                tags=tags,
            )

    async def upsert(self, memory: LongTermMemory) -> LongTermMemory:
        async with self._uow_factory() as uow:
            return await uow.long_term_memory.upsert(memory)
