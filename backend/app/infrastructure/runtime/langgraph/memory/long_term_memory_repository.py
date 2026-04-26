#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangGraph 运行时长期记忆仓储适配器。"""
from typing import Callable, List

from app.domain.models import LongTermMemory, LongTermMemorySearchQuery
from app.domain.repositories import IUnitOfWork, LongTermMemoryRepository


class LangGraphLongTermMemoryRepository(LongTermMemoryRepository):
    """基于 UoW 工厂的 LangGraph 长期记忆仓储适配器。"""

    def __init__(
            self,
            uow_factory: Callable[[], IUnitOfWork],
    ) -> None:
        self._uow_factory = uow_factory

    async def search(
            self,
            query: LongTermMemorySearchQuery,
    ) -> List[LongTermMemory]:
        async with self._uow_factory() as uow:
            return await uow.long_term_memory.search(query)

    async def upsert(self, memory: LongTermMemory) -> LongTermMemory:
        async with self._uow_factory() as uow:
            return await uow.long_term_memory.upsert(memory)
