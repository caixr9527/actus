#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆仓储 DB 实现。"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Text, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import LongTermMemory
from app.domain.repositories import LongTermMemoryRepository
from app.infrastructure.models import LongTermMemoryModel


class DBLongTermMemoryRepository(LongTermMemoryRepository):
    """长期记忆仓储 DB 实现。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def search(
            self,
            namespace_prefixes: List[str],
            query: str = "",
            limit: int = 10,
            memory_types: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
    ) -> List[LongTermMemory]:
        # 构建基础查询语句
        stmt = select(LongTermMemoryModel)

        # 处理命名空间前缀过滤：对非空且去空格后的前缀列表，使用 LIKE 'prefix%' 进行匹配
        normalized_prefixes = [str(item).strip() for item in namespace_prefixes if str(item).strip()]
        if len(normalized_prefixes) > 0:
            stmt = stmt.where(
                or_(*[LongTermMemoryModel.namespace.like(f"{prefix}%") for prefix in normalized_prefixes])
            )

        # 处理记忆类型过滤：对非空且去空格后的类型列表，使用 IN 条件匹配
        normalized_memory_types = [str(item).strip() for item in list(memory_types or []) if str(item).strip()]
        if len(normalized_memory_types) > 0:
            stmt = stmt.where(LongTermMemoryModel.memory_type.in_(normalized_memory_types))

        # 处理标签过滤：对非空且去空格后的标签列表，使用包含关系匹配（假设 tags 为数组类型字段）
        normalized_tags = [str(item).strip() for item in list(tags or []) if str(item).strip()]
        if len(normalized_tags) > 0:
            stmt = stmt.where(LongTermMemoryModel.tags.contains(normalized_tags))

        # 处理全文模糊搜索：在 summary 和 content 字段中执行不区分大小写的模糊匹配
        normalized_query = str(query or "").strip()
        if normalized_query:
            query_like = f"%{normalized_query}%"
            stmt = stmt.where(
                or_(
                    LongTermMemoryModel.summary.ilike(query_like),
                    cast(LongTermMemoryModel.content, Text).ilike(query_like),
                )
            )

        # 设置排序规则：优先按最后访问时间降序（空值排后），其次按更新时间、创建时间降序；限制返回数量
        stmt = stmt.order_by(
            LongTermMemoryModel.last_accessed_at.desc().nullslast(),
            LongTermMemoryModel.updated_at.desc(),
            LongTermMemoryModel.created_at.desc(),
        ).limit(max(int(limit or 10), 1))

        # 执行查询并获取结果
        result = await self.db_session.execute(stmt)
        records = list(result.scalars().all())
        
        # 批量更新最后访问时间
        accessed_at = datetime.now()
        for record in records:
            record.last_accessed_at = accessed_at
        
        # 转换为领域模型并返回
        return [record.to_domain() for record in records]

    async def upsert(self, memory: LongTermMemory) -> LongTermMemory:
        # 尝试根据 ID 查找现有记录（加锁以防止并发冲突）
        record: Optional[LongTermMemoryModel] = None
        if str(memory.id or "").strip():
            result = await self.db_session.execute(
                select(LongTermMemoryModel).where(LongTermMemoryModel.id == memory.id).with_for_update()
            )
            record = result.scalar_one_or_none()

        # 若未找到且存在去重键，则根据命名空间和去重键查找现有记录（加锁）
        if record is None and str(memory.dedupe_key or "").strip():
            result = await self.db_session.execute(
                select(LongTermMemoryModel)
                .where(
                    LongTermMemoryModel.namespace == memory.namespace,
                    LongTermMemoryModel.dedupe_key == memory.dedupe_key,
                )
                .with_for_update()
            )
            record = result.scalar_one_or_none()

        # 若仍未找到记录，则创建新记录；否则合并更新现有记录
        if record is None:
            record = LongTermMemoryModel.from_domain(memory)
            self.db_session.add(record)
        else:
            existing = record.to_domain()
            merged_memory = memory.model_copy(
                update={
                    "id": existing.id,
                    "created_at": existing.created_at,
                    "updated_at": datetime.now(),
                }
            )
            record.update_from_domain(merged_memory)

        # 刷新会话以持久化更改
        await self.db_session.flush()
        return record.to_domain()
