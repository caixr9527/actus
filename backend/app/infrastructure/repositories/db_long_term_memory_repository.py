#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆仓储 DB 实现。"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Text, case, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import LongTermMemory
from app.domain.repositories import LongTermMemoryRepository
from app.infrastructure.models import LongTermMemoryModel


class DBLongTermMemoryRepository(LongTermMemoryRepository):
    """长期记忆仓储 DB 实现。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    @staticmethod
    def _split_query_tokens(query: str) -> List[str]:
        normalized = str(query or "").strip()
        if not normalized:
            return []
        raw_tokens = [token.strip() for token in normalized.replace("|", " ").split() if token.strip()]
        deduped_tokens: List[str] = []
        for token in raw_tokens:
            if token not in deduped_tokens:
                deduped_tokens.append(token)
        return deduped_tokens[:12]

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

        # 处理检索查询：按词元拆分后做 OR 检索，避免整串 query 导致召回率过低。
        normalized_query = str(query or "").strip()
        query_tokens = self._split_query_tokens(normalized_query)
        if len(query_tokens) > 0:
            token_conditions = []
            for token in query_tokens:
                token_like = f"%{token}%"
                token_conditions.append(
                    or_(
                        LongTermMemoryModel.summary.ilike(token_like),
                        LongTermMemoryModel.content_text.ilike(token_like),
                        cast(LongTermMemoryModel.content, Text).ilike(token_like),
                    )
                )
            stmt = stmt.where(or_(*token_conditions))

        # 设置排序规则：先按查询匹配词元数排序，再按访问热度和新鲜度排序。
        if len(query_tokens) > 0:
            match_score = 0
            for token in query_tokens:
                token_like = f"%{token}%"
                match_score += case(
                    (
                        or_(
                            LongTermMemoryModel.summary.ilike(token_like),
                            LongTermMemoryModel.content_text.ilike(token_like),
                        ),
                        1,
                    ),
                    else_=0,
                )
            stmt = stmt.order_by(
                match_score.desc(),
                LongTermMemoryModel.last_accessed_at.desc().nullslast(),
                LongTermMemoryModel.updated_at.desc(),
                LongTermMemoryModel.created_at.desc(),
            )
        else:
            stmt = stmt.order_by(
                LongTermMemoryModel.last_accessed_at.desc().nullslast(),
                LongTermMemoryModel.updated_at.desc(),
                LongTermMemoryModel.created_at.desc(),
            )
        stmt = stmt.limit(max(int(limit or 10), 1))

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
