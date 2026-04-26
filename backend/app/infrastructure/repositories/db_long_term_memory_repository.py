#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆仓储 DB 实现。"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Text, case, cast, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.external import EmbeddingService
from app.domain.models import LongTermMemory, LongTermMemorySearchMode, LongTermMemorySearchQuery
from app.domain.models.long_term_memory import LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS
from app.domain.repositories import LongTermMemoryRepository
from app.infrastructure.external.embedding import OpenAIEmbeddingService
from app.infrastructure.models import LongTermMemoryModel
from core.config import get_settings


class DBLongTermMemoryRepository(LongTermMemoryRepository):
    """长期记忆仓储 DB 实现。"""

    def __init__(
            self,
            db_session: AsyncSession,
            embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.db_session = db_session
        self._embedding_service = embedding_service or self._build_embedding_service()

    @staticmethod
    def _build_embedding_service() -> EmbeddingService:
        settings = get_settings()
        return OpenAIEmbeddingService(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            dimensions=settings.resolved_embedding_dimensions,
        )

    @staticmethod
    def _validate_embedding_dimensions(embedding: List[float], *, context: str) -> List[float]:
        normalized_embedding = [float(item) for item in list(embedding or [])]
        if len(normalized_embedding) != LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"{context} 向量维度必须为 {LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS}，"
                f"实际为 {len(normalized_embedding)}"
            )
        return normalized_embedding

    async def _embed_text(self, text: str, *, context: str) -> List[float]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError(f"{context} 缺少可嵌入文本")

        embeddings = await self._embedding_service.embed_texts([normalized_text])
        if len(embeddings) == 0:
            raise RuntimeError(f"{context} 生成 embedding 失败")
        return self._validate_embedding_dimensions(embeddings[0], context=context)

    async def _prepare_query(self, query: LongTermMemorySearchQuery) -> LongTermMemorySearchQuery:
        if query.mode not in {LongTermMemorySearchMode.SEMANTIC, LongTermMemorySearchMode.HYBRID}:
            return query
        if query.query_embedding:
            return query.model_copy(
                update={
                    "query_embedding": self._validate_embedding_dimensions(
                        query.query_embedding,
                        context=f"{query.mode.value} query",
                    )
                }
            )
        return query.model_copy(
            update={
                "query_embedding": await self._embed_text(
                    query.query_text,
                    context=f"{query.mode.value} query",
                )
            }
        )

    async def _prepare_memory(self, memory: LongTermMemory) -> LongTermMemory:
        normalized_content_text = str(memory.content_text or "").strip() or memory.build_content_text()
        if not normalized_content_text:
            return memory.model_copy(update={"content_text": "", "embedding": None})

        normalized_embedding = (
            self._validate_embedding_dimensions(memory.embedding, context="memory")
            if memory.embedding
            else await self._embed_text(normalized_content_text, context="memory")
        )
        return memory.model_copy(
            update={
                "content_text": normalized_content_text,
                "embedding": normalized_embedding,
            }
        )

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

    @staticmethod
    def _build_keyword_conditions(query_tokens: List[str]) -> List[object]:
        conditions: List[object] = []
        for token in query_tokens:
            token_like = f"%{token}%"
            conditions.append(
                or_(
                    LongTermMemoryModel.summary.ilike(token_like),
                    LongTermMemoryModel.content_text.ilike(token_like),
                    cast(LongTermMemoryModel.content, Text).ilike(token_like),
                )
            )
        return conditions

    @classmethod
    def _build_keyword_score(cls, query_tokens: List[str]):
        if len(query_tokens) == 0:
            return literal(0.0)

        keyword_score = literal(0.0)
        for condition in cls._build_keyword_conditions(query_tokens):
            keyword_score = keyword_score + case((condition, 1.0), else_=0.0)
        return keyword_score / float(len(query_tokens))

    @staticmethod
    def _build_full_text_score(query_text: str):
        normalized_query_text = str(query_text or "").strip()
        if not normalized_query_text:
            return literal(0.0)
        ts_query = func.websearch_to_tsquery("simple", normalized_query_text)
        return func.coalesce(func.ts_rank_cd(LongTermMemoryModel.search_tsv, ts_query), 0.0)

    @staticmethod
    def _build_semantic_score(query_embedding: Optional[List[float]]):
        if not query_embedding:
            return literal(0.0)
        return case(
            (
                LongTermMemoryModel.embedding.is_not(None),
                1.0 / (1.0 + LongTermMemoryModel.embedding.cosine_distance(query_embedding)),
            ),
            else_=0.0,
        )

    @staticmethod
    def _build_recency_score():
        recency_source = func.coalesce(
            LongTermMemoryModel.last_accessed_at,
            LongTermMemoryModel.updated_at,
            LongTermMemoryModel.created_at,
        )
        age_days = func.extract("epoch", func.now() - recency_source) / 86400.0
        return 1.0 / (1.0 + age_days)

    @staticmethod
    def _build_semantic_candidate_condition():
        """语义检索只能在已有 embedding 的候选集上排序。"""
        return LongTermMemoryModel.embedding.is_not(None)

    @classmethod
    def _build_total_score(
            cls,
            query: LongTermMemorySearchQuery,
            query_tokens: List[str],
    ):
        keyword_score = cls._build_keyword_score(query_tokens)
        full_text_score = cls._build_full_text_score(query.query_text)
        semantic_score = cls._build_semantic_score(query.query_embedding)
        recency_score = cls._build_recency_score()
        confidence_score = func.coalesce(LongTermMemoryModel.confidence, 0.0)

        if query.mode == LongTermMemorySearchMode.RECENT:
            return (
                recency_score * 0.85
                + confidence_score * 0.15
            )
        if query.mode == LongTermMemorySearchMode.KEYWORD:
            return (
                keyword_score * 0.45
                + full_text_score * 0.35
                + recency_score * 0.15
                + confidence_score * 0.05
            )
        if query.mode == LongTermMemorySearchMode.SEMANTIC:
            return (
                semantic_score * 0.75
                + recency_score * 0.15
                + confidence_score * 0.10
            )
        return (
            semantic_score * 0.40
            + full_text_score * 0.25
            + keyword_score * 0.20
            + recency_score * 0.10
            + confidence_score * 0.05
        )

    async def search(
            self,
            query: LongTermMemorySearchQuery,
    ) -> List[LongTermMemory]:
        prepared_query = await self._prepare_query(query)
        query_tokens = self._split_query_tokens(prepared_query.query_text)
        stmt = select(LongTermMemoryModel)

        normalized_prefixes = [str(item).strip() for item in list(prepared_query.namespace_prefixes or []) if str(item).strip()]
        if len(normalized_prefixes) > 0:
            stmt = stmt.where(
                or_(*[LongTermMemoryModel.namespace.like(f"{prefix}%") for prefix in normalized_prefixes])
            )

        normalized_memory_types = [str(item).strip() for item in list(prepared_query.memory_types or []) if str(item).strip()]
        if len(normalized_memory_types) > 0:
            stmt = stmt.where(LongTermMemoryModel.memory_type.in_(normalized_memory_types))

        normalized_tags = [str(item).strip() for item in list(prepared_query.tags or []) if str(item).strip()]
        if len(normalized_tags) > 0:
            stmt = stmt.where(LongTermMemoryModel.tags.contains(normalized_tags))

        if prepared_query.mode == LongTermMemorySearchMode.SEMANTIC:
            stmt = stmt.where(self._build_semantic_candidate_condition())

        if prepared_query.mode in {LongTermMemorySearchMode.KEYWORD, LongTermMemorySearchMode.HYBRID} and len(query_tokens) > 0:
            stmt = stmt.where(or_(*self._build_keyword_conditions(query_tokens)))

        total_score = self._build_total_score(query=prepared_query, query_tokens=query_tokens)
        stmt = stmt.order_by(
            total_score.desc(),
            LongTermMemoryModel.last_accessed_at.desc().nullslast(),
            LongTermMemoryModel.updated_at.desc(),
            LongTermMemoryModel.created_at.desc(),
        ).limit(max(int(query.limit or 10), 1))

        result = await self.db_session.execute(stmt)
        records = list(result.scalars().all())

        accessed_at = datetime.now()
        for record in records:
            record.last_accessed_at = accessed_at

        return [record.to_domain() for record in records]

    async def upsert(self, memory: LongTermMemory) -> LongTermMemory:
        prepared_memory = await self._prepare_memory(memory)
        # 尝试根据 ID 查找现有记录（加锁以防止并发冲突）
        record: Optional[LongTermMemoryModel] = None
        if str(prepared_memory.id or "").strip():
            result = await self.db_session.execute(
                select(LongTermMemoryModel).where(LongTermMemoryModel.id == prepared_memory.id).with_for_update()
            )
            record = result.scalar_one_or_none()

        # 若未找到且存在去重键，则根据命名空间和去重键查找现有记录（加锁）
        if record is None and str(prepared_memory.dedupe_key or "").strip():
            result = await self.db_session.execute(
                select(LongTermMemoryModel)
                .where(
                    LongTermMemoryModel.namespace == prepared_memory.namespace,
                    LongTermMemoryModel.dedupe_key == prepared_memory.dedupe_key,
                )
                .with_for_update()
            )
            record = result.scalar_one_or_none()

        # 若仍未找到记录，则创建新记录；否则合并更新现有记录
        if record is None:
            record = LongTermMemoryModel.from_domain(prepared_memory)
            self.db_session.add(record)
        else:
            existing = record.to_domain()
            merged_memory = prepared_memory.model_copy(
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
