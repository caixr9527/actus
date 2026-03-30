#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆 ORM 模型。"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Float, Index, PrimaryKeyConstraint, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import LongTermMemory
from .base import Base


class LongTermMemoryModel(Base):
    """长期记忆 ORM 模型。"""

    __tablename__ = "long_term_memories"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_long_term_memories_id"),
        UniqueConstraint("namespace", "dedupe_key", name="uq_long_term_memories_namespace_dedupe_key"),
        Index("ix_long_term_memories_namespace", "namespace"),
        Index("ix_long_term_memories_memory_type", "memory_type"),
        Index("ix_long_term_memories_updated_at", "updated_at"),
    )

    # 长期记忆主键，默认使用 UUID，供跨线程稳定引用。
    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    # 长期记忆命名空间，用于按 user/session/agent 前缀隔离召回与写入范围。
    namespace: Mapped[str] = mapped_column(String(255), nullable=False)
    # 记忆类型，当前主要承载 profile、fact 等稳定记忆类别。
    memory_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text("''::character varying"),
    )
    # 面向检索与快速理解的文本摘要，结构化检索时会参与模糊匹配。
    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("''::text"),
    )
    # 长期记忆的规范化正文载荷，保存可跨线程复用的事实、偏好或约束。
    content: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    # 标签集合，用于补充记忆分类并支持结构化过滤。
    tags: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    # 来源元数据，记录提炼来源与写入上下文，便于追踪记忆出处。
    source: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    # 记忆置信度分值，表示该条长期记忆的稳定性或可信程度。
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        server_default=text("0"),
    )
    # 命名空间内的幂等去重键，配合唯一约束避免重复固化相同记忆。
    dedupe_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # 最近一次被召回或访问的时间，用于搜索结果排序与热度判断。
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # 最近一次更新的时间戳，用于排序与审计记忆的新鲜度。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    # 记忆首次创建时间，用于保留长期记忆的初始落库时间。
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, memory: LongTermMemory) -> "LongTermMemoryModel":
        return cls(
            **memory.model_dump(
                mode="python",
                exclude={"content", "tags", "source"},
            ),
            **memory.model_dump(
                mode="json",
                include={"content", "tags", "source"},
            ),
        )

    def to_domain(self) -> LongTermMemory:
        return LongTermMemory.model_validate(self, from_attributes=True)

    def update_from_domain(self, memory: LongTermMemory) -> None:
        base_data = memory.model_dump(
            mode="python",
            exclude={"content", "tags", "source"},
        )
        json_data = memory.model_dump(
            mode="json",
            include={"content", "tags", "source"},
        )
        for field, value in {**base_data, **json_data}.items():
            setattr(self, field, value)
