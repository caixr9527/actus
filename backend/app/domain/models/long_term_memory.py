#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆领域模型。"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)

LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS = 1536


class LongTermMemorySearchMode(str, Enum):
    """长期记忆检索模式。"""

    RECENT = "recent"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class LongTermMemorySearchQuery(BaseModel):
    """长期记忆检索请求。"""

    user_id: str
    namespace_prefixes: List[str] = Field(default_factory=list)
    query_text: str = ""
    # semantic / hybrid 查询允许只提供 query_text，由底层仓储统一补齐 query_embedding。
    query_embedding: Optional[List[float]] = None
    limit: int = 10
    memory_types: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    mode: LongTermMemorySearchMode

    @model_validator(mode="after")
    def validate_query_shape(self) -> "LongTermMemorySearchQuery":
        if not str(self.user_id or "").strip():
            raise ValueError("长期记忆检索必须提供 user_id")
        normalized_query_text = str(self.query_text or "").strip()
        has_embedding = bool(self.query_embedding)

        if self.mode == LongTermMemorySearchMode.RECENT:
            if normalized_query_text:
                raise ValueError("recent 检索模式不接受 query_text")
            if has_embedding:
                raise ValueError("recent 检索模式不接受 query_embedding")
            return self

        if self.mode == LongTermMemorySearchMode.KEYWORD:
            if not normalized_query_text:
                raise ValueError("keyword 检索模式要求提供 query_text")
            if has_embedding:
                raise ValueError("keyword 检索模式不接受 query_embedding")
            return self

        if self.mode == LongTermMemorySearchMode.SEMANTIC:
            if not normalized_query_text and not has_embedding:
                raise ValueError("semantic 检索模式要求提供 query_text 或 query_embedding")
            return self

        if not normalized_query_text:
            raise ValueError("hybrid 检索模式要求提供 query_text")
        return self


class LongTermMemory(BaseModel):
    """跨线程持久化的长期记忆记录。"""

    # 长期记忆主键，默认使用 UUID，供跨线程稳定引用。
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # 用户归属是长期记忆权限边界，namespace 只能作为检索组织字段。
    user_id: str
    tenant_id: str = ""
    scope: Literal["user", "session", "workspace", "run"] = "user"
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    run_id: Optional[str] = None
    # 长期记忆命名空间，用于按 user/session/agent 前缀隔离召回与写入范围。
    namespace: str
    # 记忆类型，当前主要承载 profile、fact 等稳定记忆类别。
    memory_type: str
    # 面向检索与快速理解的文本摘要，结构化检索时会参与模糊匹配。
    summary: str = ""
    # 长期记忆的规范化正文载荷，保存可跨线程复用的事实、偏好或约束。
    content: Dict[str, Any] = Field(default_factory=dict)
    # 面向全文检索与向量嵌入的归一化纯文本。
    content_text: str = ""
    # 标签集合，用于补充记忆分类并支持结构化过滤。
    tags: List[str] = Field(default_factory=list)
    # 来源元数据，记录提炼来源与写入上下文，便于追踪记忆出处。
    source: Dict[str, Any] = Field(default_factory=dict)
    origin: DataOrigin = DataOrigin.LONG_TERM_MEMORY
    trust_level: DataTrustLevel = DataTrustLevel.SYSTEM_GENERATED
    privacy_level: PrivacyLevel = PrivacyLevel.SENSITIVE
    retention_policy: RetentionPolicyKind = RetentionPolicyKind.USER_MEMORY
    # 记忆置信度分值，表示该条长期记忆的稳定性或可信程度。
    confidence: float = 0.0
    # 命名空间内的幂等去重键，配合唯一语义避免重复固化相同记忆。
    dedupe_key: Optional[str] = None
    # 语义检索向量；生成失败时允许为空。
    embedding: Optional[List[float]] = None
    # 最近一次被召回或访问的时间，用于搜索结果排序与热度判断。
    last_accessed_at: Optional[datetime] = None
    # 最近一次更新的时间戳，用于排序与审计记忆的新鲜度。
    updated_at: datetime = Field(default_factory=datetime.now)
    # 记忆首次创建时间，用于保留长期记忆的初始落库时间。
    created_at: datetime = Field(default_factory=datetime.now)

    def build_content_text(self) -> str:
        """构建面向全文检索与向量嵌入的归一化文本。"""
        content_parts: List[str] = [str(self.summary or "").strip()]
        for key, value in dict(self.content or {}).items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            normalized_value = value if isinstance(value, str) else str(value)
            normalized_value = normalized_value.strip()
            if normalized_value:
                content_parts.append(f"{normalized_key}: {normalized_value}")
        content_parts.extend([str(tag).strip() for tag in list(self.tags or []) if str(tag).strip()])
        return "\n".join([part for part in content_parts if part]).strip()
