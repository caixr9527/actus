#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆领域模型。"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LongTermMemory(BaseModel):
    """跨线程持久化的长期记忆记录。"""

    # 长期记忆主键，默认使用 UUID，供跨线程稳定引用。
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
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
