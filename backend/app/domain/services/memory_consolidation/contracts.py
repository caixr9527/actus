#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆沉淀服务合同。

本模块只定义领域输入/输出结构，不引用 LangGraph state、Ollama 或任何 infrastructure 实现。
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class MemoryConsolidationInput(BaseModel):
    """一次运行结束后的沉淀输入。

    `consolidate_memory_node` 负责把 graph state 映射为该合同，领域服务只消费这里的稳定字段。
    """

    session_id: str = ""
    user_id: str = ""
    run_id: str = ""
    thread_id: str = ""
    user_message: str = ""
    assistant_message: str = ""
    previous_conversation_summary: str = ""
    message_window: List[Dict[str, Any]] = Field(default_factory=list)
    selected_artifacts: List[str] = Field(default_factory=list)
    goal: str = ""
    completed_step_count: int = 0
    total_step_count: int = 0
    facts_in_session: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    pending_memory_writes: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryConsolidationStats(BaseModel):
    """沉淀过程统计，用于日志追踪与测试断言。"""

    input_candidate_count: int = 0
    kept_candidate_count: int = 0
    dropped_invalid_count: int = 0
    dropped_sensitive_count: int = 0
    dropped_low_confidence_count: int = 0
    deduped_count: int = 0
    merged_profile_count: int = 0
    trimmed_message_count: int = 0


class MemoryConsolidationResult(BaseModel):
    """一次沉淀后的领域结果。

    该结果由节点层回写 graph state；服务自身不直接修改运行时状态。
    """

    message_window: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_summary: str = ""
    facts_in_session: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    memory_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    degraded: bool = False
    degrade_reason: str = ""
    stats: MemoryConsolidationStats = Field(default_factory=MemoryConsolidationStats)
