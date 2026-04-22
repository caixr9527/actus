#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层记忆召回与等待消息窗口 helper。

本模块只保留 LangGraph 节点层仍需直接使用的记忆召回查询与等待恢复消息窗口追加逻辑。
候选治理、消息窗口压缩与 conversation_summary 生成统一由 domain memory_consolidation 服务负责。
"""

import hashlib
from typing import Any, Dict, List, Optional

from app.domain.models import (
    LongTermMemory,
    LongTermMemorySearchMode,
    LongTermMemorySearchQuery,
)
from app.domain.services.runtime.contracts.langgraph_settings import (
    MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    MESSAGE_WINDOW_MAX_MESSAGE_CHARS,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import (
    normalize_message_window_entry,
)
from .working_memory import _ensure_working_memory


def _build_memory_query(state: PlannerReActLangGraphState) -> str:
    working_memory = _ensure_working_memory(state)
    parts = [
        str(state.get("user_message") or "").strip(),
        str(state.get("conversation_summary") or "").strip(),
        str(working_memory.get("goal") or "").strip(),
    ]
    return " | ".join([item for item in parts if item][:3])

def _build_memory_namespace_prefixes(state: PlannerReActLangGraphState) -> List[str]:
    prefixes: List[str] = []
    user_id = str(state.get("user_id") or "").strip()
    session_id = str(state.get("session_id") or "").strip()
    if user_id:
        prefixes.append(f"user/{user_id}/")
    if session_id:
        prefixes.append(f"session/{session_id}/")
    prefixes.append("agent/planner_react/")
    return prefixes

def _dedupe_recalled_memories(memories: List[LongTermMemory]) -> List[LongTermMemory]:
    """按 id/dedupe_key 去重不同召回策略返回的记忆。"""
    deduped_memories: List[LongTermMemory] = []
    seen_keys: set[str] = set()
    for memory in memories:
        dedupe_key = str(memory.id or "").strip() or str(memory.dedupe_key or "").strip()
        if not dedupe_key:
            dedupe_key = hashlib.sha1(memory.model_dump_json().encode("utf-8")).hexdigest()
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped_memories.append(memory)
    return deduped_memories

def _build_memory_recall_queries(state: PlannerReActLangGraphState) -> List[LongTermMemorySearchQuery]:
    """按记忆类型拆分召回策略，避免一个 search 兜底所有长期记忆。"""
    namespace_prefixes = _build_memory_namespace_prefixes(state)
    recall_query = _build_memory_query(state)
    return [
        LongTermMemorySearchQuery(
            namespace_prefixes=namespace_prefixes,
            limit=3,
            memory_types=["profile"],
            mode=LongTermMemorySearchMode.RECENT,
        ),
        LongTermMemorySearchQuery(
            namespace_prefixes=namespace_prefixes,
            limit=3,
            memory_types=["instruction"],
            mode=LongTermMemorySearchMode.RECENT,
        ),
        LongTermMemorySearchQuery(
            namespace_prefixes=namespace_prefixes,
            query_text=recall_query,
            limit=4,
            memory_types=["fact"],
            mode=LongTermMemorySearchMode.HYBRID,
        ),
    ]

def _append_message_window_entry(
        message_window: List[Dict[str, Any]],
        *,
        role: str,
        message: str,
        attachments: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    next_entry = normalize_message_window_entry(
        {
            "role": role,
            "message": message,
            "attachment_paths": list(attachments or []),
        },
        default_role=role,
        max_message_chars=MESSAGE_WINDOW_MAX_MESSAGE_CHARS,
        max_attachment_paths=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )
    if next_entry is None:
        return list(message_window)

    # 创建消息窗口的副本以避免直接修改原列表
    updated_window = list(message_window)

    # 检查是否与最后一条消息完全重复（角色、内容、附件均一致），若是则避免重复添加
    if updated_window:
        latest_entry = normalize_message_window_entry(
            dict(updated_window[-1]),
            default_role=role,
            max_message_chars=MESSAGE_WINDOW_MAX_MESSAGE_CHARS,
            max_attachment_paths=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
        )
        if latest_entry == next_entry:
            return updated_window

    # 将新条目添加到消息窗口末尾
    updated_window.append(next_entry)
    return updated_window
