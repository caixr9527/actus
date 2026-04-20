#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层记忆上下文召回节点。"""

import logging
from typing import List, Optional

from app.domain.models import LongTermMemory
from app.domain.repositories import LongTermMemoryRepository
from app.domain.services.runtime.contracts.runtime_logging import (
    elapsed_ms,
    log_runtime,
    now_perf,
)
from app.domain.services.runtime.langgraph_state import (
    PlannerReActLangGraphState,
    normalize_retrieved_memories,
)
from app.domain.services.runtime.normalizers import normalize_text_list

from .memory_helpers import _build_memory_recall_queries, _dedupe_recalled_memories
from .working_memory import _ensure_working_memory

logger = logging.getLogger(__name__)


async def recall_memory_context_node(
        state: PlannerReActLangGraphState,
        long_term_memory_repository: Optional[LongTermMemoryRepository] = None,
) -> PlannerReActLangGraphState:
    """统一整理线程级短期记忆，为后续 planner/react 节点提供稳定输入。"""
    started_at = now_perf()
    log_runtime(
        logger,
        logging.INFO,
        "开始召回记忆上下文",
        state=state,
        existing_memory_count=len(list(state.get("retrieved_memories") or [])),
        has_repository=long_term_memory_repository is not None,
    )
    plan = state.get("plan")
    working_memory = _ensure_working_memory(state)
    if not str(working_memory.get("goal") or "").strip():
        working_memory["goal"] = str(getattr(plan, "goal", "") or state.get("user_message") or "")
    if not list(working_memory.get("open_questions") or []):
        working_memory["open_questions"] = normalize_text_list(state.get("session_open_questions"))

    retrieved_memories = list(state.get("retrieved_memories") or [])
    recall_cost_ms = 0
    if long_term_memory_repository is not None:
        try:
            recall_started_at = now_perf()
            recalled_memories: List[LongTermMemory] = []
            for query in _build_memory_recall_queries(state):
                recalled_memories.extend(await long_term_memory_repository.search(query))
            recalled_memories = _dedupe_recalled_memories(recalled_memories)
            retrieved_memories = normalize_retrieved_memories(
                [memory.model_dump(mode="json") for memory in recalled_memories]
            )
            recall_cost_ms = elapsed_ms(recall_started_at)
        except Exception as e:
            log_runtime(
                logger,
                logging.WARNING,
                "记忆召回失败，回退线程快照",
                state=state,
                error=str(e),
                recall_elapsed_ms=recall_cost_ms,
                elapsed_ms=elapsed_ms(started_at),
            )
    # P3-一次性收口：planner 前禁止把 profile 记忆回写到 working_memory，避免跨任务偏好污染计划。

    log_runtime(
        logger,
        logging.INFO,
        "记忆召回完成",
        state=state,
        recalled_memory_count=len(retrieved_memories),
        open_question_count=len(list(working_memory.get("open_questions") or [])),
        preference_count=len(dict(working_memory.get("user_preferences") or {})),
        recall_elapsed_ms=recall_cost_ms,
        elapsed_ms=elapsed_ms(started_at),
    )

    return {
        **state,
        "working_memory": working_memory,
        "retrieved_memories": retrieved_memories,
    }
