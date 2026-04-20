#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层记忆收敛节点。

本模块只承载 consolidate_memory 节点实现，不改消息窗口压缩与长期记忆写入语义。
"""

import logging
from typing import Dict, List, Optional

from app.domain.models import LongTermMemory
from app.domain.repositories import LongTermMemoryRepository
from app.domain.services.runtime.contracts.runtime_logging import (
    elapsed_ms,
    log_runtime,
    now_perf,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import normalize_file_path_list
from .memory_helpers import (
    _append_message_window_entry,
    _build_conversation_summary,
    _compact_message_window,
    _govern_memory_candidates,
)

logger = logging.getLogger(__name__)


async def consolidate_memory_node(
        state: PlannerReActLangGraphState,
        long_term_memory_repository: Optional[LongTermMemoryRepository] = None,
) -> PlannerReActLangGraphState:
    """统一收敛线程级短期记忆，压缩消息窗口并记录压缩元数据。"""
    started_at = now_perf()
    log_runtime(
        logger,
        logging.INFO,
        "开始收敛记忆",
        state=state,
        pending_memory_write_count=len(list(state.get("pending_memory_writes") or [])),
        message_window_size=len(list(state.get("message_window") or [])),
    )
    message_window = _append_message_window_entry(
        list(state.get("message_window") or []),
        role="assistant",
        message=str(state.get("final_message") or ""),
        attachments=list(state.get("selected_artifacts") or []),
    )

    compacted_message_window, trimmed_message_count = _compact_message_window(message_window)

    pending_memory_writes, candidate_stats = _govern_memory_candidates(
        list(state.get("pending_memory_writes") or [])
    )
    remaining_memory_writes: List[Dict[str, object]] = []
    persisted_memory_ids: List[str] = []
    write_cost_ms = 0

    if long_term_memory_repository is None:
        remaining_memory_writes = pending_memory_writes
    else:
        write_started_at = now_perf()
        for item in pending_memory_writes:
            try:
                memory = LongTermMemory.model_validate(item)
                persisted_memory = await long_term_memory_repository.upsert(memory)
                persisted_memory_ids.append(persisted_memory.id)
            except Exception as e:
                log_runtime(
                    logger,
                    logging.WARNING,
                    "记忆写入失败，保留待重试",
                    state=state,
                    error=str(e),
                )
                if isinstance(item, dict):
                    remaining_memory_writes.append(item)
        write_cost_ms = elapsed_ms(write_started_at)

    next_state: PlannerReActLangGraphState = {
        **state,
        "message_window": compacted_message_window,
        "conversation_summary": _build_conversation_summary(
            {
                **state,
                "message_window": compacted_message_window,
            },
            trimmed_message_count=trimmed_message_count,
        ),
        "pending_memory_writes": remaining_memory_writes,
        "selected_artifacts": normalize_file_path_list(state.get("selected_artifacts")),
    }
    log_runtime(
        logger,
        logging.INFO,
        "记忆收敛完成",
        state=next_state,
        compacted_message_window_size=len(compacted_message_window),
        trimmed_message_count=trimmed_message_count,
        kept_candidate_count=candidate_stats["kept_count"],
        persisted_memory_count=len(persisted_memory_ids),
        remaining_memory_write_count=len(remaining_memory_writes),
        write_elapsed_ms=write_cost_ms,
        elapsed_ms=elapsed_ms(started_at),
    )
    return next_state
