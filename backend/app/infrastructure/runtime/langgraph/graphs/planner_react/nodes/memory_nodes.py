#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层记忆收敛节点。

本模块只承载 consolidate_memory 节点实现，不改消息窗口压缩与长期记忆写入语义。
"""

import logging
from typing import Any, Dict, List, Optional

from app.domain.models import ExecutionStatus
from app.domain.models import LongTermMemory
from app.domain.repositories import LongTermMemoryRepository
from app.domain.services.memory_consolidation import (
    MemoryConsolidationInput,
    MemoryConsolidationService,
)
from app.domain.services.runtime.contracts.data_access_contract import (
    DataClassificationPolicy,
    DataOrigin,
    normalize_tenant_id,
)
from app.domain.services.runtime.contracts.runtime_logging import (
    elapsed_ms,
    log_runtime,
    now_perf,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import normalize_file_path_list
from .working_memory import _ensure_working_memory

logger = logging.getLogger(__name__)


def _build_memory_consolidation_input(state: PlannerReActLangGraphState) -> MemoryConsolidationInput:
    """把 LangGraph state 适配为领域沉淀输入，避免 domain 服务依赖具体图状态结构。"""
    working_memory = _ensure_working_memory(state)
    plan = state.get("plan")
    step_states = list(state.get("step_states") or [])
    completed_step_count = sum(
        1
        for item in step_states
        if str(item.get("status") or "") == ExecutionStatus.COMPLETED.value
    )
    total_step_count = len(step_states) if len(step_states) > 0 else len(getattr(plan, "steps", []) or [])
    final_message = str(state.get("final_message") or "").strip()
    return MemoryConsolidationInput(
        session_id=str(state.get("session_id") or "").strip(),
        user_id=str(state.get("user_id") or "").strip(),
        run_id=str(state.get("run_id") or "").strip(),
        thread_id=str(state.get("thread_id") or "").strip(),
        user_message=str(state.get("user_message") or "").strip(),
        assistant_message=final_message,
        previous_conversation_summary=str(state.get("conversation_summary") or "").strip(),
        message_window=[
            item
            for item in list(state.get("message_window") or [])
            if isinstance(item, dict)
        ],
        selected_artifacts=normalize_file_path_list(state.get("selected_artifacts")),
        goal=str(working_memory.get("goal") or getattr(plan, "goal", "") or "").strip(),
        completed_step_count=completed_step_count,
        total_step_count=total_step_count,
        facts_in_session=[
            str(item or "").strip()
            for item in list(working_memory.get("facts_in_session") or [])
            if str(item or "").strip()
        ],
        user_preferences=dict(working_memory.get("user_preferences") or {}),
        pending_memory_writes=[
            item
            for item in list(state.get("pending_memory_writes") or [])
            if isinstance(item, dict)
        ],
    )


def _build_working_memory_with_consolidation_result(
        state: PlannerReActLangGraphState,
        *,
        facts_in_session: List[str],
        user_preferences: Dict[str, Any],
) -> Dict[str, Any]:
    """把沉淀结果中的会话事实和偏好回写到 working_memory。"""
    working_memory = dict(_ensure_working_memory(state))
    working_memory["facts_in_session"] = list(facts_in_session or [])
    working_memory["user_preferences"] = dict(user_preferences or {})
    return working_memory


def _build_owned_memory_candidate(
        item: Dict[str, object],
        *,
        state: PlannerReActLangGraphState,
        data_retention_policy_service: DataClassificationPolicy,
) -> Dict[str, object]:
    """长期记忆写入前补齐强归属字段，namespace 不再承担权限边界。"""
    user_id = str(state.get("user_id") or "").strip()
    session_id = str(state.get("session_id") or "").strip() or None
    run_id = str(state.get("run_id") or "").strip() or None
    tenant_id = normalize_tenant_id(user_id)
    classification = data_retention_policy_service.classify_data(
        tenant_id=tenant_id,
        origin=DataOrigin.LONG_TERM_MEMORY,
    )
    candidate = dict(item)
    candidate["user_id"] = user_id
    candidate["tenant_id"] = classification.tenant_id
    candidate.setdefault("scope", "user")
    candidate.setdefault("session_id", session_id)
    candidate.setdefault("run_id", run_id)
    candidate["origin"] = classification.origin
    candidate["trust_level"] = classification.trust_level
    candidate["privacy_level"] = classification.privacy_level
    candidate["retention_policy"] = classification.retention_policy
    return candidate


async def consolidate_memory_node(
        state: PlannerReActLangGraphState,
        long_term_memory_repository: Optional[LongTermMemoryRepository] = None,
        memory_consolidation_service: Optional[MemoryConsolidationService] = None,
        data_retention_policy_service: Optional[DataClassificationPolicy] = None,
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
    consolidation_input = _build_memory_consolidation_input(state)
    consolidation_service = memory_consolidation_service or MemoryConsolidationService()
    if data_retention_policy_service is None:
        raise ValueError("data_retention_policy_service 不能为空")
    consolidation_result = await consolidation_service.consolidate(consolidation_input)
    pending_memory_writes = list(consolidation_result.memory_candidates or [])
    remaining_memory_writes: List[Dict[str, object]] = []
    persisted_memory_ids: List[str] = []
    write_cost_ms = 0

    if long_term_memory_repository is None:
        remaining_memory_writes = pending_memory_writes
    else:
        write_started_at = now_perf()
        for item in pending_memory_writes:
            try:
                memory = LongTermMemory.model_validate(
                    _build_owned_memory_candidate(
                        item,
                        state=state,
                        data_retention_policy_service=data_retention_policy_service,
                    )
                )
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
        "message_window": list(consolidation_result.message_window or []),
        "conversation_summary": consolidation_result.conversation_summary,
        "working_memory": _build_working_memory_with_consolidation_result(
            state,
            facts_in_session=list(consolidation_result.facts_in_session or []),
            user_preferences=dict(consolidation_result.user_preferences or {}),
        ),
        "pending_memory_writes": remaining_memory_writes,
        "selected_artifacts": normalize_file_path_list(state.get("selected_artifacts")),
    }
    log_runtime(
        logger,
        logging.INFO,
        "记忆收敛完成",
        state=next_state,
        compacted_message_window_size=len(consolidation_result.message_window),
        trimmed_message_count=consolidation_result.stats.trimmed_message_count,
        kept_candidate_count=consolidation_result.stats.kept_candidate_count,
        dropped_invalid_count=consolidation_result.stats.dropped_invalid_count,
        dropped_sensitive_count=consolidation_result.stats.dropped_sensitive_count,
        dropped_low_confidence_count=consolidation_result.stats.dropped_low_confidence_count,
        deduped_count=consolidation_result.stats.deduped_count,
        merged_profile_count=consolidation_result.stats.merged_profile_count,
        memory_consolidation_degraded=consolidation_result.degraded,
        memory_consolidation_degrade_reason=consolidation_result.degrade_reason,
        persisted_memory_count=len(persisted_memory_ids),
        remaining_memory_write_count=len(remaining_memory_writes),
        write_elapsed_ms=write_cost_ms,
        elapsed_ms=elapsed_ms(started_at),
    )
    return next_state
