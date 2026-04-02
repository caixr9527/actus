#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 节点实现。"""
import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langgraph.types import interrupt

from app.domain.external import LLM
from app.domain.models import (
    DoneEvent,
    ExecutionStatus,
    File,
    LongTermMemory,
    LongTermMemorySearchMode,
    LongTermMemorySearchQuery,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Step,
    StepOutcome,
    StepEvent,
    StepEventStatus,
    TitleEvent,
    ToolEvent,
    normalize_wait_payload,
    resolve_wait_resume_message,
)
from app.domain.repositories import LongTermMemoryRepository
from app.domain.services.prompts import CREATE_PLAN_PROMPT, EXECUTION_PROMPT, UPDATE_PLAN_PROMPT, SYSTEM_PROMPT, \
    PLANNER_SYSTEM_PROMPT, SUMMARIZE_PROMPT
from app.domain.services.runtime import SkillGraphRuntime
from app.domain.services.runtime.langgraph_events import append_events
from app.domain.services.runtime.langgraph_state import GraphStateContractMapper, PlannerReActLangGraphState
from app.domain.services.tools import BaseTool
from .live_events import emit_live_events
from .parsers import (
    build_fallback_plan_title,
    format_attachments_for_prompt,
    build_step_from_payload,
    merge_attachment_paths,
    normalize_attachments,
    safe_parse_json,
)
from .tools import execute_step_with_prompt

logger = logging.getLogger(__name__)

PLANNER_EXECUTE_STEP_SKILL_ID = "planner_react.execute_step"
MESSAGE_WINDOW_MAX_ITEMS = 100
MESSAGE_WINDOW_MAX_MESSAGE_CHARS = 500
MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS = 8
MEMORY_CANDIDATE_MIN_CONFIDENCE = 0.3
CONVERSATION_SUMMARY_MAX_PARTS = 4
PROMPT_CONTEXT_BRIEF_LIMIT = 5
PROMPT_CONTEXT_ARTIFACT_LIMIT = 10
STEP_EXECUTION_TIMEOUT_SECONDS = 180


def _ensure_working_memory(state: PlannerReActLangGraphState) -> Dict[str, Any]:
    working_memory = dict(state.get("working_memory") or {})
    working_memory.setdefault("goal", "")
    working_memory.setdefault("constraints", [])
    working_memory.setdefault("decisions", [])
    working_memory.setdefault("open_questions", [])
    working_memory.setdefault("user_preferences", {})
    working_memory.setdefault("facts_in_session", [])
    return working_memory


def _append_unique_text_item(items: List[Any], value: str) -> List[str]:
    normalized_items = [str(item).strip() for item in items if str(item).strip()]
    normalized_value = str(value).strip()
    if normalized_value and normalized_value not in normalized_items:
        normalized_items.append(normalized_value)
    return normalized_items


def _truncate_text(value: Any, *, max_chars: int) -> str:
    normalized_value = str(value or "").strip()
    if len(normalized_value) <= max_chars:
        return normalized_value
    return normalized_value[:max_chars]


def _normalize_step_result_text(value: Any, *, fallback: str = "") -> str:
    """统一规整步骤结果，避免把 None 写成字符串 'None'。"""
    if value is None:
        return str(fallback or "").strip()

    normalized_value = str(value).strip()
    if not normalized_value or normalized_value.lower() == "none":
        return str(fallback or "").strip()
    return normalized_value


def _normalize_text_items(raw: Any) -> List[str]:
    """统一规整 LLM/工具返回的字符串列表。"""
    if not isinstance(raw, list):
        return []
    normalized_items: List[str] = []
    for item in raw:
        normalized_item = str(item or "").strip()
        if normalized_item and normalized_item not in normalized_items:
            normalized_items.append(normalized_item)
    return normalized_items


def _normalize_context_artifact_refs(raw: Any) -> List[str]:
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    normalized_items = [str(item).strip() for item in items if str(item).strip()]
    return list(dict.fromkeys(normalized_items))


def _get_step_outcome_summary(step: Optional[Step]) -> str:
    """读取步骤结果摘要。"""
    if step is None or step.outcome is None:
        return ""
    return _normalize_step_result_text(step.outcome.summary)


def _get_step_artifacts(step: Optional[Step]) -> List[str]:
    """读取步骤产物列表。"""
    if step is None or step.outcome is None:
        return []
    return _normalize_attachment_paths(step.outcome.produced_artifacts)


def _reduce_state_with_events(
        state: PlannerReActLangGraphState,
        *,
        updates: Dict[str, Any],
        events: Optional[List[Any]] = None,
) -> PlannerReActLangGraphState:
    """把新增事件立即收敛回 graph state，避免图内后续节点读到过期 step 状态。"""
    new_events = list(events or [])
    next_state: PlannerReActLangGraphState = {
        **state,
        **updates,
        "emitted_events": append_events(state.get("emitted_events"), *new_events),
    }
    if len(new_events) == 0:
        return next_state
    return GraphStateContractMapper.apply_emitted_events(state=next_state)


def _hydrate_step_outcome(raw: Any) -> Optional[StepOutcome]:
    """把 dict/领域对象统一规整为 StepOutcome。"""
    if raw is None:
        return None
    if isinstance(raw, StepOutcome):
        return raw
    if not isinstance(raw, dict):
        return None
    try:
        return StepOutcome.model_validate(raw)
    except Exception:
        return None


def _outcome_is_reusable(outcome: Optional[StepOutcome]) -> bool:
    if outcome is None or not outcome.done:
        return False
    if _normalize_step_result_text(outcome.summary):
        return True
    return len(_normalize_attachment_paths(outcome.produced_artifacts)) > 0


def _merge_step_outcome_into_working_memory(
        working_memory: Dict[str, Any],
        outcome: StepOutcome,
) -> Dict[str, Any]:
    """将步骤结果沉淀到工作记忆，供后续 step / replan 使用。"""
    updated_working_memory = dict(working_memory or {})
    updated_working_memory.setdefault("decisions", [])
    updated_working_memory.setdefault("open_questions", [])
    updated_working_memory.setdefault("facts_in_session", [])

    summary = _normalize_step_result_text(outcome.summary)
    if summary:
        updated_working_memory["decisions"] = _append_unique_text_item(
            list(updated_working_memory.get("decisions") or []),
            summary,
        )

    for open_question in list(outcome.open_questions or []):
        updated_working_memory["open_questions"] = _append_unique_text_item(
            list(updated_working_memory.get("open_questions") or []),
            open_question,
        )

    for fact in list(outcome.facts_learned or []):
        updated_working_memory["facts_in_session"] = _append_unique_text_item(
            list(updated_working_memory.get("facts_in_session") or []),
            fact,
        )
    return updated_working_memory


def _build_reused_step_outcome(
        source_outcome: StepOutcome,
        *,
        reused_from_run_id: str,
        reused_from_step_id: str,
) -> StepOutcome:
    """为复用场景生成带来源标记的 outcome。"""
    return StepOutcome(
        done=True,
        summary=_normalize_step_result_text(source_outcome.summary),
        produced_artifacts=_normalize_attachment_paths(source_outcome.produced_artifacts),
        blockers=_normalize_text_items(list(source_outcome.blockers or [])),
        facts_learned=_normalize_text_items(list(source_outcome.facts_learned or [])),
        open_questions=_normalize_text_items(list(source_outcome.open_questions or [])),
        next_hint=_normalize_step_result_text(source_outcome.next_hint),
        reused_from_run_id=reused_from_run_id,
        reused_from_step_id=reused_from_step_id,
    )


def _find_reusable_step_outcome(
        state: PlannerReActLangGraphState,
        step: Step,
        plan: Plan,
) -> Optional[Tuple[StepOutcome, str, str]]:
    """仅在当前 run 内按 objective_key 查找可复用的步骤结果。"""
    if not str(step.objective_key or "").strip():
        return None

    current_run_id = str(state.get("run_id") or "").strip()

    # 先检查当前 run 已完成的步骤，避免同一轮里重复执行等价目标。
    for candidate in list(plan.steps or []):
        if str(candidate.id or "").strip() == str(step.id or "").strip():
            continue
        if str(candidate.objective_key or "").strip() != step.objective_key:
            continue
        if candidate.status != ExecutionStatus.COMPLETED:
            continue
        candidate_outcome = _hydrate_step_outcome(candidate.outcome)
        if not _outcome_is_reusable(candidate_outcome):
            continue
        if not current_run_id:
            continue
        return candidate_outcome, current_run_id, str(candidate.id)

    return None


async def guard_step_reuse_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """在真实执行前做当前 run 内复用，命中时直接跳过执行节点。"""
    plan = state.get("plan")
    if plan is None:
        return state

    step = plan.get_next_step()
    if step is None:
        return state

    reusable_step = _find_reusable_step_outcome(state=state, step=step, plan=plan)
    graph_metadata = dict(state.get("graph_metadata") or {})
    graph_metadata["step_reuse_hit"] = False
    graph_metadata.pop("step_reuse_source_run_id", None)
    graph_metadata.pop("step_reuse_source_step_id", None)

    if reusable_step is None:
        return {
            **state,
            "graph_metadata": graph_metadata,
        }

    source_outcome, reused_from_run_id, reused_from_step_id = reusable_step
    step.outcome = _build_reused_step_outcome(
        source_outcome,
        reused_from_run_id=reused_from_run_id,
        reused_from_step_id=reused_from_step_id,
    )
    step.status = ExecutionStatus.COMPLETED

    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=StepEventStatus.COMPLETED,
    )

    await emit_live_events(completed_event)

    next_step = plan.get_next_step()
    working_memory = _merge_step_outcome_into_working_memory(
        _ensure_working_memory(state),
        step.outcome,
    )
    step_local_memory = dict(state.get("step_local_memory") or {})
    step_local_memory["current_step_id"] = str(step.id)
    step_local_memory["observation_summary"] = _normalize_step_result_text(step.outcome.summary)
    step_local_memory["pending_findings"] = list(step.outcome.produced_artifacts or [])
    graph_metadata["step_reuse_hit"] = True
    graph_metadata["step_reuse_source_run_id"] = reused_from_run_id
    graph_metadata["step_reuse_source_step_id"] = reused_from_step_id
    graph_metadata["last_step_execution_mode"] = "reused"

    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "last_executed_step": step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
            "working_memory": working_memory,
            "step_local_memory": step_local_memory,
            "graph_metadata": graph_metadata,
            "final_message": _normalize_step_result_text(step.outcome.summary),
            "selected_artifacts": list(
                dict.fromkeys(
                    list(state.get("selected_artifacts") or [])
                    + list(step.outcome.produced_artifacts or [])
                )
            ),
            "pending_interrupt": {},
        },
        events=[completed_event],
    )


def _build_execution_context_block(state: PlannerReActLangGraphState) -> Dict[str, Any]:
    """统一构造 planner / execute_step / replan 的结构化上下文块。"""
    # 提取并过滤有效的步骤状态列表
    step_states = [dict(item) for item in list(state.get("step_states") or []) if isinstance(item, dict)]
    # 筛选出所有状态为“已完成”的步骤
    completed_steps = [
        item for item in step_states
        if str(item.get("status") or "") == ExecutionStatus.COMPLETED.value
    ]
    # 获取最后执行的步骤对象
    last_step = state.get("last_executed_step")
    # 确保工作记忆已初始化
    working_memory = _ensure_working_memory(state)
    
    # 收集所有待解决的开放性问题，来源包括会话级、工作记忆及最后一步的执行结果
    open_questions = _normalize_text_items(state.get("session_open_questions"))
    open_questions.extend(_normalize_text_items(working_memory.get("open_questions")))
    if last_step is not None and last_step.outcome is not None:
        open_questions.extend(_normalize_text_items(last_step.outcome.open_questions))

    blockers = _normalize_text_items(state.get("session_blockers"))
    if last_step is not None and last_step.outcome is not None:
        blockers.extend(_normalize_text_items(last_step.outcome.blockers))

    selected_artifacts = _normalize_context_artifact_refs(state.get("selected_artifacts"))
    historical_artifact_refs = _normalize_context_artifact_refs(state.get("historical_artifact_refs"))

    recent_run_briefs = [
        {
            "run_id": str(item.get("run_id") or "").strip(),
            "title": str(item.get("title") or "").strip(),
            "goal": str(item.get("goal") or "").strip(),
            "status": str(item.get("status") or "").strip(),
            "final_answer_summary": _truncate_text(item.get("final_answer_summary"), max_chars=200),
        }
        for item in list(state.get("recent_run_briefs") or [])[:PROMPT_CONTEXT_BRIEF_LIMIT]
        if isinstance(item, dict) and str(item.get("run_id") or "").strip()
    ]
    recent_attempt_briefs = [
        {
            "run_id": str(item.get("run_id") or "").strip(),
            "title": str(item.get("title") or "").strip(),
            "goal": str(item.get("goal") or "").strip(),
            "status": str(item.get("status") or "").strip(),
            "final_answer_summary": _truncate_text(item.get("final_answer_summary"), max_chars=200),
        }
        for item in list(state.get("recent_attempt_briefs") or [])[:PROMPT_CONTEXT_BRIEF_LIMIT]
        if isinstance(item, dict) and str(item.get("run_id") or "").strip()
    ]

    # 构建并返回统一的执行上下文块
    return {
        "conversation_summary": str(state.get("conversation_summary") or "").strip(),
        "completed_steps": completed_steps,
        "last_step_result": _get_step_outcome_summary(last_step),
        "retrieved_memories": list(state.get("retrieved_memories") or []),
        "open_questions": list(dict.fromkeys(open_questions)),
        "blockers": list(dict.fromkeys(blockers)),
        "selected_artifacts": list(dict.fromkeys(selected_artifacts))[:PROMPT_CONTEXT_ARTIFACT_LIMIT],
        "historical_artifact_refs": list(dict.fromkeys(historical_artifact_refs))[:PROMPT_CONTEXT_ARTIFACT_LIMIT],
        "recent_run_briefs": recent_run_briefs,
        "recent_attempt_briefs": recent_attempt_briefs,
    }


def _append_execution_context_to_prompt(prompt: str, state: PlannerReActLangGraphState) -> str:
    """将统一上下文块追加到 prompt，避免各节点各自拼装 ad hoc 文本。"""
    context_block = _build_execution_context_block(state)
    context_json = json.dumps(context_block, ensure_ascii=False, indent=2)
    return f"{prompt}\n\n已知上下文:\n```json\n{context_json}\n```"


def _dedupe_replanned_steps(existing_steps: List[Step], new_steps: List[Step]) -> List[Step]:
    """重规划时确保新步骤 ID 不与已保留步骤冲突。"""
    seen_step_ids = {
        str(step.id).strip()
        for step in existing_steps
        if str(step.id).strip()
    }
    deduped_steps: List[Step] = []
    for step in new_steps:
        normalized_step = step.model_copy(deep=True)
        step_id = str(normalized_step.id).strip()
        if not step_id or step_id in seen_step_ids:
            normalized_step.id = str(uuid.uuid4())
            step_id = normalized_step.id
        seen_step_ids.add(step_id)
        deduped_steps.append(normalized_step)
    return deduped_steps


def _normalize_attachment_paths(raw: Any) -> List[str]:
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    normalized_items = [str(item).strip() for item in items if str(item).strip()]
    return list(dict.fromkeys(normalized_items))[:MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS]


def _normalize_message_window_entry(
        raw_entry: Dict[str, Any],
        *,
        default_role: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_entry, dict):
        return None

    normalized_role = str(raw_entry.get("role") or default_role).strip() or default_role
    normalized_message = _truncate_text(
        raw_entry.get("message"),
        max_chars=MESSAGE_WINDOW_MAX_MESSAGE_CHARS,
    )
    normalized_attachments = _normalize_attachment_paths(raw_entry.get("attachment_paths"))
    input_part_count_raw = raw_entry.get("input_part_count")
    try:
        input_part_count = max(int(input_part_count_raw or 0), 0)
    except Exception:
        input_part_count = 0

    if not normalized_message and len(normalized_attachments) == 0 and input_part_count == 0:
        return None

    normalized_entry: Dict[str, Any] = {
        "role": normalized_role,
        "message": normalized_message,
    }
    if len(normalized_attachments) > 0:
        normalized_entry["attachment_paths"] = normalized_attachments
    if input_part_count > 0:
        normalized_entry["input_part_count"] = input_part_count
    return normalized_entry


def _build_memory_context_version(state: PlannerReActLangGraphState) -> str:
    return ":".join(
        [
            str(state.get("thread_id") or ""),
            str(int(state.get("execution_count") or 0)),
            str(len(state.get("message_window") or [])),
            str(len(state.get("retrieved_memories") or [])),
        ]
    )


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


def _build_memory_dedupe_key(*, namespace: str, memory_type: str, content: Dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "namespace": namespace,
            "memory_type": memory_type,
            "content": content,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_memory_fact_items(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    normalized_items: List[str] = []
    for item in raw:
        normalized_text = str(item or "").strip()
        if normalized_text and normalized_text not in normalized_items:
            normalized_items.append(normalized_text)
    return normalized_items


def _normalize_memory_preferences(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    normalized_preferences: Dict[str, Any] = {}
    for key, value in raw.items():
        normalized_key = str(key or "").strip()
        if not normalized_key:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized_preferences[normalized_key] = value
            continue
        if value is None:
            continue
        normalized_value = str(value).strip()
        if normalized_value:
            normalized_preferences[normalized_key] = normalized_value
    return normalized_preferences


def _build_outcome_fact_text(
        state: PlannerReActLangGraphState,
        summary_message: str,
) -> str:
    normalized_summary = str(summary_message or "").strip()
    if normalized_summary:
        return normalized_summary

    working_memory = _ensure_working_memory(state)
    decisions = [str(item or "").strip() for item in list(working_memory.get("decisions") or []) if str(item or "").strip()]
    if len(decisions) > 0:
        return decisions[-1]

    last_step = state.get("last_executed_step")
    return _get_step_outcome_summary(last_step)


def _build_outcome_memory_candidate(
        state: PlannerReActLangGraphState,
        summary_message: str,
) -> Optional[Dict[str, Any]]:
    outcome_text = _build_outcome_fact_text(state, summary_message)
    if not outcome_text:
        return None

    session_id = str(state.get("session_id") or "").strip()
    run_id = str(state.get("run_id") or "").strip()
    thread_id = str(state.get("thread_id") or "").strip()
    namespace = f"session/{session_id}/fact"
    content = {
        "text": outcome_text[:2000],
        "source_kind": "task_outcome",
    }
    return {
        "namespace": namespace,
        "memory_type": "fact",
        "summary": outcome_text[:120],
        "content": content,
        "tags": ["task_outcome"],
        "source": {
            "session_id": session_id,
            "run_id": run_id,
            "thread_id": thread_id,
            "stage": "summarize",
        },
        "confidence": 0.5,
        "dedupe_key": _build_memory_dedupe_key(
            namespace=namespace,
            memory_type="fact",
            content=content,
        ),
    }


def _merge_memory_candidates(
        current_candidates: List[Dict[str, Any]],
        new_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in [*current_candidates, *new_candidates]:
        if not isinstance(item, dict):
            continue
        dedupe_key = str(item.get("dedupe_key") or item.get("id") or "").strip()
        if not dedupe_key:
            dedupe_key = hashlib.sha1(
                json.dumps(item, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        merged.append(item)
    return merged


def _normalize_memory_candidate(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    namespace = str(item.get("namespace") or "").strip()
    memory_type = str(item.get("memory_type") or "").strip().lower()
    if not namespace or memory_type not in {"profile", "fact", "instruction"}:
        return None

    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    summary = _truncate_text(item.get("summary"), max_chars=120)
    if not summary and isinstance(content.get("text"), str):
        summary = _truncate_text(content.get("text"), max_chars=120)
    if not summary and len(content) == 0:
        return None

    try:
        confidence = float(item.get("confidence")) if item.get("confidence") is not None else 0.6
    except Exception:
        confidence = 0.6
    confidence = max(0.0, min(confidence, 1.0))

    tags = [str(tag).strip() for tag in list(item.get("tags") or []) if str(tag).strip()]
    normalized_tags = list(dict.fromkeys(tags))[:8]
    source = item.get("source") if isinstance(item.get("source"), dict) else {}

    dedupe_key = str(item.get("dedupe_key") or "").strip()
    if not dedupe_key:
        dedupe_key = _build_memory_dedupe_key(
            namespace=namespace,
            memory_type=memory_type,
            content=content or {"summary": summary},
        )

    normalized_candidate = {
        "namespace": namespace,
        "memory_type": memory_type,
        "summary": summary,
        "content": content,
        "tags": normalized_tags,
        "source": source,
        "confidence": confidence,
        "dedupe_key": dedupe_key,
    }
    if item.get("id"):
        normalized_candidate["id"] = str(item.get("id"))
    return normalized_candidate


def _merge_profile_candidates(base_item: Dict[str, Any], incoming_item: Dict[str, Any]) -> Dict[str, Any]:
    merged_content = {
        **dict(base_item.get("content") or {}),
        **dict(incoming_item.get("content") or {}),
    }
    merged_tags = list(
        dict.fromkeys(
            [
                *list(base_item.get("tags") or []),
                *list(incoming_item.get("tags") or []),
            ]
        )
    )[:8]
    merged_source = {
        **dict(base_item.get("source") or {}),
        **dict(incoming_item.get("source") or {}),
    }
    merged_summary = str(incoming_item.get("summary") or base_item.get("summary") or "用户偏好")
    merged_confidence = max(
        float(base_item.get("confidence") or 0.0),
        float(incoming_item.get("confidence") or 0.0),
    )
    merged_item = {
        **base_item,
        "summary": _truncate_text(merged_summary or "用户偏好", max_chars=120),
        "content": merged_content,
        "tags": merged_tags,
        "source": merged_source,
        "confidence": merged_confidence,
        "dedupe_key": _build_memory_dedupe_key(
            namespace=str(base_item.get("namespace") or ""),
            memory_type="profile",
            content=merged_content,
        ),
    }
    return merged_item


def _govern_memory_candidates(
        candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "input_count": len(list(candidates or [])),
        "kept_count": 0,
        "dropped_invalid_count": 0,
        "dropped_low_confidence_count": 0,
        "deduped_count": 0,
        "merged_profile_count": 0,
    }
    governed: List[Dict[str, Any]] = []
    dedupe_keys: set[str] = set()
    profile_index_by_namespace: Dict[str, int] = {}

    for raw_item in list(candidates or []):
        normalized_item = _normalize_memory_candidate(raw_item)
        if normalized_item is None:
            stats["dropped_invalid_count"] += 1
            continue
        if float(normalized_item.get("confidence") or 0.0) < MEMORY_CANDIDATE_MIN_CONFIDENCE:
            stats["dropped_low_confidence_count"] += 1
            continue

        if normalized_item["memory_type"] == "profile":
            namespace = normalized_item["namespace"]
            existing_index = profile_index_by_namespace.get(namespace)
            if existing_index is not None:
                governed[existing_index] = _merge_profile_candidates(
                    governed[existing_index],
                    normalized_item,
                )
                stats["merged_profile_count"] += 1
                continue
            profile_index_by_namespace[namespace] = len(governed)

        dedupe_key = str(normalized_item.get("dedupe_key") or "")
        if dedupe_key in dedupe_keys:
            stats["deduped_count"] += 1
            continue
        dedupe_keys.add(dedupe_key)
        governed.append(normalized_item)

    stats["kept_count"] = len(governed)
    return governed, stats


def _build_memory_candidates(state: PlannerReActLangGraphState) -> List[Dict[str, Any]]:
    working_memory = _ensure_working_memory(state)
    user_id = str(state.get("user_id") or "").strip()
    session_id = str(state.get("session_id") or "").strip()
    run_id = str(state.get("run_id") or "").strip()
    thread_id = str(state.get("thread_id") or "").strip()
    candidates: List[Dict[str, Any]] = []

    user_preferences = dict(working_memory.get("user_preferences") or {})
    if user_preferences:
        namespace = f"user/{user_id}/profile" if user_id else f"session/{session_id}/profile"
        candidates.append(
            {
                "namespace": namespace,
                "memory_type": "profile",
                "summary": "用户偏好",
                "content": user_preferences,
                "tags": list(user_preferences.keys())[:5],
                "source": {
                    "session_id": session_id,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "stage": "summarize",
                },
                "confidence": 0.8,
                "dedupe_key": _build_memory_dedupe_key(
                    namespace=namespace,
                    memory_type="profile",
                    content=user_preferences,
                ),
            }
        )

    for fact in list(working_memory.get("facts_in_session") or []):
        normalized_fact = str(fact or "").strip()
        if not normalized_fact:
            continue
        namespace = f"session/{session_id}/fact"
        content = {"text": normalized_fact}
        candidates.append(
            {
                "namespace": namespace,
                "memory_type": "fact",
                "summary": normalized_fact[:120],
                "content": content,
                "tags": ["fact"],
                "source": {
                    "session_id": session_id,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "stage": "summarize",
                },
                "confidence": 0.6,
                "dedupe_key": _build_memory_dedupe_key(
                    namespace=namespace,
                    memory_type="fact",
                    content=content,
                ),
            }
        )

    return candidates


def _build_model_memory_candidates(
        state: PlannerReActLangGraphState,
        raw_candidates: Any,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_candidates, list):
        return []

    session_id = str(state.get("session_id") or "").strip()
    run_id = str(state.get("run_id") or "").strip()
    thread_id = str(state.get("thread_id") or "").strip()
    user_id = str(state.get("user_id") or "").strip()
    normalized_candidates: List[Dict[str, Any]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        memory_type = str(item.get("memory_type") or "fact").strip().lower()
        if memory_type not in {"profile", "fact", "instruction"}:
            continue
        summary = str(item.get("summary") or "").strip()
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        if not summary and not content:
            continue
        namespace = str(item.get("namespace") or "").strip()
        if not namespace:
            if memory_type == "profile":
                namespace = f"user/{user_id}/profile" if user_id else f"session/{session_id}/profile"
            elif memory_type == "instruction":
                namespace = "agent/planner_react/instruction"
            else:
                namespace = f"session/{session_id}/fact"
        tags = [str(tag).strip() for tag in list(item.get("tags") or []) if str(tag).strip()]
        confidence_raw = item.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else 0.6
        except Exception:
            confidence = 0.6
        normalized_candidates.append(
            {
                "namespace": namespace,
                "memory_type": memory_type,
                "summary": summary[:120],
                "content": content,
                "tags": tags[:8],
                "source": {
                    "session_id": session_id,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "stage": "summarize",
                    "source_type": "llm_extract",
                },
                "confidence": max(0.0, min(confidence, 1.0)),
                "dedupe_key": _build_memory_dedupe_key(
                    namespace=namespace,
                    memory_type=memory_type,
                    content=content or {"summary": summary[:120]},
                ),
            }
        )
    return normalized_candidates


def _append_message_window_entry(
        message_window: List[Dict[str, Any]],
        *,
        role: str,
        message: str,
        attachments: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    next_entry = _normalize_message_window_entry(
        {
            "role": role,
            "message": message,
            "attachment_paths": list(attachments or []),
        },
        default_role=role,
    )
    if next_entry is None:
        return list(message_window)

    # 创建消息窗口的副本以避免直接修改原列表
    updated_window = list(message_window)

    # 检查是否与最后一条消息完全重复（角色、内容、附件均一致），若是则避免重复添加
    if updated_window:
        latest_entry = _normalize_message_window_entry(
            dict(updated_window[-1]),
            default_role=role,
        )
        if latest_entry == next_entry:
            return updated_window

    # 将新条目添加到消息窗口末尾
    updated_window.append(next_entry)
    return updated_window


def _compact_message_window(
        message_window: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    normalized_window: List[Dict[str, Any]] = []
    for item in list(message_window or []):
        normalized_item = _normalize_message_window_entry(item, default_role="assistant")
        if normalized_item is None:
            continue
        if normalized_window and normalized_window[-1] == normalized_item:
            continue
        normalized_window.append(normalized_item)

    if len(normalized_window) <= MESSAGE_WINDOW_MAX_ITEMS:
        return list(normalized_window), 0

    trimmed_count = len(normalized_window) - MESSAGE_WINDOW_MAX_ITEMS
    return list(normalized_window[-MESSAGE_WINDOW_MAX_ITEMS:]), trimmed_count


def _build_conversation_summary(state: PlannerReActLangGraphState) -> str:
    # 获取之前的对话摘要
    previous_summary = str(state.get("conversation_summary") or "").strip()
    # 确保工作记忆存在并初始化默认值
    working_memory = _ensure_working_memory(state)
    # 获取当前计划对象
    plan = state.get("plan")
    # 获取步骤状态列表
    step_states = list(state.get("step_states") or [])
    # 统计已完成的步骤数量
    completed_steps = sum(1 for item in step_states if str(item.get("status") or "") == ExecutionStatus.COMPLETED.value)
    # 计算总步骤数：优先使用 step_states 长度，若为空则使用 plan.steps 长度
    total_steps = len(step_states) if len(step_states) > 0 else len(getattr(plan, "steps", []) or [])
    parts: List[str] = []
    # 如果存在之前的摘要，则添加到部分列表中
    if previous_summary:
        parts.append(previous_summary)

    # 构建目标字符串：优先从工作记忆获取，其次从 plan 获取，最后从用户消息获取
    goal = str(working_memory.get("goal") or getattr(plan, "goal", "") or state.get("user_message") or "").strip()
    if goal:
        parts.append(f"目标:{goal}")

    # 如果总步骤数大于 0，添加进度信息
    if total_steps > 0:
        parts.append(f"进度:{completed_steps}/{total_steps}")

    trimmed_message_count = int((state.get("graph_metadata") or {}).get("memory_trimmed_message_count") or 0)
    if trimmed_message_count > 0:
        parts.append(f"裁剪:{trimmed_message_count}条消息")

    # 获取最终消息并截断至 120 字符
    final_message = str(state.get("final_message") or "").strip()
    if final_message:
        parts.append(f"结果:{final_message[:120]}")

    # 返回最近若干个部分，用 " | " 连接，避免摘要无限膨胀
    return " | ".join(parts[-CONVERSATION_SUMMARY_MAX_PARTS:])


def _extract_profile_preferences(retrieved_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    preferences: Dict[str, Any] = {}
    for item in retrieved_memories:
        if not isinstance(item, dict):
            continue
        if str(item.get("memory_type") or "").strip().lower() != "profile":
            continue
        content = item.get("content")
        if isinstance(content, dict):
            preferences.update(content)
    return preferences


async def _build_message(llm: LLM, user_message_prompt: str, input_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if getattr(llm, "multimodal", False) and input_parts is not None and len(input_parts) > 0:
        multiplexed_message = await llm.format_multiplexed_message(input_parts)
        user_content = [
            {"type": "text", "text": user_message_prompt},
            *multiplexed_message,
        ]
    else:
        user_content = [{"type": "text", "text": user_message_prompt}]
    return user_content


def _normalize_interrupt_request(raw: Any) -> Dict[str, Any]:
    return normalize_wait_payload(raw)


def _resume_value_to_message(payload: Dict[str, Any], value: Any) -> str:
    return resolve_wait_resume_message(payload, value)


def _build_wait_resume_step_summary(step: Step, resumed_message: str) -> str:
    step_label = str(step.title or step.description or "当前步骤").strip() or "当前步骤"
    normalized_message = str(resumed_message or "").strip()
    if normalized_message:
        return f"{step_label}已收到用户回复：{normalized_message}"
    return f"{step_label}已完成用户交互"


async def create_or_reuse_plan_node(state: PlannerReActLangGraphState, llm: LLM) -> PlannerReActLangGraphState:
    """创建计划或复用已有计划。"""
    plan = state.get("plan")
    if plan is not None and len(plan.steps) > 0 and not plan.done:
        next_step = plan.get_next_step()
        working_memory = _ensure_working_memory(state)
        if not str(working_memory.get("goal") or "").strip():
            working_memory["goal"] = str(plan.goal or state.get("user_message") or "")
        planner_local_memory = dict(state.get("planner_local_memory") or {})
        planner_local_memory["plan_brief"] = str(plan.message or plan.title or "")
        return {
            **state,
            "working_memory": working_memory,
            "planner_local_memory": planner_local_memory,
            "current_step_id": next_step.id if next_step is not None else None,
        }

    user_message = state.get("user_message", "").strip()

    input_parts = list(state.get("input_parts") or [])
    attachments = [part.get("sandbox_filepath") for part in input_parts]
    user_message_prompt = CREATE_PLAN_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
    )
    user_message_prompt = _append_execution_context_to_prompt(user_message_prompt, state)

    user_content = await _build_message(llm, user_message_prompt, input_parts)

    logger.info("planner 计划")
    llm_message = await llm.invoke(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))

    title = str(parsed.get("title") or build_fallback_plan_title(user_message))
    language = str(parsed.get("language") or "zh")
    goal = str(parsed.get("goal") or user_message)
    planner_message = str(parsed.get("message") or user_message or "已生成任务计划")
    working_memory = _ensure_working_memory(state)
    working_memory["goal"] = goal
    planner_local_memory = dict(state.get("planner_local_memory") or {})
    planner_local_memory["plan_brief"] = planner_message or title
    planner_local_memory["plan_assumptions"] = _append_unique_text_item(
        list(planner_local_memory.get("plan_assumptions") or []),
        str(parsed.get("assumption") or ""),
    )
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or raw_steps is None or len(raw_steps) == 0:
        plan = Plan(
            title=title,
            goal=goal,
            language=language,
            message=planner_message,
            steps=[],
            status=ExecutionStatus.COMPLETED,
        )
        planner_events: List[Any] = [
            TitleEvent(title=title),
            MessageEvent(role="assistant", message=planner_message)
        ]
        await emit_live_events(*planner_events)
        return _reduce_state_with_events(
            state,
            updates={
                "plan": plan,
                "working_memory": working_memory,
                "planner_local_memory": planner_local_memory,
                "current_step_id": None,
                "final_message": planner_message,
                "graph_metadata": {
                    **dict(state.get("graph_metadata") or {}),
                },
                "step_states": [],
            },
            events=planner_events,
        )
    else:
        steps = [build_step_from_payload(item, index) for index, item in enumerate(raw_steps)]
        plan = Plan(
            title=title,
            goal=goal,
            language=language,
            message=planner_message,
            steps=steps,
            status=ExecutionStatus.PENDING,
        )
        next_step = plan.get_next_step()

        planner_events: List[Any] = [
            TitleEvent(title=title),
            PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.CREATED),
            MessageEvent(role="assistant", message=planner_message)
        ]
        await emit_live_events(*planner_events)
        return _reduce_state_with_events(
            state,
            updates={
                "plan": plan,
                "working_memory": working_memory,
                "planner_local_memory": planner_local_memory,
                "current_step_id": next_step.id if next_step is not None else None,
                "graph_metadata": {
                    **dict(state.get("graph_metadata") or {}),
                },
            },
            events=planner_events,
        )


async def execute_step_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
        skill_runtime: Optional[SkillGraphRuntime] = None,
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
) -> PlannerReActLangGraphState:
    """执行单个步骤，完成后交给 replan 节点更新后续计划。"""
    plan = state.get("plan")
    if plan is None:
        return state

    step = plan.get_next_step()
    if step is None:
        return state

    step.status = ExecutionStatus.RUNNING
    started_event = StepEvent(step=step.model_copy(deep=True), status=StepEventStatus.STARTED)
    await emit_live_events(started_event)

    user_message = str(state.get("user_message", ""))
    language = plan.language or "zh"
    input_parts = list(state.get("input_parts") or [])
    attachments = [part.get("sandbox_filepath") for part in input_parts]

    user_message_prompt = EXECUTION_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
        language=language,
        step=step.description,
    )
    user_message_prompt = _append_execution_context_to_prompt(user_message_prompt, state)
    user_content = await _build_message(llm, user_message_prompt, input_parts)

    llm_message: Optional[Dict[str, Any]] = None
    tool_events: List[ToolEvent] = []

    try:
        async with asyncio.timeout(STEP_EXECUTION_TIMEOUT_SECONDS):
            # 若运行时已注入工具能力，则优先走“提示词 + 工具循环”路径。
            if runtime_tools:
                llm_message, tool_events = await execute_step_with_prompt(
                    llm=llm,
                    step=step,
                    runtime_tools=runtime_tools,
                    max_tool_iterations=max_tool_iterations,
                    on_tool_event=emit_live_events,
                    user_content=user_content,
                )

            # 无工具能力时保持原 BE-LG-10 skill 路径。
            if llm_message is None and skill_runtime is not None:
                try:
                    skill_result = await skill_runtime.execute_skill(
                        skill_id=PLANNER_EXECUTE_STEP_SKILL_ID,
                        payload={
                            "session_id": str(state.get("session_id") or ""),
                            "user_message": user_message,
                            "step_description": step.description,
                            "language": language,
                            "attachments": attachments,
                            "execution_context": _build_execution_context_block(state),
                        },
                    )
                    llm_message = {
                        "success": bool(getattr(skill_result, "success", True)),
                        "result": str(getattr(skill_result, "result", "") or f"已完成步骤：{step.description}"),
                        "attachments": normalize_attachments(getattr(skill_result, "attachments", [])),
                    }
                except Exception as e:
                    logger.warning("执行步骤 Skill 运行失败，回退默认执行链路: %s", e)
    except TimeoutError:
        logger.warning("步骤执行超时: step_id=%s description=%s", str(step.id or ""), str(step.description or ""))
        llm_message = {
            "success": False,
            "result": f"步骤执行超时：{step.description}",
            "attachments": [],
            "blockers": [f"当前步骤超过 {STEP_EXECUTION_TIMEOUT_SECONDS} 秒未完成"],
            "next_hint": "请缩小当前步骤范围后重试",
        }

    if llm_message is None:
        llm_message = {
            "success": False,
            "result": f"步骤执行失败：{step.description}",
            "attachments": [],
        }

    interrupt_request = _normalize_interrupt_request(llm_message.get("interrupt_request"))
    if interrupt_request:
        step_local_memory = dict(state.get("step_local_memory") or {})
        step_local_memory["current_step_id"] = str(step.id)
        step_local_memory["pending_interrupt"] = interrupt_request
        step_local_memory["waiting_step_id"] = str(step.id)
        step_local_memory["waiting_step_title"] = str(step.title or "").strip()
        step_local_memory["waiting_step_description"] = str(step.description or "").strip()
        graph_metadata = dict(state.get("graph_metadata") or {})
        graph_metadata["waiting_interrupt_kind"] = interrupt_request.get("kind") or "input_text"
        graph_metadata["waiting_interrupt_prompt"] = interrupt_request.get("prompt") or ""
        graph_metadata["step_reuse_hit"] = False
        graph_metadata["last_step_execution_mode"] = "executed"
        return _reduce_state_with_events(
            state,
            updates={
                "plan": plan,
                "current_step_id": step.id,
                "step_local_memory": step_local_memory,
                "graph_metadata": graph_metadata,
                "pending_interrupt": interrupt_request,
            },
            events=[started_event, *tool_events],
        )

    step_success = bool(llm_message.get("success", True))
    step_summary = _normalize_step_result_text(
        llm_message.get("result"),
        fallback=f"已完成步骤：{step.description}" if step_success else f"步骤执行失败：{step.description}",
    )
    model_attachment_paths = normalize_attachments(llm_message.get("attachments"))
    step.outcome = StepOutcome(
        done=step_success,
        summary=step_summary,
        produced_artifacts=model_attachment_paths,
        blockers=_normalize_text_items(llm_message.get("blockers")),
        facts_learned=_normalize_text_items(llm_message.get("facts_learned")),
        open_questions=_normalize_text_items(llm_message.get("open_questions")),
        next_hint=_normalize_step_result_text(llm_message.get("next_hint")),
    )
    step.status = ExecutionStatus.COMPLETED if step_success else ExecutionStatus.FAILED

    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=step.status,
    )
    final_step_events: List[Any] = [completed_event]
    # if step.outcome is not None and step.outcome.summary:
    #     final_step_events.append(
    #         MessageEvent(
    #             role="assistant",
    #             message=step.outcome.summary,
    #             attachments=[File(filepath=filepath) for filepath in step.outcome.produced_artifacts],
    #         )
    #     )

    await emit_live_events(*final_step_events)

    events: List[Any] = [started_event, *tool_events, *final_step_events]
    next_step = plan.get_next_step()
    graph_metadata = dict(state.get("graph_metadata") or {})
    graph_metadata["step_reuse_hit"] = False
    graph_metadata["last_step_execution_mode"] = "executed"
    working_memory = _merge_step_outcome_into_working_memory(
        _ensure_working_memory(state),
        step.outcome,
    )
    step_local_memory = dict(state.get("step_local_memory") or {})
    step_local_memory["current_step_id"] = str(step.id)
    step_local_memory["observation_summary"] = step_summary
    step_local_memory["pending_findings"] = list(step.outcome.produced_artifacts or [])
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "last_executed_step": step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
            "working_memory": working_memory,
            "step_local_memory": step_local_memory,
            "graph_metadata": graph_metadata,
            "final_message": step_summary,
            "selected_artifacts": list(
                dict.fromkeys(
                    list(state.get("selected_artifacts") or [])
                    + list(step.outcome.produced_artifacts or [])
                )
            ),
            "pending_interrupt": {},
        },
        events=events,
    )


async def wait_for_human_node(
        state: PlannerReActLangGraphState,
) -> PlannerReActLangGraphState:
    """在等待节点中恢复用户输入，并完成当前等待中的步骤。"""
    interrupt_request = _normalize_interrupt_request(state.get("pending_interrupt"))
    if not interrupt_request:
        return {
            **state,
            "pending_interrupt": {},
        }

    resume_value = interrupt(interrupt_request)
    resumed_message = _resume_value_to_message(interrupt_request, resume_value)
    message_window = list(state.get("message_window") or [])
    if resumed_message:
        message_window = _append_message_window_entry(
            message_window,
            role="user",
            message=resumed_message,
            attachments=[],
        )

    graph_metadata = dict(state.get("graph_metadata") or {})
    graph_metadata["last_resume_value"] = resume_value
    graph_metadata["last_resumed_at"] = datetime.now().isoformat()
    graph_metadata.pop("waiting_interrupt_kind", None)
    graph_metadata.pop("waiting_interrupt_prompt", None)
    graph_metadata.pop("pending_interrupts", None)

    step_local_memory = dict(state.get("step_local_memory") or {})
    step_local_memory.pop("pending_interrupt", None)
    waiting_step_id = str(
        step_local_memory.pop("waiting_step_id", "") or state.get("current_step_id") or ""
    ).strip()
    step_local_memory.pop("waiting_step_title", None)
    step_local_memory.pop("waiting_step_description", None)

    plan = state.get("plan")
    if plan is None or not waiting_step_id:
        return {
            **state,
            "user_message": resumed_message,
            "input_parts": [],
            "message_window": message_window,
            "graph_metadata": graph_metadata,
            "step_local_memory": step_local_memory,
            "pending_interrupt": {},
        }

    waiting_step: Optional[Step] = None
    for candidate in list(plan.steps or []):
        if str(candidate.id or "").strip() == waiting_step_id:
            waiting_step = candidate
            break

    if waiting_step is None:
        return {
            **state,
            "user_message": resumed_message,
            "input_parts": [],
            "message_window": message_window,
            "graph_metadata": graph_metadata,
            "step_local_memory": step_local_memory,
            "pending_interrupt": {},
        }

    waiting_step.outcome = StepOutcome(
        done=True,
        summary=_build_wait_resume_step_summary(waiting_step, resumed_message),
    )
    waiting_step.status = ExecutionStatus.COMPLETED
    completed_event = StepEvent(
        step=waiting_step.model_copy(deep=True),
        status=StepEventStatus.COMPLETED,
    )
    await emit_live_events(completed_event)
    next_step = plan.get_next_step()
    working_memory = _merge_step_outcome_into_working_memory(
        _ensure_working_memory(state),
        waiting_step.outcome,
    )
    step_local_memory["current_step_id"] = str(waiting_step.id)
    step_local_memory["observation_summary"] = _normalize_step_result_text(waiting_step.outcome.summary)
    step_local_memory["pending_findings"] = []

    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "user_message": resumed_message,
            "input_parts": [],
            "message_window": message_window,
            "graph_metadata": graph_metadata,
            "step_local_memory": step_local_memory,
            "pending_interrupt": {},
            "last_executed_step": waiting_step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
            "working_memory": working_memory,
            "final_message": _normalize_step_result_text(waiting_step.outcome.summary),
        },
        events=[completed_event],
    )


async def recall_memory_context_node(
        state: PlannerReActLangGraphState,
        long_term_memory_repository: Optional[LongTermMemoryRepository] = None,
) -> PlannerReActLangGraphState:
    """统一整理线程级短期记忆，为后续 planner/react 节点提供稳定输入。"""
    plan = state.get("plan")
    working_memory = _ensure_working_memory(state)
    if not str(working_memory.get("goal") or "").strip():
        working_memory["goal"] = str(getattr(plan, "goal", "") or state.get("user_message") or "")
    if not list(working_memory.get("open_questions") or []):
        working_memory["open_questions"] = _normalize_text_items(state.get("session_open_questions"))

    retrieved_memories = list(state.get("retrieved_memories") or [])
    if long_term_memory_repository is not None:
        try:
            recalled_memories: List[LongTermMemory] = []
            for query in _build_memory_recall_queries(state):
                recalled_memories.extend(await long_term_memory_repository.search(query))
            recalled_memories = _dedupe_recalled_memories(recalled_memories)
            retrieved_memories = [memory.model_dump(mode="json") for memory in recalled_memories]
        except Exception as e:
            logger.warning("长期记忆召回失败，回退已有线程态记忆快照: %s", e)
    if not dict(working_memory.get("user_preferences") or {}):
        working_memory["user_preferences"] = _extract_profile_preferences(retrieved_memories)

    graph_metadata = dict(state.get("graph_metadata") or {})
    graph_metadata["memory_recall_prepared_at"] = datetime.now().isoformat()
    graph_metadata["memory_recall_count"] = len(retrieved_memories)
    graph_metadata["memory_recall_namespaces"] = _build_memory_namespace_prefixes(state)
    graph_metadata["context_recent_run_brief_count"] = len(state.get("recent_run_briefs") or [])
    graph_metadata["context_recent_attempt_brief_count"] = len(state.get("recent_attempt_briefs") or [])
    graph_metadata["context_selected_artifact_count"] = len(state.get("selected_artifacts") or [])
    graph_metadata["context_historical_artifact_count"] = len(state.get("historical_artifact_refs") or [])

    return {
        **state,
        "working_memory": working_memory,
        "retrieved_memories": retrieved_memories,
        "planner_local_memory": dict(state.get("planner_local_memory") or {}),
        "step_local_memory": dict(state.get("step_local_memory") or {}),
        "summary_local_memory": dict(state.get("summary_local_memory") or {}),
        "memory_context_version": _build_memory_context_version(state),
        "graph_metadata": graph_metadata,
    }


async def replan_node(state: PlannerReActLangGraphState, llm: LLM) -> PlannerReActLangGraphState:
    """根据最新步骤执行结果更新后续未完成步骤。"""
    plan = state.get("plan")
    last_step = state.get("last_executed_step")
    if plan is None or last_step is None:
        return state

    prompt = UPDATE_PLAN_PROMPT.format(step=last_step.model_dump_json(), plan=plan.model_dump_json())
    prompt = _append_execution_context_to_prompt(prompt, state)
    logger.info("replan 计划")
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))

    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list):
        return state

    new_steps = [build_step_from_payload(item, index) for index, item in enumerate(raw_steps)]
    first_pending_index: Optional[int] = None
    for index, current_step in enumerate(plan.steps):
        if not current_step.done:
            first_pending_index = index
            break

    if first_pending_index is not None:
        updated_steps = plan.steps[:first_pending_index]
        # 重规划返回的步骤可能复用了历史 step_id，需要统一做去重，避免覆盖已完成步骤。
        updated_steps.extend(_dedupe_replanned_steps(updated_steps, new_steps))
        plan.steps = updated_steps

    next_step = plan.get_next_step()
    updated_event = PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.UPDATED)
    await emit_live_events(updated_event)
    planner_local_memory = dict(state.get("planner_local_memory") or {})
    planner_local_memory["replan_rationale"] = _get_step_outcome_summary(last_step)
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "planner_local_memory": planner_local_memory,
            "step_local_memory": {},
            "current_step_id": next_step.id if next_step is not None else None,
        },
        events=[updated_event],
    )


async def summarize_node(state: PlannerReActLangGraphState, llm: LLM) -> PlannerReActLangGraphState:
    """在所有步骤完成后汇总结果。"""
    plan = state.get("plan")
    if plan is None:
        return state
    plan_snapshot = plan.model_dump(mode="json") if plan is not None else {}
    final_message = str(state.get("final_message") or "")
    user_message = str(state.get("user_message") or "")
    execution_count = int(state.get("execution_count") or 0)
    summarize_prompt = SUMMARIZE_PROMPT.format(user_message=user_message,
                                               execution_count=execution_count,
                                               final_message=final_message,
                                               plan_snapshot=json.dumps(plan_snapshot, ensure_ascii=False))
    logger.info("总结计划")
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": summarize_prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))
    summary_message = str(parsed.get("message") or "")
    extracted_facts = _normalize_memory_fact_items(parsed.get("facts_in_session"))
    extracted_preferences = _normalize_memory_preferences(parsed.get("user_preferences"))
    model_memory_candidates = _build_model_memory_candidates(
        state=state,
        raw_candidates=parsed.get("memory_candidates"),
    )
    # 附件处理
    summary_attachment_refs = merge_attachment_paths(normalize_attachments(parsed.get("attachments")))
    summary_attachment_paths = [File(filepath=filepath) for filepath in summary_attachment_refs]

    final_events: List[Any] = [
        MessageEvent(role="assistant", message=summary_message, attachments=summary_attachment_paths)]

    plan.status = ExecutionStatus.COMPLETED
    final_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.COMPLETED))

    await emit_live_events(*final_events)
    working_memory = _ensure_working_memory(state)
    working_memory["facts_in_session"] = list(working_memory.get("facts_in_session") or [])
    for fact in extracted_facts:
        working_memory["facts_in_session"] = _append_unique_text_item(
            list(working_memory.get("facts_in_session") or []),
            fact,
        )
    if extracted_preferences:
        merged_preferences = dict(working_memory.get("user_preferences") or {})
        merged_preferences.update(extracted_preferences)
        working_memory["user_preferences"] = merged_preferences

    summary_local_memory = dict(state.get("summary_local_memory") or {})
    summary_local_memory["answer_outline"] = summary_message
    summary_local_memory["selected_artifacts"] = summary_attachment_refs
    next_state_for_memory: PlannerReActLangGraphState = {
        **state,
        "working_memory": working_memory,
        "final_message": summary_message,
    }
    memory_candidates = _build_memory_candidates(next_state_for_memory)
    if len(memory_candidates) == 0:
        outcome_candidate = _build_outcome_memory_candidate(
            next_state_for_memory,
            summary_message=summary_message,
        )
        if outcome_candidate is not None:
            memory_candidates = [outcome_candidate]
    memory_candidates = _merge_memory_candidates(memory_candidates, model_memory_candidates)
    if len(model_memory_candidates) > 0:
        summary_local_memory["memory_candidates_reason"] = "基于总结阶段的结构化提炼生成长期记忆候选"
    elif extracted_facts or extracted_preferences:
        summary_local_memory["memory_candidates_reason"] = "从总结阶段提炼出的稳定事实与偏好生成长期记忆候选"
    elif len(memory_candidates) > 0:
        summary_local_memory["memory_candidates_reason"] = "使用本轮任务结果生成保守的会话级长期记忆候选"
    else:
        summary_local_memory["memory_candidates_reason"] = ""
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "current_step_id": None,
            "final_message": summary_message,
            "working_memory": working_memory,
            "selected_artifacts": list(
                dict.fromkeys(
                    list(state.get("selected_artifacts") or [])
                    + summary_attachment_refs
                )
            ),
            "pending_memory_writes": _merge_memory_candidates(
                list(state.get("pending_memory_writes") or []),
                memory_candidates,
            ),
            "summary_local_memory": summary_local_memory,
        },
        events=final_events,
    )


async def consolidate_memory_node(
        state: PlannerReActLangGraphState,
        long_term_memory_repository: Optional[LongTermMemoryRepository] = None,
) -> PlannerReActLangGraphState:
    """统一收敛线程级短期记忆，压缩消息窗口并记录压缩元数据。"""
    # 获取并初始化摘要本地记忆
    summary_local_memory = dict(state.get("summary_local_memory") or {})
    
    # 将最终消息和选中的附件添加到消息窗口中
    message_window = _append_message_window_entry(
        list(state.get("message_window") or []),
        role="assistant",
        message=str(state.get("final_message") or ""),
        attachments=list(summary_local_memory.get("selected_artifacts") or []),
    )
    
    # 压缩消息窗口，防止超出最大长度限制
    compacted_message_window, trimmed_message_count = _compact_message_window(message_window)
    
    # 更新图元数据，记录记忆压缩相关信息
    graph_metadata = dict(state.get("graph_metadata") or {})
    graph_metadata["memory_compacted"] = True
    graph_metadata["memory_last_compaction_at"] = datetime.now().isoformat()
    graph_metadata["memory_message_window_size"] = len(compacted_message_window)
    graph_metadata["memory_trimmed_message_count"] = trimmed_message_count

    # 处理待写入的长期记忆候选项
    pending_memory_writes, candidate_stats = _govern_memory_candidates(
        list(state.get("pending_memory_writes") or [])
    )
    graph_metadata["memory_candidate_input_count"] = candidate_stats["input_count"]
    graph_metadata["memory_candidate_kept_count"] = candidate_stats["kept_count"]
    graph_metadata["memory_candidate_dropped_invalid_count"] = candidate_stats["dropped_invalid_count"]
    graph_metadata["memory_candidate_dropped_low_confidence_count"] = candidate_stats["dropped_low_confidence_count"]
    graph_metadata["memory_candidate_deduped_count"] = candidate_stats["deduped_count"]
    graph_metadata["memory_candidate_profile_merge_count"] = candidate_stats["merged_profile_count"]
    remaining_memory_writes: List[Dict[str, Any]] = []
    persisted_memory_ids: List[str] = []
    
    if long_term_memory_repository is None:
        # 若未提供长期记忆仓库，则跳过写入，保留候选项供后续重试
        graph_metadata["memory_write_skipped"] = len(pending_memory_writes) > 0
        remaining_memory_writes = pending_memory_writes
    else:
        # 遍历所有待写入的记忆候选项，尝试持久化
        for item in pending_memory_writes:
            try:
                memory = LongTermMemory.model_validate(item)
                persisted_memory = await long_term_memory_repository.upsert(memory)
                persisted_memory_ids.append(persisted_memory.id)
            except Exception as e:
                logger.warning("长期记忆写入失败，保留候选待后续重试：%s", e)
                if isinstance(item, dict):
                    remaining_memory_writes.append(item)
        graph_metadata["memory_write_count"] = len(persisted_memory_ids)
        graph_metadata["memory_write_ids"] = persisted_memory_ids

    # 构建下一个状态对象，更新消息窗口、对话摘要、清理临时记忆并保留未成功写入的记忆候选
    next_state: PlannerReActLangGraphState = {
        **state,
        "message_window": compacted_message_window,
        "conversation_summary": _build_conversation_summary(
            {
                **state,
                "message_window": compacted_message_window,
                "graph_metadata": graph_metadata,
            }
        ),
        "pending_memory_writes": remaining_memory_writes,
        "planner_local_memory": {},
        "step_local_memory": {},
        "summary_local_memory": {},
        "selected_artifacts": list(dict.fromkeys(list(state.get("selected_artifacts") or []) + list(summary_local_memory.get("selected_artifacts") or []))),
        "graph_metadata": graph_metadata,
    }
    return next_state


async def finalize_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """结束节点，追加 done 事件。"""
    events = list(state.get("emitted_events") or [])
    if events and isinstance(events[-1], DoneEvent):
        return state

    done_event = DoneEvent()
    await emit_live_events(done_event)
    return _reduce_state_with_events(
        state,
        updates={},
        events=[done_event],
    )
