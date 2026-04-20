#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层记忆与消息窗口 helper。

本模块只负责记忆查询构造、候选规整、证据过滤和消息窗口压缩，
不决定 planner/replan/summary/consolidate 的节点流转。
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from app.domain.models import (
    ExecutionStatus,
    LongTermMemory,
    LongTermMemorySearchMode,
    LongTermMemorySearchQuery,
    Step,
    StepOutcome,
    ToolEvent,
)
from app.domain.services.runtime.contracts.langgraph_settings import (
    CONVERSATION_SUMMARY_MAX_PARTS,
    MEMORY_CANDIDATE_MIN_CONFIDENCE,
    MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    MESSAGE_WINDOW_MAX_ITEMS,
    MESSAGE_WINDOW_MAX_MESSAGE_CHARS,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import (
    normalize_message_window_entry,
    normalize_text_list,
    normalize_step_result_text,
    truncate_text,
)
from .working_memory import _ensure_working_memory


def _truncate_text(value: Any, *, max_chars: int) -> str:
    return truncate_text(value, max_chars=max_chars)


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
    return normalize_text_list(raw)

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

def _normalize_summary_evidence_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())

def _extract_summary_evidence_fragments(raw: Any) -> List[str]:
    fragments: List[str] = []
    if isinstance(raw, str):
        normalized = _normalize_summary_evidence_text(raw)
        if normalized:
            fragments.append(normalized)
        return fragments
    if isinstance(raw, (int, float, bool)):
        normalized = _normalize_summary_evidence_text(raw)
        if normalized:
            fragments.append(normalized)
        return fragments
    if isinstance(raw, list):
        for item in raw:
            fragments.extend(_extract_summary_evidence_fragments(item))
        return fragments
    if isinstance(raw, dict):
        for key, value in raw.items():
            if str(key or "").strip():
                fragments.extend(_extract_summary_evidence_fragments(key))
            fragments.extend(_extract_summary_evidence_fragments(value))
        return fragments
    return fragments

def _collect_successful_tool_event_evidence(state: PlannerReActLangGraphState) -> List[str]:
    evidence: List[str] = []
    for event in reversed(list(state.get("emitted_events") or [])):
        if not isinstance(event, ToolEvent):
            continue
        status_value = str(
            getattr(getattr(event, "status", ""), "value", getattr(event, "status", "")) or "").strip().lower()
        if status_value != "called":
            continue
        function_result = getattr(event, "function_result", None)
        if function_result is None or not bool(getattr(function_result, "success", False)):
            continue
        evidence.extend(
            _extract_summary_evidence_fragments(
                {
                    "function_name": str(getattr(event, "function_name", "") or "").strip(),
                    "function_args": getattr(event, "function_args", {}) or {},
                    "message": str(getattr(function_result, "message", "") or "").strip(),
                    "data": getattr(function_result, "data", {}) or {},
                }
            )
        )
        if len(evidence) >= 40:
            break
    return evidence[:40]

def _collect_summary_evidence_texts(
        *,
        state: PlannerReActLangGraphState,
        last_executed_step: Optional[Step],
) -> List[str]:
    plan_value = state.get("plan")
    goal_value = (
        str(plan_value.get("goal") or "").strip()
        if isinstance(plan_value, dict)
        else str(getattr(plan_value, "goal", "") or "").strip()
    )
    evidence: List[str] = []
    evidence.extend(
        _extract_summary_evidence_fragments(
            {
                "user_message": str(state.get("user_message") or "").strip(),
                "goal": goal_value,
            }
        )
    )
    if last_executed_step is not None and last_executed_step.outcome is not None:
        evidence.extend(
            _extract_summary_evidence_fragments(
                {
                    "summary": last_executed_step.outcome.summary,
                    "delivery_text": last_executed_step.outcome.delivery_text,
                    "blockers": list(last_executed_step.outcome.blockers or []),
                    "facts_learned": list(last_executed_step.outcome.facts_learned or []),
                    "next_hint": last_executed_step.outcome.next_hint,
                    "produced_artifacts": list(last_executed_step.outcome.produced_artifacts or []),
                }
            )
        )
    recent_action_digest = state.get("recent_action_digest")
    if isinstance(recent_action_digest, dict):
        evidence.extend(_extract_summary_evidence_fragments(recent_action_digest.get("payload")))
    evidence.extend(_collect_successful_tool_event_evidence(state))
    deduped: List[str] = []
    for item in evidence:
        if not item or item in deduped:
            continue
        deduped.append(item)
    return deduped[:80]

def _memory_item_has_execution_evidence(item_text: str, evidence_texts: List[str]) -> bool:
    normalized_item_text = _normalize_summary_evidence_text(item_text)
    if not normalized_item_text:
        return False

    for evidence_text in evidence_texts:
        if not evidence_text:
            continue
        if normalized_item_text in evidence_text:
            return True

    keyword_tokens = [token for token in re.findall(r"[a-z0-9_./:-]+", normalized_item_text) if len(token) >= 4]
    for token in keyword_tokens:
        if any(token in evidence_text for evidence_text in evidence_texts):
            return True

    zh_tokens = re.findall(r"[\u4e00-\u9fff]{4,}", normalized_item_text)
    for token in zh_tokens:
        if any(token in evidence_text for evidence_text in evidence_texts):
            return True
    return False

def _filter_summary_facts_by_evidence(facts: List[str], evidence_texts: List[str]) -> List[str]:
    filtered: List[str] = []
    for fact in facts:
        if _memory_item_has_execution_evidence(fact, evidence_texts):
            filtered.append(fact)
    return filtered

def _filter_model_memory_candidates_by_evidence(
        candidates: List[Dict[str, Any]],
        evidence_texts: List[str],
) -> Tuple[List[Dict[str, Any]], int]:
    filtered: List[Dict[str, Any]] = []
    dropped_count = 0
    for item in candidates:
        if not isinstance(item, dict):
            dropped_count += 1
            continue
        memory_type = str(item.get("memory_type") or "").strip().lower()
        if memory_type not in {"fact", "instruction", "profile"}:
            filtered.append(item)
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        candidate_fragments = [str(item.get("summary") or "").strip(), str(content.get("text") or "").strip()]
        if memory_type == "profile":
            candidate_fragments.extend(
                [
                    " ".join([str(key).strip(), str(value).strip()]).strip()
                    for key, value in content.items()
                ]
            )
        candidate_text = " ".join([fragment for fragment in candidate_fragments if fragment]).strip()
        if _memory_item_has_execution_evidence(candidate_text, evidence_texts):
            filtered.append(item)
            continue
        dropped_count += 1
    return filtered, dropped_count

def _preference_item_has_execution_evidence(
        *,
        key: str,
        value: Any,
        evidence_texts: List[str],
) -> bool:
    candidate_text = " ".join([str(key or "").strip(), str(value or "").strip()]).strip()
    if _memory_item_has_execution_evidence(candidate_text, evidence_texts):
        return True
    if _memory_item_has_execution_evidence(str(value or "").strip(), evidence_texts):
        return True

    normalized_key = _normalize_summary_evidence_text(key)
    normalized_value = _normalize_summary_evidence_text(value)
    evidence_blob = " ".join(evidence_texts)
    if normalized_key in {"language", "lang", "语言"}:
        if normalized_value in {"zh", "zh-cn", "chinese", "中文"} and (
                "中文" in evidence_blob or "chinese" in evidence_blob
        ):
            return True
        if normalized_value in {"en", "en-us", "english", "英文"} and (
                "英文" in evidence_blob or "english" in evidence_blob
        ):
            return True
    if normalized_key in {"response_style", "style", "回复风格", "风格"}:
        if normalized_value in {"concise", "brief", "简洁", "简明"} and (
                "简洁" in evidence_blob or "简明" in evidence_blob or "concise" in evidence_blob or "brief" in evidence_blob
        ):
            return True
        if normalized_value in {"detailed", "详细"} and ("详细" in evidence_blob or "detailed" in evidence_blob):
            return True
    return False

def _filter_preferences_by_evidence(
        preferences: Dict[str, Any],
        evidence_texts: List[str],
) -> Tuple[Dict[str, Any], int]:
    filtered: Dict[str, Any] = {}
    dropped_count = 0
    for key, value in dict(preferences or {}).items():
        if _preference_item_has_execution_evidence(key=str(key or ""), value=value, evidence_texts=evidence_texts):
            filtered[str(key)] = value
            continue
        dropped_count += 1
    return filtered, dropped_count

def _build_outcome_fact_text(
        state: PlannerReActLangGraphState,
        summary_message: str,
) -> str:
    normalized_summary = str(summary_message or "").strip()
    if normalized_summary:
        return normalized_summary

    working_memory = _ensure_working_memory(state)
    decisions = [str(item or "").strip() for item in list(working_memory.get("decisions") or []) if
                 str(item or "").strip()]
    if len(decisions) > 0:
        return decisions[-1]

    last_step = state.get("last_executed_step")
    if last_step is None or last_step.outcome is None:
        return ""
    return normalize_step_result_text(last_step.outcome.summary)

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

def _compact_message_window(
        message_window: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    normalized_window: List[Dict[str, Any]] = []
    for item in list(message_window or []):
        normalized_item = normalize_message_window_entry(
            item,
            default_role="assistant",
            max_message_chars=MESSAGE_WINDOW_MAX_MESSAGE_CHARS,
            max_attachment_paths=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
        )
        if normalized_item is None:
            continue
        if normalized_window and normalized_window[-1] == normalized_item:
            continue
        normalized_window.append(normalized_item)

    if len(normalized_window) <= MESSAGE_WINDOW_MAX_ITEMS:
        return list(normalized_window), 0

    trimmed_count = len(normalized_window) - MESSAGE_WINDOW_MAX_ITEMS
    return list(normalized_window[-MESSAGE_WINDOW_MAX_ITEMS:]), trimmed_count

def _build_conversation_summary(
        state: PlannerReActLangGraphState,
        *,
        trimmed_message_count: int = 0,
) -> str:
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

    if trimmed_message_count > 0:
        parts.append(f"裁剪:{trimmed_message_count}条消息")

    # 获取最终消息并截断至 120 字符
    final_message = str(state.get("final_message") or "").strip()
    if final_message:
        parts.append(f"结果:{final_message[:120]}")

    # 返回最近若干个部分，用 " | " 连接，避免摘要无限膨胀
    return " | ".join(parts[-CONVERSATION_SUMMARY_MAX_PARTS:])
