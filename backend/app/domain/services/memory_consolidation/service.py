#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆沉淀领域服务。

服务负责消息窗口压缩、最小 conversation_summary 生成、候选治理与 provider 降级。
这里不接收 LangGraph state，也不直接请求 Ollama。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.domain.services.memory_consolidation.contracts import (
    MemoryConsolidationInput,
    MemoryConsolidationResult,
    MemoryConsolidationStats,
)
from app.domain.services.memory_consolidation.ports import MemoryConsolidationProvider
from app.domain.services.memory_consolidation.settings import (
    CONVERSATION_SUMMARY_MAX_PARTS,
    MEMORY_CANDIDATE_MIN_CONFIDENCE,
    MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    MESSAGE_WINDOW_MAX_ITEMS,
    MESSAGE_WINDOW_MAX_MESSAGE_CHARS,
)
from app.domain.services.runtime.contracts.sensitive_data_policy import (
    assert_memory_content_safe,
    assert_memory_payload_safe,
    detect_sensitive_text,
)
from app.domain.services.runtime.normalizers import normalize_message_window_entry

_MEMORY_SOURCE_ALLOWLIST = {"kind", "stage", "run_id", "step_id", "session_id", "thread_id"}


@dataclass(frozen=True)
class _MemoryCandidateGovernanceResult:
    candidate: Optional[Dict[str, Any]] = None
    reason: str = ""


class MemoryConsolidationService:
    """统一沉淀短期上下文、会话摘要与待写入长期记忆候选。"""

    def __init__(self, provider: Optional[MemoryConsolidationProvider] = None) -> None:
        self._provider = provider

    async def consolidate(self, payload: MemoryConsolidationInput) -> MemoryConsolidationResult:
        """执行沉淀；provider 不存在或失败时使用规则降级结果。"""
        baseline_result = self._build_rule_based_result(payload)
        if self._provider is None:
            return baseline_result

        try:
            provider_result = await self._provider.consolidate(payload)
        except Exception as exc:
            return baseline_result.model_copy(
                update={
                    "degraded": True,
                    "degrade_reason": f"provider_error:{exc.__class__.__name__}",
                }
            )

        return self._merge_provider_result(
            baseline_result=baseline_result,
            provider_result=provider_result,
        )

    def _build_rule_based_result(self, payload: MemoryConsolidationInput) -> MemoryConsolidationResult:
        """构建不依赖模型的最小沉淀结果，保证主流程稳定收尾。"""
        message_window = self._append_message_window_entry(
            list(payload.message_window or []),
            role="assistant",
            message=payload.assistant_message,
            attachments=list(payload.selected_artifacts or []),
        )
        compacted_message_window, trimmed_message_count = self._compact_message_window(message_window)
        memory_candidates, candidate_stats = self._govern_memory_candidates(
            list(payload.pending_memory_writes or [])
        )
        stats = MemoryConsolidationStats(
            input_candidate_count=candidate_stats["input_count"],
            kept_candidate_count=candidate_stats["kept_count"],
            dropped_invalid_count=candidate_stats["dropped_invalid_count"],
            dropped_sensitive_count=candidate_stats["dropped_sensitive_count"],
            dropped_low_confidence_count=candidate_stats["dropped_low_confidence_count"],
            deduped_count=candidate_stats["deduped_count"],
            merged_profile_count=candidate_stats["merged_profile_count"],
            trimmed_message_count=trimmed_message_count,
        )
        return MemoryConsolidationResult(
            message_window=compacted_message_window,
            conversation_summary=self._build_conversation_summary(
                payload=payload,
                trimmed_message_count=trimmed_message_count,
            ),
            facts_in_session=list(payload.facts_in_session or []),
            user_preferences=dict(payload.user_preferences or {}),
            memory_candidates=memory_candidates,
            degraded=False,
            degrade_reason="",
            stats=stats,
        )

    def _merge_provider_result(
            self,
            *,
            baseline_result: MemoryConsolidationResult,
            provider_result: MemoryConsolidationResult,
    ) -> MemoryConsolidationResult:
        """合并 provider 结果；候选仍统一经过领域治理，避免污染待写入队列。"""
        provider_candidates, candidate_stats = self._govern_memory_candidates(
            list(provider_result.memory_candidates or [])
        )
        merged_candidates, merged_stats = self._govern_memory_candidates(
            [*baseline_result.memory_candidates, *provider_candidates]
        )
        stats = baseline_result.stats.model_copy(
            update={
                "input_candidate_count": merged_stats["input_count"],
                "kept_candidate_count": merged_stats["kept_count"],
                "dropped_invalid_count": (
                        baseline_result.stats.dropped_invalid_count
                        + candidate_stats["dropped_invalid_count"]
                        + merged_stats["dropped_invalid_count"]
                ),
                "dropped_sensitive_count": (
                        baseline_result.stats.dropped_sensitive_count
                        + candidate_stats["dropped_sensitive_count"]
                        + merged_stats["dropped_sensitive_count"]
                ),
                "dropped_low_confidence_count": (
                        baseline_result.stats.dropped_low_confidence_count
                        + candidate_stats["dropped_low_confidence_count"]
                        + merged_stats["dropped_low_confidence_count"]
                ),
                "deduped_count": baseline_result.stats.deduped_count + merged_stats["deduped_count"],
                "merged_profile_count": (
                        baseline_result.stats.merged_profile_count
                        + candidate_stats["merged_profile_count"]
                        + merged_stats["merged_profile_count"]
                ),
            }
        )
        return baseline_result.model_copy(
            update={
                "conversation_summary": provider_result.conversation_summary
                                        or baseline_result.conversation_summary,
                "facts_in_session": list(provider_result.facts_in_session or baseline_result.facts_in_session),
                "user_preferences": dict(provider_result.user_preferences or baseline_result.user_preferences),
                "memory_candidates": merged_candidates,
                "degraded": bool(provider_result.degraded),
                "degrade_reason": str(provider_result.degrade_reason or ""),
                "stats": stats,
            }
        )

    @staticmethod
    def _append_message_window_entry(
            message_window: List[Dict[str, Any]],
            *,
            role: str,
            message: str,
            attachments: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """追加消息窗口条目，并避免与最后一条完全重复。"""
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

        updated_window = list(message_window)
        if updated_window:
            latest_entry = normalize_message_window_entry(
                dict(updated_window[-1]),
                default_role=role,
                max_message_chars=MESSAGE_WINDOW_MAX_MESSAGE_CHARS,
                max_attachment_paths=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
            )
            if latest_entry == next_entry:
                return updated_window

        updated_window.append(next_entry)
        return updated_window

    @staticmethod
    def _compact_message_window(
            message_window: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """归一化并裁剪近期消息窗口。"""
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

    @staticmethod
    def _build_conversation_summary(
            *,
            payload: MemoryConsolidationInput,
            trimmed_message_count: int = 0,
    ) -> str:
        """生成最小可用会话摘要，后续 provider 可覆盖为更高质量摘要。"""
        parts: List[str] = []
        previous_summary = str(payload.previous_conversation_summary or "").strip()
        if previous_summary:
            parts.append(previous_summary)

        goal = str(payload.goal or payload.user_message or "").strip()
        if goal:
            parts.append(f"目标:{goal}")

        if payload.total_step_count > 0:
            parts.append(f"进度:{payload.completed_step_count}/{payload.total_step_count}")

        if trimmed_message_count > 0:
            parts.append(f"裁剪:{trimmed_message_count}条消息")

        final_message = str(payload.assistant_message or "").strip()
        if final_message:
            parts.append(f"结果:{final_message[:120]}")

        return " | ".join(parts[-CONVERSATION_SUMMARY_MAX_PARTS:])

    @staticmethod
    def _build_memory_dedupe_key(*, namespace: str, memory_type: str, content: Dict[str, Any]) -> str:
        """根据命名空间、类型与内容生成稳定去重键。"""
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

    @staticmethod
    def _normalize_sensitive_string(value: Any) -> str:
        return assert_memory_content_safe(str(value or "").strip())

    @staticmethod
    def _truncate_memory_summary(value: Any) -> str:
        return str(value or "").strip()

    @classmethod
    def _normalize_memory_tags(cls, raw_tags: Any) -> List[str]:
        normalized_tags: List[str] = []
        for raw_tag in list(raw_tags or []):
            normalized_tag = cls._normalize_sensitive_string(raw_tag)
            if normalized_tag:
                normalized_tags.append(normalized_tag)
        return list(dict.fromkeys(normalized_tags))

    @classmethod
    def _normalize_memory_source(cls, raw_source: Any) -> Dict[str, str]:
        if not isinstance(raw_source, dict):
            return {}
        # source 只允许少量结构化上下文字段落库；但拒写判断必须先覆盖完整 provider payload。
        try:
            serialized_source = json.dumps(raw_source, ensure_ascii=False, sort_keys=True)
        except TypeError:
            raise ValueError("长期记忆 source 包含不可序列化元数据")
        if detect_sensitive_text(serialized_source).has_secret:
            raise ValueError("长期记忆 source 包含不允许保存的敏感凭证")
        normalized_source: Dict[str, str] = {}
        for key in _MEMORY_SOURCE_ALLOWLIST:
            raw_value = raw_source.get(key)
            if raw_value is None:
                continue
            normalized_value = cls._normalize_sensitive_string(raw_value)
            if normalized_value:
                normalized_source[key] = normalized_value
        return normalized_source

    def _normalize_memory_candidate(self, item: Dict[str, Any]) -> _MemoryCandidateGovernanceResult:
        """规整长期记忆候选，过滤缺少命名空间、类型或内容的无效项。"""
        if not isinstance(item, dict):
            return _MemoryCandidateGovernanceResult(reason="invalid")

        namespace = str(item.get("namespace") or "").strip()
        memory_type = str(item.get("memory_type") or "").strip().lower()
        if not namespace or memory_type not in {"profile", "fact", "instruction"}:
            return _MemoryCandidateGovernanceResult(reason="invalid")

        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        try:
            safe_content = assert_memory_payload_safe(content)
            safe_summary = assert_memory_content_safe(str(item.get("summary") or ""))
            normalized_tags = self._normalize_memory_tags(item.get("tags"))
            normalized_source = self._normalize_memory_source(item.get("source"))
            raw_dedupe_key = str(item.get("dedupe_key") or "").strip()
            safe_dedupe_key = assert_memory_content_safe(raw_dedupe_key) if raw_dedupe_key else ""
        except ValueError:
            return _MemoryCandidateGovernanceResult(reason="sensitive")

        content = safe_content if isinstance(safe_content, dict) else {}
        summary = self._truncate_memory_summary(safe_summary)
        if not summary and isinstance(content.get("text"), str):
            summary = self._truncate_memory_summary(content.get("text"))
        if not summary and len(content) == 0:
            return _MemoryCandidateGovernanceResult(reason="invalid")

        try:
            confidence = float(item.get("confidence")) if item.get("confidence") is not None else 0.6
        except Exception:
            confidence = 0.6
        confidence = max(0.0, min(confidence, 1.0))

        dedupe_key = safe_dedupe_key
        if not dedupe_key:
            dedupe_key = self._build_memory_dedupe_key(
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
            "source": normalized_source,
            "confidence": confidence,
            "dedupe_key": dedupe_key,
        }
        if item.get("id"):
            normalized_candidate["id"] = str(item.get("id"))
        return _MemoryCandidateGovernanceResult(candidate=normalized_candidate)

    def _merge_profile_candidates(
            self,
            base_item: Dict[str, Any],
            incoming_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """同一 profile namespace 下合并偏好内容，避免多条碎片化 profile 记忆。"""
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
        merged_summary = self._truncate_memory_summary(
            incoming_item.get("summary") or base_item.get("summary") or "用户偏好"
        )
        merged_confidence = max(
            float(base_item.get("confidence") or 0.0),
            float(incoming_item.get("confidence") or 0.0),
        )
        return {
            **base_item,
            "summary": merged_summary or "用户偏好",
            "content": merged_content,
            "tags": merged_tags,
            "source": merged_source,
            "confidence": merged_confidence,
            "dedupe_key": self._build_memory_dedupe_key(
                namespace=str(base_item.get("namespace") or ""),
                memory_type="profile",
                content=merged_content,
            ),
        }

    def _govern_memory_candidates(
            self,
            candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """治理待写入候选：规整、过滤低置信度、去重并合并 profile。"""
        stats = {
            "input_count": len(list(candidates or [])),
            "kept_count": 0,
            "dropped_invalid_count": 0,
            "dropped_sensitive_count": 0,
            "dropped_low_confidence_count": 0,
            "deduped_count": 0,
            "merged_profile_count": 0,
        }
        governed: List[Dict[str, Any]] = []
        dedupe_keys: set[str] = set()
        profile_index_by_namespace: Dict[str, int] = {}

        for raw_item in list(candidates or []):
            governance_result = self._normalize_memory_candidate(raw_item)
            if governance_result.candidate is None:
                if governance_result.reason == "sensitive":
                    stats["dropped_sensitive_count"] += 1
                else:
                    stats["dropped_invalid_count"] += 1
                continue
            normalized_item = governance_result.candidate
            if float(normalized_item.get("confidence") or 0.0) < MEMORY_CANDIDATE_MIN_CONFIDENCE:
                stats["dropped_low_confidence_count"] += 1
                continue

            if normalized_item["memory_type"] == "profile":
                namespace = normalized_item["namespace"]
                existing_index = profile_index_by_namespace.get(namespace)
                if existing_index is not None:
                    governed[existing_index] = self._merge_profile_candidates(
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
