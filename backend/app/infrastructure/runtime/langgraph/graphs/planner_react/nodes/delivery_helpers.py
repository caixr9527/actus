#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层附件选择与步骤复用 helper。

本模块只处理最终附件引用选择、步骤结果复用和附件交付偏好，
不决定节点路由，也不触发工具执行。
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.domain.models import (
    ExecutionStatus,
    SelectedArtifactRevisionResult,
    Step,
    StepOutcome,
)
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionSourceKind,
    ArtifactStorageBackend,
)
from app.domain.services.runtime.contracts.langgraph_settings import (
    ATTACHMENT_DELIVERY_ALLOW_PATTERN,
    ATTACHMENT_DELIVERY_DENY_PATTERN,
    MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import (
    append_unique_text,
    is_attachment_filepath,
    normalize_controlled_value,
    normalize_file_path_list,
    normalize_optional_bool,
    normalize_step_result_text,
    normalize_text_list,
)
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import normalize_attachments
from .working_memory import _ensure_working_memory
from ..parsers import merge_attachment_paths

logger = logging.getLogger(__name__)

_RUNTIME_TEMP_ATTACHMENT_NAME_PATTERN = re.compile(
    r"(^|/)(temp_response\.json|response\.json|final_output\.txt|directory_info\.txt|.*\.tmp|.*\.log)$",
    re.IGNORECASE,
)


def _infer_step_attachment_delivery_preference(
        *,
        user_message: str,
        normalized_execution: Dict[str, Any],
) -> Optional[bool]:
    """推断本步骤结果是否更适合附件交付。

    判定顺序：
    1. 优先尊重执行结果里的显式字段；
    2. 其次根据用户本轮消息做轻量推断；
    3. 都没有时返回 `None`，交给后续 summary/final 阶段继续决策。
    """
    # P3-CASE3 修复：常量在 settings，判定逻辑集中在 nodes。
    explicit_preference = normalize_optional_bool((normalized_execution or {}).get("deliver_result_as_attachment"))
    if explicit_preference is not None:
        return explicit_preference
    normalized_message = user_message.strip().lower()
    if not normalized_message:
        return None
    if ATTACHMENT_DELIVERY_DENY_PATTERN.search(normalized_message):
        return False
    if ATTACHMENT_DELIVERY_ALLOW_PATTERN.search(normalized_message):
        return True
    return None


def _is_completed_status(value: Any) -> bool:
    return normalize_controlled_value(value, ExecutionStatus) == ExecutionStatus.COMPLETED.value


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
    if len(list(getattr(outcome, "evidence_backed_facts", []) or [])) > 0:
        return True
    return len(
        normalize_file_path_list(
            outcome.produced_artifacts,
            max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
        )
    ) > 0


def _is_successful_step_outcome(status: Any, outcome_raw: Any) -> bool:
    if not _is_completed_status(status):
        return False
    outcome = _hydrate_step_outcome(outcome_raw)
    return outcome is not None and bool(outcome.done)


def _normalize_successful_outcome_artifacts(status: Any, outcome_raw: Any) -> List[str]:
    if not _is_successful_step_outcome(status, outcome_raw):
        return []
    outcome = _hydrate_step_outcome(outcome_raw)
    if outcome is None:
        return []
    return normalize_file_path_list(
        outcome.produced_artifacts,
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )


def _collect_current_run_artifacts(state: PlannerReActLangGraphState) -> List[str]:
    artifact_groups: List[List[str]] = []
    for step_state in list(state.get("step_states") or []):
        if not isinstance(step_state, dict):
            continue
        artifact_groups.append(
            _normalize_successful_outcome_artifacts(
                step_state.get("status"),
                step_state.get("outcome"),
            )
        )
    last_step = state.get("last_executed_step")
    artifact_groups.append(
        _normalize_successful_outcome_artifacts(
            getattr(last_step, "status", None),
            getattr(last_step, "outcome", None),
        )
    )
    normalized_groups = [
        normalize_file_path_list(group, max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS)
        for group in artifact_groups
    ]
    return normalize_file_path_list(
        merge_attachment_paths(*normalized_groups),
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )


def _collect_available_file_context_refs(state: PlannerReActLangGraphState) -> List[str]:
    """统一收口当前运行里可直接消费的文件路径，供执行器判断是否允许文件工具。"""
    return normalize_file_path_list(
        merge_attachment_paths(
            [
                ref
                for ref in list(state.get("selected_artifacts") or [])
                if is_attachment_filepath(ref)
            ],
            _collect_current_run_artifacts(state),
        ),
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )


def _has_explicit_attachment_payload(parsed_attachments: Any) -> bool:
    if isinstance(parsed_attachments, str):
        return bool(parsed_attachments.strip())
    if isinstance(parsed_attachments, list):
        return len(parsed_attachments) > 0
    return False


def _extract_summary_available_artifacts(summary_context_packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    summary_context = summary_context_packet.get("summary_evidence_context")
    if not isinstance(summary_context, dict):
        return []
    available_artifacts = summary_context.get("available_artifacts")
    if not isinstance(available_artifacts, list):
        return []
    return [dict(item) for item in available_artifacts if isinstance(item, dict)]


def _is_deliverable_available_artifact(item: Dict[str, Any]) -> bool:
    storage_ref = item.get("storage_ref")
    if not isinstance(storage_ref, dict):
        return False
    storage_backend = str(storage_ref.get("storage_backend") or "").strip()
    source_kind = str(item.get("source_kind") or "").strip()
    return (
        item.get("version_locked") is True
        and item.get("delivery_candidate") is True
        and str(item.get("delivery_state") or "").strip() == ArtifactDeliveryState.CANDIDATE.value
        and storage_backend == ArtifactStorageBackend.FILE_STORAGE.value
        and source_kind != ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT.value
    )


def _selected_revision_from_available_artifact(
        item: Dict[str, Any],
        *,
        selected_reason: str,
) -> Dict[str, Any] | None:
    try:
        selected = SelectedArtifactRevisionResult(
            artifact_id=str(item.get("artifact_id") or ""),
            revision_id=str(item.get("revision_id") or ""),
            content_hash=str(item.get("content_hash") or ""),
            path=str(item.get("path") or ""),
            artifact_type=item.get("artifact_type") or "file",
            delivery_state=item.get("delivery_state") or "candidate",
            session_id=str(item.get("session_id") or ""),
            run_id=str(item.get("run_id") or "") or None,
            source_run_id=str(item.get("source_run_id") or item.get("run_id") or "") or None,
            source_step_id=str(item.get("source_step_id") or "") or None,
            source_event_id=str(item.get("source_event_id") or "") or None,
            source_kind=item.get("source_kind"),
            selected_reason=selected_reason,
            selected_at=datetime.now(timezone.utc),
        )
    except Exception:
        return None
    return selected.model_dump(mode="json")


def _dedupe_selected_revision_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (
            str(item.get("artifact_id") or ""),
            str(item.get("revision_id") or ""),
            str(item.get("content_hash") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS]


def _resolve_summary_selected_artifact_revisions(
        *,
        summary_context_packet: Dict[str, Any],
        parsed_attachments: Any,
        previous_selected_artifacts: Any = None,
) -> List[Dict[str, Any]]:
    """只从 EvidenceDigest.available_artifacts 中选择最终附件 revision。"""
    candidates = [
        item
        for item in _extract_summary_available_artifacts(summary_context_packet)
        if _is_deliverable_available_artifact(item)
    ]
    if not candidates:
        return []

    candidates_by_path: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        path = str(item.get("path") or "").strip()
        if path and path not in candidates_by_path:
            candidates_by_path[path] = item

    requested_paths = normalize_file_path_list(
        normalize_attachments(parsed_attachments),
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )
    selected: List[Dict[str, Any]] = []
    for path in requested_paths:
        candidate = candidates_by_path.get(path)
        if candidate is None:
            continue
        selected_item = _selected_revision_from_available_artifact(
            candidate,
            selected_reason="summary_explicit_attachment",
        )
        if selected_item is not None:
            selected.append(selected_item)

    if selected or _has_explicit_attachment_payload(parsed_attachments):
        return _dedupe_selected_revision_results(selected)

    fallback_paths = normalize_file_path_list(
        previous_selected_artifacts,
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )
    for path in fallback_paths:
        candidate = candidates_by_path.get(path)
        if candidate is None:
            continue
        selected_item = _selected_revision_from_available_artifact(
            candidate,
            selected_reason="summary_previous_selection",
        )
        if selected_item is not None:
            selected.append(selected_item)

    return _dedupe_selected_revision_results(selected)


def _filter_runtime_temp_attachment_refs(refs: List[str]) -> List[str]:
    """过滤运行时临时调试文件，避免被误交付到最终结果。"""
    return [
        ref for ref in normalize_file_path_list(refs, max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS)
        if not _RUNTIME_TEMP_ATTACHMENT_NAME_PATTERN.search(str(ref or "").strip())
    ]


def _resolve_attachment_delivery_preference_for_summary(
        *,
        state: PlannerReActLangGraphState,
        last_executed_step: Optional[Step],
) -> Optional[bool]:
    if last_executed_step is not None and last_executed_step.outcome is not None:
        step_preference = normalize_optional_bool(last_executed_step.outcome.deliver_result_as_attachment)
        if step_preference is not None:
            return step_preference

    working_memory = _ensure_working_memory(state)
    delivery_controls = dict(working_memory.get("delivery_controls") or {})
    source_step_id = str(delivery_controls.get("source_step_id") or "").strip()
    if last_executed_step is not None and source_step_id and source_step_id != str(last_executed_step.id or "").strip():
        return None
    return normalize_optional_bool(delivery_controls.get("deliver_result_as_attachment"))


def _merge_step_outcome_into_working_memory(
        working_memory: Dict[str, Any],
        *,
        step: Optional[Step],
) -> Dict[str, Any]:
    """将步骤结果沉淀到工作记忆，供后续 step / replan 使用。"""
    updated_working_memory = dict(working_memory or {})
    updated_working_memory.setdefault("decisions", [])
    updated_working_memory.setdefault("open_questions", [])
    updated_working_memory.setdefault("facts_in_session", [])
    if step is None or step.outcome is None:
        return updated_working_memory
    outcome = step.outcome

    summary = normalize_step_result_text(outcome.summary)
    if summary:
        updated_working_memory["decisions"] = append_unique_text(
            list(updated_working_memory.get("decisions") or []),
            summary,
        )

    for open_question in list(outcome.open_questions or []):
        updated_working_memory["open_questions"] = append_unique_text(
            list(updated_working_memory.get("open_questions") or []),
            open_question,
        )

    for projection in list(getattr(outcome, "evidence_backed_facts", []) or []):
        fact = str(getattr(projection, "text", "") or "").strip()
        if not fact:
            continue
        updated_working_memory["facts_in_session"] = append_unique_text(
            list(updated_working_memory.get("facts_in_session") or []),
            fact,
        )

    # P3-CASE3 修复：把“本步骤是否允许最终附件交付”写入工作记忆，供 summarize 阶段硬门禁使用。
    updated_working_memory["delivery_controls"] = {
        "source_step_id": str(step.id or ""),
        "deliver_result_as_attachment": outcome.deliver_result_as_attachment,
    }

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
        summary=normalize_step_result_text(source_outcome.summary),
        produced_artifacts=normalize_file_path_list(
            source_outcome.produced_artifacts,
            max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
        ),
        blockers=normalize_text_list(list(source_outcome.blockers or [])),
        evidence_backed_facts=list(getattr(source_outcome, "evidence_backed_facts", []) or []),
        facts_learned=[
            str(item.text or "").strip()
            for item in list(getattr(source_outcome, "evidence_backed_facts", []) or [])
            if str(item.text or "").strip()
        ],
        open_questions=normalize_text_list(list(source_outcome.open_questions or [])),
        deliver_result_as_attachment=source_outcome.deliver_result_as_attachment,
        next_hint=normalize_step_result_text(source_outcome.next_hint),
        reused_from_run_id=reused_from_run_id,
        reused_from_step_id=reused_from_step_id,
    )


def _find_reusable_step_outcome(
        state: PlannerReActLangGraphState,
) -> Optional[Tuple[StepOutcome, str, str]]:
    """仅在当前 run 内按 objective_key 查找可复用的步骤结果。"""
    plan = state.get("plan")
    if plan is None:
        return None

    step = plan.get_next_step()
    if step is None:
        return None

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
