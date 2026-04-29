#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层附件选择与步骤复用 helper。

本模块只处理最终附件引用选择、步骤结果复用和附件交付偏好，
不决定节点路由，也不触发工具执行。
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.domain.models import (
    ExecutionStatus,
    Step,
    StepOutcome,
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
from app.domain.services.workspace_runtime.context import RuntimeContextService
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
    if normalize_step_result_text(outcome.summary):
        return True
    if len(normalize_text_list(list(outcome.facts_learned or []))) > 0:
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


def _resolve_current_run_attachment_candidates(state: PlannerReActLangGraphState) -> List[str]:
    """汇总当前运行中可验证的最终附件候选。

    Phase B 起，summary 不再依赖历史重正文载荷，而是只从当前运行真实产出的
    文件路径中选择可交付附件。
    """
    return _filter_runtime_temp_attachment_refs(
        normalize_file_path_list(
            merge_attachment_paths(
                list(state.get("selected_artifacts") or []),
                _collect_current_run_artifacts(state),
            ),
            max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
        )
    )


def _filter_attachment_refs_by_authoritative_paths(
        refs: List[str],
        authoritative_paths: List[str],
) -> List[str]:
    normalized_refs = normalize_file_path_list(
        refs,
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )
    if len(authoritative_paths) == 0:
        return _filter_runtime_temp_attachment_refs(normalized_refs)
    allowed_paths = set(
        normalize_file_path_list(
            authoritative_paths,
            max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
        )
    )
    return _filter_runtime_temp_attachment_refs([ref for ref in normalized_refs if ref in allowed_paths])


async def _resolve_summary_attachment_refs(
        state: PlannerReActLangGraphState,
        parsed_attachments: Any,
        runtime_context_service: RuntimeContextService,
) -> List[str]:
    workspace_artifact_paths: List[str] = []
    if runtime_context_service is not None:
        workspace_artifact_paths = await runtime_context_service.list_workspace_artifact_paths()

    explicit_attachment_refs = _filter_attachment_refs_by_authoritative_paths(
        normalize_attachments(parsed_attachments),
        workspace_artifact_paths,
    )
    current_run_artifact_refs = _filter_attachment_refs_by_authoritative_paths(
        _resolve_current_run_attachment_candidates(state),
        workspace_artifact_paths,
    )
    # Phase B：summary 附件只允许来自当前运行可验证产物，避免外部伪路径或陈旧附件污染最终结果。
    known_attachment_refs = normalize_file_path_list(
        current_run_artifact_refs,
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )
    if len(explicit_attachment_refs) > 0:
        resolved_explicit_refs = [ref for ref in explicit_attachment_refs if ref in known_attachment_refs]
        if len(resolved_explicit_refs) > 0:
            return resolved_explicit_refs

    if len(current_run_artifact_refs) > 0:
        return _filter_runtime_temp_attachment_refs(current_run_artifact_refs)

    return []


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

    for fact in list(outcome.facts_learned or []):
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
        facts_learned=normalize_text_list(list(source_outcome.facts_learned or [])),
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
