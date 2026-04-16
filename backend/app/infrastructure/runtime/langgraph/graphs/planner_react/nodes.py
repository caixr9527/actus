#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 节点实现。"""
import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from langgraph.types import interrupt

from app.domain.external import LLM
from app.domain.models import (
    DoneEvent,
    ErrorEvent,
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
    StepDeliveryContextState,
    StepDeliveryRole,
    StepOutputMode,
    StepOutcome,
    StepTaskModeHint,
    StepEvent,
    StepEventStatus,
    TitleEvent,
    ToolEvent,
    normalize_wait_payload,
    resolve_wait_resume_message,
)
from app.domain.repositories import LongTermMemoryRepository
from app.domain.services.prompts import (
    CREATE_PLAN_PROMPT,
    DIRECT_ANSWER_PROMPT,
    EXECUTION_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SUMMARIZE_PROMPT,
    SYSTEM_PROMPT,
    UPDATE_PLAN_PROMPT,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.runtime import SkillGraphRuntime
from app.domain.services.workspace_runtime.policies import (
    collect_step_contract_hard_issues,
    compile_step_contracts,
)
from app.domain.services.runtime.langgraph_events import append_events
from app.domain.services.runtime.langgraph_state import (
    GraphStateContractMapper,
    PlannerReActLangGraphState,
    get_graph_control,
    normalize_retrieved_memories,
    replace_graph_control,
)
from app.domain.services.runtime.normalizers import (
    append_unique_text,
    build_delivery_text,
    is_attachment_filepath,
    normalize_message_window_entry,
    normalize_optional_bool,
    normalize_controlled_value,
    normalize_delivery_payload,
    normalize_execution_response,
    normalize_file_path_list,
    normalize_step_result_text,
    normalize_text_list,
    truncate_text,
)
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    format_attachments_for_prompt,
    normalize_attachments,
    safe_parse_json,
)
from .language_checker import build_direct_path_copy, infer_working_language_from_message
from .live_events import emit_live_events
from .parsers import (
    build_fallback_plan_title,
    build_step_from_payload,
    extract_write_file_paths_from_tool_events,
    merge_attachment_paths,
)
from app.domain.services.runtime.contracts.runtime_logging import describe_llm_runtime, elapsed_ms, log_runtime, now_perf
from app.domain.services.runtime.contracts.langgraph_settings import (
    ATTACHMENT_DELIVERY_ALLOW_PATTERN,
    ATTACHMENT_DELIVERY_DENY_PATTERN,
    CONVERSATION_SUMMARY_MAX_PARTS,
    MEMORY_CANDIDATE_MIN_CONFIDENCE,
    MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    MESSAGE_WINDOW_MAX_ITEMS,
    MESSAGE_WINDOW_MAX_MESSAGE_CHARS,
    REPLAN_META_VALIDATION_ALLOW_PATTERN,
    REPLAN_META_VALIDATION_DENY_PATTERN,
    REPLAN_META_VALIDATION_STEP_PATTERN,
    STEP_EXECUTION_TIMEOUT_SECONDS,
)
from .tools import (
    classify_confirmed_user_task_mode,
    classify_step_task_mode,
    execute_step_with_prompt,
    has_available_file_context,
    infer_entry_strategy,
    requests_plan_only,
)
from .replan import ReplanMergeEngine

logger = logging.getLogger(__name__)
_REPLAN_MERGE_ENGINE = ReplanMergeEngine(
    replan_meta_validation_step_pattern=REPLAN_META_VALIDATION_STEP_PATTERN,
    replan_meta_validation_allow_pattern=REPLAN_META_VALIDATION_ALLOW_PATTERN,
    replan_meta_validation_deny_pattern=REPLAN_META_VALIDATION_DENY_PATTERN,
)


def _get_control_metadata(state: PlannerReActLangGraphState) -> Dict[str, Any]:
    return get_graph_control(state.get("graph_metadata"))


def _replace_control_metadata(state: PlannerReActLangGraphState, control: Dict[str, Any]) -> Dict[str, Any]:
    return replace_graph_control(state.get("graph_metadata"), control)


def _is_direct_wait_execute_step(step: Step, control: Dict[str, Any]) -> bool:
    """识别 direct_wait 确认后的真实执行步骤。"""
    return (
            str(control.get("entry_strategy") or "").strip() == "direct_wait"
            and str(step.id or "").strip() == "direct-wait-execute"
    )


def _clear_direct_wait_control_state(control: Dict[str, Any]) -> None:
    """清理 direct_wait 专属控制字段，避免取消后误伤后续重规划链路。"""
    if str(control.get("entry_strategy") or "").strip() == "direct_wait":
        control.pop("entry_strategy", None)
    control.pop("skip_replan_when_plan_finished", None)
    control.pop("direct_wait_original_message", None)
    control.pop("direct_wait_execute_task_mode", None)
    control.pop("direct_wait_original_task_executed", None)


def _clear_plan_only_control_state(control: Dict[str, Any]) -> None:
    """清理仅规划模式的控制字段，避免跨 run 残留。"""
    control.pop("plan_only", None)


def _infer_step_attachment_delivery_preference(
        *,
        user_message: str,
        normalized_execution: Dict[str, Any],
) -> Optional[bool]:
    # P3-CASE3 修复：常量在 settings，判定逻辑集中在 nodes。
    explicit_preference = normalize_optional_bool((normalized_execution or {}).get("deliver_result_as_attachment"))
    if explicit_preference is not None:
        return explicit_preference
    normalized_message = _truncate_text(user_message, max_chars=600).strip().lower()
    if not normalized_message:
        return None
    if ATTACHMENT_DELIVERY_DENY_PATTERN.search(normalized_message):
        return False
    if ATTACHMENT_DELIVERY_ALLOW_PATTERN.search(normalized_message):
        return True
    return None

# 状态与交付 helper：负责工作记忆默认结构，以及“轻 summary / 重交付正文”分轨。
def _ensure_working_memory(state: PlannerReActLangGraphState) -> Dict[str, Any]:
    working_memory = dict(state.get("working_memory") or {})
    working_memory.setdefault("goal", "")
    working_memory.setdefault("constraints", [])
    working_memory.setdefault("decisions", [])
    working_memory.setdefault("open_questions", [])
    working_memory.setdefault("user_preferences", {})
    working_memory.setdefault("facts_in_session", [])
    # 轻 summary 与重交付正文分轨：最终长正文只放在 working_memory，不进入 final_message 热路径。
    working_memory.setdefault(
        "final_delivery_payload",
        {
            "text": "",
            "sections": [],
            "source_refs": [],
        },
    )
    return working_memory


def _truncate_text(value: Any, *, max_chars: int) -> str:
    return truncate_text(value, max_chars=max_chars)


async def _build_prompt_context_packet_async(
        *,
        stage: str,
        state: PlannerReActLangGraphState,
        runtime_context_service: RuntimeContextService,
        step: Optional[Step] = None,
        task_mode: str = "",
) -> Dict[str, Any]:
    """统一从上下文服务异步构造 Prompt 数据包。"""
    return await runtime_context_service.build_packet_async(
        stage=stage,  # type: ignore[arg-type]
        state=state,
        step=step,
        task_mode=task_mode,
    )


def _extract_prompt_context_state_updates(
        *,
        runtime_context_service: RuntimeContextService,
        context_packet: Dict[str, Any],
) -> Dict[str, Any]:
    """只回写 digest 与 task_mode，避免节点直接操心字段细节。"""
    return runtime_context_service.extract_state_updates(context_packet)


def _append_prompt_context_to_prompt(prompt: str, context_packet: Dict[str, Any]) -> str:
    """将结构化 context packet 追加到 Prompt，避免节点手写上下文拼装。"""
    context_json = json.dumps(context_packet, ensure_ascii=False, indent=2)
    return f"{prompt}\n\n已知上下文:\n```json\n{context_json}\n```"


def _collect_recent_search_queries_from_tool_events(tool_events: List[ToolEvent]) -> List[str]:
    queries: List[str] = []
    for event in tool_events:
        if str(event.function_name or "").strip().lower() != "search_web":
            continue
        query = str((event.function_args or {}).get("query") or (event.function_args or {}).get("q") or "").strip()
        if not query or query in queries:
            continue
        queries.append(query)
    return queries


def _build_step_delivery_payload(step: Optional[Step]) -> Dict[str, Any]:
    """仅为显式 final 交付步骤保留重正文，避免中间内联步骤误覆盖最终正文。"""
    if step is None or step.outcome is None or not step.outcome.done:
        return {}
    outcome = step.outcome

    output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
    if output_mode != StepOutputMode.INLINE.value:
        return {}
    delivery_role = normalize_controlled_value(getattr(step, "delivery_role", None), StepDeliveryRole)
    if delivery_role != StepDeliveryRole.FINAL.value:
        return {}

    delivery_text = normalize_step_result_text(outcome.delivery_text)
    if not delivery_text:
        return {}

    return normalize_delivery_payload(
        {
            "text": delivery_text,
            "sections": [],
            "source_refs": normalize_file_path_list(
                outcome.produced_artifacts,
                max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
            ),
        }
    )


def _sanitize_step_delivery_text(step: Step, raw_delivery_text: Any) -> str:
    """只有显式 inline 交付步骤才允许保留 delivery_text。"""
    delivery_text = normalize_step_result_text(raw_delivery_text)
    if not delivery_text:
        return ""

    output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
    delivery_role = normalize_controlled_value(getattr(step, "delivery_role", None), StepDeliveryRole)
    if output_mode != StepOutputMode.INLINE.value:
        return ""
    if delivery_role not in {
        StepDeliveryRole.INTERMEDIATE.value,
        StepDeliveryRole.FINAL.value,
    }:
        return ""
    return delivery_text


def _is_intermediate_delivery_step(step: Optional[Step]) -> bool:
    """识别“预览/草稿”类中间交付步骤。"""
    if step is None:
        return False
    return normalize_controlled_value(getattr(step, "delivery_role", None), StepDeliveryRole) == (
        StepDeliveryRole.INTERMEDIATE.value
    )


def _build_intermediate_round_summary_prompt(context_packet: Dict[str, Any]) -> str:
    """预览/草稿轮仍需进入 summary，但只允许输出轻量收尾。"""
    return SUMMARIZE_PROMPT.format(
        context_packet=json.dumps(context_packet, ensure_ascii=False)
    ) + """

补充限制：
- 当前这轮的最后一步属于“预览/草稿”步骤，不是最终定稿。
- `message` 只能写 1 到 2 句轻量收尾，说明“已生成草稿/预览”，并提示用户下一轮可以继续提出修改意见。
- 不要重复草稿正文，不要把步骤结果重新展开成大段内容。
- 如果存在历史最终正文载荷，也不要复用为本轮给用户的最终消息。
"""


def _build_intermediate_round_summary_fallback(step: Optional[Step]) -> str:
    """给预览/草稿轮提供稳定的轻量收尾兜底文案。"""
    step_label = str(getattr(step, "title", None) or getattr(step, "description", None) or "").strip()
    if step_label:
        return f"已生成{step_label}，如需调整方向或补充要求，请直接告诉我。"
    return "已生成预览草稿，如需调整方向或补充要求，请直接告诉我。"


# 附件 helper：只负责当前 run 内附件引用的收集、合并与最终交付选择。
def _is_completed_status(value: Any) -> bool:
    return normalize_controlled_value(value, ExecutionStatus) == ExecutionStatus.COMPLETED.value


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


def _collect_last_successful_step_artifacts(state: PlannerReActLangGraphState) -> List[str]:
    step_states = list(state.get("step_states") or [])
    for step_state in reversed(step_states):
        if not isinstance(step_state, dict):
            continue
        normalized = _normalize_successful_outcome_artifacts(
            step_state.get("status"),
            step_state.get("outcome"),
        )
        if normalized:
            return normalized

    last_step = state.get("last_executed_step")
    if isinstance(last_step, Step):
        normalized = _normalize_successful_outcome_artifacts(last_step.status, last_step.outcome)
        if normalized:
            return normalized
    return []


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


def _resolve_final_delivery_source_refs(state: PlannerReActLangGraphState) -> List[str]:
    """最终重交付正文的附件来源以 final_delivery_payload 为真相源。"""
    working_memory = _ensure_working_memory(state)
    delivery_payload = normalize_delivery_payload(working_memory.get("final_delivery_payload"))
    return normalize_file_path_list(
        delivery_payload.get("source_refs"),
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
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
        return normalized_refs
    allowed_paths = set(
        normalize_file_path_list(
            authoritative_paths,
            max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
        )
    )
    return [ref for ref in normalized_refs if ref in allowed_paths]


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
    final_delivery_source_refs = _filter_attachment_refs_by_authoritative_paths(
        _resolve_final_delivery_source_refs(state),
        workspace_artifact_paths,
    )
    # P3-一次性收口：summary 附件只允许来自最终交付真相源，不再 fallback 到中间步骤产物。
    known_attachment_refs = normalize_file_path_list(
        final_delivery_source_refs,
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )
    if len(explicit_attachment_refs) > 0:
        resolved_explicit_refs = [ref for ref in explicit_attachment_refs if ref in known_attachment_refs]
        if len(resolved_explicit_refs) > 0:
            return resolved_explicit_refs

    if len(final_delivery_source_refs) > 0:
        return final_delivery_source_refs

    return []


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


def _build_direct_plan_title(user_message: str, fallback: str) -> str:
    normalized_message = str(user_message or "").strip()
    if not normalized_message:
        return fallback
    return normalized_message[:40]


def _resolve_direct_delivery_context_state(task_mode: str) -> str:
    """直达路径下，只有纯 general 任务才可直接组织最终正文，其余模式需先准备上下文。"""
    normalized_task_mode = normalize_controlled_value(task_mode, StepTaskModeHint)
    if normalized_task_mode == StepTaskModeHint.GENERAL.value:
        return StepDeliveryContextState.READY.value
    return StepDeliveryContextState.NEEDS_PREPARATION.value


def _should_skip_summary_llm_for_final_delivery(
        *,
        summarize_intermediate_round: bool,
        last_executed_step: Optional[Step],
        deterministic_delivery_text: str,
) -> bool:
    if summarize_intermediate_round:
        return False
    if not deterministic_delivery_text:
        return False
    if last_executed_step is None:
        return False
    step_status = normalize_controlled_value(getattr(last_executed_step, "status", None), ExecutionStatus)
    output_mode = normalize_controlled_value(getattr(last_executed_step, "output_mode", None), StepOutputMode)
    delivery_role = normalize_controlled_value(getattr(last_executed_step, "delivery_role", None), StepDeliveryRole)
    return (
        step_status == ExecutionStatus.COMPLETED.value
        and output_mode == StepOutputMode.INLINE.value
        and delivery_role == StepDeliveryRole.FINAL.value
    )


# Summary helper：负责总结阶段的快照裁剪、结果复用和最终轻量输入构造。
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
    if normalize_step_result_text(outcome.delivery_text):
        return True
    return len(
        normalize_file_path_list(
            outcome.produced_artifacts,
            max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
        )
    ) > 0


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
    updated_working_memory.setdefault(
        "final_delivery_payload",
        {
            "text": "",
            "sections": [],
            "source_refs": [],
        },
    )
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

    delivery_payload = _build_step_delivery_payload(step)
    if delivery_payload:
        updated_working_memory["final_delivery_payload"] = delivery_payload
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
        delivery_text=normalize_step_result_text(source_outcome.delivery_text),
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


# 记忆 helper：负责总结后的事实/偏好提炼、候选规整与去重治理。
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
        status_value = str(getattr(getattr(event, "status", ""), "value", getattr(event, "status", "")) or "").strip().lower()
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


def _build_step_label(step: Step, default: str = "当前步骤") -> str:
    return str(step.title or step.description or default).strip() or default


def _build_wait_resume_step_summary(step: Step, resumed_message: str) -> str:
    step_label = _build_step_label(step)
    normalized_message = str(resumed_message or "").strip()
    if normalized_message:
        return f"{step_label}已收到用户回复：{normalized_message}"
    return f"{step_label}已完成用户交互"


def _build_wait_cancel_step_summary(step: Step, resumed_message: str) -> str:
    step_label = _build_step_label(step)
    normalized_message = str(resumed_message or "").strip()
    if normalized_message:
        return f"{step_label}已被用户取消：{normalized_message}"
    return f"{step_label}已被用户取消，等待重新规划"


def _resolve_wait_resume_branch(
        payload: Dict[str, Any],
        resume_value: Any,
) -> Literal["confirm_continue", "confirm_cancel", "select", "input_text"]:
    """显式区分不同等待态恢复分支，避免不同 UI 交互共用一条模糊路径。"""
    kind = str(payload.get("kind") or "").strip()
    if kind == "confirm":
        if resume_value == payload.get("cancel_resume_value"):
            return "confirm_cancel"
        return "confirm_continue"
    if kind == "select":
        return "select"
    return "input_text"


def _build_wait_resume_outcome(
        step: Step,
        *,
        branch: Literal["confirm_continue", "select", "input_text"],
        resumed_message: str,
) -> StepOutcome:
    """按恢复分支生成步骤结果，保证确认/选择/文本输入语义清晰。"""
    step_label = _build_step_label(step)
    normalized_message = str(resumed_message or "").strip()

    if branch == "confirm_continue":
        if normalized_message:
            summary = f"{step_label}已确认继续：{normalized_message}"
        else:
            summary = f"{step_label}已确认继续执行"
    elif branch == "select":
        if normalized_message:
            summary = f"{step_label}已收到用户选择：{normalized_message}"
        else:
            summary = f"{step_label}已完成用户选择"
    else:
        if normalized_message:
            summary = f"{step_label}已收到用户输入：{normalized_message}"
        else:
            summary = _build_wait_resume_step_summary(step, resumed_message)

    return StepOutcome(
        done=True,
        summary=summary,
    )


# 入口路由节点：决定 direct_answer / direct_wait / direct_execute / planner 主链入口。
async def entry_router_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """P0 入口轻路由：先判断是否需要完整记忆召回和 Planner。"""
    control = _get_control_metadata(state)
    user_message = str(state.get("user_message") or "")
    plan = state.get("plan")
    control["entry_strategy"] = infer_entry_strategy(
        user_message=user_message,
        has_input_parts=bool(list(state.get("input_parts") or [])),
        has_active_plan=bool(plan is not None and len(list(plan.steps or [])) > 0 and not plan.done),
    )
    # “只给步骤，不执行”是当前 run 的显式用户意图，入口即写入控制态，供 planner 路由收口。
    if requests_plan_only(user_message):
        control["plan_only"] = True
    else:
        _clear_plan_only_control_state(control)

    log_runtime(
        logger,
        logging.INFO,
        "入口路由完成",
        state=state,
        entry_strategy=str(control.get("entry_strategy") or ""),
        plan_only=bool(control.get("plan_only")),
    )
    return {
        **state,
        "graph_metadata": _replace_control_metadata(state, control),
    }


async def direct_answer_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
) -> PlannerReActLangGraphState:
    """直接回答类任务跳过 Planner 和工具循环。"""
    started_at = now_perf()
    user_message = str(state.get("user_message") or "").strip()
    language = infer_working_language_from_message(user_message)
    direct_copy = build_direct_path_copy(language)
    prompt = DIRECT_ANSWER_PROMPT.format(message=user_message)
    llm_runtime = describe_llm_runtime(llm)
    llm_started_at = now_perf()
    llm_message = await llm.invoke(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))
    final_message = str(parsed.get("message") or user_message or "").strip()
    plan = Plan(
        title=_build_direct_plan_title(user_message, direct_copy["direct_answer_fallback"]),
        goal=user_message,
        language=language,
        message=final_message,
        steps=[],
        status=ExecutionStatus.COMPLETED,
    )
    events: List[Any] = [
        TitleEvent(title=plan.title),
        MessageEvent(role="assistant", message=final_message, stage="final"),
    ]
    await emit_live_events(*events)
    control = _get_control_metadata(state)
    control["entry_strategy"] = "direct_answer"
    control["skip_replan_when_plan_finished"] = True
    log_runtime(
        logger,
        logging.INFO,
        "直接回复完成",
        state=state,
        stage_name="router",
        model_name=llm_runtime["model_name"],
        max_tokens=llm_runtime["max_tokens"],
        llm_elapsed_ms=elapsed_ms(llm_started_at),
        elapsed_ms=elapsed_ms(started_at),
    )
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "current_step_id": None,
            "final_message": final_message,
            "graph_metadata": _replace_control_metadata(state, control),
        },
        events=events,
    )


async def direct_wait_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """直接构造前置确认等待步骤，避免为等待任务先走重 Planner。"""
    user_message = str(state.get("user_message") or "").strip()
    language = infer_working_language_from_message(user_message)
    direct_copy = build_direct_path_copy(language)
    execute_task_mode = classify_confirmed_user_task_mode(user_message)
    step_wait = Step(
        id="direct-wait-confirm",
        title=direct_copy["direct_wait_title"],
        description=direct_copy["direct_wait_description"],
        task_mode_hint="human_wait",
        output_mode="none",
        artifact_policy="forbid_file_output",
        delivery_role="none",
        status=ExecutionStatus.RUNNING,
    )
    step_execute = Step(
        id="direct-wait-execute",
        title=direct_copy["direct_wait_execute_title"],
        # 第二步只表达“开始执行原任务”，不再把“先确认”语义写回步骤文本。
        description=direct_copy["direct_wait_execute_title"],
        task_mode_hint=execute_task_mode,
        # direct_wait 的真实执行步骤本身就承担最终正文，不应再丢回轻摘要链路。
        output_mode="inline",
        artifact_policy="default",
        delivery_role="final",
        # direct_wait 的真实执行步骤可能仍需先搜索/读取页面；是否可直接交付由 readiness 决定。
        delivery_context_state=_resolve_direct_delivery_context_state(execute_task_mode),
        status=ExecutionStatus.PENDING,
    )
    plan = Plan(
        title=_build_direct_plan_title(user_message, direct_copy["direct_wait_fallback"]),
        goal=user_message,
        language=language,
        message=direct_copy["direct_wait_message"],
        steps=[step_wait, step_execute],
        status=ExecutionStatus.PENDING,
    )
    control = _get_control_metadata(state)
    control["entry_strategy"] = "direct_wait"
    control["skip_replan_when_plan_finished"] = True
    # 保留原始请求，供确认后直接执行与最终总结使用。
    control["direct_wait_original_message"] = user_message
    # 在入口处就确定真实执行模式，避免确认后再次被等待语义误判。
    control["direct_wait_execute_task_mode"] = execute_task_mode
    # 只有真实执行步骤收尾后，才允许 direct_wait 路径进入总结。
    control["direct_wait_original_task_executed"] = False
    events: List[Any] = [
        TitleEvent(title=plan.title),
        PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.CREATED),
    ]
    await emit_live_events(*events)
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "current_step_id": step_wait.id,
            "pending_interrupt": {
                "kind": "confirm",
                "prompt": direct_copy["direct_wait_prompt"],
                "confirm_resume_value": True,
                "cancel_resume_value": False,
                "confirm_label": direct_copy["direct_wait_confirm_label"],
                "cancel_label": direct_copy["direct_wait_cancel_label"],
            },
            "graph_metadata": _replace_control_metadata(state, control),
        },
        events=events,
    )


async def direct_execute_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """直接进入单步执行，跳过 Planner。"""
    user_message = str(state.get("user_message") or "").strip()
    language = infer_working_language_from_message(user_message)
    direct_copy = build_direct_path_copy(language)
    execute_task_mode = classify_confirmed_user_task_mode(user_message)
    step = Step(
        id="direct-execute-step",
        title=direct_copy["direct_execute_step_title"],
        description=user_message,
        task_mode_hint=execute_task_mode,
        # direct_execute 是单步执行路径，本步骤直接承担最终正文交付。
        output_mode="inline",
        artifact_policy="default",
        delivery_role="final",
        # 单步直达路径也可能要先检索/读取再输出最终正文，不能一律当成已准备好上下文。
        delivery_context_state=_resolve_direct_delivery_context_state(execute_task_mode),
        status=ExecutionStatus.PENDING,
    )
    plan = Plan(
        title=_build_direct_plan_title(user_message, direct_copy["direct_execute_fallback"]),
        goal=user_message,
        language=language,
        message=direct_copy["direct_execute_message"],
        steps=[step],
        status=ExecutionStatus.PENDING,
    )
    control = _get_control_metadata(state)
    control["entry_strategy"] = "direct_execute"
    control["skip_replan_when_plan_finished"] = True
    events: List[Any] = [
        TitleEvent(title=plan.title),
        PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.CREATED),
    ]
    await emit_live_events(*events)
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "current_step_id": step.id,
            "graph_metadata": _replace_control_metadata(state, control),
        },
        events=events,
    )


# 规划前置节点：先召回记忆，再创建或复用当前计划。
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


async def create_or_reuse_plan_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
        runtime_context_service: RuntimeContextService,
) -> PlannerReActLangGraphState:
    """创建计划或复用已有计划。"""
    started_at = now_perf()
    control = _get_control_metadata(state)
    plan_only = bool(control.get("plan_only"))
    plan = state.get("plan")
    if plan is not None and len(plan.steps) > 0 and not plan.done:
        next_step = plan.get_next_step()
        working_memory = _ensure_working_memory(state)
        if not str(working_memory.get("goal") or "").strip():
            working_memory["goal"] = str(plan.goal or state.get("user_message") or "")
        resumed_from_cancelled_plan = bool(control.pop("continued_from_cancelled_plan", False))

        reuse_events: List[Any] = []
        if resumed_from_cancelled_plan:
            reuse_events.append(
                PlanEvent(
                    plan=plan.model_copy(deep=True),
                    status=PlanEventStatus.UPDATED,
                )
            )
        log_runtime(
            logger,
            logging.INFO,
            "继续复用已有计划" if resumed_from_cancelled_plan else "复用已有计划",
            state=state,
            plan_title=str(plan.title or ""),
            step_count=len(list(plan.steps or [])),
            next_step_id=str(next_step.id or "") if next_step is not None else "",
            elapsed_ms=elapsed_ms(started_at),
        )
        if len(reuse_events) > 0:
            await emit_live_events(*reuse_events)
        return _reduce_state_with_events(
            state,
            updates={
                "plan": plan,
                "working_memory": working_memory,
                "current_step_id": next_step.id if next_step is not None else None,
                "graph_metadata": _replace_control_metadata(state, control),
            },
            events=reuse_events,
        )

    user_message = state.get("user_message", "").strip()

    input_parts = list(state.get("input_parts") or [])
    attachments = [part.get("sandbox_filepath") for part in input_parts]
    planner_context_packet = await _build_prompt_context_packet_async(
        stage="planner",
        state=state,
        runtime_context_service=runtime_context_service,
    )
    planner_context_updates = _extract_prompt_context_state_updates(
        runtime_context_service=runtime_context_service,
        context_packet=planner_context_packet,
    )
    user_message_prompt = CREATE_PLAN_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
    )
    user_message_prompt = _append_prompt_context_to_prompt(user_message_prompt, planner_context_packet)

    user_content = await _build_message(llm, user_message_prompt, input_parts)
    llm_runtime = describe_llm_runtime(llm)

    log_runtime(
        logger,
        logging.INFO,
        "开始创建计划",
        state=state,
        stage_name="planner",
        model_name=llm_runtime["model_name"],
        max_tokens=llm_runtime["max_tokens"],
        attachment_count=len(attachments),
        context_memory_count=len(list(state.get("retrieved_memories") or [])),
        context_recent_run_count=len(list(state.get("recent_run_briefs") or [])),
        context_recent_attempt_count=len(list(state.get("recent_attempt_briefs") or [])),
    )
    llm_started_at = now_perf()
    llm_message = await llm.invoke(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        tools=[],
        response_format={"type": "json_object"},
    )
    llm_cost_ms = elapsed_ms(llm_started_at)
    parsed = safe_parse_json(llm_message.get("content"))

    title = str(parsed.get("title") or build_fallback_plan_title(user_message))
    language = str(parsed.get("language") or "zh")
    goal = str(parsed.get("goal") or user_message)
    planner_message = str(parsed.get("message") or user_message or "已生成任务计划")
    working_memory = _ensure_working_memory(state)
    working_memory["goal"] = goal
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or raw_steps is None or len(raw_steps) == 0:
        log_runtime(
            logger,
            logging.INFO,
            "计划创建完成，无需步骤",
            state=state,
            plan_title=title,
            language=language,
            message_length=len(planner_message),
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
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
                **planner_context_updates,
                "current_step_id": None,
                "final_message": planner_message,
                "graph_metadata": dict(state.get("graph_metadata") or {}),
                "step_states": [],
            },
            events=planner_events,
        )
    else:
        steps = [
            build_step_from_payload(
                item,
                index,
                user_message=user_message,
            )
            for index, item in enumerate(raw_steps)
        ]
        # P3-一次性收口：计划步骤入图前统一编译结构化契约，纠偏语义冲突步骤。
        compiled_steps, contract_issues, corrected_count = compile_step_contracts(
            steps=steps,
            user_message=user_message,
        )
        contract_issues.extend(collect_step_contract_hard_issues(steps=compiled_steps))
        if corrected_count > 0:
            log_runtime(
                logger,
                logging.INFO,
                "计划步骤契约已自动纠偏",
                state=state,
                corrected_step_count=corrected_count,
            )
        if contract_issues:
            log_runtime(
                logger,
                logging.WARNING,
                "计划步骤契约校验失败",
                state=state,
                issue_count=len(contract_issues),
                issue_codes=[item.issue_code for item in contract_issues],
            )
            compiled_steps = []
        log_runtime(
            logger,
            logging.INFO,
            "计划创建完成",
            state=state,
            plan_title=title,
            language=language,
            step_count=len(compiled_steps),
            next_step_id=str(compiled_steps[0].id or "") if len(compiled_steps) > 0 else "",
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
        plan = Plan(
            title=title,
            goal=goal,
            language=language,
            message=planner_message,
            steps=compiled_steps,
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
                **planner_context_updates,
                # 仅规划模式在 planner 后直接收尾，不进入 execute。
                "current_step_id": None if plan_only else next_step.id if next_step is not None else None,
                "final_message": planner_message if plan_only else "",
                "graph_metadata": _replace_control_metadata(state, control),
            },
            events=planner_events,
        )


# 步骤复用节点：在真实执行前先做当前 run 内的等价目标复用。
async def guard_step_reuse_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """在真实执行前做当前 run 内复用，命中时直接跳过执行节点。"""
    plan = state.get("plan")
    if plan is None:
        return state

    step = plan.get_next_step()
    if step is None:
        return state

    log_runtime(
        logger,
        logging.INFO,
        "开始检查步骤复用",
        state=state,
        step_id=str(step.id or ""),
        objective_key=str(step.objective_key or ""),
    )

    reusable_step = _find_reusable_step_outcome(state=state)
    control = _get_control_metadata(state)
    control["step_reuse_hit"] = False

    if reusable_step is None:
        log_runtime(
            logger,
            logging.INFO,
            "步骤复用未命中",
            state=state,
            step_id=str(step.id or ""),
        )
        return {
            **state,
            "graph_metadata": _replace_control_metadata(state, control),
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
        step=step,
    )
    control["step_reuse_hit"] = True
    log_runtime(
        logger,
        logging.INFO,
        "步骤复用命中",
        state=state,
        step_id=str(step.id or ""),
        source_run_id=reused_from_run_id,
        source_step_id=reused_from_step_id,
        artifact_count=len(list(step.outcome.produced_artifacts or [])),
    )

    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "last_executed_step": step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
            "working_memory": working_memory,
            "graph_metadata": _replace_control_metadata(state, control),
            "final_message": normalize_step_result_text(step.outcome.summary),
            "selected_artifacts": list(state.get("selected_artifacts") or []),
            "pending_interrupt": {},
        },
        events=[completed_event],
    )


# 执行节点：负责单步工具执行与步骤结果落账。
async def execute_step_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
        runtime_context_service: RuntimeContextService,
        skill_runtime: Optional[SkillGraphRuntime] = None,
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
) -> PlannerReActLangGraphState:
    """执行单个步骤；当前批次未完成时继续跑后续步骤，整批完成后再统一重规划。"""
    started_at = now_perf()
    plan = state.get("plan")
    if plan is None:
        return state

    step = plan.get_next_step()
    if step is None:
        return state

    control = _get_control_metadata(state)
    is_direct_wait_execute_step = _is_direct_wait_execute_step(step, control)
    explicit_direct_wait_task_mode = str(control.get("direct_wait_execute_task_mode") or "").strip()
    task_mode = explicit_direct_wait_task_mode if is_direct_wait_execute_step and explicit_direct_wait_task_mode else classify_step_task_mode(
        step)
    step.status = ExecutionStatus.RUNNING
    started_event = StepEvent(step=step.model_copy(deep=True), status=StepEventStatus.STARTED)
    await emit_live_events(started_event)
    log_runtime(
        logger,
        logging.INFO,
        "开始执行步骤",
        state=state,
        step_id=str(step.id or ""),
        step_title=str(step.title or step.description or ""),
        task_mode=task_mode,
        attachment_count=len(list(state.get("input_parts") or [])),
        runtime_tool_count=len(list(runtime_tools or [])),
        has_skill_runtime=skill_runtime is not None,
    )

    user_message = str(state.get("user_message", ""))
    if is_direct_wait_execute_step:
        # 确认后的执行阶段必须继续消费原始请求，而不是“继续/确认”这类恢复文本。
        user_message = str(control.get("direct_wait_original_message") or user_message).strip()
    language = plan.language or "zh"
    input_parts = list(state.get("input_parts") or [])
    attachments = [part.get("sandbox_filepath") for part in input_parts]
    available_file_context_refs = _collect_available_file_context_refs(state)
    available_file_context = has_available_file_context(
        user_message=user_message,
        attachment_paths=attachments,
        artifact_paths=available_file_context_refs,
    )

    user_message_prompt = EXECUTION_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
        language=language,
        step=step.description,
        delivery_role=str(getattr(step, "delivery_role", "") or "none"),
        delivery_context_state=str(getattr(step, "delivery_context_state", "") or "none"),
    )
    execute_context_packet = await _build_prompt_context_packet_async(
        stage="execute",
        state=state,
        runtime_context_service=runtime_context_service,
        step=step,
        task_mode=task_mode,
    )
    execute_context_updates = _extract_prompt_context_state_updates(
        runtime_context_service=runtime_context_service,
        context_packet=execute_context_packet,
    )
    user_message_prompt = _append_prompt_context_to_prompt(user_message_prompt, execute_context_packet)
    user_content = await _build_message(llm, user_message_prompt, input_parts)

    llm_message: Optional[Dict[str, Any]] = None
    tool_events: List[ToolEvent] = []
    tool_cost_ms = 0
    skill_cost_ms = 0

    try:
        async with asyncio.timeout(STEP_EXECUTION_TIMEOUT_SECONDS):
            # 只要执行器能力已初始化（即便当前列表为空），都统一走 execute_step_with_prompt，
            # 这样“无工具纯文本完成”分支才能真正生效。
            if runtime_tools is not None:
                tool_started_at = now_perf()
                llm_message, tool_events = await execute_step_with_prompt(
                    llm=llm,
                    step=step,
                    runtime_tools=runtime_tools or [],
                    max_tool_iterations=max_tool_iterations,
                    task_mode=task_mode,
                    on_tool_event=emit_live_events,
                    user_content=user_content,
                    user_message=user_message,
                    attachment_paths=attachments,
                    artifact_paths=available_file_context_refs,
                    has_available_file_context=available_file_context,
                )
                tool_cost_ms = elapsed_ms(tool_started_at)

            # 暂时忽略, 因为技能能力目前尚未实现。
            # if llm_message is None and skill_runtime is not None:
            #     try:
            #         log_runtime(
            #             logger,
            #             logging.INFO,
            #             "开始执行技能",
            #             state=state,
            #             step_id=str(step.id or ""),
            #             skill_id=PLANNER_EXECUTE_STEP_SKILL_ID,
            #         )
            #         skill_started_at = now_perf()
            #         skill_result = await skill_runtime.execute_skill(
            #             skill_id=PLANNER_EXECUTE_STEP_SKILL_ID,
            #             payload={
            #                 "session_id": str(state.get("session_id") or ""),
            #                 "user_message": user_message,
            #                 "step_description": step.description,
            #                 "language": language,
            #                 "attachments": attachments,
            #                 "execution_context": execute_context_packet,
            #             },
            #         )
            #         skill_cost_ms = elapsed_ms(skill_started_at)
            #         llm_message = {
            #             "success": bool(getattr(skill_result, "success", True)),
            #             "result": str(getattr(skill_result, "result", "") or f"步骤执行完成：{step.description}"),
            #             "attachments": normalize_attachments(getattr(skill_result, "attachments", [])),
            #         }
            #     except Exception as e:
            #         log_runtime(
            #             logger,
            #             logging.WARNING,
            #             "技能执行失败，回退默认链路",
            #             state=state,
            #             step_id=str(step.id or ""),
            #             skill_id=PLANNER_EXECUTE_STEP_SKILL_ID,
            #             error=str(e),
            #             elapsed_ms=elapsed_ms(started_at),
            #         )
    except TimeoutError:
        log_runtime(
            logger,
            logging.WARNING,
            "步骤执行超时",
            state=state,
            step_id=str(step.id or ""),
            step_description=str(step.description or ""),
            timeout_seconds=STEP_EXECUTION_TIMEOUT_SECONDS,
            elapsed_ms=elapsed_ms(started_at),
        )
        llm_message = {
            "success": False,
            "summary": f"步骤执行超时：{step.description}",
            "delivery_text": "",
            "attachments": [],
            "blockers": [f"当前步骤超过 {STEP_EXECUTION_TIMEOUT_SECONDS} 秒未完成"],
            "next_hint": "请缩小当前步骤范围后重试",
        }

    if llm_message is None:
        llm_message = {
            "success": False,
            "summary": f"步骤执行失败：{step.description}",
            "delivery_text": "",
            "attachments": [],
        }

    runtime_recent_action = runtime_context_service.normalize_runtime_recent_action(
        llm_message.get("runtime_recent_action")
    )
    recent_search_queries = _collect_recent_search_queries_from_tool_events(tool_events)
    if recent_search_queries:
        runtime_recent_action["recent_search_queries"] = recent_search_queries
    interrupt_request = _normalize_interrupt_request(llm_message.get("interrupt_request"))
    if interrupt_request:
        log_runtime(
            logger,
            logging.INFO,
            "步骤请求进入等待",
            state=state,
            step_id=str(step.id or ""),
            interrupt_kind=str(interrupt_request.get("kind") or ""),
            tool_elapsed_ms=tool_cost_ms,
            skill_elapsed_ms=skill_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
        control = _get_control_metadata(state)
        control["step_reuse_hit"] = False
        interrupt_context_packet = await _build_prompt_context_packet_async(
            stage="execute",
            state={
                **state,
                **runtime_context_service.merge_runtime_recent_action(
                    state_updates=execute_context_updates,
                    task_mode=task_mode,
                    runtime_recent_action=runtime_recent_action,
                ),
                "plan": plan,
                "current_step_id": step.id,
                "pending_interrupt": interrupt_request,
            },
            runtime_context_service=runtime_context_service,
            step=step,
            task_mode=task_mode,
        )
        interrupt_context_updates = runtime_context_service.merge_runtime_recent_action(
            state_updates=_extract_prompt_context_state_updates(
                runtime_context_service=runtime_context_service,
                context_packet=interrupt_context_packet,
            ),
            task_mode=task_mode,
            runtime_recent_action=runtime_recent_action,
        )
        return _reduce_state_with_events(
            state,
            updates={
                "plan": plan,
                **interrupt_context_updates,
                "current_step_id": step.id,
                "user_message": user_message,
                "graph_metadata": _replace_control_metadata(state, control),
                "pending_interrupt": interrupt_request,
            },
            events=[started_event, *tool_events],
        )

    normalized_execution = normalize_execution_response(llm_message)
    step_success = bool(normalized_execution.get("success", True))
    step_summary = normalize_step_result_text(
        normalized_execution.get("summary"),
        fallback=f"步骤执行完成：{step.description}" if step_success else f"步骤执行失败：{step.description}",
    )
    step_delivery_text = normalize_step_result_text(
        _sanitize_step_delivery_text(step, normalized_execution.get("delivery_text")),
        fallback=(
            step_summary
            if normalize_controlled_value(getattr(step, "delivery_role", None), StepDeliveryRole)
               == StepDeliveryRole.FINAL.value
               and normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
               == StepOutputMode.INLINE.value
            else ""
        ),
    )
    step_deliver_result_as_attachment = _infer_step_attachment_delivery_preference(
        user_message=user_message,
        normalized_execution=normalized_execution,
    )
    model_attachment_paths = normalize_attachments(normalized_execution.get("attachments"))
    tool_attachment_paths = extract_write_file_paths_from_tool_events(tool_events)
    step_attachment_paths = normalize_file_path_list(
        merge_attachment_paths(model_attachment_paths, tool_attachment_paths),
        max_items=MESSAGE_WINDOW_MAX_ATTACHMENT_PATHS,
    )
    step.outcome = StepOutcome(
        done=step_success,
        summary=step_summary,
        delivery_text=step_delivery_text,
        produced_artifacts=step_attachment_paths,
        blockers=normalize_text_list(normalized_execution.get("blockers")),
        facts_learned=normalize_text_list(normalized_execution.get("facts_learned")),
        open_questions=normalize_text_list(normalized_execution.get("open_questions")),
        deliver_result_as_attachment=step_deliver_result_as_attachment,
        next_hint=normalize_step_result_text(normalized_execution.get("next_hint")),
    )
    step.status = ExecutionStatus.COMPLETED if step_success else ExecutionStatus.FAILED

    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=step.status,
    )
    final_step_events: List[Any] = [completed_event]

    await emit_live_events(*final_step_events)

    events: List[Any] = [started_event, *tool_events, *final_step_events]
    next_step = plan.get_next_step()
    control["step_reuse_hit"] = False
    if is_direct_wait_execute_step:
        # 只要真实执行步骤已经完成收尾，就视为原始任务已被实际执行过。
        control["direct_wait_original_task_executed"] = True
        control.pop("direct_wait_execute_task_mode", None)
        control.pop("direct_wait_original_message", None)
    working_memory = _merge_step_outcome_into_working_memory(
        _ensure_working_memory(state),
        step=step,
    )
    log_runtime(
        logger,
        logging.INFO,
        "步骤执行完成",
        state=state,
        step_id=str(step.id or ""),
        status=step.status.value,
        success=step_success,
        artifact_count=len(step_attachment_paths),
        blocker_count=len(list(step.outcome.blockers or [])),
        open_question_count=len(list(step.outcome.open_questions or [])),
        next_step_id=str(next_step.id or "") if next_step is not None else "",
        tool_elapsed_ms=tool_cost_ms,
        skill_elapsed_ms=skill_cost_ms,
        elapsed_ms=elapsed_ms(started_at),
    )
    completed_context_packet = await _build_prompt_context_packet_async(
        stage="execute",
        state={
            **state,
            **runtime_context_service.merge_runtime_recent_action(
                state_updates=execute_context_updates,
                task_mode=task_mode,
                runtime_recent_action=runtime_recent_action,
            ),
            "plan": plan,
            "last_executed_step": step.model_copy(deep=True),
            "working_memory": working_memory,
            "final_message": step_summary,
            "pending_interrupt": {},
            "emitted_events": append_events(state.get("emitted_events"), *events),
        },
        runtime_context_service=runtime_context_service,
        step=step,
        task_mode=task_mode,
    )
    completed_context_updates = runtime_context_service.merge_runtime_recent_action(
        state_updates=_extract_prompt_context_state_updates(
            runtime_context_service=runtime_context_service,
            context_packet=completed_context_packet,
        ),
        task_mode=task_mode,
        runtime_recent_action=runtime_recent_action,
    )
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            **completed_context_updates,
            "last_executed_step": step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
            "user_message": user_message,
            "working_memory": working_memory,
            "graph_metadata": _replace_control_metadata(state, control),
            "final_message": step_summary,
            "selected_artifacts": list(state.get("selected_artifacts") or []),
            "pending_interrupt": {},
        },
        events=events,
    )


# 等待恢复节点：负责 human_wait 的暂停恢复与后续链路衔接。
async def wait_for_human_node(
        state: PlannerReActLangGraphState,
) -> PlannerReActLangGraphState:
    """在等待节点中恢复用户输入，并回到当前批次继续执行剩余步骤。"""
    started_at = now_perf()
    interrupt_request = _normalize_interrupt_request(state.get("pending_interrupt"))
    if not interrupt_request:
        log_runtime(
            logger,
            logging.INFO,
            "跳过恢复处理",
            state=state,
            reason="没有待恢复的中断",
            elapsed_ms=elapsed_ms(started_at),
        )
        return {
            **state,
            "pending_interrupt": {},
        }

    log_runtime(
        logger,
        logging.INFO,
        "开始处理等待恢复",
        state=state,
        interrupt_kind=str(interrupt_request.get("kind") or ""),
    )

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

    control = _get_control_metadata(state)
    control.pop("wait_resume_action", None)

    waiting_step_id = str(state.get("current_step_id") or "").strip()

    plan = state.get("plan")
    if plan is None or not waiting_step_id:
        log_runtime(
            logger,
            logging.INFO,
            "恢复完成，但未绑定步骤",
            state=state,
            resumed_message_length=len(resumed_message),
            elapsed_ms=elapsed_ms(started_at),
        )
        return {
            **state,
            "user_message": resumed_message,
            "input_parts": [],
            "message_window": message_window,
            "graph_metadata": _replace_control_metadata(state, control),
            "pending_interrupt": {},
        }

    waiting_step: Optional[Step] = None
    for candidate in list(plan.steps or []):
        if str(candidate.id or "").strip() == waiting_step_id:
            waiting_step = candidate
            break

    if waiting_step is None:
        log_runtime(
            logger,
            logging.WARNING,
            "恢复时未找到等待步骤",
            state=state,
            waiting_step_id=waiting_step_id,
            elapsed_ms=elapsed_ms(started_at),
        )
        return {
            **state,
            "user_message": resumed_message,
            "input_parts": [],
            "message_window": message_window,
            "graph_metadata": _replace_control_metadata(state, control),
            "pending_interrupt": {},
        }

    resume_branch = _resolve_wait_resume_branch(interrupt_request, resume_value)

    if resume_branch == "confirm_cancel":
        waiting_step.outcome = StepOutcome(
            done=False,
            summary=_build_wait_cancel_step_summary(waiting_step, resumed_message),
        )
        waiting_step.status = ExecutionStatus.CANCELLED
        cancelled_event = StepEvent(
            step=waiting_step.model_copy(deep=True),
            status=StepEventStatus.CANCELLED,
        )
        await emit_live_events(cancelled_event)
        working_memory = _merge_step_outcome_into_working_memory(
            _ensure_working_memory(state),
            step=waiting_step,
        )
        # direct_wait 已被用户取消，后续会转入新的重规划链路，必须清掉原链路的控制语义。
        _clear_direct_wait_control_state(control)
        control["wait_resume_action"] = "replan"
        log_runtime(
            logger,
            logging.INFO,
            "等待恢复收到取消确认，转入重规划",
            state=state,
            step_id=str(waiting_step.id or ""),
            resumed_message_length=len(resumed_message),
            elapsed_ms=elapsed_ms(started_at),
        )
        return _reduce_state_with_events(
            state,
            updates={
                "plan": plan,
                "user_message": resumed_message,
                "input_parts": [],
                "message_window": message_window,
                "graph_metadata": _replace_control_metadata(state, control),
                "pending_interrupt": {},
                "last_executed_step": waiting_step.model_copy(deep=True),
                "execution_count": int(state.get("execution_count", 0)) + 1,
                "current_step_id": None,
                "working_memory": working_memory,
                "final_message": normalize_step_result_text(waiting_step.outcome.summary),
            },
            events=[cancelled_event],
        )

    waiting_step.outcome = _build_wait_resume_outcome(
        waiting_step,
        branch=resume_branch,
        resumed_message=resumed_message,
    )
    waiting_step.status = ExecutionStatus.COMPLETED
    completed_event = StepEvent(
        step=waiting_step.model_copy(deep=True),
        status=StepEventStatus.COMPLETED,
    )
    await emit_live_events(completed_event)
    next_step = plan.get_next_step()
    next_user_message = resumed_message
    if str(waiting_step.id or "").strip() == "direct-wait-confirm":
        # synthetic confirm 只负责授权继续执行，后续节点仍应保留原始请求作为当前任务。
        next_user_message = str(control.get("direct_wait_original_message") or resumed_message).strip()
    else:
        # P3-一次性收口：常规 human_wait 收到用户补充后先重规划，避免沿用等待前已写死的后续步骤参数。
        control["wait_resume_action"] = "replan"
        next_step = None
    working_memory = _merge_step_outcome_into_working_memory(
        _ensure_working_memory(state),
        step=waiting_step,
    )
    log_runtime(
        logger,
        logging.INFO,
        "等待恢复完成",
        state=state,
        step_id=str(waiting_step.id or ""),
        resumed_message_length=len(resumed_message),
        next_step_id=str(next_step.id or "") if next_step is not None else "",
        elapsed_ms=elapsed_ms(started_at),
    )

    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "user_message": next_user_message,
            "input_parts": [],
            "message_window": message_window,
            "graph_metadata": _replace_control_metadata(state, control),
            "pending_interrupt": {},
            "last_executed_step": waiting_step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
            "working_memory": working_memory,
            "final_message": normalize_step_result_text(waiting_step.outcome.summary),
        },
        events=[completed_event],
    )


# 重规划节点：当前批次跑完后再决定下一批步骤。
async def replan_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
        runtime_context_service: RuntimeContextService,
) -> PlannerReActLangGraphState:
    """在当前批次执行完成后，基于最新结果生成下一批步骤。"""
    started_at = now_perf()
    plan = state.get("plan")
    last_step = state.get("last_executed_step")
    if plan is None or last_step is None:
        return state

    replan_context_packet = await _build_prompt_context_packet_async(
        stage="replan",
        state=state,
        runtime_context_service=runtime_context_service,
        step=last_step,
        task_mode=state.get("task_mode") or normalize_controlled_value(
            getattr(last_step, "task_mode_hint", None),
            StepTaskModeHint,
        ),
    )
    replan_context_updates = _extract_prompt_context_state_updates(
        runtime_context_service=runtime_context_service,
        context_packet=replan_context_packet,
    )
    # replan 仅消费摘要字段，避免把完整步骤和整份计划 JSON 再次塞入 Prompt。
    current_step_snapshot = (
        dict(replan_context_packet.get("current_step") or {})
        if isinstance(replan_context_packet, dict)
        else {}
    )
    stable_background = (
        dict(replan_context_packet.get("stable_background") or {})
        if isinstance(replan_context_packet, dict)
        else {}
    )
    prompt = UPDATE_PLAN_PROMPT.format(
        current_step=json.dumps(current_step_snapshot, ensure_ascii=False),
        plan_snapshot=json.dumps(
            dict(stable_background.get("plan_snapshot") or {}),
            ensure_ascii=False,
        ),
    )
    prompt = _append_prompt_context_to_prompt(prompt, replan_context_packet)
    llm_runtime = describe_llm_runtime(llm)
    log_runtime(
        logger,
        logging.INFO,
        "开始重规划",
        state=state,
        stage_name="replan",
        model_name=llm_runtime["model_name"],
        max_tokens=llm_runtime["max_tokens"],
        last_step_id=str(last_step.id or ""),
        current_step_count=len(list(plan.steps or [])),
    )
    user_message = str(state.get("user_message") or getattr(plan, "goal", "") or "")
    llm_cost_ms_total = 0
    new_steps: List[Step] = []
    replan_prompt = prompt
    for attempt in range(2):
        llm_started_at = now_perf()
        llm_message = await llm.invoke(
            messages=[{"role": "user", "content": replan_prompt}],
            tools=[],
            response_format={"type": "json_object"},
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        llm_cost_ms_total += llm_cost_ms
        parsed = safe_parse_json(llm_message.get("content"))

        raw_steps = parsed.get("steps")
        if not isinstance(raw_steps, list):
            log_runtime(
                logger,
                logging.WARNING,
                "重规划返回无效结果",
                state=state,
                response_keys=sorted(parsed.keys()),
                llm_elapsed_ms=llm_cost_ms_total,
                elapsed_ms=elapsed_ms(started_at),
            )
            return state

        candidate_steps = [
            build_step_from_payload(
                item,
                index,
                user_message=user_message,
            )
            for index, item in enumerate(raw_steps)
        ]
        # P3-一次性收口：重规划步骤同样走统一契约编译，避免新批次继续放大结构化矛盾。
        candidate_steps, contract_issues, corrected_count = compile_step_contracts(
            steps=candidate_steps,
            user_message=user_message,
        )
        contract_issues.extend(collect_step_contract_hard_issues(steps=candidate_steps))
        if corrected_count > 0:
            log_runtime(
                logger,
                logging.INFO,
                "重规划步骤契约已自动纠偏",
                state=state,
                corrected_step_count=corrected_count,
                attempt=attempt + 1,
            )
        if contract_issues:
            log_runtime(
                logger,
                logging.WARNING,
                "重规划步骤契约校验失败",
                state=state,
                issue_count=len(contract_issues),
                issue_codes=[item.issue_code for item in contract_issues],
                attempt=attempt + 1,
            )
            candidate_steps = []
        filtered_steps, dropped_drift_steps = _REPLAN_MERGE_ENGINE.filter_replan_drift_steps(
            candidate_steps,
            user_message=user_message,
        )
        if dropped_drift_steps > 0:
            log_runtime(
                logger,
                logging.WARNING,
                "重规划已拦截漂移元步骤",
                state=state,
                dropped_step_count=dropped_drift_steps,
                attempt=attempt + 1,
            )
        if filtered_steps:
            new_steps = filtered_steps
            break
        if dropped_drift_steps == 0:
            break
        if attempt == 0:
            replan_prompt = replan_prompt + """

补充限制（必须遵守）：
- 只生成直接推进用户业务目标的步骤。
- 禁止生成“测试工具可用性 / 验证工具 / 探活 / smoke test”这类元步骤。
"""
            continue
        log_runtime(
            logger,
            logging.WARNING,
            "重规划结果全部被判定为漂移步骤，已保持原计划不变",
            state=state,
            llm_elapsed_ms=llm_cost_ms_total,
            elapsed_ms=elapsed_ms(started_at),
        )
        return state

    updated_steps, merge_mode = _REPLAN_MERGE_ENGINE.merge_replanned_steps_into_plan(plan, new_steps)
    plan.steps = updated_steps

    next_step = plan.get_next_step()
    updated_event = PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.UPDATED)
    await emit_live_events(updated_event)
    control = _get_control_metadata(state)
    control.pop("wait_resume_action", None)
    log_runtime(
        logger,
        logging.INFO,
        "重规划完成",
        state=state,
        new_step_count=len(new_steps),
        total_step_count=len(list(plan.steps or [])),
        merge_mode=merge_mode,
        next_step_id=str(next_step.id or "") if next_step is not None else "",
        llm_elapsed_ms=llm_cost_ms_total,
        elapsed_ms=elapsed_ms(started_at),
    )
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            **replan_context_updates,
            "current_step_id": next_step.id if next_step is not None else None,
            "graph_metadata": _replace_control_metadata(state, control),
        },
        events=[updated_event],
    )


# 总结节点：生成轻 summary，并输出最终重交付正文与附件。
async def summarize_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
        runtime_context_service: RuntimeContextService,
) -> PlannerReActLangGraphState:
    """在所有步骤完成后汇总结果。"""
    started_at = now_perf()
    plan = state.get("plan")
    if plan is None:
        return state
    control = _get_control_metadata(state)
    if str(control.get("entry_strategy") or "").strip() == "direct_wait" and not bool(
            control.get("direct_wait_original_task_executed")
    ):
        error_message = "运行时异常：direct_wait 已完成确认，但原始任务尚未执行，已阻止错误总结。"
        plan.status = ExecutionStatus.FAILED
        plan.error = error_message
        final_events: List[Any] = [
            ErrorEvent(error=error_message, error_key="direct_wait_unexecuted"),
            MessageEvent(role="assistant", message=error_message, stage="final"),
        ]
        await emit_live_events(*final_events)
        log_runtime(
            logger,
            logging.WARNING,
            "阻断未执行原任务的 direct_wait 错误总结",
            state=state,
            error_key="direct_wait_unexecuted",
            elapsed_ms=elapsed_ms(started_at),
        )
        return _reduce_state_with_events(
            state,
            updates={
                "plan": plan,
                "current_step_id": None,
                "final_message": error_message,
            },
            events=final_events,
        )
    working_memory = _ensure_working_memory(state)
    final_message = str(state.get("final_message") or "")
    last_executed_step = state.get("last_executed_step")
    summarize_intermediate_round = _is_intermediate_delivery_step(last_executed_step)
    deterministic_delivery_text = (
        ""
        if summarize_intermediate_round
        else build_delivery_text(
            working_memory.get("final_delivery_payload"),
            fallback="",
        )
    )
    skip_summary_llm = _should_skip_summary_llm_for_final_delivery(
        summarize_intermediate_round=summarize_intermediate_round,
        last_executed_step=last_executed_step,
        deterministic_delivery_text=deterministic_delivery_text,
    )
    summary_context_updates: Dict[str, Any] = {}
    summary_context_packet: Dict[str, Any] = {}
    if not skip_summary_llm:
        summary_context_packet = await _build_prompt_context_packet_async(
            stage="summary",
            state=state,
            runtime_context_service=runtime_context_service,
            task_mode=state.get("task_mode") or normalize_controlled_value(
                getattr(last_executed_step, "task_mode_hint", None),
                StepTaskModeHint,
            ),
        )
        summary_context_updates = _extract_prompt_context_state_updates(
            runtime_context_service=runtime_context_service,
            context_packet=summary_context_packet,
        )
    llm_runtime = describe_llm_runtime(llm)
    summarize_prompt = (
        _build_intermediate_round_summary_prompt(summary_context_packet)
        if summarize_intermediate_round
        else SUMMARIZE_PROMPT.format(
            context_packet=json.dumps(summary_context_packet, ensure_ascii=False)
        )
    )
    parsed: Dict[str, Any] = {}
    llm_cost_ms = 0
    if skip_summary_llm:
        log_runtime(
            logger,
            logging.INFO,
            "命中确定性交付正文，跳过总结模型调用",
            state=state,
            stage_name="summary",
            deterministic_delivery_text_length=len(deterministic_delivery_text),
            execution_count=int(state.get("execution_count") or 0),
            step_count=len(list(plan.steps or [])),
        )
        summary_message = final_message
    else:
        log_runtime(
            logger,
            logging.INFO,
            "开始生成总结",
            state=state,
            stage_name="summary",
            model_name=llm_runtime["model_name"],
            max_tokens=llm_runtime["max_tokens"],
            execution_count=int(state.get("execution_count") or 0),
            step_count=len(list(plan.steps or [])),
            final_message_length=len(final_message),
            intermediate_round=summarize_intermediate_round,
        )
        llm_started_at = now_perf()
        llm_message = await llm.invoke(
            messages=[{"role": "user", "content": summarize_prompt}],
            tools=[],
            response_format={"type": "json_object"},
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        parsed = safe_parse_json(llm_message.get("content"))
        summary_message = str(
            parsed.get("message")
            or (
                _build_intermediate_round_summary_fallback(last_executed_step)
                if summarize_intermediate_round
                else final_message
            )
            or ""
        ).strip()
    extracted_facts = _normalize_memory_fact_items(parsed.get("facts_in_session"))
    extracted_preferences = _normalize_memory_preferences(parsed.get("user_preferences"))
    model_memory_candidates = _build_model_memory_candidates(
        state=state,
        raw_candidates=parsed.get("memory_candidates"),
    )
    # P3-1A 收敛修复：总结阶段的事实与记忆候选必须绑定执行证据，避免“总结幻觉写入记忆”。
    summary_evidence_texts = _collect_summary_evidence_texts(
        state=state,
        last_executed_step=last_executed_step,
    )
    extracted_facts = _filter_summary_facts_by_evidence(extracted_facts, summary_evidence_texts)
    extracted_preferences, dropped_extracted_preferences = _filter_preferences_by_evidence(
        extracted_preferences,
        summary_evidence_texts,
    )
    # P3-一次性收口：历史偏好也必须绑定本轮证据，避免跨任务污染回流到长期记忆。
    existing_preferences, dropped_existing_preferences = _filter_preferences_by_evidence(
        dict(working_memory.get("user_preferences") or {}),
        summary_evidence_texts,
    )
    working_memory["user_preferences"] = existing_preferences
    model_memory_candidates, dropped_model_memory_candidates = _filter_model_memory_candidates_by_evidence(
        model_memory_candidates,
        summary_evidence_texts,
    )
    if dropped_model_memory_candidates > 0 or dropped_extracted_preferences > 0 or dropped_existing_preferences > 0:
        log_runtime(
            logger,
            logging.INFO,
            "总结记忆已按执行证据过滤",
            state=state,
            dropped_memory_candidate_count=dropped_model_memory_candidates,
            dropped_extracted_preference_count=dropped_extracted_preferences,
            dropped_existing_preference_count=dropped_existing_preferences,
            evidence_count=len(summary_evidence_texts),
        )
    # P3-CASE3 修复：执行阶段已显式声明“不要作为最终附件”时，summary 禁止任何 fallback 附件回填。
    attachment_delivery_preference = _resolve_attachment_delivery_preference_for_summary(
        state=state,
        last_executed_step=last_executed_step,
    )
    if attachment_delivery_preference is False:
        summary_attachment_refs = []
        log_runtime(
            logger,
            logging.INFO,
            "总结附件已按步骤偏好禁用",
            state=state,
            step_id=str(getattr(last_executed_step, "id", "") or ""),
        )
    else:
        # 附件处理
        summary_attachment_refs = await _resolve_summary_attachment_refs(
            state,
            parsed.get("attachments"),
            runtime_context_service=runtime_context_service,
        )
    summary_attachment_paths = [File(filepath=filepath) for filepath in summary_attachment_refs]

    final_events: List[Any] = [
        MessageEvent(
            role="assistant",
            # 预览/草稿轮仍然需要轻量收尾，但不能把旧的最终正文或草稿正文重新当成本轮最终消息发给用户。
            message=(
                summary_message
                if summarize_intermediate_round
                else (
                    deterministic_delivery_text
                    if deterministic_delivery_text
                    else build_delivery_text(
                        working_memory.get("final_delivery_payload"),
                        fallback=summary_message,
                    )
                )
            ),
            attachments=summary_attachment_paths,
            stage="final",
        )]

    plan.status = ExecutionStatus.COMPLETED
    final_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.COMPLETED))

    await emit_live_events(*final_events)
    working_memory["facts_in_session"] = list(working_memory.get("facts_in_session") or [])
    for fact in extracted_facts:
        working_memory["facts_in_session"] = append_unique_text(
            list(working_memory.get("facts_in_session") or []),
            fact,
        )
    if extracted_preferences:
        merged_preferences = dict(existing_preferences)
        merged_preferences.update(extracted_preferences)
        working_memory["user_preferences"] = merged_preferences

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
        outcome_text = str(
            ((outcome_candidate or {}).get("content") or {}).get("text")
            or (outcome_candidate or {}).get("summary")
            or ""
        ).strip()
        if outcome_candidate is not None and _memory_item_has_execution_evidence(outcome_text, summary_evidence_texts):
            memory_candidates = [outcome_candidate]
        elif outcome_candidate is not None:
            log_runtime(
                logger,
                logging.INFO,
                "任务结果候选缺少执行证据，已跳过入库",
                state=state,
                dropped_outcome_candidate=True,
            )
    memory_candidates = _merge_memory_candidates(memory_candidates, model_memory_candidates)
    log_runtime(
        logger,
        logging.INFO,
        "总结生成完成",
        state=state,
        attachment_count=len(summary_attachment_refs),
        fact_count=len(extracted_facts),
        preference_count=len(extracted_preferences),
        memory_candidate_count=len(memory_candidates),
        llm_elapsed_ms=llm_cost_ms,
        elapsed_ms=elapsed_ms(started_at),
    )
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            **summary_context_updates,
            "current_step_id": None,
            "final_message": summary_message,
            "working_memory": working_memory,
            "selected_artifacts": list(summary_attachment_refs),
            "pending_memory_writes": _merge_memory_candidates(
                list(state.get("pending_memory_writes") or []),
                memory_candidates,
            ),
        },
        events=final_events,
    )


# 记忆节点：收敛消息窗口与长期记忆候选。
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
    # 将最终消息和选中的附件添加到消息窗口中
    message_window = _append_message_window_entry(
        list(state.get("message_window") or []),
        role="assistant",
        message=str(state.get("final_message") or ""),
        attachments=list(state.get("selected_artifacts") or []),
    )

    # 压缩消息窗口，防止超出最大长度限制
    compacted_message_window, trimmed_message_count = _compact_message_window(message_window)

    # 处理待写入的长期记忆候选项
    pending_memory_writes, candidate_stats = _govern_memory_candidates(
        list(state.get("pending_memory_writes") or [])
    )
    remaining_memory_writes: List[Dict[str, Any]] = []
    persisted_memory_ids: List[str] = []
    write_cost_ms = 0

    if long_term_memory_repository is None:
        # 若未提供长期记忆仓库，则跳过写入，保留候选项供后续重试
        remaining_memory_writes = pending_memory_writes
    else:
        # 遍历所有待写入的记忆候选项，尝试持久化
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

    # 构建下一个状态对象，更新消息窗口、对话摘要、清理临时记忆并保留未成功写入的记忆候选
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


# 收尾节点：统一补发 done 事件。
async def finalize_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """结束节点，追加 done 事件。"""
    started_at = now_perf()
    events = list(state.get("emitted_events") or [])
    if events and isinstance(events[-1], DoneEvent):
        log_runtime(
            logger,
            logging.INFO,
            "结束事件已存在，跳过收尾",
            state=state,
            reason="已存在完成事件",
            elapsed_ms=elapsed_ms(started_at),
        )
        return state

    done_event = DoneEvent()
    await emit_live_events(done_event)
    log_runtime(
        logger,
        logging.INFO,
        "流程收尾完成",
        state=state,
        emitted_event_count=len(events) + 1,
        elapsed_ms=elapsed_ms(started_at),
    )
    return _reduce_state_with_events(
        state,
        updates={},
        events=[done_event],
    )
