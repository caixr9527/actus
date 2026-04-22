#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层总结节点。

本模块只承载 summary 节点实现，不改 summary/final delivery 的业务语义。
"""

import json
import logging
import sys
from typing import Any, Dict, List

from app.domain.external import LLM
from app.domain.models import (
    ErrorEvent,
    ExecutionStatus,
    File,
    MessageEvent,
    PlanEvent,
    PlanEventStatus,
    StepTaskModeHint,
)
from app.domain.services.prompts import SUMMARIZE_PROMPT
from app.domain.services.runtime.contracts.runtime_logging import (
    describe_llm_runtime,
    elapsed_ms,
    log_runtime,
    now_perf,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import (
    build_delivery_text,
    normalize_controlled_value,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    normalize_attachments,
    safe_parse_json,
)
from ..live_events import emit_live_events
from .control_state import get_control_metadata as _get_control_metadata
from .delivery_helpers import (
    _build_intermediate_round_summary_fallback,
    _build_intermediate_round_summary_prompt,
    _is_intermediate_delivery_step,
    _resolve_attachment_delivery_preference_for_summary,
    _resolve_summary_attachment_refs,
    _should_skip_summary_llm_for_final_delivery,
)
from .prompt_context_helpers import (
    _build_prompt_context_packet_async,
    _extract_prompt_context_state_updates,
)
from .state_reducer import _reduce_state_with_events
from .working_memory import _ensure_working_memory

logger = logging.getLogger(__name__)


def _resolve_emit_live_events():
    """统一从 nodes 包级入口解析事件发送函数，保持聚合入口的可替换性。"""
    package_module = sys.modules.get(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes"
    )
    if package_module is not None:
        package_emit_live_events = getattr(package_module, "emit_live_events", None)
        if callable(package_emit_live_events):
            return package_emit_live_events
    return emit_live_events


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
        await _resolve_emit_live_events()(*final_events)
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
        summary_attachment_refs = await _resolve_summary_attachment_refs(
            state,
            parsed.get("attachments"),
            runtime_context_service=runtime_context_service,
        )
        log_runtime(
            logger,
            logging.INFO,
            "总结附件过滤完成",
            state=state,
            raw_attachment_count=len(normalize_attachments(parsed.get("attachments"))),
            final_attachment_count=len(summary_attachment_refs),
            final_attachment_paths=summary_attachment_refs,
        )
    summary_attachment_paths = [File(filepath=filepath) for filepath in summary_attachment_refs]

    final_events: List[Any] = [
        MessageEvent(
            role="assistant",
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
        )
    ]

    plan.status = ExecutionStatus.COMPLETED
    final_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.COMPLETED))

    await _resolve_emit_live_events()(*final_events)
    log_runtime(
        logger,
        logging.INFO,
        "总结生成完成",
        state=state,
        attachment_count=len(summary_attachment_refs),
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
        },
        events=final_events,
    )
