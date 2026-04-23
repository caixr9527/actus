#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层总结节点。

本模块负责多步执行链的最终交付收口：
- 统一生成轻量最终总结；
- 统一组织最终面向用户的正文与附件；
- 不再复用步骤阶段残留文本作为 summary 真相源。
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
    TextStreamChannel,
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
    normalize_controlled_value,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    normalize_attachments,
    safe_parse_json,
)
from ..live_events import emit_live_events
from ..streaming import build_text_stream_events
from .control_state import get_control_metadata as _get_control_metadata
from .delivery_helpers import (
    _resolve_attachment_delivery_preference_for_summary,
    _resolve_summary_attachment_refs,
)
from .prompt_context_helpers import (
    _build_prompt_context_packet_async,
    _extract_prompt_context_state_updates,
)
from .state_reducer import _reduce_state_with_events
from .working_memory import _ensure_working_memory

logger = logging.getLogger(__name__)

_GENERIC_STEP_SUMMARY_TEXTS = {
    "步骤完成",
    "已完成",
    "已结束步骤",
    "当前步骤已完成",
    "该步骤已完成",
    "已成功完成该步骤",
    "任务处理完成",
}


def _build_failure_final_answer_text(
        *,
        plan: Any,
        last_executed_step: Any,
) -> str:
    """失败态最终正文兜底。

    约束：
    - 当最后一步失败或计划整体失败时，最终正文必须明确告知“未完整完成”；
    - 不能继续输出“已完成调研/已整理结果”这类伪成功文案。
    """
    step_outcome = getattr(last_executed_step, "outcome", None)
    step_label = str(
        getattr(last_executed_step, "title", None)
        or getattr(last_executed_step, "description", None)
        or ""
    ).strip()
    blockers = list(getattr(step_outcome, "blockers", []) or [])
    first_blocker = str(blockers[0] or "").strip() if len(blockers) > 0 else ""
    if step_label and first_blocker:
        return f"任务未完整完成。最后一步“{step_label}”执行失败，原因：{first_blocker}"
    if step_label:
        return f"任务未完整完成。最后一步“{step_label}”执行失败。"
    plan_title = str(getattr(plan, "title", "") or "").strip()
    if plan_title:
        return f"任务未完整完成：{plan_title}"
    return "任务未完整完成。"


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


def _build_summary_message_fallback(
        *,
        state: PlannerReActLangGraphState,
        last_executed_step: Any,
) -> str:
    """为 summary 节点提供稳定轻量总结兜底。

    约束：
    - 不能直接复用进入 summary 之前残留的 `final_message`，避免把 planner/step 阶段文本误当最终总结；
    - 最终轮优先基于最后一步的执行摘要或步骤标题生成稳定轻总结。
    """
    step_outcome = getattr(last_executed_step, "outcome", None)
    outcome_summary = str(getattr(step_outcome, "summary", "") or "").strip()
    if outcome_summary and outcome_summary not in _GENERIC_STEP_SUMMARY_TEXTS:
        return outcome_summary

    step_label = str(
        getattr(last_executed_step, "title", None)
        or getattr(last_executed_step, "description", None)
        or ""
    ).strip()
    if step_label:
        return f"已完成{step_label}。"

    plan = state.get("plan")
    plan_title = str(getattr(plan, "title", "") or "").strip()
    if plan_title:
        return f"已完成{plan_title}。"
    return "任务已完成。"


async def summarize_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
        runtime_context_service: RuntimeContextService,
) -> PlannerReActLangGraphState:
    """在所有步骤完成后统一收口最终交付。

    业务约束：
    - `summary_node` 是多步执行链里最终用户正文的唯一组织出口；
    - state 内的 `final_message` 只保存轻量最终总结，供会话历史/记忆/运行摘要使用；
    - `final_answer_text` 保存最终面向用户的正文，且只允许由 final/direct 阶段写入。
    """
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
        final_stream_events = build_text_stream_events(
            channel=TextStreamChannel.FINAL_MESSAGE,
            text=error_message,
            state=state,
            stage="final",
        )
        final_events: List[Any] = [
            ErrorEvent(error=error_message, error_key="direct_wait_unexecuted"),
            MessageEvent(role="assistant", message=error_message, stage="final"),
        ]
        await _resolve_emit_live_events()(*final_stream_events, *final_events)
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
                # 错误总结分支同样要保持最终正文真相源一致，避免投影和跨轮锚点出现空值分叉。
                "final_answer_text": error_message,
            },
            events=final_events,
        )
    working_memory = _ensure_working_memory(state)
    last_executed_step = state.get("last_executed_step")
    summary_context_updates: Dict[str, Any] = {}
    summary_context_packet: Dict[str, Any] = {}
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
    summarize_prompt = SUMMARIZE_PROMPT.format(
        context_packet=json.dumps(summary_context_packet, ensure_ascii=False)
    )
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
        previous_final_message_length=len(str(state.get("final_message") or "")),
    )
    llm_started_at = now_perf()
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": summarize_prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    llm_cost_ms = elapsed_ms(llm_started_at)
    parsed: Dict[str, Any] = safe_parse_json(llm_message.get("content"))
    has_failed_last_step = bool(
        last_executed_step is not None
        and str(getattr(last_executed_step, "status", "") or "") == ExecutionStatus.FAILED.value
    )
    has_failed_plan = str(getattr(plan, "status", "") or "") == ExecutionStatus.FAILED.value
    if has_failed_last_step or has_failed_plan:
        failure_text = _build_failure_final_answer_text(
            plan=plan,
            last_executed_step=last_executed_step,
        )
        summary_message = failure_text
        final_answer_text = failure_text
        log_runtime(
            logger,
            logging.WARNING,
            "总结阶段已按失败态收紧输出口径",
            state=state,
            failed_plan=has_failed_plan,
            failed_last_step=has_failed_last_step,
            step_id=str(getattr(last_executed_step, "id", "") or ""),
        )
    else:
        summary_message = str(
            parsed.get("message")
            or _build_summary_message_fallback(
                state=state,
                last_executed_step=last_executed_step,
            )
            or ""
        ).strip()
        final_answer_text = str(
            parsed.get("final_answer_text")
            or summary_message
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
    final_answer_text_to_emit = final_answer_text
    # final_message 流事件只做临时展示，不进入 state 的 emitted_events。
    # 最终 MessageEvent(stage="final") 仍是历史落账和前端 timeline 的唯一真相源。
    final_stream_events = build_text_stream_events(
        channel=TextStreamChannel.FINAL_MESSAGE,
        text=final_answer_text_to_emit,
        state=state,
        stage="final",
    )

    final_events: List[Any] = [
        MessageEvent(
            role="assistant",
            message=final_answer_text_to_emit,
            attachments=summary_attachment_paths,
            stage="final",
        )
    ]

    plan.status = ExecutionStatus.COMPLETED
    final_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.COMPLETED))

    await _resolve_emit_live_events()(*final_stream_events, *final_events)
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
            # 最终正文真相源只允许来自 summary/direct 阶段，不再借道 working_memory。
            "final_answer_text": final_answer_text_to_emit,
            "working_memory": working_memory,
            "selected_artifacts": list(summary_attachment_refs),
        },
        events=final_events,
    )
