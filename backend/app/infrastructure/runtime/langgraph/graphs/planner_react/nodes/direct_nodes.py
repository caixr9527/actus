#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层 direct path 节点。

本模块只承载 direct_answer / direct_wait / direct_execute 及入口路由节点，
不改 planner 主链和执行链语义。
"""

import logging
from typing import Any, List

from app.domain.external import LLM
from app.domain.models import (
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Step,
    TextStreamChannel,
    TitleEvent,
)
from app.domain.services.prompts import DIRECT_ANSWER_PROMPT, SYSTEM_PROMPT
from app.domain.services.runtime.contracts.runtime_logging import (
    describe_llm_runtime,
    elapsed_ms,
    log_runtime,
    now_perf,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.policies import (
    classify_confirmed_user_task_mode,
    infer_entry_strategy,
    requests_plan_only,
)
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import safe_parse_json
from ..language_checker import build_direct_path_copy, infer_working_language_from_message
from ..live_events import emit_live_events
from ..streaming import build_text_stream_events
from .prompt_context_helpers import _append_prompt_context_to_prompt
from .control_state import (
    clear_plan_only_control_state as _clear_plan_only_control_state,
    get_control_metadata as _get_control_metadata,
    replace_control_metadata as _replace_control_metadata,
)
from .state_reducer import _reduce_state_with_events
logger = logging.getLogger(__name__)


def _count_recent_messages(state: PlannerReActLangGraphState) -> int:
    return len(list(state.get("message_window") or []))


def _count_prior_turn_messages(state: PlannerReActLangGraphState) -> int:
    """只统计当前轮之前的消息，避免把当前用户输入误记成“已有历史”。"""
    normalized_messages = list(state.get("message_window") or [])
    current_user_message = str(state.get("user_message") or "").strip()
    if not normalized_messages:
        return 0
    if current_user_message and str(normalized_messages[-1].get("message") or "").strip() == current_user_message:
        return max(len(normalized_messages) - 1, 0)
    return len(normalized_messages)


def _has_prior_turn_context(state: PlannerReActLangGraphState) -> bool:
    return _count_prior_turn_messages(state) > 0


def _build_direct_plan_title(user_message: str, fallback: str) -> str:
    normalized_message = str(user_message or "").strip()
    if not normalized_message:
        return fallback
    return normalized_message[:40]

async def entry_router_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """P0 入口轻路由：先判断是否需要完整记忆召回和 Planner。"""
    control = _get_control_metadata(state)
    user_message = str(state.get("user_message") or "")
    plan = state.get("plan")
    recent_message_count = _count_recent_messages(state)
    prior_turn_message_count = _count_prior_turn_messages(state)
    has_prior_turn_context = _has_prior_turn_context(state)
    has_recent_run_brief = len(list(state.get("recent_run_briefs") or [])) > 0
    # 追问类短消息需要一个明确历史锚点才允许走 direct_answer，避免无上下文时泛化回答。
    has_contextual_followup_anchor = any(
        (
            has_prior_turn_context,
            bool(str(state.get("conversation_summary") or "").strip()),
            has_recent_run_brief,
        )
    )
    control["entry_strategy"] = infer_entry_strategy(
        user_message=user_message,
        has_input_parts=bool(list(state.get("input_parts") or [])),
        has_active_plan=bool(plan is not None and len(list(plan.steps or [])) > 0 and not plan.done),
        has_contextual_followup_anchor=has_contextual_followup_anchor,
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
        recent_message_count=recent_message_count,
        prior_turn_message_count=prior_turn_message_count,
        has_prior_turn_context=has_prior_turn_context,
        has_conversation_summary=bool(str(state.get("conversation_summary") or "").strip()),
        has_recent_run_brief=has_recent_run_brief,
        has_contextual_followup_anchor=has_contextual_followup_anchor,
    )
    return {
        **state,
        "graph_metadata": _replace_control_metadata(state, control),
    }

async def direct_answer_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
        runtime_context_service: RuntimeContextService,
) -> PlannerReActLangGraphState:
    """直接回答类任务跳过 Planner 和工具循环。"""
    started_at = now_perf()
    user_message = str(state.get("user_message") or "").strip()
    language = infer_working_language_from_message(user_message)
    direct_copy = build_direct_path_copy(language)
    # direct_answer 只需要历史对话锚点，不读取 workspace snapshot，避免直答路径引入环境 I/O。
    direct_context_packet = runtime_context_service.build_packet(
        stage="direct_answer",
        state=state,
        task_mode="general",
    )
    prompt = _append_prompt_context_to_prompt(
        DIRECT_ANSWER_PROMPT.format(message=user_message),
        direct_context_packet,
    )
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
    # direct_answer 的流式正文仍属于 final_message 通道。
    # text_stream_* 只做临时展示，不进入 state 的 emitted_events。
    direct_answer_stream_events = build_text_stream_events(
        channel=TextStreamChannel.FINAL_MESSAGE,
        text=final_message,
        state=state,
        stage="final",
    )
    events: List[Any] = [
        TitleEvent(title=plan.title),
        MessageEvent(role="assistant", message=final_message, stage="final"),
    ]
    await emit_live_events(*direct_answer_stream_events, *events)
    control = _get_control_metadata(state)
    control["entry_strategy"] = "direct_answer"
    control["skip_replan_when_plan_finished"] = True
    stable_background = dict(direct_context_packet.get("stable_background") or {})
    topic_anchor = dict(stable_background.get("topic_anchor") or {})
    topic_anchor_source = str(topic_anchor.get("source") or "").strip()
    topic_anchor_text = str(topic_anchor.get("text") or "").strip()
    prior_turn_message_count = _count_prior_turn_messages(state)
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
        direct_answer_context_has_conversation_summary=bool(
            str(stable_background.get("conversation_summary") or "").strip()
        ),
        direct_answer_context_has_recent_run_brief=bool(list(stable_background.get("recent_run_briefs") or [])),
        direct_answer_context_has_prior_turn_context=prior_turn_message_count > 0,
        direct_answer_context_prior_turn_message_count=prior_turn_message_count,
        direct_answer_context_recent_message_count=len(list(stable_background.get("recent_messages") or [])),
        direct_answer_context_recent_run_brief_count=len(list(stable_background.get("recent_run_briefs") or [])),
        direct_answer_context_topic_anchor_source=topic_anchor_source or "none",
        direct_answer_context_topic_anchor_preview=topic_anchor_text[:160],
    )
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "current_step_id": None,
            "final_message": final_message,
            # direct_answer 直接产生最终面向用户的正文，因此同步更新最终正文真相源。
            "final_answer_text": final_message,
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
        status=ExecutionStatus.RUNNING,
    )
    step_execute = Step(
        id="direct-wait-execute",
        title=direct_copy["direct_wait_execute_title"],
        # 第二步只表达“开始执行原任务”，不再把“先确认”语义写回步骤文本。
        description=direct_copy["direct_wait_execute_title"],
        task_mode_hint=execute_task_mode,
        # Step 只负责执行，最终正文统一由 summary_node 组织。
        output_mode="none",
        artifact_policy="default",
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
        # direct_execute 也只执行当前任务，最终正文统一由 summary_node 输出。
        output_mode="none",
        artifact_policy="default",
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
