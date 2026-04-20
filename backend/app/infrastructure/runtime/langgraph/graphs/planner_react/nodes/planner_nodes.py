#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层 planner 节点。

本模块只承载 create_or_reuse_plan_node，不改 planner 业务语义。
"""

import logging
import sys
from typing import Any, Dict, List

from app.domain.external import LLM
from app.domain.models import (
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    TitleEvent,
)
from app.domain.services.prompts import (
    CREATE_PLAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from app.domain.services.runtime.contracts.runtime_logging import (
    describe_llm_runtime,
    elapsed_ms,
    log_runtime,
    now_perf,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import normalize_success_criteria
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.policies import (
    collect_step_contract_hard_issues,
    compile_step_contracts,
)
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    format_attachments_for_prompt,
    safe_parse_json,
)
from ..live_events import emit_live_events
from ..parsers import build_fallback_plan_title, build_step_from_payload
from .control_state import (
    get_control_metadata as _get_control_metadata,
    replace_control_metadata as _replace_control_metadata,
)
from .state_reducer import _reduce_state_with_events
from .working_memory import _ensure_working_memory

logger = logging.getLogger(__name__)


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


def _resolve_prompt_context_builder():
    package_module = sys.modules.get(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes"
    )
    if package_module is not None:
        package_builder = getattr(package_module, "_build_prompt_context_packet_async", None)
        if callable(package_builder):
            return package_builder
    from .prompt_context_helpers import _build_prompt_context_packet_async
    return _build_prompt_context_packet_async


def _resolve_prompt_context_extractor():
    package_module = sys.modules.get(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes"
    )
    if package_module is not None:
        package_extractor = getattr(package_module, "_extract_prompt_context_state_updates", None)
        if callable(package_extractor):
            return package_extractor
    from .prompt_context_helpers import _extract_prompt_context_state_updates
    return _extract_prompt_context_state_updates


def _resolve_prompt_context_appender():
    package_module = sys.modules.get(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes"
    )
    if package_module is not None:
        package_appender = getattr(package_module, "_append_prompt_context_to_prompt", None)
        if callable(package_appender):
            return package_appender
    from .prompt_context_helpers import _append_prompt_context_to_prompt
    return _append_prompt_context_to_prompt


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
            await _resolve_emit_live_events()(*reuse_events)
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

    build_prompt_context_packet_async = _resolve_prompt_context_builder()
    extract_prompt_context_state_updates = _resolve_prompt_context_extractor()
    append_prompt_context_to_prompt = _resolve_prompt_context_appender()

    planner_context_packet = await build_prompt_context_packet_async(
        stage="planner",
        state=state,
        runtime_context_service=runtime_context_service,
    )
    planner_context_updates = extract_prompt_context_state_updates(
        runtime_context_service=runtime_context_service,
        context_packet=planner_context_packet,
    )
    user_message_prompt = CREATE_PLAN_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
    )
    user_message_prompt = append_prompt_context_to_prompt(user_message_prompt, planner_context_packet)

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
            MessageEvent(role="assistant", message=planner_message),
        ]
        await _resolve_emit_live_events()(*planner_events)
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

    steps = [
        build_step_from_payload(
            item,
            index,
            user_message=user_message,
        )
        for index, item in enumerate(raw_steps)
    ]
    criteria_missing_count = 0
    criteria_filtered_count = 0
    for index, raw_item in enumerate(raw_steps):
        if not isinstance(raw_item, dict):
            criteria_missing_count += 1
            continue
        step_description = str(steps[index].description or "").strip() if index < len(steps) else ""
        normalized_criteria, criteria_metrics = normalize_success_criteria(
            raw_item.get("success_criteria"),
            fallback_description=step_description,
        )
        raw_criteria = raw_item.get("success_criteria")
        raw_has_criteria = isinstance(raw_criteria, list) and len(raw_criteria) > 0
        if not raw_has_criteria:
            criteria_missing_count += 1
        if raw_has_criteria:
            criteria_filtered_count += int(criteria_metrics.get("filtered_low_value_count", 0))
            criteria_filtered_count += int(criteria_metrics.get("filtered_too_short_count", 0))
        steps[index].success_criteria = list(normalized_criteria)
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
        success_criteria_missing_count=criteria_missing_count,
        success_criteria_filtered_count=criteria_filtered_count,
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
        MessageEvent(role="assistant", message=planner_message),
    ]
    await _resolve_emit_live_events()(*planner_events)
    return _reduce_state_with_events(
        state,
        updates={
            "plan": plan,
            "working_memory": working_memory,
            **planner_context_updates,
            "current_step_id": None if plan_only else next_step.id if next_step is not None else None,
            "final_message": planner_message if plan_only else "",
            "graph_metadata": _replace_control_metadata(state, control),
        },
        events=planner_events,
    )
