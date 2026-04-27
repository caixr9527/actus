#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层重规划节点。

本模块只承载当前批次执行完成后的重规划节点实现，不改重规划业务语义。
"""

import json
import logging
from typing import List

from app.domain.external import LLM
from app.domain.models import PlanEvent, PlanEventStatus, Step, StepTaskModeHint
from app.domain.services.prompts import UPDATE_PLAN_PROMPT
from app.domain.services.runtime.contracts.langgraph_settings import (
    REPLAN_META_VALIDATION_ALLOW_PATTERN,
    REPLAN_META_VALIDATION_DENY_PATTERN,
    REPLAN_META_VALIDATION_STEP_PATTERN,
)
from app.domain.services.runtime.contracts.runtime_logging import (
    describe_llm_runtime,
    elapsed_ms,
    log_runtime,
    now_perf,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import (
    normalize_controlled_value,
    normalize_success_criteria,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.policies import (
    collect_step_contract_hard_issues,
    compile_step_contracts,
    filter_final_delivery_steps,
)
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import safe_parse_json
from ..live_events import emit_live_events
from ..parsers import build_step_from_payload
from ..replan import ReplanMergeEngine
from .control_state import (
    get_control_metadata as _get_control_metadata,
    replace_control_metadata as _replace_control_metadata,
)
from .prompt_context_helpers import (
    _append_prompt_context_to_prompt,
    _build_prompt_context_packet_async,
    _extract_prompt_context_state_updates,
)
from .state_reducer import _reduce_state_with_events

logger = logging.getLogger(__name__)
_REPLAN_MERGE_ENGINE = ReplanMergeEngine(
    replan_meta_validation_step_pattern=REPLAN_META_VALIDATION_STEP_PATTERN,
    replan_meta_validation_allow_pattern=REPLAN_META_VALIDATION_ALLOW_PATTERN,
    replan_meta_validation_deny_pattern=REPLAN_META_VALIDATION_DENY_PATTERN,
)


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
        criteria_missing_count = 0
        criteria_filtered_count = 0
        for index, raw_item in enumerate(raw_steps):
            if not isinstance(raw_item, dict):
                criteria_missing_count += 1
                continue
            step_description = str(candidate_steps[index].description or "").strip() if index < len(
                candidate_steps) else ""
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
            candidate_steps[index].success_criteria = list(normalized_criteria)
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
                success_criteria_missing_count=criteria_missing_count,
                success_criteria_filtered_count=criteria_filtered_count,
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
        candidate_steps, dropped_final_delivery_steps = filter_final_delivery_steps(
            steps=candidate_steps,
            user_message=user_message,
        )
        if dropped_final_delivery_steps > 0:
            log_runtime(
                logger,
                logging.WARNING,
                "重规划已过滤最终交付型步骤",
                state=state,
                dropped_final_delivery_steps=dropped_final_delivery_steps,
                remaining_step_count=len(candidate_steps),
                success_criteria_missing_count=criteria_missing_count,
                success_criteria_filtered_count=criteria_filtered_count,
                attempt=attempt + 1,
            )
            if len(candidate_steps) == 0:
                next_step = plan.get_next_step()
                control = _get_control_metadata(state)
                control.pop("wait_resume_action", None)
                control.pop("entry_upgrade", None)
                log_runtime(
                    logger,
                    logging.INFO,
                    "重规划仅返回最终交付型步骤，保持原计划并交给后续路由",
                    state=state,
                    dropped_final_delivery_steps=dropped_final_delivery_steps,
                    remaining_step_count=0,
                    next_step_id=str(next_step.id or "") if next_step is not None else "",
                    llm_elapsed_ms=llm_cost_ms_total,
                    elapsed_ms=elapsed_ms(started_at),
                )
                return {
                    **state,
                    **replan_context_updates,
                    "current_step_id": next_step.id if next_step is not None else None,
                    "graph_metadata": _replace_control_metadata(state, control),
                }
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
                success_criteria_missing_count=criteria_missing_count,
                success_criteria_filtered_count=criteria_filtered_count,
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
    # entry_upgrade 是一次性运行态信号；进入 Planner 后即完成消费，避免后续步骤反复重规划。
    control.pop("entry_upgrade", None)
    log_runtime(
        logger,
        logging.INFO,
        "重规划完成",
        state=state,
        new_step_count=len(new_steps),
        total_step_count=len(list(plan.steps or [])),
        merge_mode=merge_mode,
        success_criteria_missing_count=0 if len(new_steps) == 0 else len(
            [step for step in new_steps if len(list(step.success_criteria or [])) == 0]
        ),
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
