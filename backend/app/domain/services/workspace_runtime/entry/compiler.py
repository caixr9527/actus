#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 任务入口编译器。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

from app.domain.models import Plan, StepTaskModeHint
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.normalizers import normalize_controlled_value

from . import reason_codes as rc
from .contracts import (
    EntryContextProfile,
    EntryContract,
    EntryRiskLevel,
    EntryRoute,
    EntrySourceSnapshot,
    EntryToolBudget,
    EntryUpgradePolicy,
)
from .signals import (
    collect_entry_signals,
    score_complexity,
    score_context_need,
    score_freshness,
    score_risk,
    score_tool_need,
)

logger = logging.getLogger(__name__)


def _has_active_plan(plan: Any) -> bool:
    if not isinstance(plan, Plan):
        return False
    return len(list(plan.steps or [])) > 0 and not plan.done


def _deduplicate_reason_codes(values: Iterable[str]) -> List[str]:
    reason_codes: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in reason_codes:
            reason_codes.append(normalized)
    return reason_codes


def _risk_level(risk_score: int) -> EntryRiskLevel:
    if risk_score >= 4:
        return EntryRiskLevel.HIGH
    if risk_score > 0:
        return EntryRiskLevel.MEDIUM
    return EntryRiskLevel.LOW


def _upgrade_policy(route: EntryRoute) -> EntryUpgradePolicy:
    if route != EntryRoute.ATOMIC_ACTION:
        return EntryUpgradePolicy()
    return EntryUpgradePolicy(
        allow_upgrade=True,
        max_tool_calls_before_upgrade=3,
        upgrade_on_second_tool_family=True,
        upgrade_on_user_confirmation_required=True,
        upgrade_on_file_output_required=True,
        upgrade_on_open_questions=True,
    )


def _contract(
        *,
        route: EntryRoute,
        task_mode: str,
        context_profile: EntryContextProfile,
        tool_budget: EntryToolBudget,
        needs_summary: bool,
        plan_only: bool,
        risk_level: EntryRiskLevel,
        complexity_score: int,
        tool_need_score: int,
        freshness_score: int,
        context_need_score: int,
        reason_codes: Iterable[str],
        source: EntrySourceSnapshot,
) -> EntryContract:
    normalized_task_mode = normalize_controlled_value(task_mode, StepTaskModeHint) or StepTaskModeHint.GENERAL.value
    return EntryContract(
        route=route,
        task_mode=StepTaskModeHint(normalized_task_mode),
        context_profile=context_profile,
        tool_budget=tool_budget,
        needs_summary=needs_summary,
        plan_only=plan_only,
        risk_level=risk_level,
        complexity_score=complexity_score,
        tool_need_score=tool_need_score,
        freshness_score=freshness_score,
        context_need_score=context_need_score,
        reason_codes=_deduplicate_reason_codes(reason_codes),
        upgrade_policy=_upgrade_policy(route),
        source=source,
    )


def _log_compiled_contract(contract: EntryContract) -> None:
    """记录入口编译决策，不写入原文，避免日志泄露用户完整输入。"""
    log_runtime(
        logger,
        logging.INFO,
        "入口合同编译完成",
        entry_route=contract.route.value,
        task_mode=contract.task_mode.value,
        context_profile=contract.context_profile.value,
        tool_budget=contract.tool_budget.value,
        plan_only=contract.plan_only,
        risk_level=contract.risk_level.value,
        complexity_score=contract.complexity_score,
        tool_need_score=contract.tool_need_score,
        freshness_score=contract.freshness_score,
        context_need_score=contract.context_need_score,
        reason_codes=contract.reason_codes,
        message_length=len(contract.source.user_message),
        has_input_parts=contract.source.has_input_parts,
        has_active_plan=contract.source.has_active_plan,
        has_contextual_followup_anchor=contract.source.contextual_followup_anchor,
    )


def _needs_planner(signals: Dict[str, Any], *, complexity_score: int, risk_score: int) -> bool:
    if signals["has_plan_only_signal"]:
        return True
    if signals["has_planning_signal"] or signals["has_comparison_signal"]:
        return True
    if signals["has_search_signal"] and signals["has_synthesis_signal"]:
        return True
    if signals["has_search_signal"] and signals["has_read_action_signal"]:
        return True
    if signals["has_web_reading_signal"] and signals["has_synthesis_signal"] and not (
            signals["has_url"] or signals["has_tool_reference"]
    ):
        return True
    if complexity_score >= 4:
        return True
    if risk_score >= 4:
        return True
    return False


def _planner_reason_codes(signals: Dict[str, Any], *, complexity_score: int, risk_score: int) -> List[str]:
    reasons: List[str] = []
    if signals["has_plan_only_signal"]:
        reasons.append(rc.PLAN_ONLY_REQUIRES_PLANNER)
    if signals["has_planning_signal"]:
        reasons.append(rc.PLANNING_REQUIRES_PLANNER)
    if signals["has_comparison_signal"]:
        reasons.append(rc.COMPARISON_REQUIRES_PLANNER)
    if signals["has_search_signal"] and signals["has_synthesis_signal"]:
        reasons.append(rc.SEARCH_AND_SYNTHESIS_REQUIRES_PLANNER)
    if signals["has_search_signal"] and signals["has_read_action_signal"]:
        reasons.append(rc.SEARCH_AND_READ_REQUIRES_PLANNER)
    if complexity_score >= 4:
        reasons.append(rc.MULTI_STEP_REQUIRES_PLANNER)
    if risk_score >= 4:
        reasons.append(rc.HIGH_RISK_REQUIRES_PLANNER)
    return reasons or [rc.LOW_CONFIDENCE_REQUIRES_PLANNER]


def _atomic_reason_codes(signals: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if signals["has_url"]:
        reasons.append(rc.SINGLE_URL_ATOMIC_ACTION)
    if (signals["has_absolute_path"] or signals["has_file_signal"]) and not signals["has_url"]:
        reasons.append(rc.SINGLE_FILE_READ_ATOMIC_ACTION)
    if signals["has_search_signal"]:
        reasons.append(rc.SINGLE_SEARCH_ATOMIC_ACTION)
    if not reasons:
        reasons.append(rc.SINGLE_TOOL_ATOMIC_ACTION)
    return reasons


class EntryCompiler:
    """把原始运行状态编译为入口执行合同。"""

    def compile(
            self,
            *,
            user_message: str,
            has_input_parts: bool,
            has_active_plan: bool,
            contextual_followup_anchor: bool,
    ) -> EntryContract:
        source = EntrySourceSnapshot(
            user_message=str(user_message or "").strip(),
            has_input_parts=has_input_parts,
            has_active_plan=has_active_plan,
            contextual_followup_anchor=contextual_followup_anchor,
        )
        signals = collect_entry_signals(source.user_message)
        complexity_score = score_complexity(signals)
        tool_need_score = score_tool_need(signals)
        freshness_score = score_freshness(signals)
        context_need_score = score_context_need(
            signals,
            contextual_followup_anchor=contextual_followup_anchor,
        )
        risk_score = score_risk(signals)
        risk_level = _risk_level(risk_score)
        task_mode = str(signals.get("task_mode") or StepTaskModeHint.GENERAL.value)
        plan_only = bool(signals["has_plan_only_signal"])

        if has_active_plan:
            contract = _contract(
                route=EntryRoute.RESUME_PLAN,
                task_mode=task_mode,
                context_profile=EntryContextProfile.FULL,
                tool_budget=EntryToolBudget.PLANNER_CONTROLLED,
                needs_summary=True,
                plan_only=False,
                risk_level=risk_level,
                complexity_score=complexity_score,
                tool_need_score=tool_need_score,
                freshness_score=freshness_score,
                context_need_score=context_need_score,
                reason_codes=[rc.ACTIVE_PLAN_RESUME],
                source=source,
            )
            _log_compiled_contract(contract)
            return contract
        if has_input_parts:
            contract = _contract(
                route=EntryRoute.PLANNED_TASK,
                task_mode=task_mode,
                context_profile=EntryContextProfile.FULL,
                tool_budget=EntryToolBudget.PLANNER_CONTROLLED,
                needs_summary=True,
                plan_only=plan_only,
                risk_level=risk_level,
                complexity_score=complexity_score,
                tool_need_score=tool_need_score,
                freshness_score=freshness_score,
                context_need_score=context_need_score,
                reason_codes=[rc.INPUT_ATTACHMENT_REQUIRES_PLANNER],
                source=source,
            )
            _log_compiled_contract(contract)
            return contract
        if int(signals["char_count"]) == 0:
            contract = _contract(
                route=EntryRoute.PLANNED_TASK,
                task_mode=task_mode,
                context_profile=EntryContextProfile.FULL,
                tool_budget=EntryToolBudget.PLANNER_CONTROLLED,
                needs_summary=True,
                plan_only=False,
                risk_level=risk_level,
                complexity_score=complexity_score,
                tool_need_score=tool_need_score,
                freshness_score=freshness_score,
                context_need_score=context_need_score,
                reason_codes=[rc.EMPTY_MESSAGE_REQUIRES_PLANNER],
                source=source,
            )
            _log_compiled_contract(contract)
            return contract

        needs_planner = _needs_planner(signals, complexity_score=complexity_score, risk_score=risk_score)
        if signals["needs_human_wait"]:
            if needs_planner or (tool_need_score > 0 and (complexity_score >= 3 or int(signals["char_count"]) >= 48)):
                contract = _contract(
                    route=EntryRoute.PLANNED_TASK,
                    task_mode=task_mode,
                    context_profile=EntryContextProfile.FULL,
                    tool_budget=EntryToolBudget.PLANNER_CONTROLLED,
                    needs_summary=True,
                    plan_only=plan_only,
                    risk_level=risk_level,
                    complexity_score=complexity_score,
                    tool_need_score=tool_need_score,
                    freshness_score=freshness_score,
                    context_need_score=context_need_score,
                    reason_codes=[rc.COMPLEX_WAIT_REQUIRES_PLANNER, *_planner_reason_codes(
                        signals,
                        complexity_score=complexity_score,
                        risk_score=risk_score,
                    )],
                    source=source,
                )
                _log_compiled_contract(contract)
                return contract
            contract = _contract(
                route=EntryRoute.WAIT,
                task_mode=task_mode,
                context_profile=EntryContextProfile.WORKSPACE if tool_need_score > 0 else EntryContextProfile.MINIMAL_HISTORY,
                tool_budget=EntryToolBudget.SMALL_LOOP if tool_need_score > 0 else EntryToolBudget.NONE,
                needs_summary=True,
                plan_only=False,
                risk_level=risk_level,
                complexity_score=complexity_score,
                tool_need_score=tool_need_score,
                freshness_score=freshness_score,
                context_need_score=context_need_score,
                reason_codes=[rc.WAIT_BEFORE_ACTION],
                source=source,
            )
            _log_compiled_contract(contract)
            return contract

        if needs_planner:
            contract = _contract(
                route=EntryRoute.PLANNED_TASK,
                task_mode=task_mode,
                context_profile=EntryContextProfile.FULL,
                tool_budget=EntryToolBudget.PLANNER_CONTROLLED,
                needs_summary=True,
                plan_only=plan_only,
                risk_level=risk_level,
                complexity_score=complexity_score,
                tool_need_score=tool_need_score,
                freshness_score=freshness_score,
                context_need_score=context_need_score,
                reason_codes=_planner_reason_codes(signals, complexity_score=complexity_score, risk_score=risk_score),
                source=source,
            )
            _log_compiled_contract(contract)
            return contract

        if tool_need_score > 0 or freshness_score > 0:
            reason_codes = _atomic_reason_codes(signals)
            if freshness_score > 0:
                reason_codes.insert(0, rc.FRESHNESS_REQUIRES_TOOL)
            contract = _contract(
                route=EntryRoute.ATOMIC_ACTION,
                task_mode=task_mode,
                context_profile=EntryContextProfile.WORKSPACE,
                tool_budget=EntryToolBudget.SMALL_LOOP if tool_need_score > 2 else EntryToolBudget.SINGLE_CALL,
                needs_summary=True,
                plan_only=False,
                risk_level=risk_level,
                complexity_score=complexity_score,
                tool_need_score=tool_need_score,
                freshness_score=freshness_score,
                context_need_score=context_need_score,
                reason_codes=reason_codes,
                source=source,
            )
            _log_compiled_contract(contract)
            return contract

        if signals["is_phatic"]:
            contract = _contract(
                route=EntryRoute.ANSWER,
                task_mode=StepTaskModeHint.GENERAL.value,
                context_profile=EntryContextProfile.NONE,
                tool_budget=EntryToolBudget.NONE,
                needs_summary=False,
                plan_only=False,
                risk_level=risk_level,
                complexity_score=complexity_score,
                tool_need_score=tool_need_score,
                freshness_score=freshness_score,
                context_need_score=context_need_score,
                reason_codes=[rc.PHATIC_DIRECT_ANSWER],
                source=source,
            )
            _log_compiled_contract(contract)
            return contract
        if (
                contextual_followup_anchor
                and signals["has_contextual_followup_signal"]
                and complexity_score < 3
                and int(signals["char_count"]) < 80
        ):
            contract = _contract(
                route=EntryRoute.ANSWER,
                task_mode=StepTaskModeHint.GENERAL.value,
                context_profile=EntryContextProfile.MINIMAL_HISTORY,
                tool_budget=EntryToolBudget.NONE,
                needs_summary=False,
                plan_only=False,
                risk_level=risk_level,
                complexity_score=complexity_score,
                tool_need_score=tool_need_score,
                freshness_score=freshness_score,
                context_need_score=context_need_score,
                reason_codes=[rc.CONTEXTUAL_FOLLOWUP_DIRECT_ANSWER],
                source=source,
            )
            _log_compiled_contract(contract)
            return contract

        contract = _contract(
            route=EntryRoute.ANSWER,
            task_mode=StepTaskModeHint.GENERAL.value,
            context_profile=EntryContextProfile.NONE,
            tool_budget=EntryToolBudget.NONE,
            needs_summary=False,
            plan_only=False,
            risk_level=risk_level,
            complexity_score=complexity_score,
            tool_need_score=tool_need_score,
            freshness_score=freshness_score,
            context_need_score=context_need_score,
            reason_codes=[rc.SIMPLE_DIRECT_ANSWER],
            source=source,
        )
        _log_compiled_contract(contract)
        return contract

    def compile_state(self, state: Dict[str, Any]) -> EntryContract:
        """从 LangGraph state 编译入口合同。"""
        message_window = list(state.get("message_window") or [])
        current_user_message = str(state.get("user_message") or "").strip()
        if message_window and current_user_message and str(message_window[-1].get("message") or "").strip() == current_user_message:
            prior_turn_count = max(len(message_window) - 1, 0)
        else:
            prior_turn_count = len(message_window)
        contextual_followup_anchor = any(
            (
                prior_turn_count > 0,
                bool(str(state.get("conversation_summary") or "").strip()),
                len(list(state.get("recent_run_briefs") or [])) > 0,
            )
        )
        return self.compile(
            user_message=current_user_message,
            has_input_parts=bool(list(state.get("input_parts") or [])),
            has_active_plan=_has_active_plan(state.get("plan")),
            contextual_followup_anchor=contextual_followup_anchor,
        )
