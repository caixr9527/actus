#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务模式与入口路由策略。"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from app.domain.models import Step, StepTaskModeHint
from app.domain.services.runtime.normalizers import normalize_controlled_value, normalize_success_criteria
from app.domain.services.runtime.contracts.langgraph_settings import (
    ABSOLUTE_PATH_PATTERN,
    ACTION_PATTERN,
    CODING_PATTERN,
    COMPARISON_PATTERN,
    CONTEXTUAL_FOLLOWUP_PATTERN,
    FILE_FUNCTION_NAMES,
    FILE_PATTERN,
    PHATIC_PATTERN,
    PLAN_ONLY_PATTERN,
    PLANNING_PATTERN,
    READ_ACTION_PATTERN,
    SEARCH_FUNCTION_NAMES,
    SEARCH_PATTERN,
    SEQUENCE_PATTERN,
    SHELL_COMMAND_PATTERN,
    SYNTHESIS_PATTERN,
    TASK_MODE_ALLOWED_FUNCTIONS,
    TASK_MODE_ALLOWED_PREFIXES,
    TOOL_REFERENCE_PATTERN,
    WRITE_ACTION_DENY_PATTERN,
    WRITE_ACTION_PATTERN,
    URL_PATTERN,
    WAIT_PATTERN,
    WAIT_REQUEST_PATTERN,
    WEB_READING_PATTERN,
    BROWSER_INTERACTION_PATTERN,
    CODE_BLOCK_PATTERN,
    NUMBERED_LIST_PATTERN,
)

def normalize_intent_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def classify_file_access_intent(value: str) -> str:
    """把文件访问语义分成三态，供执行期精确治理。"""
    signals = analyze_text_intent(value)
    if has_environment_write_intent(signals):
        return "write_intent"
    if bool(signals.get("has_read_action_signal")):
        return "read_only_intent"
    return "unknown"


def has_environment_write_target_signal(signals: Dict[str, Any]) -> bool:
    """判断文本是否命中了真实环境副作用载体。"""
    return any(
        (
            bool(signals.get("has_absolute_path")),
            bool(signals.get("has_file_signal")),
            bool(signals.get("has_shell_command")),
            bool(signals.get("has_code_block")),
            bool(signals.get("has_coding_signal")),
            bool(signals.get("has_tool_reference")),
        )
    )


def has_environment_write_intent(value_or_signals: str | Dict[str, Any]) -> bool:
    """统一判断“工程写副作用”，避免把纯语义编辑误判成 coding。"""
    signals = (
        value_or_signals
        if isinstance(value_or_signals, dict)
        else analyze_text_intent(str(value_or_signals or ""))
    )
    return bool(signals.get("has_write_action_signal")) and has_environment_write_target_signal(signals)


def has_explicit_wait_semantics(value: str) -> bool:
    normalized_text = normalize_intent_text(value)
    if not normalized_text:
        return False
    return bool(
        WAIT_PATTERN.search(normalized_text)
        or WAIT_REQUEST_PATTERN.search(normalized_text)
    )


def analyze_text_intent(value: str) -> Dict[str, Any]:
    normalized_text = normalize_intent_text(value)
    if not normalized_text:
        return {
            "text": "",
            "char_count": 0,
            "has_url": False,
            "has_absolute_path": False,
            "has_shell_command": False,
            "has_code_block": False,
            "has_numbered_list": False,
            "has_sequence_marker": False,
            "is_phatic": False,
            "needs_human_wait": False,
            "has_browser_interaction_signal": False,
            "has_web_reading_signal": False,
            "has_read_action_signal": False,
            "has_search_signal": False,
            "has_planning_signal": False,
            "has_plan_only_signal": False,
            "has_synthesis_signal": False,
            "has_comparison_signal": False,
            "has_file_signal": False,
            "has_coding_signal": False,
            "has_action_signal": False,
            "has_contextual_followup_signal": False,
            "has_write_action_signal": False,
            "has_tool_reference": False,
            "clause_count": 0,
        }

    clause_count = len(
        [
            segment
            for segment in re.split(r"[。！？!?；;\n]+", normalized_text)
            if str(segment).strip()
        ]
    )
    return {
        "text": normalized_text,
        "char_count": len(normalized_text),
        "has_url": bool(URL_PATTERN.search(normalized_text)),
        "has_absolute_path": bool(ABSOLUTE_PATH_PATTERN.search(normalized_text)),
        "has_shell_command": bool(SHELL_COMMAND_PATTERN.search(normalized_text)),
        "has_code_block": bool(CODE_BLOCK_PATTERN.search(normalized_text)),
        "has_numbered_list": bool(NUMBERED_LIST_PATTERN.search(normalized_text)),
        "has_sequence_marker": bool(SEQUENCE_PATTERN.search(normalized_text)),
        "is_phatic": bool(PHATIC_PATTERN.match(normalized_text)),
        "needs_human_wait": has_explicit_wait_semantics(normalized_text),
        "has_browser_interaction_signal": bool(BROWSER_INTERACTION_PATTERN.search(normalized_text)),
        "has_web_reading_signal": bool(WEB_READING_PATTERN.search(normalized_text)),
        "has_read_action_signal": bool(READ_ACTION_PATTERN.search(normalized_text)),
        "has_search_signal": bool(SEARCH_PATTERN.search(normalized_text)),
        "has_planning_signal": bool(PLANNING_PATTERN.search(normalized_text)),
        "has_plan_only_signal": bool(PLAN_ONLY_PATTERN.search(normalized_text)),
        "has_synthesis_signal": bool(SYNTHESIS_PATTERN.search(normalized_text)),
        "has_comparison_signal": bool(COMPARISON_PATTERN.search(normalized_text)),
        "has_file_signal": bool(FILE_PATTERN.search(normalized_text)),
        "has_coding_signal": bool(CODING_PATTERN.search(normalized_text)),
        "has_action_signal": bool(ACTION_PATTERN.search(normalized_text)),
        "has_contextual_followup_signal": bool(CONTEXTUAL_FOLLOWUP_PATTERN.search(normalized_text)),
        # P3-1A 收敛修复：统一识别“写入/改写/执行命令”等写副作用意图，供只读任务硬拦截复用。
        "has_write_action_signal": bool(
            WRITE_ACTION_PATTERN.search(normalized_text)
            and not WRITE_ACTION_DENY_PATTERN.search(normalized_text)
        ),
        "has_tool_reference": bool(TOOL_REFERENCE_PATTERN.search(normalized_text)),
        "clause_count": clause_count,
    }


def _has_direct_execution_need(signals: Dict[str, Any]) -> bool:
    return any(
        (
            signals["has_tool_reference"],
            signals["has_url"],
            signals["has_absolute_path"],
            signals["has_shell_command"],
            signals["has_code_block"],
            signals["has_browser_interaction_signal"],
            signals["has_web_reading_signal"],
            signals["has_search_signal"],
            signals["has_file_signal"],
            signals["has_coding_signal"],
        )
    )


def _should_force_planner_for_tool_task(signals: Dict[str, Any]) -> bool:
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
    if signals["clause_count"] >= 3 and (
            signals["has_search_signal"]
            or signals["has_web_reading_signal"]
            or signals["has_planning_signal"]
            or signals["has_comparison_signal"]
    ):
        return True
    return False


def requests_plan_only(user_message: str) -> bool:
    return bool(analyze_text_intent(user_message)["has_plan_only_signal"])


def infer_entry_strategy(
        *,
        user_message: str,
        has_input_parts: bool,
        has_active_plan: bool,
        has_contextual_followup_anchor: bool = False,
) -> str:
    if has_active_plan:
        return "create_plan_or_reuse"
    if has_input_parts:
        return "recall_memory_context"

    signals = analyze_text_intent(user_message)
    if signals["char_count"] == 0:
        return "recall_memory_context"
    needs_planner = _should_force_planner_for_tool_task(signals)
    is_structurally_multi_step = (
            signals["has_numbered_list"]
            or signals["has_sequence_marker"]
            or signals["clause_count"] >= 3
            or signals["char_count"] >= 120
    )
    has_direct_execution_need = _has_direct_execution_need(signals)
    if signals["needs_human_wait"]:
        # 复杂/规划型请求即便包含“先确认”语义，也应先走 planner 主链，
        # 由 planner 决定是否进入 input_text/confirm 的 human_wait，而不是硬落 direct_wait。
        if needs_planner:
            return "recall_memory_context"
        if has_direct_execution_need and (is_structurally_multi_step or signals["char_count"] >= 48):
            return "recall_memory_context"
        return "direct_wait"
    if signals["is_phatic"] and not has_direct_execution_need:
        return "direct_answer"
    # 追问/展开类短消息在存在历史锚点时应继续走 direct_answer。
    # 这里仍显式排除工具、文件、检索和复杂多步骤信号，避免把真实执行请求误降级成直答。
    if (
            has_contextual_followup_anchor
            and signals["has_contextual_followup_signal"]
            and not has_direct_execution_need
            and not needs_planner
            and not signals["needs_human_wait"]
            and not signals["has_numbered_list"]
            and signals["clause_count"] <= 2
            and signals["char_count"] < 80
    ):
        return "direct_answer"
    if needs_planner or is_structurally_multi_step:
        return "recall_memory_context"
    if has_direct_execution_need:
        return "direct_execute"
    return "direct_answer"


def classify_task_mode_from_signals(signals: Dict[str, Any]) -> str:
    scores = {
        "web_reading": 0,
        "browser_interaction": 0,
        "coding": 0,
        "file_processing": 0,
        "research": 0,
    }
    if signals["has_browser_interaction_signal"]:
        scores["browser_interaction"] += 5
    if signals["has_web_reading_signal"]:
        scores["web_reading"] += 3
    if signals["has_url"]:
        scores["web_reading"] += 2
        scores["research"] += 2
    if signals["has_shell_command"] or signals["has_code_block"] or signals["has_coding_signal"]:
        scores["coding"] += 3
    if has_environment_write_intent(signals):
        # P3-一次性收口：创建/写入/修改/执行类任务属于可变更环境操作，
        # 不能继续按只读 file_processing 处理，否则执行约束层会误伤 shell/write 能力。
        scores["coding"] += 3
    if signals["has_absolute_path"] or signals["has_file_signal"]:
        scores["file_processing"] += 3
    if signals["has_search_signal"]:
        scores["research"] += 3
    if signals["has_tool_reference"]:
        if any(name in signals["text"] for name in
               ("browser_click", "browser_input", "browser_scroll", "browser_press_key", "browser_select_option")):
            scores["browser_interaction"] += 3
        if any(
                name in signals["text"]
                for name in (
                        "browser_view",
                        "browser_navigate",
                        "browser_restart",
                        "browser_read_current_page_structured",
                        "browser_extract_main_content",
                        "browser_extract_cards",
                        "browser_find_link_by_text",
                        "browser_find_actionable_elements",
                )
        ):
            scores["web_reading"] += 2
        if "shell_" in signals["text"]:
            scores["coding"] += 2
        if any(name in signals["text"] for name in FILE_FUNCTION_NAMES):
            scores["file_processing"] += 2
        if any(name in signals["text"] for name in SEARCH_FUNCTION_NAMES):
            scores["research"] += 2
    if scores["browser_interaction"] == 0 and scores["web_reading"] > 0:
        scores["research"] += 1

    ranked_modes = sorted(
        scores.items(),
        key=lambda item: (
            item[1],
            {
                "web_reading": 5,
                "browser_interaction": 4,
                "coding": 3,
                "file_processing": 2,
                "research": 1,
            }.get(item[0], 0),
        ),
        reverse=True,
    )
    if ranked_modes and ranked_modes[0][1] > 0:
        return ranked_modes[0][0]
    # 无显式工具/文件/浏览器信号时，规划/对比/总结类请求默认按 research 执行，
    # 避免误落 general 后在直达路径上漂移到 shell。
    if signals["has_planning_signal"] or signals["has_comparison_signal"] or signals["has_synthesis_signal"]:
        return "research"
    return "general"


def build_step_candidate_text(step: Step) -> str:
    normalized_criteria, _ = normalize_success_criteria(
        getattr(step, "success_criteria", []),
        fallback_description=str(step.description or "").strip(),
    )
    candidate_parts = [
        str(step.title or "").strip(),
        str(step.description or "").strip(),
        *normalized_criteria,
    ]
    return " ".join([part for part in candidate_parts if part])


def classify_step_task_mode(step: Step) -> str:
    structured_hint = normalize_controlled_value(getattr(step, "task_mode_hint", None), StepTaskModeHint)
    if structured_hint:
        return structured_hint
    signals = analyze_text_intent(build_step_candidate_text(step))
    if signals["needs_human_wait"]:
        return "human_wait"
    return classify_task_mode_from_signals(signals)


def classify_confirmed_user_task_mode(user_message: str) -> str:
    signals = analyze_text_intent(user_message)
    return classify_task_mode_from_signals(signals)


def is_allowed_in_task_mode(function_name: str, task_mode: str) -> bool:
    normalized_name = function_name.strip().lower()
    allowed_functions = set(TASK_MODE_ALLOWED_FUNCTIONS.get(task_mode, TASK_MODE_ALLOWED_FUNCTIONS["general"]))
    allowed_prefixes = TASK_MODE_ALLOWED_PREFIXES.get(task_mode, TASK_MODE_ALLOWED_PREFIXES["general"])
    if normalized_name in allowed_functions:
        return True
    return any(normalized_name.startswith(prefix) for prefix in allowed_prefixes)


def build_task_mode_disallowed_names(
        available_function_names: List[str],
        *,
        task_mode: str,
) -> set[str]:
    blocked_names: set[str] = set()
    for function_name in available_function_names:
        if not function_name:
            continue
        if is_allowed_in_task_mode(function_name, task_mode):
            continue
        blocked_names.add(function_name)
    return blocked_names
