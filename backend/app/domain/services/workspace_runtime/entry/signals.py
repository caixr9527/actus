#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""入口编译信号与评分。

本模块只负责把用户输入转成可解释评分，不直接决定入口路由。
"""

from __future__ import annotations

import re
from typing import Any, Dict

from app.domain.services.workspace_runtime.policies import (
    analyze_text_intent,
    classify_confirmed_user_task_mode,
    has_environment_write_intent,
)

FRESHNESS_PATTERN = re.compile(
    r"(最新|当前|现在|今天|今日|今年|最近|近况|新闻|价格|政策|版本|榜单|可用|available|latest|current|today|recent|news|price|version)",
    re.IGNORECASE,
)
EXPLICIT_STEP_SEQUENCE_PATTERN = re.compile(
    r"(然后|下一步|第二步|接着|then|next\s+step|second\s+step)",
    re.IGNORECASE,
)
LEADING_STEP_SEQUENCE_PATTERN = re.compile(r"(先|first)", re.IGNORECASE)
REPEAT_SAME_ACTION_PATTERN = re.compile(
    r"(再次|再读取|再搜索|重复|同一个问题|同一个\s*URL|同一个文件|again|repeat|same\s+(?:question|url|file))",
    re.IGNORECASE,
)
NO_CHANGE_PATTERN = re.compile(r"(不要换|do\s+not\s+change)", re.IGNORECASE)


def collect_entry_signals(user_message: str) -> Dict[str, Any]:
    """汇总入口编译需要的文本信号。"""
    signals = analyze_text_intent(user_message)
    text = str(signals.get("text") or "")
    signals["has_freshness_signal"] = bool(FRESHNESS_PATTERN.search(text))
    has_following_step_marker = bool(EXPLICIT_STEP_SEQUENCE_PATTERN.search(text))
    has_repeat_same_action_intent = (
        bool(REPEAT_SAME_ACTION_PATTERN.search(text))
        or (has_following_step_marker and bool(NO_CHANGE_PATTERN.search(text)))
    )
    signals["has_explicit_step_sequence"] = (
        has_following_step_marker
        or (has_repeat_same_action_intent and bool(LEADING_STEP_SEQUENCE_PATTERN.search(text)))
    )
    signals["has_repeat_same_action_intent"] = has_repeat_same_action_intent
    signals["has_environment_write_intent"] = has_environment_write_intent(signals)
    signals["task_mode"] = classify_confirmed_user_task_mode(user_message)
    return signals


def score_complexity(signals: Dict[str, Any]) -> int:
    score = 0
    if signals["has_numbered_list"]:
        score += 2
    if signals["has_sequence_marker"]:
        score += 2
    if int(signals["clause_count"]) >= 3:
        score += 2
    if int(signals["char_count"]) >= 120:
        score += 2
    if signals["has_planning_signal"]:
        score += 3
    if signals["has_comparison_signal"]:
        score += 3
    if signals["has_synthesis_signal"]:
        score += 2
    return score


def score_tool_need(signals: Dict[str, Any]) -> int:
    score = 0
    for key in (
        "has_tool_reference",
        "has_url",
        "has_absolute_path",
        "has_shell_command",
        "has_code_block",
        "has_browser_interaction_signal",
        "has_web_reading_signal",
        "has_search_signal",
        "has_file_signal",
        "has_coding_signal",
    ):
        if signals[key]:
            score += 2
    if signals["has_read_action_signal"]:
        score += 1
    return score


def score_freshness(signals: Dict[str, Any]) -> int:
    score = 0
    if signals["has_freshness_signal"]:
        score += 3
    if signals["has_search_signal"]:
        score += 1
    return score


def score_context_need(signals: Dict[str, Any], *, contextual_followup_anchor: bool) -> int:
    score = 0
    if signals["has_contextual_followup_signal"]:
        score += 2
    if contextual_followup_anchor:
        score += 2
    return score


def score_risk(signals: Dict[str, Any]) -> int:
    score = 0
    if signals["has_environment_write_intent"]:
        score += 4
    if signals["has_shell_command"]:
        score += 2
    if signals["has_browser_interaction_signal"]:
        score += 1
    return score
