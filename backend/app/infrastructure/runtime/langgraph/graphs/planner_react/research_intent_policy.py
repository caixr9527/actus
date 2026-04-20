#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究链路意图判定策略。"""

from __future__ import annotations

import re

from app.domain.models import Step
from app.domain.services.workspace_runtime.policies import (
    analyze_text_intent as _analyze_text_intent,
    build_step_candidate_text as _build_step_candidate_text,
)

# 单页读取动作：强调“对某个已给定 URL 做读取/提取”，避免绑定具体业务场景词。
_EXPLICIT_SINGLE_PAGE_ACTION_PATTERN = re.compile(
    r"(读取|抓取|打开|查看|访问|提取|解析|浏览|点击|进入|跳转|read|fetch|open|visit|extract|parse|crawl)",
    re.IGNORECASE,
)

# 多来源研究信号：仅拦截明确“多来源/来源对比”语义，避免误伤“单页读取后总结要点”。
_MULTI_SOURCE_RESEARCH_PATTERN = re.compile(
    r"(多来源|多个来源|不同来源|跨来源|多站点|多渠道|来源对比|对比来源|对比.{0,8}来源|汇总.{0,8}来源|compare\s+sources?)",
    re.IGNORECASE,
)


def is_explicit_single_page_fetch_intent(step: Step, *, explicit_url: str = "") -> bool:
    """判断步骤是否属于“显式 URL 的单页读取”意图。支持 URL 来自步骤或当前轮用户输入。"""
    step_text = str(_build_step_candidate_text(step) or "").strip()
    if not step_text:
        return False

    signals = _analyze_text_intent(step_text)
    has_explicit_url = bool(str(explicit_url or "").strip())
    if (not bool(signals.get("has_url"))) and (not has_explicit_url):
        return False

    # 出现明确多来源研究语义时，禁止判定为单页读取，避免把 research 锁到 fetch 单页。
    if _MULTI_SOURCE_RESEARCH_PATTERN.search(step_text):
        return False

    # 有 URL + 明确读取动作，判定为单页读取。
    if _EXPLICIT_SINGLE_PAGE_ACTION_PATTERN.search(step_text):
        return True

    # 兜底：网页读取信号存在且非搜索/综合语义，可视为单页读取。
    return (
            bool(signals.get("has_web_reading_signal"))
            and (not bool(signals.get("has_search_signal")))
            and (not bool(signals.get("has_synthesis_signal")))
            and (not bool(signals.get("has_comparison_signal")))
    )
