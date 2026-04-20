#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究链路 snippet 证据判断。"""

from __future__ import annotations

from typing import Any, Dict, List

from app.domain.models import SearchResults, Step
from app.domain.services.runtime.normalizers import extract_url_domain
from app.infrastructure.runtime.langgraph.graphs.planner_react.research_intent_policy import (
    is_explicit_single_page_fetch_intent,
)

_MAX_EVIDENCE_ITEMS = 5
_MIN_USEFUL_SNIPPET_LENGTH = 48


def extract_search_evidence_items(*, tool_result_data: Any) -> List[Dict[str, str]]:
    """从 search_web 结果中提取可注入上下文的摘要证据。"""
    raw_items: List[Dict[str, str]] = []
    if isinstance(tool_result_data, SearchResults):
        for item in list(tool_result_data.results or []):
            raw_items.append(
                {
                    "title": str(getattr(item, "title", "") or "").strip(),
                    "url": str(getattr(item, "url", "") or "").strip(),
                    "snippet": str(getattr(item, "snippet", "") or "").strip(),
                }
            )
    elif isinstance(tool_result_data, dict):
        for item in list(tool_result_data.get("results") or []):
            if not isinstance(item, dict):
                continue
            raw_items.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "snippet": str(item.get("snippet") or item.get("content") or "").strip(),
                }
            )
    evidence_items: List[Dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in raw_items:
        url = str(item.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        evidence_items.append(item)
        if len(evidence_items) >= _MAX_EVIDENCE_ITEMS:
            break
    return evidence_items


def evaluate_search_snippet_sufficiency(
        *,
        step: Step,
        evidence_items: List[Dict[str, str]],
) -> Dict[str, Any]:
    """判断 search_web 返回的 snippet 是否足以支撑当前步骤。"""
    useful_items = [
        item for item in list(evidence_items or [])
        if len(str(item.get("snippet") or "").strip()) >= _MIN_USEFUL_SNIPPET_LENGTH
    ]
    domains = {
        extract_url_domain(str(item.get("url") or "").strip())
        for item in useful_items
        if str(item.get("url") or "").strip()
    }
    domains.discard("")
    explicit_single_page_fetch = is_explicit_single_page_fetch_intent(step, explicit_url="")
    snippet_sufficient = bool(useful_items) and not explicit_single_page_fetch
    need_fetch = explicit_single_page_fetch or not snippet_sufficient
    reason_code = "snippet_sufficient"
    if explicit_single_page_fetch:
        reason_code = "step_requires_page_content"
    elif not useful_items:
        reason_code = "snippet_insufficient"
    elif len(domains) < 2:
        reason_code = "snippet_source_coverage_thin"
    recommended_items = useful_items if useful_items else list(evidence_items or [])
    return {
        "snippet_sufficient": snippet_sufficient,
        "need_fetch": need_fetch,
        "reason_code": reason_code,
        "recommended_fetch_urls": [
            str(item.get("url") or "").strip()
            for item in recommended_items[:3]
            if str(item.get("url") or "").strip()
        ],
        "useful_snippet_count": len(useful_items),
        "distinct_domain_count": len(domains),
        "evidence_items": list(evidence_items or []),
    }
