#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究链路诊断：区分搜索质量、抓取质量与证据充分性。"""

from __future__ import annotations

import re
from typing import Any, Dict

from app.domain.services.runtime.normalizers import normalize_url_value
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution_state import ExecutionState

_LOW_VALUE_TITLE_PATTERN = re.compile(
    r"(首页|登录|注册|搜索结果|列表页|频道页|专题页|门票预订|攻略详情)",
    re.IGNORECASE,
)


def build_research_diagnosis(*, state: ExecutionState) -> Dict[str, Any]:
    """根据当前执行态输出研究诊断结果。"""
    candidate_url_count = len(list(state.research_candidate_urls or []))
    fetched_url_count = len(list(state.research_fetched_urls or []))
    fetched_domain_count = len(list(state.research_fetched_domains or []))
    last_search_evidence_quality = dict(state.last_search_evidence_quality or {})
    last_fetch_quality = dict(state.last_fetch_quality or {})
    diagnosis_code = "research_in_progress"
    diagnosis_message = "研究链路仍在推进中。"

    if candidate_url_count == 0 and int(state.search_invocation_count or 0) > 0:
        diagnosis_code = "search_low_recall"
        diagnosis_message = "搜索已执行，但候选来源过少或偏题。"
    elif bool(state.research_snippet_sufficient):
        diagnosis_code = "search_snippet_sufficient"
        diagnosis_message = "搜索摘要已提供足够证据，当前步骤不一定需要继续抓取正文。"
    elif bool(last_search_evidence_quality) and bool(last_search_evidence_quality.get("need_fetch")):
        diagnosis_code = "search_snippet_insufficient"
        diagnosis_message = "搜索已执行，但摘要证据不足，建议继续抓取单页正文。"
    elif bool(last_fetch_quality) and not bool(last_fetch_quality.get("is_useful")):
        diagnosis_code = "fetch_low_value"
        diagnosis_message = "页面抓取已执行，但正文价值不足，未拿到有效信息。"
    elif int(state.consecutive_fetch_failure_count or 0) >= 2 and fetched_url_count == 0:
        diagnosis_code = "fetch_unavailable"
        diagnosis_message = "页面抓取连续失败，当前来源无法稳定读取。"
    elif fetched_url_count > 0 and fetched_domain_count > 0 and float(state.research_coverage_score or 0.0) >= 0.9:
        diagnosis_code = "research_complete"
        diagnosis_message = "研究证据已达到当前步骤要求。"
    elif fetched_url_count > 0:
        diagnosis_code = "evidence_insufficient"
        diagnosis_message = "已抓取部分来源，但关键证据仍不足。"

    return {
        "code": diagnosis_code,
        "message": diagnosis_message,
        "candidate_url_count": candidate_url_count,
        "fetched_url_count": fetched_url_count,
        "fetched_domain_count": fetched_domain_count,
        "coverage_score": float(state.research_coverage_score or 0.0),
        "last_search_evidence_quality": last_search_evidence_quality,
        "last_fetch_quality": last_fetch_quality,
    }


def evaluate_fetch_result_quality(*, fetched_result: Any, fallback_url: str = "") -> Dict[str, Any]:
    """判断 fetch_page 的成功结果是否真的有信息增量。"""
    data = fetched_result if isinstance(fetched_result, dict) else {}
    final_url = normalize_url_value(
        data.get("final_url") or data.get("url") or fallback_url,
    )
    title = str(data.get("title") or "").strip()
    content = str(data.get("content") or "").strip()
    normalized_content = re.sub(r"\s+", " ", content)
    content_length = len(normalized_content)
    title_low_value = bool(title and _LOW_VALUE_TITLE_PATTERN.search(title))
    content_low_value = content_length < 120
    is_useful = bool(final_url) and (not content_low_value) and (not title_low_value)
    return {
        "url": final_url,
        "title": title,
        "content_length": content_length,
        "is_useful": is_useful,
        "low_value_reason": _build_low_value_reason(
            title_low_value=title_low_value,
            content_low_value=content_low_value,
        ),
    }


def _build_low_value_reason(*, title_low_value: bool, content_low_value: bool) -> str:
    reasons: list[str] = []
    if title_low_value:
        reasons.append("页面标题疑似聚合页或落地页")
    if content_low_value:
        reasons.append("正文长度不足")
    return "；".join(reasons)
