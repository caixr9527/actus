#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具结果状态归并（reducer）。"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.domain.models import SearchResults, Step, ToolResult
from app.domain.services.runtime.normalizers import extract_url_domain, normalize_url_value
from app.domain.services.workspace_runtime.policies import (
    attach_browser_degrade_payload as _attach_browser_degrade_payload,
    build_tool_feedback_content as _build_tool_feedback_content,
    build_browser_high_level_failure_key as _build_browser_high_level_failure_key,
    build_tool_fingerprint as _build_tool_fingerprint,
    build_browser_observation_fingerprint as _build_browser_observation_fingerprint,
    build_recent_blocked_tool_call as _build_recent_blocked_tool_call,
    build_recent_failed_action as _build_recent_failed_action,
    extract_browser_tool_state as _extract_browser_tool_state,
)
from .execution_context import ExecutionContext
from .execution_state import ExecutionState
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.contracts.langgraph_settings import (
    BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
    BROWSER_NO_PROGRESS_LIMIT,
    RESEARCH_MIN_DOMAIN_COUNT,
    RESEARCH_MIN_FETCH_COUNT,
    RESEARCH_PROGRESS_MAX_TRACK_ITEMS,
    BROWSER_PROGRESS_FUNCTIONS,
    TOOL_FAILURE_LIMIT,
)


@dataclass(slots=True)
class ToolEffectsResult:
    tool_result: ToolResult
    loop_break_reason: str


def apply_tool_result_effects(
    *,
    logger: logging.Logger,
    step: Step,
    function_name: str,
    normalized_function_name: str,
    function_args: Dict[str, Any],
    tool_result: ToolResult,
    loop_break_reason: str,
    browser_route_state_key: str,
    execution_context: ExecutionContext,
    execution_state: ExecutionState,
) -> ToolEffectsResult:
    if bool(tool_result.success):
        execution_state.consecutive_failure_count = 0
        execution_state.last_successful_tool_call = {
            "function_name": str(function_name or "").strip().lower(),
            "function_args": dict(function_args or {}),
            "message": str(tool_result.message or "").strip(),
            "data": tool_result.data,
            # P3-一次性收口：重复成功兜底保留标准化反馈文本，避免后续“成功但无可交付内容”。
            "feedback_content": _build_tool_feedback_content(function_name, tool_result),
        }
        execution_state.last_successful_tool_fingerprint = _build_tool_fingerprint(
            str(normalized_function_name or "").strip().lower(),
            dict(function_args or {}),
        )
    else:
        execution_state.runtime_recent_action["last_failed_action"] = _build_recent_failed_action(
            function_name=function_name,
            tool_result=tool_result,
        )
        blocked_tool_call = _build_recent_blocked_tool_call(
            function_name=function_name,
            tool_result=tool_result,
            loop_break_reason=loop_break_reason,
        )
        if blocked_tool_call:
            execution_state.runtime_recent_action["last_blocked_tool_call"] = blocked_tool_call
        execution_state.consecutive_failure_count += 1
        if (
            normalized_function_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES
            and loop_break_reason != "browser_high_level_retry_blocked"
        ):
            execution_state.failed_browser_high_level_keys.add(
                _build_browser_high_level_failure_key(
                    function_name=normalized_function_name,
                    function_args=function_args,
                    browser_route_state_key=browser_route_state_key,
                )
            )
            degrade_reason = f"{normalized_function_name}_failed"
            tool_result = _attach_browser_degrade_payload(
                tool_result,
                function_name=function_name,
                degrade_reason=degrade_reason,
                browser_page_type=execution_state.browser_page_type,
                browser_url=execution_state.last_browser_route_url,
                browser_title=execution_state.last_browser_route_title,
            )
            log_runtime(
                logger,
                logging.INFO,
                "浏览器高阶能力失败，允许降级为其他浏览器能力",
                step_id=str(step.id or ""),
                function_name=function_name,
                degrade_reason=degrade_reason,
            )

    if execution_context.research_route_enabled and normalized_function_name == "search_web" and bool(tool_result.success):
        extracted_query = _extract_search_query(tool_result, function_args=function_args)
        if extracted_query:
            _append_unique_text(
                execution_state.research_query_history,
                extracted_query,
                max_items=RESEARCH_PROGRESS_MAX_TRACK_ITEMS,
            )
        candidate_urls = _extract_search_result_urls(tool_result)
        execution_state.research_candidate_urls = candidate_urls
        execution_state.research_search_ready = len(candidate_urls) > 0
        execution_state.research_total_search_results += len(candidate_urls)
        for domain in _extract_domains(candidate_urls):
            _append_unique_text(
                execution_state.research_candidate_domains,
                domain,
                max_items=RESEARCH_PROGRESS_MAX_TRACK_ITEMS,
            )
    elif execution_context.research_route_enabled and normalized_function_name == "fetch_page" and bool(tool_result.success):
        execution_state.research_fetch_completed = True
        execution_state.research_fetch_success_count += 1
        execution_state.research_cross_domain_repeat_blocks = 0
        fetched_url = _extract_fetched_page_url(tool_result)
        if fetched_url:
            _append_unique_text(
                execution_state.research_fetched_urls,
                fetched_url,
                max_items=RESEARCH_PROGRESS_MAX_TRACK_ITEMS,
            )
            domain = _extract_domain(fetched_url)
            if domain:
                _append_unique_text(
                    execution_state.research_fetched_domains,
                    domain,
                    max_items=RESEARCH_PROGRESS_MAX_TRACK_ITEMS,
                )
        execution_state.research_coverage_score = _estimate_research_coverage_score(execution_state)
    if execution_context.research_route_enabled:
        execution_state.research_coverage_score = _estimate_research_coverage_score(execution_state)
        execution_state.runtime_recent_action["research_progress"] = _build_research_progress_snapshot(execution_state)
        if normalized_function_name in {"search_web", "fetch_page"}:
            research_progress = dict(execution_state.runtime_recent_action.get("research_progress") or {})
            log_runtime(
                logger,
                logging.INFO,
                "研究链路进展已更新",
                step_id=str(step.id or ""),
                function_name=function_name,
                query_count=int(research_progress.get("query_count") or 0),
                candidate_url_count=int(research_progress.get("candidate_url_count") or 0),
                fetched_url_count=int(research_progress.get("fetched_url_count") or 0),
                candidate_domain_count=int(research_progress.get("candidate_domain_count") or 0),
                fetched_domain_count=int(research_progress.get("fetched_domain_count") or 0),
                coverage_score=float(research_progress.get("coverage_score") or 0.0),
                missing_signal_count=len(list(research_progress.get("missing_signals") or [])),
            )

    browser_tool_state = _extract_browser_tool_state(tool_result)
    if browser_tool_state["page_type"]:
        execution_state.browser_page_type = browser_tool_state["page_type"]
    if browser_tool_state["url"]:
        execution_state.last_browser_route_url = browser_tool_state["url"]
    if browser_tool_state["title"]:
        execution_state.last_browser_route_title = browser_tool_state["title"]
    if browser_tool_state["selector"]:
        execution_state.last_browser_route_selector = str(browser_tool_state["selector"])
    if browser_tool_state["index"] is not None:
        execution_state.last_browser_route_index = int(browser_tool_state["index"])

    if bool(tool_result.success):
        if normalized_function_name == "browser_read_current_page_structured":
            execution_state.browser_structured_ready = True
        elif normalized_function_name == "browser_extract_main_content":
            execution_state.browser_main_content_ready = True
        elif normalized_function_name == "browser_extract_cards":
            execution_state.browser_cards_ready = True
        elif normalized_function_name == "browser_find_link_by_text":
            execution_state.browser_link_match_ready = True
        elif normalized_function_name == "browser_find_actionable_elements":
            execution_state.browser_actionables_ready = True

    if normalized_function_name in BROWSER_PROGRESS_FUNCTIONS and bool(tool_result.success):
        browser_observation_fingerprint = _build_browser_observation_fingerprint(tool_result)
        if browser_observation_fingerprint == execution_state.last_browser_observation_fingerprint:
            execution_state.browser_no_progress_count += 1
        else:
            execution_state.browser_no_progress_count = 0
            execution_state.last_browser_observation_fingerprint = browser_observation_fingerprint
        if execution_state.browser_no_progress_count >= BROWSER_NO_PROGRESS_LIMIT:
            loop_break_reason = "browser_no_progress"

    return ToolEffectsResult(
        tool_result=tool_result,
        loop_break_reason=loop_break_reason,
    )


def extract_interrupt_request(tool_result: ToolResult) -> Optional[Dict[str, Any]]:
    interrupt_request = tool_result.data.get("interrupt") if isinstance(tool_result.data, dict) else None
    return interrupt_request if isinstance(interrupt_request, dict) and interrupt_request else None


def build_interrupt_payload(
    *,
    interrupt_request: Dict[str, Any],
    runtime_recent_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "success": True,
        "interrupt_request": interrupt_request,
        "summary": "",
        "result": "",
        "delivery_text": "",
        "attachments": [],
        "runtime_recent_action": runtime_recent_action or {},
    }


def reached_tool_failure_limit(execution_state: ExecutionState) -> bool:
    return execution_state.consecutive_failure_count >= TOOL_FAILURE_LIMIT


def _extract_search_result_urls(tool_result: ToolResult) -> List[str]:
    data = tool_result.data
    urls: List[str] = []

    if isinstance(data, SearchResults):
        for item in list(data.results or []):
            url = str(getattr(item, "url", "") or "").strip()
            if url:
                urls.append(url)
    elif isinstance(data, dict):
        raw_results = data.get("results") or []
        if isinstance(raw_results, list):
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if url:
                    urls.append(url)

    return _rank_and_limit_candidate_urls(urls, limit=8)


def _extract_search_query(tool_result: ToolResult, *, function_args: Dict[str, Any]) -> str:
    data = tool_result.data
    if isinstance(data, SearchResults):
        query = str(data.query or "").strip()
        if query:
            return query
    elif isinstance(data, dict):
        query = str(data.get("query") or "").strip()
        if query:
            return query
    return str(function_args.get("query") or "").strip()


def _extract_fetched_page_url(tool_result: ToolResult) -> str:
    data = tool_result.data
    if hasattr(data, "final_url") and hasattr(data, "url"):
        final_url = str(getattr(data, "final_url", "") or "").strip()
        if final_url:
            return final_url
        return str(getattr(data, "url", "") or "").strip()
    if isinstance(data, dict):
        final_url = str(data.get("final_url") or "").strip()
        if final_url:
            return final_url
        return str(data.get("url") or "").strip()
    return ""


def _rank_and_limit_candidate_urls(urls: List[str], *, limit: int) -> List[str]:
    deduped_urls: List[str] = []
    seen_urls: set[str] = set()
    for url in urls:
        normalized_url = normalize_url_value(url)
        if not normalized_url or normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        deduped_urls.append(normalized_url)
    if len(deduped_urls) <= 1:
        return deduped_urls[:limit]

    # 优先不同域名，降低单域结果污染带来的信息盲区。
    domain_first_urls: List[str] = []
    same_domain_urls: List[str] = []
    seen_domains: set[str] = set()
    for url in deduped_urls:
        domain = _extract_domain(url)
        if domain and domain not in seen_domains:
            seen_domains.add(domain)
            domain_first_urls.append(url)
        else:
            same_domain_urls.append(url)
    return (domain_first_urls + same_domain_urls)[:limit]


def _extract_domain(url: str) -> str:
    return extract_url_domain(url)


def _extract_domains(urls: List[str]) -> List[str]:
    domains: List[str] = []
    seen_domains: set[str] = set()
    for url in urls:
        domain = _extract_domain(url)
        if not domain or domain in seen_domains:
            continue
        seen_domains.add(domain)
        domains.append(domain)
    return domains


def _append_unique_text(target: List[str], value: str, *, max_items: int) -> None:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return
    if normalized_value in target:
        return
    target.append(normalized_value)
    if len(target) > max_items:
        del target[: len(target) - max_items]


def _estimate_research_coverage_score(state: ExecutionState) -> float:
    fetch_factor = min(float(state.research_fetch_success_count) / float(max(RESEARCH_MIN_FETCH_COUNT, 1)), 1.0)
    domain_factor = min(float(len(state.research_fetched_domains)) / float(max(RESEARCH_MIN_DOMAIN_COUNT, 1)), 1.0)
    candidate_factor = min(float(len(state.research_candidate_urls)) / 5.0, 1.0)
    query_factor = min(float(len(state.research_query_history)) / 2.0, 1.0)
    # 检索质量收口：覆盖度不仅看抓取与域名，也看查询是否充分展开。
    score = (0.45 * fetch_factor) + (0.30 * domain_factor) + (0.15 * candidate_factor) + (0.10 * query_factor)
    return round(score, 3)


def _build_research_progress_snapshot(state: ExecutionState) -> Dict[str, Any]:
    missing_signals: List[str] = []
    query_count = len(state.research_query_history)
    candidate_url_count = len(state.research_candidate_urls)
    fetched_url_count = len(state.research_fetched_urls)
    candidate_domain_count = len(state.research_candidate_domains)
    fetched_domain_count = len(state.research_fetched_domains)
    low_recall = candidate_url_count <= 1
    low_domain_diversity = candidate_domain_count <= 1 and query_count > 0

    if query_count == 0:
        missing_signals.append("先执行一次高质量搜索并返回候选链接")
    elif low_recall:
        missing_signals.append("候选链接过少，请改写关键词提高召回")
    elif low_domain_diversity:
        missing_signals.append("候选来源过于集中，请补充不同站点检索词")
    if state.research_fetch_success_count < RESEARCH_MIN_FETCH_COUNT:
        missing_signals.append(f"至少读取 {RESEARCH_MIN_FETCH_COUNT} 个来源页面正文")
    if fetched_domain_count < RESEARCH_MIN_DOMAIN_COUNT:
        missing_signals.append(f"至少覆盖 {RESEARCH_MIN_DOMAIN_COUNT} 个不同站点来源")
    if candidate_url_count == 0:
        missing_signals.append("补充可抓取候选链接")
    return {
        "query_count": query_count,
        "candidate_url_count": candidate_url_count,
        "fetched_url_count": fetched_url_count,
        "candidate_domain_count": candidate_domain_count,
        "fetched_domain_count": fetched_domain_count,
        "fetch_success_count": int(state.research_fetch_success_count),
        "total_search_result_count": int(state.research_total_search_results),
        "coverage_score": float(state.research_coverage_score),
        "is_low_recall": low_recall,
        "is_low_domain_diversity": low_domain_diversity,
        "missing_signals": missing_signals,
        "latest_query": str(state.research_query_history[-1] if state.research_query_history else ""),
        "latest_fetched_url": str(state.research_fetched_urls[-1] if state.research_fetched_urls else ""),
    }
