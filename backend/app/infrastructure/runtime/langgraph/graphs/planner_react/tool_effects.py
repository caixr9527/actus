#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具结果状态归并（reducer）。"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.domain.models import SearchResults, Step, ToolResult
from app.domain.services.runtime.contracts.langgraph_settings import (
    BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
    BROWSER_NO_PROGRESS_LIMIT,
    RESEARCH_MIN_DOMAIN_COUNT,
    RESEARCH_MIN_FETCH_COUNT,
    RESEARCH_PROGRESS_MAX_TRACK_ITEMS,
    BROWSER_PROGRESS_FUNCTIONS,
    TOOL_FAILURE_LIMIT,
)
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.normalizers import extract_url_domain, normalize_url_value
from app.domain.services.workspace_runtime.policies import (
    attach_browser_degrade_payload as _attach_browser_degrade_payload,
    build_search_fingerprint as _build_search_fingerprint,
    build_tool_feedback_content as _build_tool_feedback_content,
    build_browser_high_level_failure_key as _build_browser_high_level_failure_key,
    build_tool_fingerprint as _build_tool_fingerprint,
    build_browser_observation_fingerprint as _build_browser_observation_fingerprint,
    build_recent_blocked_tool_call as _build_recent_blocked_tool_call,
    build_recent_failed_action as _build_recent_failed_action,
    extract_browser_tool_state as _extract_browser_tool_state,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.research_diagnosis import build_research_diagnosis
from app.infrastructure.runtime.langgraph.graphs.planner_react.research_diagnosis import evaluate_fetch_result_quality
from app.infrastructure.runtime.langgraph.graphs.planner_react.research_snippet_policy import (
    evaluate_search_snippet_sufficiency,
    extract_search_evidence_items,
)
from .constraint_engine.reason_codes import (
    REASON_BROWSER_CLICK_TARGET_BLOCKED,
    REASON_BROWSER_HIGH_LEVEL_RETRY_BLOCKED,
    REASON_RESEARCH_ROUTE_CROSS_DOMAIN_FETCH_LIMIT,
    REASON_RESEARCH_SEARCH_TO_FETCH_REWRITE,
)
from .execution_context import ExecutionContext
from .execution_state import ExecutionState


@dataclass(slots=True)
class ToolEffectsResult:
    """effects 域输出。

    业务含义：
    - `tool_result` 是更新完降级信息、recent_action、研究/浏览器进展后的最终结果；
    - `loop_break_reason` 可能在 effects 中被补写或重写，用于后续 convergence 统一判断。
    """
    tool_result: ToolResult
    loop_break_reason: str


def apply_rewrite_effects(
        *,
        rewrite_reason: str,
        rewrite_metadata: Dict[str, Any],
        execution_state: ExecutionState,
) -> None:
    """把 rewrite 产生的状态写回统一收口到 effects 域。

    当前只处理“显式 URL 的 search -> fetch 改写”：
    - 标记该 explicit URL 已被消费过；
    - 避免同一步骤内对同一个 explicit URL 反复触发 rewrite。
    """
    if str(rewrite_reason or "") != REASON_RESEARCH_SEARCH_TO_FETCH_REWRITE:
        return
    rewrite_source = str((rewrite_metadata or {}).get("rewrite_source") or "")
    if rewrite_source != "explicit":
        return
    rewrite_url_key = _normalize_fetch_dedupe_key((rewrite_metadata or {}).get("rewrite_url"))
    if rewrite_url_key:
        execution_state.research_explicit_rewrite_url_keys.add(rewrite_url_key)


def apply_tool_preinvoke_effects(
        *,
        normalized_function_name: str,
        function_args: Dict[str, Any],
        execution_state: ExecutionState,
) -> None:
    """在真实执行前统一入账本轮调用计数。

    业务含义：
    - 只对最终执行目标入账一次；
    - 更新通用同工具重复计数，以及 research 场景的 search/fetch 细粒度重复计数；
    - 为 repeat_loop_policy 与后续日志提供稳定输入。
    """
    tool_fingerprint = _build_tool_fingerprint(
        str(normalized_function_name or "").strip().lower(),
        dict(function_args or {}),
    )
    if tool_fingerprint == execution_state.last_tool_fingerprint:
        execution_state.same_tool_repeat_count += 1
    else:
        execution_state.same_tool_repeat_count = 1
        execution_state.last_tool_fingerprint = tool_fingerprint

    normalized_name = str(normalized_function_name or "").strip().lower()
    if normalized_name == "search_web":
        execution_state.search_invocation_count += 1
        search_fingerprint = _build_search_fingerprint(function_args)
        execution_state.search_repeat_counter[search_fingerprint] = (
                execution_state.search_repeat_counter.get(search_fingerprint, 0) + 1
        )
    elif normalized_name == "fetch_page":
        fetch_fingerprint = _normalize_fetch_dedupe_key(function_args.get("url"))
        if fetch_fingerprint:
            execution_state.fetch_repeat_counter[fetch_fingerprint] = (
                    execution_state.fetch_repeat_counter.get(fetch_fingerprint, 0) + 1
            )


def apply_tool_result_effects(
        *,
        logger: logging.Logger,
        step: Step,
        function_name: str,
        normalized_function_name: str,
        function_args: Dict[str, Any],
        tool_result: ToolResult,
        loop_break_reason: str,
        guard_reason_code: str = "",
        browser_route_state_key: str,
        execution_context: ExecutionContext,
        execution_state: ExecutionState,
        tool_executed: bool = True,
) -> ToolEffectsResult:
    """把工具执行后的状态变化统一写回 `ExecutionState`。

    实现语义：
    - success 路径：更新最近成功调用、研究/浏览器进展与覆盖度；
    - failure 路径：沉淀 recent_action、失败计数、黑名单与降级信息；
    - 这是 executor 与 convergence 之间唯一的状态归并入口。
    """
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
        effective_block_reason = str(loop_break_reason or "").strip()
        # soft block 不会携带 loop_break_reason，约束 reason_code 通过 guard_reason_code 透传。
        if not effective_block_reason and (not tool_executed):
            effective_block_reason = str(guard_reason_code or "").strip()
        if effective_block_reason == REASON_RESEARCH_ROUTE_CROSS_DOMAIN_FETCH_LIMIT:
            # 约束层只返回信号，累计计数在 effects 域统一持久化。
            next_repeat_count = execution_state.research_cross_domain_repeat_blocks + 1
            if isinstance(tool_result.data, dict):
                raw_repeat_count = tool_result.data.get("cross_domain_repeat_block_count")
                if isinstance(raw_repeat_count, int) and raw_repeat_count >= 1:
                    next_repeat_count = raw_repeat_count
                elif isinstance(raw_repeat_count, str) and raw_repeat_count.strip().isdigit():
                    parsed_repeat_count = int(raw_repeat_count.strip())
                    if parsed_repeat_count >= 1:
                        next_repeat_count = parsed_repeat_count
            execution_state.research_cross_domain_repeat_blocks = next_repeat_count
            log_runtime(
                logger,
                logging.INFO,
                "研究链路跨域重复阻断计数已更新",
                step_id=str(step.id or ""),
                function_name=function_name,
                reason_code=effective_block_reason,
                repeat_count=execution_state.research_cross_domain_repeat_blocks,
            )
        execution_state.runtime_recent_action["last_failed_action"] = _build_recent_failed_action(
            function_name=function_name,
            tool_result=tool_result,
        )
        if effective_block_reason in {REASON_BROWSER_CLICK_TARGET_BLOCKED, REASON_BROWSER_HIGH_LEVEL_RETRY_BLOCKED}:
            execution_state.runtime_recent_action["last_blocked_tool_call"] = {
                "function_name": str(function_name or "").strip(),
                "reason": effective_block_reason,
                "message": str(tool_result.message or "").strip()[:160],
            }
        blocked_tool_call = _build_recent_blocked_tool_call(
            function_name=function_name,
            tool_result=tool_result,
            loop_break_reason=effective_block_reason,
        )
        if blocked_tool_call:
            execution_state.runtime_recent_action["last_blocked_tool_call"] = blocked_tool_call
        execution_state.consecutive_failure_count += 1
        if execution_context.research_route_enabled and normalized_function_name == "fetch_page":
            # 仅“真实执行 fetch_page 失败”才累计 fetch 失败计数。
            if tool_executed:
                execution_state.consecutive_fetch_failure_count += 1
            # P3-一次性收口：fetch_page 失败 URL 进入黑名单，仅在真实执行失败时写入，避免 guard 误污染。
            if tool_executed:
                failed_url_key = _normalize_fetch_dedupe_key(function_args.get("url"))
                if failed_url_key:
                    execution_state.research_failed_fetch_url_keys.add(failed_url_key)
        if (
                normalized_function_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES
                and effective_block_reason != REASON_BROWSER_HIGH_LEVEL_RETRY_BLOCKED
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

    if execution_context.research_route_enabled and normalized_function_name == "search_web" and bool(
            tool_result.success):
        # P3-一次性收口：search 成功扩召回后应重置 fetch 连续失败计数，
        # 避免因历史失败残留导致后续长期停留在 search-only 循环。
        execution_state.consecutive_fetch_failure_count = 0
        extracted_query = _extract_search_query(tool_result, function_args=function_args)
        if extracted_query:
            _append_unique_text(
                execution_state.research_query_history,
                extracted_query,
                max_items=RESEARCH_PROGRESS_MAX_TRACK_ITEMS,
            )
        candidate_urls = _extract_search_result_urls(tool_result)
        search_evidence_items = extract_search_evidence_items(tool_result_data=tool_result.data)
        snippet_quality = evaluate_search_snippet_sufficiency(
            step=step,
            evidence_items=search_evidence_items,
        )
        execution_state.research_candidate_urls = candidate_urls
        execution_state.research_search_evidence_items = list(search_evidence_items or [])
        execution_state.research_search_ready = len(candidate_urls) > 0
        execution_state.research_total_search_results += len(candidate_urls)
        execution_state.research_snippet_sufficient = bool(snippet_quality.get("snippet_sufficient"))
        execution_state.research_recommended_fetch_urls = list(snippet_quality.get("recommended_fetch_urls") or [])
        execution_state.last_search_evidence_quality = {
            "snippet_count": len(list(search_evidence_items or [])),
            "useful_snippet_count": int(snippet_quality.get("useful_snippet_count") or 0),
            "distinct_domain_count": int(snippet_quality.get("distinct_domain_count") or 0),
            "need_fetch": bool(snippet_quality.get("need_fetch")),
            "reason_code": str(snippet_quality.get("reason_code") or ""),
        }
        for domain in _extract_domains(candidate_urls):
            _append_unique_text(
                execution_state.research_candidate_domains,
                domain,
                max_items=RESEARCH_PROGRESS_MAX_TRACK_ITEMS,
            )
    elif execution_context.research_route_enabled and normalized_function_name == "fetch_page" and bool(
            tool_result.success):
        execution_state.last_fetch_quality = evaluate_fetch_result_quality(
            fetched_result=tool_result.data,
            fallback_url=str(function_args.get("url") or ""),
        )
        if not bool(execution_state.last_fetch_quality.get("is_useful")):
            failed_url_key = _normalize_fetch_dedupe_key(function_args.get("url"))
            if failed_url_key:
                execution_state.research_failed_fetch_url_keys.add(failed_url_key)
            execution_state.runtime_recent_action["research_diagnosis"] = build_research_diagnosis(
                state=execution_state,
            )
            log_runtime(
                logger,
                logging.INFO,
                "页面抓取成功但正文价值不足",
                step_id=str(step.id or ""),
                function_name=function_name,
                diagnosis_code=str(execution_state.runtime_recent_action["research_diagnosis"].get("code") or ""),
                low_value_reason=str(execution_state.last_fetch_quality.get("low_value_reason") or ""),
                content_length=int(execution_state.last_fetch_quality.get("content_length") or 0),
            )
            return ToolEffectsResult(
                tool_result=ToolResult(
                    success=False,
                    message="页面已读取，但正文价值不足，未拿到可用于当前研究步骤的有效信息。",
                    data={
                        "research_diagnosis": dict(execution_state.runtime_recent_action["research_diagnosis"]),
                    },
                ),
                loop_break_reason=loop_break_reason,
            )
        execution_state.research_fetch_completed = True
        execution_state.research_fetch_success_count += 1
        execution_state.consecutive_fetch_failure_count = 0
        execution_state.research_cross_domain_repeat_blocks = 0
        fetched_url = _extract_fetched_page_url(tool_result)
        # 抓取成功后，若此前误入失败黑名单，立即清理对应 key。
        successful_url_key = _normalize_fetch_dedupe_key(fetched_url or function_args.get("url"))
        if successful_url_key and successful_url_key in execution_state.research_failed_fetch_url_keys:
            execution_state.research_failed_fetch_url_keys.discard(successful_url_key)
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
        execution_state.runtime_recent_action["research_diagnosis"] = build_research_diagnosis(
            state=execution_state,
        )
        if normalized_function_name in {"search_web", "fetch_page"}:
            research_progress = dict(execution_state.runtime_recent_action.get("research_progress") or {})
            research_diagnosis = dict(execution_state.runtime_recent_action.get("research_diagnosis") or {})
            log_runtime(
                logger,
                logging.INFO,
                "研究链路进展已更新",
                step_id=str(step.id or ""),
                function_name=function_name,
                query_count=int(research_progress.get("query_count") or 0),
                candidate_url_count=int(research_progress.get("candidate_url_count") or 0),
                useful_snippet_count=int(execution_state.last_search_evidence_quality.get("useful_snippet_count") or 0),
                fetched_url_count=int(research_progress.get("fetched_url_count") or 0),
                candidate_domain_count=int(research_progress.get("candidate_domain_count") or 0),
                fetched_domain_count=int(research_progress.get("fetched_domain_count") or 0),
                coverage_score=float(research_progress.get("coverage_score") or 0.0),
                missing_signal_count=len(list(research_progress.get("missing_signals") or [])),
                diagnosis_code=str(research_diagnosis.get("code") or ""),
            )
            log_runtime(
                logger,
                logging.INFO,
                "研究链路状态已写回执行态",
                step_id=str(step.id or ""),
                function_name=function_name,
                success=bool(tool_result.success),
                research_search_ready=bool(execution_state.research_search_ready),
                research_snippet_sufficient=bool(execution_state.research_snippet_sufficient),
                research_fetch_completed=bool(execution_state.research_fetch_completed),
                same_tool_repeat_count=execution_state.same_tool_repeat_count,
                diagnosis_code=str(research_diagnosis.get("code") or ""),
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


def _normalize_fetch_dedupe_key(url: Any) -> str:
    return normalize_url_value(url, drop_query=True)


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
        missing_signals.append("候选链接过少，请改写主题描述提高召回")
    elif low_domain_diversity:
        missing_signals.append("候选来源过于集中，请补充不同站点的主题描述")
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
