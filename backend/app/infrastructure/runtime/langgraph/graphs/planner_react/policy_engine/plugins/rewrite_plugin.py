#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：工具改写策略插件（研究链路）。"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from app.domain.models import Step
from app.domain.services.runtime.normalizers import (
    extract_url_domain,
    normalize_url_value,
)
from ...research_url_extractor import extract_explicit_url_from_research_context
from ...execution_context import ExecutionContext
from ...execution_state import ExecutionState
from ...tool_events import ToolCallLifecycle


@dataclass(slots=True)
class RewriteDecision:
    """工具改写决策结果。"""

    lifecycle: ToolCallLifecycle
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_fetch_dedupe_key(url: Any) -> str:
    return normalize_url_value(url, drop_query=True)


def _collect_pending_candidate_urls_for_research(execution_state: ExecutionState) -> List[str]:
    fetched_keys = {
        _normalize_fetch_dedupe_key(url)
        for url in list(execution_state.research_fetched_urls or [])
        if _normalize_fetch_dedupe_key(url)
    }
    failed_keys = set(list(execution_state.research_failed_fetch_url_keys or set()))
    pending_urls: List[str] = []
    seen_pending_keys: set[str] = set()
    for raw_url in list(execution_state.research_candidate_urls or []):
        normalized_url = normalize_url_value(raw_url)
        pending_key = _normalize_fetch_dedupe_key(normalized_url)
        if not normalized_url or not pending_key:
            continue
        if pending_key in fetched_keys or pending_key in failed_keys or pending_key in seen_pending_keys:
            continue
        seen_pending_keys.add(pending_key)
        pending_urls.append(normalized_url)
    return pending_urls


def _pick_research_fetch_url_for_rewrite(
        *,
        step: Step,
        execution_context: ExecutionContext,
        execution_state: ExecutionState,
) -> tuple[str, str]:
    """返回 (rewrite_url, source)。source: candidate | explicit"""
    pending_candidate_urls = _collect_pending_candidate_urls_for_research(execution_state)
    fetched_domains = set(list(execution_state.research_fetched_domains or []))
    if len(pending_candidate_urls) > 0:
        for url in pending_candidate_urls:
            domain = extract_url_domain(url)
            if domain and domain not in fetched_domains:
                return url, "candidate"
        return pending_candidate_urls[0], "candidate"

    explicit_url = ""
    if bool(getattr(execution_context, "research_has_explicit_url", False)):
        explicit_url = extract_explicit_url_from_research_context(step=step, ctx=execution_context)
    explicit_key = _normalize_fetch_dedupe_key(explicit_url)
    explicit_blacklisted = bool(
        explicit_key and explicit_key in set(list(execution_state.research_failed_fetch_url_keys or set()))
    )
    if explicit_url and not explicit_blacklisted:
        return explicit_url, "explicit"

    return "", ""


def run_rewrite_plugin(
        *,
        lifecycle: ToolCallLifecycle,
        execution_context: ExecutionContext,
        execution_state: ExecutionState,
        step: Step,
) -> RewriteDecision:
    if not bool(getattr(execution_context, "research_route_enabled", False)):
        return RewriteDecision(lifecycle=lifecycle)
    if str(getattr(lifecycle, "normalized_function_name", "") or "").strip().lower() != "search_web":
        return RewriteDecision(lifecycle=lifecycle)
    if not bool(getattr(execution_context, "research_has_explicit_url", False)) and not bool(
            execution_state.research_search_ready):
        return RewriteDecision(lifecycle=lifecycle)

    # P3-一次性收口：连续 fetch 失败后，优先恢复为 search 扩召回，不再盲目 search->fetch 重写。
    if execution_state.consecutive_fetch_failure_count >= 2 and execution_state.research_search_ready:
        return RewriteDecision(
            lifecycle=lifecycle,
            reason="research_search_recovery_after_fetch_failures",
            metadata={
                "rewrite_from": "search_web",
                "rewrite_to": "search_web",
                "skipped_rewrite": True,
                "consecutive_fetch_failure_count": execution_state.consecutive_fetch_failure_count,
                "failed_fetch_url_count": len(list(execution_state.research_failed_fetch_url_keys or [])),
            },
        )

    rewrite_url, rewrite_source = _pick_research_fetch_url_for_rewrite(
        step=step,
        execution_context=execution_context,
        execution_state=execution_state,
    )
    if not rewrite_url:
        return RewriteDecision(lifecycle=lifecycle)

    previous_args = dict(getattr(lifecycle, "function_args", {}) or {})
    lifecycle.function_name = "fetch_page"
    lifecycle.normalized_function_name = "fetch_page"
    lifecycle.function_args = {"url": rewrite_url}
    explicit_url = extract_explicit_url_from_research_context(step=step, ctx=execution_context)
    explicit_url_key = _normalize_fetch_dedupe_key(explicit_url)
    explicit_url_blacklisted = bool(
        explicit_url_key and explicit_url_key in set(list(execution_state.research_failed_fetch_url_keys or set()))
    )
    pending_candidates = _collect_pending_candidate_urls_for_research(execution_state)
    return RewriteDecision(
        lifecycle=lifecycle,
        reason="research_search_to_fetch_rewrite",
        metadata={
            "rewrite_from": "search_web",
            "rewrite_to": "fetch_page",
            "rewrite_url": rewrite_url,
            "rewrite_source": rewrite_source,
            "had_explicit_url": bool(getattr(execution_context, "research_has_explicit_url", False)),
            "explicit_url_blacklisted": explicit_url_blacklisted,
            "failed_fetch_url_count": len(list(execution_state.research_failed_fetch_url_keys or [])),
            "previous_arg_keys": sorted(previous_args.keys()),
            "pending_candidate_count": len(pending_candidates),
            "skipped_due_to_blacklist": bool(rewrite_source == "candidate" and explicit_url_blacklisted),
            "fetched_domain_count": len(list(execution_state.research_fetched_domains or [])),
        },
    )
