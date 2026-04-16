#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：工具改写策略插件（研究链路）。"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from app.domain.models import Step
from app.domain.services.runtime.normalizers import (
    extract_url_domain,
    normalize_url_value,
)
from app.domain.services.workspace_runtime.policies import (
    build_step_candidate_text as _build_step_candidate_text,
)
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
    pending_urls: List[str] = []
    seen_pending_keys: set[str] = set()
    for raw_url in list(execution_state.research_candidate_urls or []):
        normalized_url = normalize_url_value(raw_url)
        pending_key = _normalize_fetch_dedupe_key(normalized_url)
        if not normalized_url or not pending_key:
            continue
        if pending_key in fetched_keys or pending_key in seen_pending_keys:
            continue
        seen_pending_keys.add(pending_key)
        pending_urls.append(normalized_url)
    return pending_urls


def _extract_explicit_url_from_text(step: Step, execution_context: ExecutionContext) -> str:
    step_text = _build_step_candidate_text(step)
    step_url_match = re.search(r"https?://[^\s)\]>\"']+", str(step_text or ""), re.IGNORECASE)
    if step_url_match:
        return normalize_url_value(step_url_match.group(0))
    candidate_text = "\n".join(
        [
            str((item or {}).get("text") or "").strip()
            for item in list(getattr(execution_context, "normalized_user_content", []) or [])
            if isinstance(item, dict) and str((item or {}).get("type") or "").strip().lower() == "text"
        ]
    )
    url_match = re.search(r"https?://[^\s)\]>\"']+", candidate_text, re.IGNORECASE)
    if url_match:
        return normalize_url_value(url_match.group(0))
    return ""


def _pick_research_fetch_url_for_rewrite(
        *,
        step: Step,
        execution_context: ExecutionContext,
        execution_state: ExecutionState,
) -> str:
    explicit_url = ""
    if bool(getattr(execution_context, "research_has_explicit_url", False)):
        explicit_url = _extract_explicit_url_from_text(step, execution_context)
    if explicit_url:
        return explicit_url

    pending_candidate_urls = _collect_pending_candidate_urls_for_research(execution_state)
    if len(pending_candidate_urls) == 0:
        return ""
    fetched_domains = set(list(execution_state.research_fetched_domains or []))
    for url in pending_candidate_urls:
        domain = extract_url_domain(url)
        if domain and domain not in fetched_domains:
            return url
    return pending_candidate_urls[0]


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

    rewrite_url = _pick_research_fetch_url_for_rewrite(
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
    return RewriteDecision(
        lifecycle=lifecycle,
        reason="research_search_to_fetch_rewrite",
        metadata={
            "rewrite_from": "search_web",
            "rewrite_to": "fetch_page",
            "rewrite_url": rewrite_url,
            "had_explicit_url": bool(getattr(execution_context, "research_has_explicit_url", False)),
            "previous_arg_keys": sorted(previous_args.keys()),
            "pending_candidate_count": len(_collect_pending_candidate_urls_for_research(execution_state)),
            "fetched_domain_count": len(list(execution_state.research_fetched_domains or [])),
        },
    )
