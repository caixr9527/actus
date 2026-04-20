#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束策略：research/web_reading 路由约束。"""

from __future__ import annotations

import re
from typing import Any, Optional

from app.domain.services.runtime.contracts.langgraph_settings import RESEARCH_CROSS_DOMAIN_REPEAT_BLOCK_LIMIT
from app.domain.services.runtime.contracts.langgraph_settings import RESEARCH_MIN_DOMAIN_COUNT
from app.domain.services.runtime.contracts.langgraph_settings import RESEARCH_MIN_FETCH_COUNT
from app.domain.services.runtime.normalizers import extract_url_domain, normalize_url_value
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintDecision
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintInput
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintToolResultPayload
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_RESEARCH_QUERY_STYLE_BLOCKED
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_RESEARCH_ROUTE_CROSS_DOMAIN_FETCH_LIMIT
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_RESEARCH_ROUTE_FETCH_REQUIRED
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_RESEARCH_ROUTE_SEARCH_REQUIRED
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_RESEARCH_SEARCH_TO_FETCH_REWRITE
from app.infrastructure.runtime.langgraph.graphs.planner_react.research_intent_policy import is_explicit_single_page_fetch_intent
from app.infrastructure.runtime.langgraph.graphs.planner_react.research_url_extractor import extract_explicit_url_from_research_context

_SEARCH_QUERY_CJK_KEYWORD_STACK_PATTERN = re.compile(
    r"^[\u4e00-\u9fffA-Za-z0-9]{1,8}(?:\s+[\u4e00-\u9fffA-Za-z0-9]{1,8}){4,}$"
)
_SEARCH_QUERY_CJK_COMPACT_KEYWORD_STACK_PATTERN = re.compile(
    r"^(?=.{12,}$)(?=(?:.*[\u4e00-\u9fff]){4,})(?=(?:.*[A-Za-z]){2,})[\u4e00-\u9fffA-Za-z0-9]+$"
)
_SEARCH_QUERY_NATURAL_LANGUAGE_HINT_PATTERN = re.compile(
    r"(的|了|吗|如何|怎么|哪些|是什么|以及|并且|并|与|及其|是否|请|关于|where|what|which|how|why|when|who|that)",
    re.IGNORECASE,
)
_SEARCH_QUERY_EN_STOPWORD_PATTERN = re.compile(
    r"\b(and|or|with|for|to|from|in|on|at|by|about|vs|versus)\b",
    re.IGNORECASE,
)


def evaluate_research_route_policy(constraint_input: ConstraintInput) -> Optional[ConstraintDecision]:
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    function_name = str(constraint_input.function_name or "").strip()
    function_args = dict(constraint_input.function_args or {})
    ctx = constraint_input.execution_context
    state = constraint_input.execution_state
    step = constraint_input.step

    if not bool(ctx.research_route_enabled):
        return None

    if normalized_function_name == "search_web" and _is_keyword_stacked_search_query(function_args.get("query")):
        return _hard_block(
            reason_code=REASON_RESEARCH_QUERY_STYLE_BLOCKED,
            message=(
                "search_web 的 query 必须使用单主题自然语言描述，禁止关键词堆叠。"
                " 请改为一句完整的主题描述，例如“主流 AI 编程助手及其支持的 IDE”。"
            ),
        )

    if (
            normalized_function_name == "fetch_page"
            and not ctx.research_has_explicit_url
            and not state.research_search_ready
    ):
        return _hard_block(
            reason_code=REASON_RESEARCH_ROUTE_SEARCH_REQUIRED,
            message="当前步骤属于检索/网页阅读任务，请先调用 search_web 获取候选链接，再使用 fetch_page 读取正文。",
        )

    if (
            normalized_function_name == "search_web"
            and ctx.research_has_explicit_url
            and not state.research_fetch_completed
    ):
        # P3-一次性收口：显式 URL 单页抓取场景统一走 rewrite，不在这里提前 block。
        # 否则会出现“事件/消息展示的是 search_web，但真实想要执行的是 fetch_page”的双义语义。
        explicit_url_blacklisted = _is_explicit_url_blacklisted(step=step, constraint_input=constraint_input)
        explicit_url = extract_explicit_url_from_research_context(step=step, ctx=ctx)
        explicit_url_key = _normalize_fetch_dedupe_key(explicit_url)
        explicit_url_already_rewritten = bool(
            explicit_url_key and explicit_url_key in set(list(state.research_explicit_rewrite_url_keys or set()))
        )
        explicit_single_page_fetch_intent = is_explicit_single_page_fetch_intent(
            step,
            explicit_url=explicit_url,
        )
        if (
                explicit_single_page_fetch_intent
                and (
                explicit_url_blacklisted
                or explicit_url_already_rewritten
                or state.consecutive_fetch_failure_count >= 2
        )
        ):
            return _hard_block(
                reason_code=REASON_RESEARCH_ROUTE_FETCH_REQUIRED,
                message="当前步骤已提供明确 URL，请直接调用 fetch_page 读取页面正文，不要先重复搜索。",
            )

    if (
            normalized_function_name == "search_web"
            and state.research_search_ready
            and not ctx.research_has_explicit_url
    ):
        # P3-一次性收口：普通 research 场景是否需要从 snippet 升级到 fetch，
        # 由 search evidence/snippet sufficiency 层决定，不再由执行约束层强制改写。
        return None

    if (
            normalized_function_name == "fetch_page"
            and not ctx.research_has_explicit_url
            and len(state.research_fetched_domains) < RESEARCH_MIN_DOMAIN_COUNT
    ):
        requested_url = normalize_url_value(function_args.get("url"))
        requested_domain = _extract_domain(requested_url)
        fetched_domains = set(state.research_fetched_domains)
        cross_domain_candidate = _pick_pending_cross_domain_candidate(
            constraint_input=constraint_input,
            excluded_domains=fetched_domains,
        )
        if requested_domain and requested_domain in fetched_domains and cross_domain_candidate:
            repeat_blocks = int(state.research_cross_domain_repeat_blocks) + 1
            should_break = repeat_blocks > RESEARCH_CROSS_DOMAIN_REPEAT_BLOCK_LIMIT
            return ConstraintDecision(
                action="block",
                reason_code=REASON_RESEARCH_ROUTE_CROSS_DOMAIN_FETCH_LIMIT,
                block_mode="hard_block_break" if should_break else "soft_block_continue",
                loop_break_reason=REASON_RESEARCH_ROUTE_CROSS_DOMAIN_FETCH_LIMIT if should_break else "",
                tool_result_payload=ConstraintToolResultPayload(
                    success=False,
                    message=(
                        "当前研究步骤尚未覆盖足够来源，请优先抓取不同站点候选链接，避免重复读取同域页面。"
                        f" 建议先读取: {cross_domain_candidate}"
                    ),
                    data={
                        "next_cross_domain_url": cross_domain_candidate,
                        "cross_domain_repeat_block_count": repeat_blocks,
                    },
                ),
                message_for_model=(
                    "当前研究步骤尚未覆盖足够来源，请优先抓取不同站点候选链接，避免重复读取同域页面。"
                    f" 建议先读取: {cross_domain_candidate}"
                ),
            )

    return None


def build_research_route_rewrite_decision(constraint_input: ConstraintInput) -> Optional[ConstraintDecision]:
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    ctx = constraint_input.execution_context
    state = constraint_input.execution_state
    step = constraint_input.step
    if not bool(getattr(ctx, "research_route_enabled", False)):
        return None
    if normalized_function_name != "search_web":
        return None
    if not bool(getattr(ctx, "research_has_explicit_url", False)):
        return None
    if state.consecutive_fetch_failure_count >= 2:
        return None
    rewrite_url, rewrite_source = _pick_research_fetch_url_for_rewrite(constraint_input=constraint_input)
    if not rewrite_url:
        return None
    explicit_url = extract_explicit_url_from_research_context(step=step, ctx=ctx)
    explicit_url_key = _normalize_fetch_dedupe_key(explicit_url)
    explicit_url_blacklisted = bool(
        explicit_url_key and explicit_url_key in set(list(state.research_failed_fetch_url_keys or set()))
    )
    explicit_url_rewritten = bool(
        explicit_url_key and explicit_url_key in set(list(state.research_explicit_rewrite_url_keys or set()))
    )
    pending_candidates = _collect_pending_candidate_urls(constraint_input=constraint_input)
    return ConstraintDecision(
        action="rewrite",
        reason_code=REASON_RESEARCH_SEARCH_TO_FETCH_REWRITE,
        rewrite_target={
            "function_name": "fetch_page",
            "normalized_function_name": "fetch_page",
            "function_args": {"url": rewrite_url},
        },
        metadata={
            "rewrite_from": "search_web",
            "rewrite_to": "fetch_page",
            "rewrite_url": rewrite_url,
            "rewrite_source": rewrite_source,
            "had_explicit_url": bool(getattr(ctx, "research_has_explicit_url", False)),
            "explicit_url_blacklisted": explicit_url_blacklisted,
            "explicit_url_rewritten": explicit_url_rewritten,
            "failed_fetch_url_count": len(list(state.research_failed_fetch_url_keys or [])),
            "previous_arg_keys": sorted(dict(constraint_input.function_args or {}).keys()),
            "pending_candidate_count": len(pending_candidates),
            "skipped_due_to_blacklist": bool(rewrite_source == "candidate" and explicit_url_blacklisted),
            "fetched_domain_count": len(list(state.research_fetched_domains or [])),
        },
        message_for_model="研究链路已将 search_web 改写为 fetch_page 以继续正文抓取。",
    )


def _collect_pending_candidate_urls(*, constraint_input: ConstraintInput) -> list[str]:
    state = constraint_input.execution_state
    fetched_url_set = {
        _normalize_fetch_dedupe_key(url)
        for url in list(state.research_fetched_urls or [])
        if _normalize_fetch_dedupe_key(url)
    }
    failed_url_set = set(list(state.research_failed_fetch_url_keys or set()))
    pending: list[str] = []
    seen_pending_keys: set[str] = set()
    for raw_url in list(state.research_candidate_urls or []):
        normalized = normalize_url_value(raw_url)
        pending_key = _normalize_fetch_dedupe_key(normalized)
        if (
                not normalized
                or not pending_key
                or pending_key in fetched_url_set
                or pending_key in failed_url_set
                or pending_key in seen_pending_keys
        ):
            continue
        seen_pending_keys.add(pending_key)
        pending.append(normalized)
    return pending


def _pick_pending_cross_domain_candidate(*, constraint_input: ConstraintInput, excluded_domains: set[str]) -> str:
    pending_urls = _collect_pending_candidate_urls(constraint_input=constraint_input)
    for url in pending_urls:
        domain = _extract_domain(url)
        if domain and domain not in excluded_domains:
            return url
    return ""


def _pick_research_fetch_url_for_rewrite(*, constraint_input: ConstraintInput) -> tuple[str, str]:
    ctx = constraint_input.execution_context
    state = constraint_input.execution_state
    step = constraint_input.step
    explicit_url = ""
    if bool(getattr(ctx, "research_has_explicit_url", False)):
        explicit_url = extract_explicit_url_from_research_context(step=step, ctx=ctx)
    if not is_explicit_single_page_fetch_intent(step, explicit_url=explicit_url):
        return "", ""
    explicit_key = _normalize_fetch_dedupe_key(explicit_url)
    explicit_blacklisted = bool(
        explicit_key and explicit_key in set(list(state.research_failed_fetch_url_keys or set()))
    )
    explicit_already_rewritten = bool(
        explicit_key and explicit_key in set(list(state.research_explicit_rewrite_url_keys or set()))
    )
    if explicit_url and (not explicit_blacklisted) and (not explicit_already_rewritten):
        return explicit_url, "explicit"
    return "", ""


def _normalize_fetch_dedupe_key(url: Any) -> str:
    return normalize_url_value(url, drop_query=True)


def normalize_research_fetch_dedupe_key(url: Any) -> str:
    """导出统一 dedupe key，供约束引擎记录 explicit rewrite 消费状态。"""
    return _normalize_fetch_dedupe_key(url)


def _extract_domain(url: str) -> str:
    return extract_url_domain(url)


def _is_explicit_url_blacklisted(*, step: Any, constraint_input: ConstraintInput) -> bool:
    explicit_url = extract_explicit_url_from_research_context(
        step=step,
        ctx=constraint_input.execution_context,
    )
    explicit_key = _normalize_fetch_dedupe_key(explicit_url)
    if not explicit_key:
        return False
    return explicit_key in set(list(constraint_input.execution_state.research_failed_fetch_url_keys or set()))


def _hard_block(*, reason_code: str, message: str) -> ConstraintDecision:
    return ConstraintDecision(
        action="block",
        reason_code=reason_code,
        block_mode="hard_block_break",
        loop_break_reason=reason_code,
        tool_result_payload=ConstraintToolResultPayload(success=False, message=message),
        message_for_model=message,
    )


def _is_keyword_stacked_search_query(raw_query: Any) -> bool:
    query = str(raw_query or "").strip()
    if not query:
        return False
    normalized = " ".join(query.split())
    if len(normalized) < 12:
        return False
    if "http://" in normalized or "https://" in normalized:
        return False
    if _SEARCH_QUERY_NATURAL_LANGUAGE_HINT_PATTERN.search(normalized):
        return False
    if _SEARCH_QUERY_CJK_COMPACT_KEYWORD_STACK_PATTERN.fullmatch(normalized):
        return True
    if _SEARCH_QUERY_EN_STOPWORD_PATTERN.search(normalized):
        english_tokens = [token for token in re.findall(r"[A-Za-z]+", normalized) if token]
        if len(english_tokens) >= 3:
            return False
    if _SEARCH_QUERY_CJK_KEYWORD_STACK_PATTERN.fullmatch(normalized):
        return True
    if " " not in normalized:
        return False
    tokens = [token for token in normalized.split(" ") if token]
    if len(tokens) < 4:
        return False
    short_token_count = 0
    for token in tokens:
        if any(ch in token for ch in ",，。！？?;；:/\\|()[]{}\"'“”‘’"):
            return False
        if len(token) <= 8:
            short_token_count += 1
    return len(tokens) >= 5 and short_token_count >= max(5, len(tokens) - 1)
