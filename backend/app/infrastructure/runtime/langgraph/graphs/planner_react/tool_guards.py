#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具调用前置守卫。"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from app.domain.services.runtime.normalizers import extract_url_domain, normalize_url_value
from app.domain.models import Step, ToolResult
from app.domain.services.tools import BaseTool
from app.domain.services.workspace_runtime.policies import (
    build_log_text_preview as _build_log_text_preview,
    build_browser_route_block_message as _build_browser_route_block_message,
)
from .execution_context import ExecutionContext, step_allows_user_wait
from .execution_state import ExecutionState
from .research_intent_policy import is_explicit_single_page_fetch_intent
from .research_url_extractor import extract_explicit_url_from_research_context
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.contracts.langgraph_settings import (
    ASK_USER_FUNCTION_NAME,
    READ_ONLY_FILE_FUNCTION_NAMES,
    RESEARCH_CROSS_DOMAIN_REPEAT_BLOCK_LIMIT,
    RESEARCH_MIN_DOMAIN_COUNT,
    RESEARCH_MIN_FETCH_COUNT,
)

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


@dataclass(slots=True)
class GuardDecision:
    should_skip: bool
    loop_break_reason: str = ""
    tool_result: Optional[ToolResult] = None


def evaluate_tool_guard(
    *,
    logger: logging.Logger,
    step: Step,
    task_mode: str,
    function_name: str,
    normalized_function_name: str,
    function_args: Dict[str, Any],
    matched_tool: Optional[BaseTool],
    iteration_blocked_function_names: Set[str],
    ctx: ExecutionContext,
    state: ExecutionState,
) -> GuardDecision:
    if (
        normalized_function_name == "search_web"
        and _is_keyword_stacked_search_query(function_args.get("query"))
    ):
        log_runtime(
            logger,
            logging.WARNING,
            "检索查询风格不符合自然语言约束",
            step_id=str(step.id or ""),
            function_name=function_name,
            search_query_preview=_build_log_text_preview(function_args.get("query"), max_chars=100),
        )
        return GuardDecision(
            should_skip=True,
            loop_break_reason="research_query_style_blocked",
            tool_result=ToolResult(
                success=False,
                message=(
                    "search_web 的 query 必须使用单主题自然语言描述，禁止关键词堆叠。"
                    " 请改为一句完整的主题描述，例如“主流 AI 编程助手及其支持的 IDE”。"
                ),
            ),
        )

    if matched_tool is None:
        log_runtime(
            logger,
            logging.WARNING,
            "工具调用无效",
            step_id=str(step.id or ""),
            function_name=function_name,
        )
        return GuardDecision(
            should_skip=True,
            tool_result=ToolResult(success=False, message=f"无效工具: {function_name}"),
        )

    if normalized_function_name in iteration_blocked_function_names:
        if task_mode == "human_wait" and function_name != ASK_USER_FUNCTION_NAME:
            log_runtime(
                logger,
                logging.WARNING,
                "等待步骤已拦截非等待工具",
                step_id=str(step.id or ""),
                function_name=function_name,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="human_wait_non_interrupt_tool_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤是等待用户确认/选择的步骤，只允许调用 message_ask_user 发起等待。",
                ),
            )
        if normalized_function_name in ctx.research_file_context_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "缺少明确文件上下文，已拦截文件工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="research_file_context_required",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤属于检索任务，只有在用户消息或附件中出现明确文件路径/文件名时，才能调用文件工具。",
                ),
            )
        if normalized_function_name in ctx.read_only_file_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "只读步骤已拦截写副作用工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="read_only_file_intent_write_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤是只读文件请求，请使用 read_file/list_files/find_files/search_in_file，不要写文件、改文件或执行 shell 命令。",
                ),
            )
        if task_mode == "web_reading" and normalized_function_name in READ_ONLY_FILE_FUNCTION_NAMES:
            log_runtime(
                logger,
                logging.WARNING,
                "网页读取步骤已拦截文件工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="web_reading_file_tool_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤属于网页读取任务，请优先使用 search_web、fetch_page 或浏览器高阶读取工具，不要回退到文件工具。",
                ),
            )
        if normalized_function_name in ctx.general_inline_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "内联展示步骤缺少文件上下文，已拦截文件工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                output_mode=str(getattr(step, "output_mode", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="general_inline_file_context_required",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤是直接内联展示结果的步骤，且没有可用文件上下文，请直接返回文本结果，不要继续读写文件。",
                ),
            )
        if normalized_function_name in ctx.file_processing_shell_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "文件处理步骤缺少显式命令意图，已拦截 shell 工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="file_processing_shell_explicit_required",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤属于文件处理，默认禁止调用 shell_execute。仅在用户明确要求执行命令时才允许。",
                ),
            )
        if normalized_function_name in ctx.artifact_policy_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "步骤产物策略拦截文件产出工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                artifact_policy=str(getattr(step, "artifact_policy", "") or ""),
                output_mode=str(getattr(step, "output_mode", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="artifact_policy_file_output_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤的结构化产物策略禁止文件产出。请直接返回文本结果，或先通过重规划生成允许文件产出的步骤。",
                ),
            )
        if normalized_function_name in ctx.final_delivery_search_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "最终交付步骤已拦截检索漂移",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                delivery_role=str(getattr(step, "delivery_role", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="final_delivery_search_drift_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤负责最终交付正文，请直接基于已知上下文组织答案，不要重新调用 search_web 或 fetch_page。",
                ),
            )
        if normalized_function_name in ctx.final_inline_file_output_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "最终内联交付步骤已拦截写文件漂移",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                delivery_role=str(getattr(step, "delivery_role", "") or ""),
                output_mode=str(getattr(step, "output_mode", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="final_inline_file_output_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤负责最终内联交付，且用户未明确要求文件交付。请直接输出最终文本，不要调用 write_file/replace_in_file。",
                ),
            )
        if normalized_function_name in ctx.final_delivery_shell_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "最终交付步骤已拦截 shell 漂移",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                delivery_role=str(getattr(step, "delivery_role", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="final_delivery_shell_drift_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤负责最终交付正文，请直接输出最终答案，不要调用 shell_execute。",
                ),
            )
        if ctx.browser_route_enabled and normalized_function_name.startswith("browser_"):
            log_runtime(
                logger,
                logging.WARNING,
                "浏览器固定路径拦截工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                browser_page_type=state.browser_page_type,
                browser_structured_ready=state.browser_structured_ready,
                browser_cards_ready=state.browser_cards_ready,
                browser_link_match_ready=state.browser_link_match_ready,
                browser_actionables_ready=state.browser_actionables_ready,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="browser_route_blocked",
                tool_result=ToolResult(
                    success=False,
                    message=_build_browser_route_block_message(
                        task_mode=task_mode,
                        function_name=function_name,
                        browser_page_type=state.browser_page_type,
                        browser_structured_ready=state.browser_structured_ready,
                        browser_cards_ready=state.browser_cards_ready,
                        browser_link_match_ready=state.browser_link_match_ready,
                        browser_actionables_ready=state.browser_actionables_ready,
                        last_browser_route_url=state.last_browser_route_url,
                        last_browser_route_selector=state.last_browser_route_selector,
                        last_browser_route_index=state.last_browser_route_index,
                    ),
                ),
            )

        log_runtime(
            logger,
            logging.WARNING,
            "任务模式拦截工具调用",
            step_id=str(step.id or ""),
            function_name=function_name,
            task_mode=task_mode,
        )
        return GuardDecision(
            should_skip=True,
            loop_break_reason="task_mode_tool_blocked",
            tool_result=ToolResult(
                success=False,
                message=f"当前步骤的任务模式 {task_mode} 不允许调用工具: {function_name}",
            ),
        )

    if (
        ctx.research_route_enabled
        and normalized_function_name == "fetch_page"
        and not ctx.research_has_explicit_url
        and not state.research_search_ready
    ):
        return GuardDecision(
            should_skip=True,
            loop_break_reason="research_route_search_required",
            tool_result=ToolResult(
                success=False,
                message="当前步骤属于检索/网页阅读任务，请先调用 search_web 获取候选链接，再使用 fetch_page 读取正文。",
            ),
        )
    if (
        ctx.research_route_enabled
        and normalized_function_name == "search_web"
        and ctx.research_has_explicit_url
        and not state.research_fetch_completed
    ):
        # P3-一次性收口：仅“明确单页读取意图 + explicit URL 未被消费”时强制 fetch。
        # 非单页读取场景或 explicit 已改写过时，允许 search_web 持续扩召回，避免锁死。
        explicit_url_blacklisted = _is_explicit_url_blacklisted(step=step, ctx=ctx, state=state)
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
            and (not explicit_url_blacklisted)
            and (not explicit_url_already_rewritten)
            and state.consecutive_fetch_failure_count < 2
        ):
            return GuardDecision(
                should_skip=True,
                loop_break_reason="research_route_fetch_required",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤已提供明确 URL，请直接调用 fetch_page 读取页面正文，不要先重复搜索。",
                ),
            )
        log_runtime(
            logger,
            logging.INFO,
            "显式 URL 策略已放行 search_web",
            step_id=str(step.id or ""),
            function_name=function_name,
            consecutive_fetch_failure_count=state.consecutive_fetch_failure_count,
            explicit_url_blacklisted=explicit_url_blacklisted,
            explicit_url_already_rewritten=explicit_url_already_rewritten,
            explicit_single_page_fetch_intent=explicit_single_page_fetch_intent,
            allow_search_reason=(
                "not_single_page_intent"
                if not explicit_single_page_fetch_intent
                else "explicit_blacklisted"
                if explicit_url_blacklisted
                else "explicit_already_rewritten"
                if explicit_url_already_rewritten
                else "fetch_failure_recovery"
                if state.consecutive_fetch_failure_count >= 2
                else "policy_allow"
            ),
        )
    if (
        ctx.research_route_enabled
        and normalized_function_name == "search_web"
        and state.research_search_ready
        and not ctx.research_has_explicit_url
    ):
        # P3-一次性收口：fetch 连续失败后，不再强制“先 fetch 再 search”，避免策略自锁。
        if state.consecutive_fetch_failure_count >= 2:
            log_runtime(
                logger,
                logging.INFO,
                "研究链路连续失败后允许 search_web 恢复召回",
                step_id=str(step.id or ""),
                function_name=function_name,
                consecutive_fetch_failure_count=state.consecutive_fetch_failure_count,
            )
            return GuardDecision(should_skip=False)
        pending_candidate_urls = _collect_pending_candidate_urls(state=state)
        needs_fetch_count = state.research_fetch_success_count < RESEARCH_MIN_FETCH_COUNT
        needs_domain_coverage = len(state.research_fetched_domains) < RESEARCH_MIN_DOMAIN_COUNT
        if pending_candidate_urls and (needs_fetch_count or needs_domain_coverage):
            candidate_hint = "；".join((pending_candidate_urls or state.research_candidate_urls)[:3])
            return GuardDecision(
                should_skip=True,
                loop_break_reason="research_route_fetch_required",
                tool_result=ToolResult(
                    success=False,
                    message=(
                        "已经拿到候选链接，请优先对搜索结果中的 URL 调用 fetch_page 读取正文。"
                        f" 当前已抓取 {state.research_fetch_success_count} 个来源，目标至少 {RESEARCH_MIN_FETCH_COUNT} 个；"
                        f" 已覆盖 {len(state.research_fetched_domains)} 个站点，目标至少 {RESEARCH_MIN_DOMAIN_COUNT} 个。"
                        + (f" 可用链接示例: {candidate_hint}" if candidate_hint else "")
                    ),
                ),
            )
    if (
        ctx.research_route_enabled
        and normalized_function_name == "fetch_page"
        and not ctx.research_has_explicit_url
        and len(state.research_fetched_domains) < RESEARCH_MIN_DOMAIN_COUNT
    ):
        requested_url = normalize_url_value(function_args.get("url"))
        requested_domain = _extract_domain(requested_url)
        fetched_domains = set(state.research_fetched_domains)
        cross_domain_candidate = _pick_pending_cross_domain_candidate(
            state=state,
            excluded_domains=fetched_domains,
        )
        # P3-一次性收口：覆盖不足时同域重复抓取先可恢复拦截，超过阈值后再收敛终止。
        if requested_domain and requested_domain in fetched_domains and cross_domain_candidate:
            state.research_cross_domain_repeat_blocks += 1
            should_break = state.research_cross_domain_repeat_blocks > RESEARCH_CROSS_DOMAIN_REPEAT_BLOCK_LIMIT
            loop_break_reason = "research_route_cross_domain_fetch_limit" if should_break else ""
            log_runtime(
                logger,
                logging.WARNING,
                "检索覆盖不足，已拦截同域重复抓取",
                step_id=str(step.id or ""),
                function_name=function_name,
                requested_url=requested_url,
                requested_domain=requested_domain,
                fetched_domain_count=len(fetched_domains),
                next_cross_domain_url=cross_domain_candidate,
                cross_domain_repeat_blocks=state.research_cross_domain_repeat_blocks,
                cross_domain_repeat_block_limit=RESEARCH_CROSS_DOMAIN_REPEAT_BLOCK_LIMIT,
                should_break=should_break,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason=loop_break_reason,
                tool_result=ToolResult(
                    success=False,
                    message=(
                        "当前研究步骤尚未覆盖足够来源，请优先抓取不同站点候选链接，避免重复读取同域页面。"
                        f" 建议先读取: {cross_domain_candidate}"
                    ),
                ),
            )
        # 未触发“同域重复且存在跨域候选”拦截时，清理历史累计，避免旧计数污染后续轮次。
        state.research_cross_domain_repeat_blocks = 0
    if function_name == ASK_USER_FUNCTION_NAME and not step_allows_user_wait(step, function_args):
        log_runtime(
            logger,
            logging.WARNING,
            "提前请求用户交互，已拦截",
            step_id=str(step.id or ""),
            function_name=function_name,
            step_description=str(step.description or ""),
        )
        return GuardDecision(
            should_skip=True,
            tool_result=ToolResult(
                success=False,
                message="当前步骤不允许向用户提问。请先完成当前步骤，只能在明确需要用户确认/选择/输入的步骤中使用该工具。",
            ),
        )

    return GuardDecision(should_skip=False)


def _collect_pending_candidate_urls(*, state: ExecutionState) -> list[str]:
    fetched_url_set = {
        _normalize_fetch_dedupe_key(url)
        for url in list(state.research_fetched_urls or [])
        if _normalize_fetch_dedupe_key(url)
    }
    failed_url_set = set(list(state.research_failed_fetch_url_keys or set()))
    pending: list[str] = []
    seen_pending_keys: set[str] = set()
    for raw_url in list(state.research_candidate_urls or []):
        normalized = _normalize_url(raw_url)
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


def _pick_pending_cross_domain_candidate(*, state: ExecutionState, excluded_domains: set[str]) -> str:
    pending_urls = _collect_pending_candidate_urls(state=state)
    for url in pending_urls:
        domain = _extract_domain(url)
        if domain and domain not in excluded_domains:
            return url
    return ""


def _normalize_url(url: Any) -> str:
    return normalize_url_value(url)


def _normalize_fetch_dedupe_key(url: Any) -> str:
    # 只用于“是否已读/是否重复”判定：忽略 query 降低追踪参数导致的重复抓取噪音。
    return normalize_url_value(url, drop_query=True)


def _extract_domain(url: str) -> str:
    return extract_url_domain(url)


def _is_explicit_url_blacklisted(*, step: Step, ctx: ExecutionContext, state: ExecutionState) -> bool:
    explicit_url = extract_explicit_url_from_research_context(step=step, ctx=ctx)
    explicit_key = _normalize_fetch_dedupe_key(explicit_url)
    if not explicit_key:
        return False
    return explicit_key in set(list(state.research_failed_fetch_url_keys or set()))


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
    # 关键词堆叠通常表现为大量短词并列，缺少自然语言结构词。
    return len(tokens) >= 5 and short_token_count >= max(5, len(tokens) - 1)
