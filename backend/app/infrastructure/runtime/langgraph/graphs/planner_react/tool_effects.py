#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具结果状态归并（reducer）。"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.domain.models import SearchResults, Step, ToolResult
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
        execution_state.research_candidate_urls = _extract_search_result_urls(tool_result)
        execution_state.research_search_ready = len(execution_state.research_candidate_urls) > 0
    elif execution_context.research_route_enabled and normalized_function_name == "fetch_page" and bool(tool_result.success):
        execution_state.research_fetch_completed = True

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

    deduped_urls: List[str] = []
    seen_urls: set[str] = set()
    for url in urls:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_urls.append(url)
    return deduped_urls[:5]
