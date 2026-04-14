#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""浏览器能力路由策略。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.domain.models import (
    BrowserActionableElementsResult,
    BrowserCardExtractionResult,
    BrowserLinkMatchResult,
    BrowserMainContentResult,
    BrowserPageStructuredResult,
    BrowserPageType,
    ToolResult,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.settings import (
    BROWSER_ATOMIC_FUNCTION_NAMES,
    BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
)
from .common import compact_tool_value, hash_payload


def normalize_browser_page_type(value: Any) -> str:
    if isinstance(value, BrowserPageType):
        return value.value
    normalized_value = getattr(value, "value", value)
    return str(normalized_value or "").strip().lower()


def extract_browser_tool_state(tool_result: ToolResult) -> Dict[str, Any]:
    data = tool_result.data
    if isinstance(
            data,
            (
                    BrowserPageStructuredResult,
                    BrowserMainContentResult,
                    BrowserCardExtractionResult,
                    BrowserActionableElementsResult,
            ),
    ):
        return {
            "url": str(data.url or "").strip(),
            "title": str(data.title or "").strip(),
            "page_type": normalize_browser_page_type(data.page_type),
            "selector": "",
            "index": None,
        }
    if isinstance(data, BrowserLinkMatchResult):
        return {
            "url": str(data.url or "").strip(),
            "title": str(data.matched_text or "").strip(),
            "page_type": "",
            "selector": str(data.selector or "").strip(),
            "index": data.index,
        }
    if isinstance(data, dict):
        return {
            "url": str(data.get("url") or "").strip(),
            "title": str(data.get("title") or data.get("matched_text") or "").strip(),
            "page_type": normalize_browser_page_type(data.get("page_type")),
            "selector": str(data.get("selector") or "").strip(),
            "index": data.get("index"),
        }
    return {"url": "", "title": "", "page_type": "", "selector": "", "index": None}


def build_browser_observation_fingerprint(tool_result: ToolResult) -> str:
    result_data = tool_result.data if hasattr(tool_result, "data") else None
    return hash_payload(
        {
            "message": str(getattr(tool_result, "message", "") or "").strip()[:200],
            "data": compact_tool_value(result_data),
        }
    )


def build_browser_route_state_key(
        *,
        browser_page_type: str,
        browser_url: str,
        browser_observation_fingerprint: str,
) -> str:
    return hash_payload(
        {
            "page_type": browser_page_type.strip().lower(),
            "url": browser_url.strip(),
            "observation": browser_observation_fingerprint.strip(),
        }
    )


def build_browser_high_level_failure_key(
        *,
        function_name: str,
        function_args: Dict[str, Any],
        browser_route_state_key: str,
) -> str:
    return hash_payload(
        {
            "function_name": function_name.strip().lower(),
            "page": browser_route_state_key,
            "args": compact_tool_value(function_args),
        }
    )


def is_browser_high_level_temporarily_blocked(
        *,
        function_name: str,
        function_args: Dict[str, Any],
        browser_route_state_key: str,
        failed_high_level_keys: set[str],
) -> bool:
    normalized_function_name = function_name.strip().lower()
    if normalized_function_name not in BROWSER_HIGH_LEVEL_FUNCTION_NAMES:
        return False
    return build_browser_high_level_failure_key(
        function_name=normalized_function_name,
        function_args=function_args,
        browser_route_state_key=browser_route_state_key,
    ) in failed_high_level_keys


def collect_temporarily_blocked_browser_high_level_function_names(
        *,
        browser_route_state_key: str,
        failed_high_level_keys: set[str],
) -> set[str]:
    return {
        function_name
        for function_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES
        if is_browser_high_level_temporarily_blocked(
            function_name=function_name,
            function_args={},
            browser_route_state_key=browser_route_state_key,
            failed_high_level_keys=failed_high_level_keys,
        )
    }


def build_browser_preferred_function_names(
        *,
        task_mode: str,
        available_function_names: set[str],
        browser_page_type: str,
        browser_structured_ready: bool,
        browser_main_content_ready: bool,
        browser_cards_ready: bool,
        browser_link_match_ready: bool,
        browser_actionables_ready: bool,
        failed_high_level_functions: set[str],
) -> Tuple[str, ...]:
    is_listing_page = browser_page_type in {
        BrowserPageType.LISTING.value,
        BrowserPageType.SEARCH_RESULTS.value,
    }

    ordered_candidates: Tuple[str, ...] = ()
    if task_mode == "web_reading":
        if not browser_structured_ready:
            ordered_candidates = ("browser_read_current_page_structured",)
        elif is_listing_page:
            if not browser_cards_ready:
                ordered_candidates = ("browser_extract_cards",)
            elif not browser_link_match_ready:
                ordered_candidates = ("browser_find_link_by_text",)
            else:
                ordered_candidates = ("browser_click", "browser_navigate")
        elif not browser_main_content_ready:
            ordered_candidates = ("browser_extract_main_content",)
    elif task_mode == "browser_interaction":
        if not browser_structured_ready:
            ordered_candidates = ("browser_read_current_page_structured",)
        elif is_listing_page:
            if not browser_cards_ready:
                ordered_candidates = ("browser_extract_cards",)
            elif not browser_link_match_ready:
                ordered_candidates = ("browser_find_link_by_text",)
            else:
                ordered_candidates = ("browser_click", "browser_navigate")
        elif not browser_actionables_ready:
            ordered_candidates = ("browser_find_actionable_elements",)
    else:
        return ()

    return tuple(
        function_name
        for function_name in ordered_candidates
        if function_name in available_function_names and function_name not in failed_high_level_functions
    )


def build_browser_atomic_allowlist(
        *,
        task_mode: str,
        browser_page_type: str,
        browser_structured_ready: bool,
        browser_link_match_ready: bool,
        browser_actionables_ready: bool,
        failed_high_level_functions: set[str],
) -> Tuple[str, ...]:
    is_listing_page = browser_page_type in {
        BrowserPageType.LISTING.value,
        BrowserPageType.SEARCH_RESULTS.value,
    }

    if task_mode == "web_reading":
        if not browser_structured_ready and "browser_read_current_page_structured" in failed_high_level_functions:
            return ("browser_view", "browser_scroll_down", "browser_scroll_up")
        if is_listing_page:
            if browser_link_match_ready:
                return ("browser_click", "browser_navigate")
            if any(
                    function_name in failed_high_level_functions
                    for function_name in (
                            "browser_extract_cards",
                            "browser_read_current_page_structured",
                    )
            ):
                return ("browser_view", "browser_scroll_down", "browser_scroll_up", "browser_navigate")
            return ()
        if any(
                function_name in failed_high_level_functions
                for function_name in (
                        "browser_extract_main_content",
                        "browser_read_current_page_structured",
                )
        ):
            return ("browser_view", "browser_scroll_down", "browser_scroll_up")
        return ()

    if task_mode == "browser_interaction":
        if not browser_structured_ready and "browser_read_current_page_structured" in failed_high_level_functions:
            return BROWSER_ATOMIC_FUNCTION_NAMES
        if is_listing_page:
            if browser_link_match_ready:
                return ("browser_click", "browser_navigate")
            if any(
                    function_name in failed_high_level_functions
                    for function_name in (
                            "browser_extract_cards",
                            "browser_read_current_page_structured",
                    )
            ):
                return BROWSER_ATOMIC_FUNCTION_NAMES
            return ()
        if browser_actionables_ready or any(
                function_name in failed_high_level_functions
                for function_name in (
                        "browser_find_actionable_elements",
                        "browser_read_current_page_structured",
                )
        ):
            return BROWSER_ATOMIC_FUNCTION_NAMES
        return ()

    return ()


def build_browser_capability_gap_allowlist(
        *,
        task_mode: str,
) -> Tuple[str, ...]:
    if task_mode == "web_reading":
        return ("browser_view", "browser_navigate", "browser_restart", "browser_scroll_down", "browser_scroll_up")
    if task_mode == "browser_interaction":
        return BROWSER_ATOMIC_FUNCTION_NAMES
    return ()


def build_browser_route_block_message(
        *,
        task_mode: str,
        function_name: str,
        browser_page_type: str,
        browser_structured_ready: bool,
        browser_cards_ready: bool,
        browser_link_match_ready: bool,
        browser_actionables_ready: bool,
        last_browser_route_url: str,
        last_browser_route_selector: str,
        last_browser_route_index: Optional[int],
) -> str:
    is_listing_page = browser_page_type in {
        BrowserPageType.LISTING.value,
        BrowserPageType.SEARCH_RESULTS.value,
    }
    normalized_function_name = function_name.strip().lower()

    if not browser_structured_ready:
        return "当前步骤属于浏览器任务，请先调用 browser_read_current_page_structured 判断页面类型，再决定后续动作。"

    if is_listing_page:
        if not browser_cards_ready:
            return "当前页面是列表页，请先调用 browser_extract_cards 提取候选卡片，不要直接执行原子浏览器动作。"
        if not browser_link_match_ready:
            return "当前页面是列表页，已提取候选卡片，请先调用 browser_find_link_by_text 选定目标，不要直接执行原子浏览器动作。"
        if normalized_function_name not in {"browser_click", "browser_navigate"}:
            if last_browser_route_index is not None:
                url_hint = f"；如需直接跳转可使用 browser_navigate 或 fetch_page 读取 {last_browser_route_url}" if last_browser_route_url else ""
                selector_hint = f"；匹配定位: {last_browser_route_selector}" if last_browser_route_selector else ""
                return (
                    "当前页面已完成目标链接定位，请优先使用 "
                    f"browser_click(index={last_browser_route_index}){url_hint}{selector_hint}。"
                )
            if last_browser_route_url:
                return (
                    "当前页面已完成目标链接定位，请优先使用返回结果中的 URL 调用 browser_navigate"
                    f" 或 fetch_page，当前可用 URL: {last_browser_route_url}"
                )
            return "当前页面已完成目标链接定位，请优先使用 browser_click 或 browser_navigate。"

    if task_mode == "web_reading":
        return "当前步骤属于网页阅读任务，请优先使用高阶浏览器阅读能力，只有高阶提取失败后才能退回原子浏览器动作。"
    if task_mode == "browser_interaction" and not browser_actionables_ready:
        return "当前步骤属于浏览器交互任务，请先调用 browser_find_actionable_elements 确认可交互元素，再执行原子浏览器动作。"
    return "当前浏览器原子动作尚未开放，请先完成当前阶段要求的高阶浏览器能力。"


def attach_browser_degrade_payload(
        tool_result: ToolResult,
        *,
        function_name: str,
        degrade_reason: str,
        browser_page_type: str,
        browser_url: str,
        browser_title: str,
) -> ToolResult:
    payload = tool_result.data if isinstance(tool_result.data, dict) else {}
    payload = {
        **payload,
        "degrade_reason": degrade_reason,
        "function_name": function_name,
        "page_type": browser_page_type,
        "url": browser_url,
        "title": browser_title,
    }
    tool_result.data = payload
    return tool_result


def build_browser_high_level_retry_block_message(
        *,
        function_name: str,
        function_args: Dict[str, Any],
) -> str:
    normalized_function_name = function_name.strip().lower()
    if normalized_function_name == "browser_find_link_by_text":
        query = str(function_args.get("text") or "").strip()
        if query:
            return f"当前页面中“{query}”的链接匹配刚刚失败，请先更换关键词或改变页面状态后再试。"
        return "当前页面的链接匹配刚刚失败，请先更换关键词或改变页面状态后再试。"
    return f"{function_name} 在当前页面状态下刚刚失败，请先改变页面状态或尝试其他高阶路径后再试。"


def coerce_optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped_value = value.strip()
        if stripped_value.isdigit() or (
                stripped_value.startswith("-") and stripped_value[1:].isdigit()
        ):
            return int(stripped_value)
    return None


def build_listing_click_target_block_message(
        *,
        last_browser_route_index: Optional[int],
        last_browser_route_url: str,
        last_browser_route_selector: str,
) -> str:
    if last_browser_route_index is None:
        if last_browser_route_url:
            return (
                "当前列表页已完成目标链接定位，但匹配结果没有可点击 index，"
                f"请改用 browser_navigate 或 fetch_page 读取 {last_browser_route_url}。"
            )
        return "当前列表页已完成目标链接定位，但匹配结果没有可点击 index，请改用 browser_navigate。"
    selector_hint = f"；匹配定位: {last_browser_route_selector}" if last_browser_route_selector else ""
    url_hint = f"；目标 URL: {last_browser_route_url}" if last_browser_route_url else ""
    return (
        "当前列表页已完成目标链接定位，只允许点击刚刚匹配到的目标，"
        f"请调用 browser_click(index={last_browser_route_index}){selector_hint}{url_hint}。"
    )
