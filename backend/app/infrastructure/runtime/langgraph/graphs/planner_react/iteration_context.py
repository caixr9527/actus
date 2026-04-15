#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：单轮迭代上下文构建。"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from app.domain.services.workspace_runtime.policies import (
    build_browser_atomic_allowlist as _build_browser_atomic_allowlist,
    build_browser_route_state_key as _build_browser_route_state_key,
    collect_temporarily_blocked_browser_high_level_function_names as _collect_temporarily_blocked_browser_high_level_function_names,
)
from .execution_context import ExecutionContext
from .execution_state import ExecutionState
from app.domain.services.runtime.contracts.langgraph_settings import ASK_USER_FUNCTION_NAME, BROWSER_ATOMIC_FUNCTION_NAMES


@dataclass(slots=True)
class IterationContext:
    browser_route_state_key: str
    failed_browser_high_level_functions: Set[str]
    iteration_blocked_function_names: Set[str]
    iteration_tools: List[Dict[str, Any]]


def build_iteration_context(
    *,
    task_mode: str,
    execution_context: ExecutionContext,
    execution_state: ExecutionState,
) -> IterationContext:
    # P3 重构：将“单轮状态派生 + 白名单裁剪”集中在一个构建点，主流程只做调度。
    browser_route_state_key = _build_browser_route_state_key(
        browser_page_type=execution_state.browser_page_type,
        browser_url=execution_state.last_browser_route_url,
        browser_observation_fingerprint=execution_state.last_browser_observation_fingerprint,
    )
    failed_browser_high_level_functions = _collect_temporarily_blocked_browser_high_level_function_names(
        browser_route_state_key=browser_route_state_key,
        failed_high_level_keys=execution_state.failed_browser_high_level_keys,
    )
    iteration_blocked_function_names = set(execution_context.blocked_function_names)
    iteration_blocked_function_names.update(failed_browser_high_level_functions)

    if execution_context.browser_route_enabled:
        allowed_atomic_browser_functions = set(
            _build_browser_atomic_allowlist(
                task_mode=task_mode,
                browser_page_type=execution_state.browser_page_type,
                browser_structured_ready=execution_state.browser_structured_ready,
                browser_link_match_ready=execution_state.browser_link_match_ready,
                browser_actionables_ready=execution_state.browser_actionables_ready,
                failed_high_level_functions=failed_browser_high_level_functions,
            )
        )
        for function_name in BROWSER_ATOMIC_FUNCTION_NAMES:
            if (
                function_name in execution_context.available_function_names
                and function_name not in allowed_atomic_browser_functions
            ):
                iteration_blocked_function_names.add(function_name)
            elif function_name in allowed_atomic_browser_functions:
                iteration_blocked_function_names.discard(function_name)

    iteration_tools = _filter_available_tools(
        execution_context.available_tools,
        disallowed_function_names=iteration_blocked_function_names,
        allow_ask_user=execution_context.allow_ask_user,
    )
    return IterationContext(
        browser_route_state_key=browser_route_state_key,
        failed_browser_high_level_functions=failed_browser_high_level_functions,
        iteration_blocked_function_names=iteration_blocked_function_names,
        iteration_tools=iteration_tools,
    )


def _filter_available_tools(
    available_tools: List[Dict[str, Any]],
    *,
    disallowed_function_names: Optional[Set[str]] = None,
    allow_ask_user: bool,
) -> List[Dict[str, Any]]:
    filtered_tools: List[Dict[str, Any]] = []
    blocked_names = set(disallowed_function_names or set())
    for tool_schema in available_tools:
        function_name = _extract_function_name(tool_schema)
        if function_name in blocked_names:
            continue
        if function_name == ASK_USER_FUNCTION_NAME and not allow_ask_user:
            continue
        filtered_tools.append(tool_schema)
    return filtered_tools


def _extract_function_name(tool_schema: Dict[str, Any]) -> str:
    if not isinstance(tool_schema, dict):
        return ""
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip().lower()
