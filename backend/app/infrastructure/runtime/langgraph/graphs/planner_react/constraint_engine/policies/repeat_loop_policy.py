#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束策略：重复调用收敛约束。"""

from __future__ import annotations

from typing import Any, Optional

from app.domain.services.runtime.contracts.langgraph_settings import REPEAT_TOOL_LIMIT
from app.domain.services.runtime.normalizers import normalize_url_value
from app.domain.services.workspace_runtime.policies import (
    build_search_fingerprint as _build_search_fingerprint,
    build_tool_fingerprint as _build_tool_fingerprint,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintDecision
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintInput
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintToolResultPayload
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_REPEAT_TOOL_CALL
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_RESEARCH_ROUTE_FINGERPRINT_REPEAT
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import REASON_SEARCH_REPEAT
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_argument_normalizers import (
    normalize_tool_execution_args,
)


def evaluate_repeat_loop_policy(constraint_input: ConstraintInput) -> Optional[ConstraintDecision]:
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    state = constraint_input.execution_state
    normalized_function_args = normalize_tool_execution_args(
        normalized_function_name=normalized_function_name,
        function_args=dict(constraint_input.function_args or {}),
    )

    if normalized_function_name == "search_web":
        current_fingerprint = _build_search_fingerprint(normalized_function_args)
        # P3-一次性收口：约束判定发生在 pre-invoke 入账之前，这里按“历史计数 + 当前调用”预估本轮计数，
        # 避免重复拦截晚一轮触发。
        current_repeat = int(state.search_repeat_counter.get(current_fingerprint, 0)) + 1
        if current_repeat > _resolve_search_repeat_limit(constraint_input):
            return _hard_block(
                reason_code=REASON_SEARCH_REPEAT,
                message="同一搜索查询已重复多次，请改写查询、缩小范围，或改用 fetch_page / 其他工具继续。",
            )

    if normalized_function_name == "fetch_page":
        current_url_fingerprint = normalize_url_value(normalized_function_args.get("url"), drop_query=True)
        current_repeat = int(state.fetch_repeat_counter.get(current_url_fingerprint, 0)) + 1
        if current_url_fingerprint and current_repeat > _resolve_fetch_repeat_limit(constraint_input):
            return _hard_block(
                reason_code=REASON_RESEARCH_ROUTE_FINGERPRINT_REPEAT,
                message="同一页面 URL 已重复抓取多次，请切换其他候选链接或结束当前步骤。",
            )

    predicted_same_tool_repeat_count = _predict_same_tool_repeat_count(constraint_input)
    if predicted_same_tool_repeat_count > REPEAT_TOOL_LIMIT:
        return _hard_block(
            reason_code=REASON_REPEAT_TOOL_CALL,
            message="检测到同一工具与相近参数被重复调用，请改用其他工具、调整参数，或结束当前步骤。",
        )

    return None


def _hard_block(*, reason_code: str, message: str) -> ConstraintDecision:
    return ConstraintDecision(
        action="block",
        reason_code=reason_code,
        block_mode="hard_block_break",
        loop_break_reason=reason_code,
        tool_result_payload=ConstraintToolResultPayload(success=False, message=message),
        message_for_model=message,
    )


def _resolve_search_repeat_limit(constraint_input: ConstraintInput) -> int:
    return _resolve_dynamic_limit(
        snapshot=constraint_input.external_signals_snapshot,
        key="search_repeat_limit",
        default_value=2,
    )


def _resolve_fetch_repeat_limit(constraint_input: ConstraintInput) -> int:
    return _resolve_dynamic_limit(
        snapshot=constraint_input.external_signals_snapshot,
        key="fetch_repeat_limit",
        default_value=2,
    )


def _resolve_dynamic_limit(*, snapshot: dict[str, Any], key: str, default_value: int) -> int:
    raw_value = snapshot.get(key)
    if isinstance(raw_value, bool):
        return default_value
    if isinstance(raw_value, (int, float)):
        value = int(raw_value)
        return value if value >= 1 else default_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed >= 1 else default_value
    return default_value


def _predict_same_tool_repeat_count(constraint_input: ConstraintInput) -> int:
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    state = constraint_input.execution_state
    normalized_function_args = normalize_tool_execution_args(
        normalized_function_name=normalized_function_name,
        function_args=dict(constraint_input.function_args or {}),
    )
    current_tool_fingerprint = _build_tool_fingerprint(
        normalized_function_name,
        normalized_function_args,
    )
    if current_tool_fingerprint == str(state.last_tool_fingerprint or ""):
        return int(state.same_tool_repeat_count) + 1
    return 1
