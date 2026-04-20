#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具执行参数规范化。"""

from __future__ import annotations

from typing import Any, Dict

from app.infrastructure.runtime.langgraph.graphs.planner_react.research_query_builder import build_research_query


def normalize_tool_execution_args(
        *,
        normalized_function_name: str,
        function_args: Dict[str, Any],
) -> Dict[str, Any]:
    """统一生成真实执行参数，供约束层、执行层、日志与事件回放共用同一份最终值。"""
    normalized_args = dict(function_args or {})
    normalized_name = str(normalized_function_name or "").strip().lower()
    if normalized_name != "search_web":
        return normalized_args
    normalized_search_query = _normalize_search_query(normalized_args.get("query"))
    if not normalized_search_query:
        return normalized_args
    normalized_args["query"] = normalized_search_query
    return normalized_args


def _normalize_search_query(raw_query: Any) -> str:
    """统一规整 search_web 查询，收口为单主题自然语言主查询。"""
    return build_research_query(raw_query)
