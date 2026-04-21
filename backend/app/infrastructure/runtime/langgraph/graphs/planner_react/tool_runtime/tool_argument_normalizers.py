#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具执行参数规范化。"""

from __future__ import annotations

import re
from typing import Any, Dict

from app.infrastructure.runtime.langgraph.graphs.planner_react.research.research_query_builder import (
    build_research_query,
)

_OVERWRITE_WRITE_PATTERN = re.compile(
    r"(覆盖|覆写|替换为|改为|写成|overwrite|replace\s+with|set\s+to)",
    re.IGNORECASE,
)
_APPEND_WRITE_PATTERN = re.compile(
    r"(追加|附加|加到末尾|append|append\s+to|add\s+to\s+end)",
    re.IGNORECASE,
)


def normalize_tool_execution_args(
        *,
        normalized_function_name: str,
        function_args: Dict[str, Any],
        intent_text: str = "",
) -> Dict[str, Any]:
    """统一生成真实执行参数，供约束层、执行层、日志与事件回放共用同一份最终值。"""
    normalized_args = dict(function_args or {})
    normalized_name = str(normalized_function_name or "").strip().lower()
    if normalized_name == "search_web":
        normalized_search_query = _normalize_search_query(normalized_args.get("query"))
        if not normalized_search_query:
            return normalized_args
        normalized_args["query"] = normalized_search_query
        return normalized_args
    if normalized_name == "write_file":
        return _normalize_write_file_args(normalized_args, intent_text=intent_text)
    return normalized_args


def _normalize_write_file_args(function_args: Dict[str, Any], *, intent_text: str) -> Dict[str, Any]:
    normalized_args = dict(function_args or {})
    normalized_intent_text = str(intent_text or "").strip()
    if not normalized_intent_text:
        return normalized_args
    has_overwrite_intent = bool(_OVERWRITE_WRITE_PATTERN.search(normalized_intent_text))
    has_append_intent = bool(_APPEND_WRITE_PATTERN.search(normalized_intent_text))
    if has_overwrite_intent:
        normalized_args["append"] = False
        return normalized_args
    if has_append_intent:
        normalized_args["append"] = True
    return normalized_args


def _normalize_search_query(raw_query: Any) -> str:
    """统一规整 search_web 查询，收口为单主题自然语言主查询。"""
    return build_research_query(raw_query)
