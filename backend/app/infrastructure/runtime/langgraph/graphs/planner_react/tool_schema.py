#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具 schema 解析辅助方法。"""

from typing import Any, Dict


def extract_function_name(tool_schema: Dict[str, Any]) -> str:
    if not isinstance(tool_schema, dict):
        return ""
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip().lower()

