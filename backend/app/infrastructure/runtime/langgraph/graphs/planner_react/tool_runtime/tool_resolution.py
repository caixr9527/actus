#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具解析辅助函数。"""

from __future__ import annotations

from typing import List, Optional

from app.domain.services.tools import BaseTool


def resolve_matched_tool(
        *,
        function_name: str,
        fallback_tool: Optional[BaseTool],
        runtime_tools: Optional[List[BaseTool]],
) -> Optional[BaseTool]:
    """按函数名解析当前轮真实工具实现。

    业务语义：
    - 优先复用当前已命中的 `fallback_tool`；
    - 若其不承载目标函数，再在当前轮可用工具集合中查找真实实现；
    - rewrite 二次评估与 executor 必须共用这一套解析语义，避免约束输入和真实执行目标不一致。
    """
    normalized_name = str(function_name or "").strip()
    if fallback_tool is not None and fallback_tool.has_tool(normalized_name):
        return fallback_tool
    for tool in list(runtime_tools or []):
        if tool.has_tool(normalized_name):
            return tool
    return None
