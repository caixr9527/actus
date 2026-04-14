#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/19 19:41
@Author : caixiaorong01@outlook.com
@File   : __init__.py.py
"""
from importlib import import_module

from .base import BaseTool

__all__ = [
    "BaseTool",
    "MCPClientManager",
    "MCPTool",
    "MCPCapabilityAdapter",
    "A2ATool",
    "MessageTool",
    "CapabilityBuildContext",
    "CapabilityDefinition",
    "CapabilityRegistry",
    "ToolRuntimeAdapter",
    "ToolRuntimeEventHooks",
]


_LAZY_IMPORTS = {
    "A2ATool": ("app.domain.services.tools.a2a", "A2ATool"),
    "MCPClientManager": ("app.domain.services.tools.mcp", "MCPClientManager"),
    "MCPTool": ("app.domain.services.tools.mcp", "MCPTool"),
    "MCPCapabilityAdapter": ("app.domain.services.tools.mcp_capability_adapter", "MCPCapabilityAdapter"),
    "MessageTool": ("app.domain.services.tools.message", "MessageTool"),
    "CapabilityBuildContext": ("app.domain.services.tools.capability_registry", "CapabilityBuildContext"),
    "CapabilityDefinition": ("app.domain.services.tools.capability_registry", "CapabilityDefinition"),
    "CapabilityRegistry": ("app.domain.services.tools.capability_registry", "CapabilityRegistry"),
    "ToolRuntimeAdapter": ("app.domain.services.tools.runtime_adapter", "ToolRuntimeAdapter"),
    "ToolRuntimeEventHooks": ("app.domain.services.tools.runtime_adapter", "ToolRuntimeEventHooks"),
}


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
