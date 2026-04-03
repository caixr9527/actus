#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/12 12:05
@Author : caixiaorong01@outlook.com
@File   : __init__.py.py
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .file import FileService
    from .searxng import SearXNGService
    from .shell import ShellService
    from .supervisorService import SupervisorService

__all__ = [
    "ShellService",
    "FileService",
    "SupervisorService",
    "SearXNGService",
]


def __getattr__(name: str):
    """按需导入service，避免包初始化时的循环依赖。"""
    if name == "ShellService":
        from .shell import ShellService
        return ShellService
    if name == "FileService":
        from .file import FileService
        return FileService
    if name == "SupervisorService":
        from .supervisorService import SupervisorService
        return SupervisorService
    if name == "SearXNGService":
        from .searxng import SearXNGService
        return SearXNGService
    raise AttributeError(f"module 'app.services' has no attribute '{name}'")
