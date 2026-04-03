#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/12 12:10
@Author : caixiaorong01@outlook.com
@File   : service_dependencies.py
"""
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services import ShellService, FileService, SupervisorService, SearXNGService


@lru_cache()
def get_shell_service() -> "ShellService":
    from app.services import ShellService
    return ShellService()


@lru_cache()
def get_file_service() -> "FileService":
    from app.services import FileService
    return FileService()


@lru_cache()
def get_supervisor_service() -> "SupervisorService":
    from app.services import SupervisorService
    return SupervisorService()


@lru_cache()
def get_searxng_service() -> "SearXNGService":
    from app.services import SearXNGService
    return SearXNGService()
