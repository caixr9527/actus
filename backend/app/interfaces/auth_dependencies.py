#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/14 12:06
@Author : caixiaorong01@outlook.com
@File   : auth_dependencies.py
"""
# Deprecated shim:
# - deprecated_since: 2026-03-24
# - removal_target: 2026-04-30
# - replacement: app.interfaces.dependencies.auth
# 兼容层：保留旧导入路径，内部转发到新的 dependencies.auth。
DEPRECATED_SINCE = "2026-03-24"
REMOVAL_TARGET_DATE = "2026-04-30"
REPLACEMENT_IMPORT = "app.interfaces.dependencies.auth"

from app.interfaces.dependencies.auth import (  # noqa: F401
    AuthContext,
    get_access_token_ttl_seconds,
    get_current_auth_context,
    get_current_user,
)

__all__ = [
    "AuthContext",
    "get_access_token_ttl_seconds",
    "get_current_auth_context",
    "get_current_user",
]
