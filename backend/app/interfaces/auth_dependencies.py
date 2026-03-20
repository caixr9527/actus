#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/14 12:06
@Author : caixiaorong01@outlook.com
@File   : auth_dependencies.py
"""
# 兼容层：保留旧导入路径，内部转发到新的 dependencies.auth。
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
