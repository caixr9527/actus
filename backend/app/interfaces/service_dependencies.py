#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/14 12:06
@Author : caixiaorong01@outlook.com
@File   : service_dependencies.py
"""
# 兼容层：保留旧导入路径，内部转发到新的 dependencies.services。
from app.interfaces.dependencies.services import (  # noqa: F401
    get_app_config_service,
    get_status_service,
    get_file_service,
    get_session_service,
    get_auth_service,
    get_access_token_blacklist_store,
    build_agent_service,
    get_agent_service_for_lifespan,
    clear_agent_service_for_lifespan_cache,
    get_agent_service,
)

__all__ = [
    "get_app_config_service",
    "get_status_service",
    "get_file_service",
    "get_session_service",
    "get_auth_service",
    "get_access_token_blacklist_store",
    "build_agent_service",
    "get_agent_service_for_lifespan",
    "clear_agent_service_for_lifespan_cache",
    "get_agent_service",
]
