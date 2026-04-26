#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/14 12:06
@Author : caixiaorong01@outlook.com
@File   : service_dependencies.py
"""
# Deprecated shim:
# - deprecated_since: 2026-03-24
# - removal_target: 2026-04-30
# - replacement: app.interfaces.dependencies.services
# 兼容层：保留旧导入路径，内部转发到新的 dependencies.services。
DEPRECATED_SINCE = "2026-03-24"
REMOVAL_TARGET_DATE = "2026-04-30"
REPLACEMENT_IMPORT = "app.interfaces.dependencies.services"

from app.interfaces.dependencies.services import (  # noqa: F401
    get_app_config_service,
    get_status_service,
    get_file_service,
    get_session_service,
    get_auth_service,
    get_access_token_blacklist_store,
    get_refresh_token_store,
    build_agent_service,
    get_agent_service_for_lifespan,
    clear_agent_service_for_lifespan_cache,
    get_agent_service,
    get_session_stream_facade,
)

__all__ = [
    "get_app_config_service",
    "get_status_service",
    "get_file_service",
    "get_session_service",
    "get_auth_service",
    "get_access_token_blacklist_store",
    "get_refresh_token_store",
    "build_agent_service",
    "get_agent_service_for_lifespan",
    "clear_agent_service_for_lifespan_cache",
    "get_agent_service",
    "get_session_stream_facade",
]
