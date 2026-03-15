#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/15 12:00
@Author : caixiaorong01@outlook.com
@File   : auth_guard.py
"""
from collections.abc import Iterable

from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute

from app.interfaces.dependencies.auth import get_current_auth_context, get_current_user

_AUTH_GUARD_CALLS = frozenset({get_current_user, get_current_auth_context})
_AUTH_WHITELIST = frozenset(
    {
        ("GET", "/api/status"),
        ("POST", "/api/auth/register"),
        ("POST", "/api/auth/login"),
        ("POST", "/api/auth/refresh"),
        ("POST", "/api/auth/register/send-code"),
    }
)


def _iter_route_methods(route: APIRoute | APIWebSocketRoute) -> Iterable[str]:
    if isinstance(route, APIRoute):
        for method in sorted(route.methods or set()):
            yield method.upper()
        return
    yield "WEBSOCKET"


def _route_dependency_calls(route: APIRoute | APIWebSocketRoute) -> set[object]:
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return set()
    return {
        dependency.call
        for dependency in dependant.dependencies
        if getattr(dependency, "call", None) is not None
    }


def _route_requires_auth_guard(route: APIRoute | APIWebSocketRoute) -> bool:
    path = route.path
    return not any((method, path) in _AUTH_WHITELIST for method in _iter_route_methods(route))


def _route_display(route: APIRoute | APIWebSocketRoute) -> str:
    methods = ",".join(_iter_route_methods(route))
    return f"{methods} {route.path}"


def validate_api_auth_coverage(app: FastAPI) -> None:
    """校验 /api 路由是否显式挂载认证依赖。"""
    missing_routes: list[str] = []
    for route in app.routes:
        if not isinstance(route, (APIRoute, APIWebSocketRoute)):
            continue
        if not route.path.startswith("/api/"):
            continue
        if not _route_requires_auth_guard(route):
            continue

        calls = _route_dependency_calls(route)
        if calls.intersection(_AUTH_GUARD_CALLS):
            continue
        missing_routes.append(_route_display(route))

    if not missing_routes:
        return

    details = "\n".join(f"- {route}" for route in missing_routes)
    raise RuntimeError(
        "检测到未挂载鉴权依赖的 API 路由:\n"
        f"{details}\n"
        "请为以上路由显式声明 Depends(get_current_user) 或 Depends(get_current_auth_context)。"
    )
