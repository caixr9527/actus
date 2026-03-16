#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/16 17:15
@Author : caixiaorong01@outlook.com
@File   : request_security.py
"""
from fastapi import Request, Response

from app.application.errors import BadRequestError
from core.config import get_settings

_AUTH_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "X-Frame-Options": "SAMEORIGIN",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}

_HSTS_HEADER_VALUE = "max-age=31536000; includeSubDomains"


def _resolve_request_scheme(request: Request) -> str:
    """解析请求协议，优先使用反向代理透传的 x-forwarded-proto。"""
    x_forwarded_proto = request.headers.get("x-forwarded-proto")
    if x_forwarded_proto:
        return x_forwarded_proto.split(",")[0].strip().lower()
    return request.url.scheme.strip().lower()


def require_https_request(request: Request) -> None:
    """按配置要求强制请求使用 HTTPS。"""
    settings = get_settings()
    if not settings.auth_require_https:
        return

    if _resolve_request_scheme(request) == "https":
        return

    raise BadRequestError(msg="当前环境仅允许通过 HTTPS 访问该接口")


def apply_auth_security_headers(request: Request, response: Response) -> None:
    """为认证相关响应附加统一安全头与禁缓存策略。"""
    for header_name, header_value in _AUTH_SECURITY_HEADERS.items():
        response.headers[header_name] = header_value

    if _resolve_request_scheme(request) == "https":
        response.headers["Strict-Transport-Security"] = _HSTS_HEADER_VALUE
