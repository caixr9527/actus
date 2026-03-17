#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/13 15:33
@Author : caixiaorong01@outlook.com
@File   : exception_handlers.py
"""
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.application.errors.exceptions import AppException
from app.interfaces.schemas import Response

logger = logging.getLogger(__name__)


def _build_error_response(
        *,
        status_code: int,
        code: int,
        msg: str | None = None,
        data: Any = None,
        error_key: str | None = None,
        error_params: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=Response(
            code=code,
            msg=msg,
            data={} if data is None else data,
            error_key=error_key,
            error_params=error_params,
        ).model_dump()
    )


def _resolve_http_exception_msg(detail: Any) -> str:
    if isinstance(detail, str) and detail.strip():
        return detail
    return "HTTP请求错误"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def app_exception_handler(req: Request, e: AppException) -> JSONResponse:
        """App异常处理器"""
        logger.error("AppException: code=%s error_key=%s msg=%s", e.code, e.error_key, e.msg)
        return _build_error_response(
            status_code=e.status_code,
            code=e.code,
            msg=e.msg,
            data=e.data,
            error_key=e.error_key,
            error_params=e.error_params,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(req: Request, e: HTTPException) -> JSONResponse:
        """HTTP异常处理器"""
        msg = _resolve_http_exception_msg(e.detail)
        logger.error("HTTPException: status=%s detail=%s", e.status_code, e.detail)
        return _build_error_response(
            status_code=e.status_code,
            code=e.status_code,
            msg=msg,
            error_key=f"error.http.{e.status_code}",
        )

    @app.exception_handler(Exception)
    async def exception_handler(req: Request, e: Exception) -> JSONResponse:
        """异常处理器"""
        logger.exception("Unhandled exception: %s", str(e))
        return _build_error_response(
            status_code=500,
            code=500,
            msg="服务器异常",
            error_key="error.common.internal_server_error",
        )
