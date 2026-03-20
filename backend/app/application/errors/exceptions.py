#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/13 15:55
@Author : caixiaorong01@outlook.com
@File   : exceptions.py
"""
from typing import Any, Mapping


class AppException(RuntimeError):

    def __init__(self,
                 code: int = 400,
                 status_code: int = 400,
                 msg: str | None = "应用发生错误",
                 data: Any = None,
                 error_key: str | None = None,
                 error_params: Mapping[str, Any] | None = None):
        self.code = code
        self.status_code = status_code
        self.msg = msg
        self.data = data
        self.error_key = error_key
        self.error_params = dict(error_params) if error_params is not None else None
        super().__init__(msg or "")


class BadRequestError(AppException):

    def __init__(self,
                 msg: str = "客户端请求错误,请检查后重试",
                 error_key: str = "error.common.bad_request",
                 error_params: Mapping[str, Any] | None = None):
        super().__init__(
            code=400,
            status_code=400,
            msg=msg,
            error_key=error_key,
            error_params=error_params,
        )


class UnauthorizedError(AppException):

    def __init__(self,
                 msg: str = "未授权访问",
                 error_key: str = "error.common.unauthorized",
                 error_params: Mapping[str, Any] | None = None):
        super().__init__(
            code=401,
            status_code=401,
            msg=msg,
            error_key=error_key,
            error_params=error_params,
        )


class NotFoundError(AppException):

    def __init__(self,
                 msg: str = "资源未找到",
                 error_key: str = "error.common.not_found",
                 error_params: Mapping[str, Any] | None = None):
        super().__init__(
            code=404,
            status_code=404,
            msg=msg,
            error_key=error_key,
            error_params=error_params,
        )


class ValidationError(AppException):

    def __init__(self,
                 msg: str = "请求参数错误",
                 error_key: str = "error.common.validation_failed",
                 error_params: Mapping[str, Any] | None = None):
        super().__init__(
            code=422,
            status_code=422,
            msg=msg,
            error_key=error_key,
            error_params=error_params,
        )



class TooManyRequestsError(AppException):

    def __init__(self,
                 msg: str = "请求过多,请稍后重试",
                 error_key: str = "error.common.too_many_requests",
                 error_params: Mapping[str, Any] | None = None):
        super().__init__(
            code=429,
            status_code=429,
            msg=msg,
            error_key=error_key,
            error_params=error_params,
        )


class ServerError(AppException):

    def __init__(self,
                 msg: str = "服务器异常",
                 error_key: str = "error.common.internal_server_error",
                 error_params: Mapping[str, Any] | None = None):
        super().__init__(
            code=500,
            status_code=500,
            msg=msg,
            error_key=error_key,
            error_params=error_params,
        )
