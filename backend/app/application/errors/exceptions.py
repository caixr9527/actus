#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/13 15:55
@Author : caixiaorong01@outlook.com
@File   : exceptions.py
"""
from typing import Any


class AppException(RuntimeError):

    def __init__(self,
                 code: int = 400,
                 status_code: int = 400,
                 msg: str = "应用发生错误",
                 data: Any = None):
        self.code = code
        self.status_code = status_code
        self.msg = msg
        self.data = data
        super().__init__()


class BadRequestError(AppException):

    def __init__(self, msg: str = "客户端请求错误,请检查后重试"):
        super().__init__(code=400, status_code=400, msg=msg)


class UnauthorizedError(AppException):

    def __init__(self, msg: str = "未授权访问"):
        super().__init__(code=401, status_code=401, msg=msg)


class NotFoundError(AppException):

    def __init__(self, msg: str = "资源未找到"):
        super().__init__(code=404, status_code=404, msg=msg)


class ValidationError(AppException):

    def __init__(self, msg: str = "请求参数错误"):
        super().__init__(code=422, status_code=422, msg=msg)



class TooManyRequestsError(AppException):

    def __init__(self, msg: str = "请求过多,请稍后重试"):
        super().__init__(code=429, status_code=429, msg=msg)


class ServerError(AppException):

    def __init__(self, msg: str = "服务器异常"):
        super().__init__(code=500, status_code=500, msg=msg)
