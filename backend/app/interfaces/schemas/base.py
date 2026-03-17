#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/12 17:47
@Author : caixiaorong01@outlook.com
@File   : base.py
"""
from typing import TypeVar, Generic, Optional, Any

from pydantic import BaseModel, Field

T = TypeVar('T')


class Response(BaseModel, Generic[T]):
    """基础响应结构"""
    code: int = 200
    msg: Optional[str] = "success"
    data: Optional[T] = Field(default_factory=dict)
    error_key: Optional[str] = None
    error_params: Optional[dict[str, Any]] = None

    @staticmethod
    def success(
            data: Optional[T] = None,
            msg: Optional[str] = "success",
            error_key: Optional[str] = None,
            error_params: Optional[dict[str, Any]] = None,
    ) -> "Response[T]":
        """成功响应"""
        return Response(
            code=200,
            msg=msg,
            data=data if data is not None else {},
            error_key=error_key,
            error_params=error_params,
        )

    @staticmethod
    def fail(
            code: int,
            msg: Optional[str] = None,
            data: Optional[T] = None,
            error_key: Optional[str] = None,
            error_params: Optional[dict[str, Any]] = None,
    ) -> "Response[T]":
        """失败响应"""
        return Response(
            code=code,
            msg=msg,
            data=data if data is not None else {},
            error_key=error_key,
            error_params=error_params,
        )
