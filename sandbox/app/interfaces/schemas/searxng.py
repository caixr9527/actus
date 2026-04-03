#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/4 00:01
@Author : caixiaorong01@outlook.com
@File   : searxng.py
"""
from typing import Optional

from pydantic import BaseModel, Field


class SearXNGSearchRequest(BaseModel):
    """SearXNG 搜索请求结构体"""
    query: str = Field(..., description="搜索词")
    categories: Optional[str] = Field(default=None, description="搜索分类")
    engines: Optional[str] = Field(default=None, description="指定引擎，多个用逗号分隔")
    language: Optional[str] = Field(default=None, description="搜索语言")
    page: Optional[int] = Field(default=None, description="分页页码")
    time_range: Optional[str] = Field(default=None, description="时间范围")
    safesearch: Optional[int] = Field(default=None, description="安全搜索级别")
