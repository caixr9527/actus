#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/3 23:35
@Author : caixiaorong01@outlook.com
@File   : searxng.py
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class SearXNGSearchItem(BaseModel):
    """SearXNG 搜索结果条目"""
    title: str = Field(default="", description="搜索结果标题")
    url: str = Field(default="", description="搜索结果链接")
    content: str = Field(default="", description="搜索结果摘要")
    engine: Optional[str] = Field(default=None, description="结果来源搜索引擎")
    category: Optional[str] = Field(default=None, description="结果分类")
    score: Optional[float] = Field(default=None, description="结果评分")
    published_date: Optional[str] = Field(default=None, description="发布时间")
    thumbnail: Optional[str] = Field(default=None, description="缩略图地址")


class SearXNGSearchResult(BaseModel):
    """SearXNG 搜索响应"""
    query: str = Field(default="", description="实际执行的搜索词")
    number_of_results: int = Field(default=0, description="结果总数")
    results: list[SearXNGSearchItem] = Field(default_factory=list, description="搜索结果列表")
    suggestions: list[str] = Field(default_factory=list, description="搜索建议列表")
    answers: list[str] = Field(default_factory=list, description="直接答案列表")
    corrections: list[str] = Field(default_factory=list, description="拼写纠正列表")
    infoboxes: list[dict[str, Any]] = Field(default_factory=list, description="信息框列表")
    unresponsive_engines: list[str] = Field(default_factory=list, description="无响应引擎列表")


class SearXNGStatusResult(BaseModel):
    """SearXNG 服务状态"""
    base_url: str = Field(..., description="SearXNG 服务地址")
    available: bool = Field(..., description="服务是否可用")
    status_code: int = Field(..., description="HTTP 状态码")
    content_type: Optional[str] = Field(default=None, description="响应内容类型")


class SearXNGFetchPageResult(BaseModel):
    """SearXNG 单页读取结果"""
    url: str = Field(..., description="请求的原始页面地址")
    final_url: str = Field(..., description="最终页面地址")
    status_code: int = Field(..., description="HTTP 状态码")
    content_type: Optional[str] = Field(default=None, description="响应内容类型")
    title: str = Field(default="", description="页面标题")
    content: str = Field(default="", description="提取出的正文内容")
    excerpt: str = Field(default="", description="正文摘要")
    content_length: int = Field(default=0, description="正文总长度")
    truncated: bool = Field(default=False, description="正文是否被截断")
