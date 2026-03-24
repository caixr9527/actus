#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/27 18:47
@Author : caixiaorong01@outlook.com
@File   : search.py
"""
from typing import Optional

from .base import BaseTool, tool
from app.domain.external import SearchEngine
from app.domain.models import ToolResult, SearchResults


class SearchTool(BaseTool):

    name: str = "search"

    def __init__(self, search_engine: SearchEngine) -> None:
        super().__init__()
        self.search_engine = search_engine

    @tool(
        name="search_web",
        description="""
            搜索网络获取信息，并利用搜索结果来回答用户的问题。
            当你需要回答有关当前事件的问题时非常有用。
            输入应该是一个搜索查询。
        """,
        parameters={
            "query": {
                "type": "string",
                "description": "针对搜索引擎优化的查询字符串。请提取问题中核心实体和关键词（3-5个）,避免使用完整的自然语言问句（例如将'今天北京的天气怎么样' 改为 '北京 天气'"
            },
            "data_range": {
                "type": "string",
                "enum": ["all", "past_hour", "past_day", "past_week", "past_month", "past_year"],
                "description": "(可选)搜索结果的时间范围过滤。当用户询问特定时效的新闻或事件时（如'昨天'、'上周'），必须指定此参数。默认为 'all'"
            }
        },
        required=["query"]
    )
    async def search_web(self, query: str, data_range: Optional[str] = None) -> ToolResult[SearchResults]:
        # 显式关键字传参，避免 SearchEngine 协议参数名演进时出现位置参数误用。
        return await self.search_engine.invoke(query=query, date_range=data_range)
