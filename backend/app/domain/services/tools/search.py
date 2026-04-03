#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/27 18:47
@Author : caixiaorong01@outlook.com
@File   : search.py
"""
from typing import Optional

from app.domain.external import Sandbox
from app.domain.models import ToolResult, SearchResults, SearchResultItem
from .base import BaseTool, tool


class SearchTool(BaseTool):
    name: str = "search"

    def __init__(self, sandbox: Sandbox) -> None:
        super().__init__()
        self.sandbox = sandbox

    @staticmethod
    def _normalize_total_results(value: object) -> int:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        normalized = str(value).strip().replace(",", "")
        if not normalized:
            return 0
        try:
            return int(float(normalized))
        except ValueError:
            return 0

    @classmethod
    def _to_search_results(cls, payload: object, time_range: Optional[str]) -> SearchResults:
        data = payload if isinstance(payload, dict) else {}
        raw_results = data.get("results") or []
        results = [
            SearchResultItem(
                url=str(item.get("url") or ""),
                title=str(item.get("title") or ""),
                snippet=str(item.get("content") or ""),
            )
            for item in raw_results
            if isinstance(item, dict)
        ]
        return SearchResults(
            query=str(data.get("query") or ""),
            date_range=time_range,
            total_results=cls._normalize_total_results(data.get("number_of_results")),
            results=results,
        )

    async def _search_via_searxng(
            self,
            query: str,
            categories: Optional[str] = None,
            engines: Optional[str] = None,
            language: Optional[str] = None,
            page: Optional[int] = None,
            time_range: Optional[str] = None,
            safesearch: Optional[int] = None,
    ) -> ToolResult[SearchResults]:
        sandbox_result = await self.sandbox.search_searxng(
            query=query,
            categories=categories,
            engines=engines,
            language=language,
            page=page,
            time_range=time_range,
            safesearch=safesearch,
        )
        if not sandbox_result.success:
            return ToolResult(
                success=False,
                message=sandbox_result.message,
                data=None,
            )
        return ToolResult(
            success=True,
            message=sandbox_result.message,
            data=self._to_search_results(payload=sandbox_result.data, time_range=time_range),
        )

    @tool(
        name="search_web",
        description="""
            当你需要回答有关当前事件、网页信息检索、课程推荐、资料收集等问题时非常有用。
            输入应该是一个搜索查询。
        """,
        parameters={
            "query": {
                "type": "string",
                "description": "针对搜索引擎优化的查询字符串。请提取问题中核心实体和关键词（3-5个）,避免使用完整的自然语言问句（例如将'今天北京的天气怎么样' 改为 '北京 天气'"
            },
            "categories": {
                "type": "string",
                "description": "(可选)SearXNG 搜索分类，例如 `general`、`news`。多个分类使用逗号分隔。"
            },
            "engines": {
                "type": "string",
                "description": "(可选)指定使用的搜索引擎，多个引擎使用逗号分隔。"
            },
            "language": {
                "type": "string",
                "description": "(可选)结果语言，例如 `zh-CN`、`en-US`。"
            },
            "page": {
                "type": "integer",
                "description": "(可选)结果页码，从 1 开始。"
            },
            "time_range": {
                "type": "string",
                "enum": ["day", "month", "year"],
                "description": "(可选)SearXNG 时间范围过滤。"
            },
            "safesearch": {
                "type": "integer",
                "enum": [0, 1, 2],
                "description": "(可选)SearXNG 安全搜索级别。"
            }
        },
        required=["query"]
    )
    async def search_web(
            self,
            query: str,
            categories: Optional[str] = None,
            engines: Optional[str] = None,
            language: Optional[str] = None,
            page: Optional[int] = None,
            time_range: Optional[str] = None,
            safesearch: Optional[int] = None,
    ) -> ToolResult[SearchResults]:
        return await self._search_via_searxng(
            query=query,
            categories=categories,
            engines=engines,
            language=language,
            page=page,
            time_range=time_range,
            safesearch=safesearch,
        )
