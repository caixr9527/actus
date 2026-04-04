#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/27 18:47
@Author : caixiaorong01@outlook.com
@File   : search.py
"""
from typing import Optional

from app.domain.external import Sandbox
from app.domain.models import FetchedPage, SearchResultItem, SearchResults, ToolResult
from .base import BaseTool, tool


class SearchTool(BaseTool):
    name: str = "search"

    def __init__(self, sandbox: Sandbox) -> None:
        super().__init__()
        self.sandbox = sandbox

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
            total_results=len(results),
            results=results,
        )

    @staticmethod
    def _to_fetched_page(payload: object, max_chars: Optional[int]) -> FetchedPage:
        data = payload if isinstance(payload, dict) else {}
        return FetchedPage(
            url=str(data.get("url") or ""),
            final_url=str(data.get("final_url") or ""),
            status_code=int(data.get("status_code") or 0),
            content_type=str(data.get("content_type") or ""),
            title=str(data.get("title") or ""),
            content=str(data.get("content") or ""),
            excerpt=str(data.get("excerpt") or ""),
            content_length=int(data.get("content_length") or 0),
            truncated=bool(data.get("truncated", False)),
            max_chars=max_chars,
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

    async def _fetch_page_via_searxng(
            self,
            url: str,
            max_chars: Optional[int] = None,
    ) -> ToolResult[FetchedPage]:
        sandbox_result = await self.sandbox.fetch_searxng_page(
            url=url,
            max_chars=max_chars,
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
            data=self._to_fetched_page(payload=sandbox_result.data, max_chars=max_chars),
        )

    @tool(
        name="search_web",
        description="""
            当你需要回答有关当前事件、网页信息检索、课程推荐、资料收集等问题时非常有用。
            输入应该是一个搜索查询。

            注意事项：
            - 优先把问题改写成适合搜索引擎的短查询，只保留核心实体、主题词和限定词，不要直接使用完整口语问句
            - 该工具用于“找结果链接和摘要”，不是直接读取网页全文；如果已经拿到候选链接并需要查看正文，应继续使用 `fetch_page`
            - 当结果为空、相关性差、中文结果不足或时效性不理想时，可以显式指定 `engines`，优先尝试 `bing` 或 `baidu`
            - 当需要缩小范围时，优先结合 `language`、`categories`、`time_range` 调整，而不是反复使用同一个宽泛查询
            - 单次搜索先尽量聚焦一个主题；如果用户问题包含多个独立主题，应拆成多次搜索分别进行
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
                "description": "(可选)指定使用的搜索引擎，多个引擎使用逗号分隔。例如 `google`、`bing`、`duckduckgo`、`baidu`"
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

    @tool(
        name="fetch_page",
        description="""
            读取单个网页 URL 的标题和正文内容。

            使用场景：
            - 你已经通过 `search_web` 拿到候选链接，需要进一步读取某一个结果页的正文
            - 用户明确给出了一个具体网页 URL，需要提取该页面内容

            强约束：
            - 该工具一次只读取一个 URL
            - 只用于读取单个页面内容，不用于站点遍历、批量抓取、递归爬取或自动跟进多个链接
            - 如果需要比较多个页面，必须由你显式逐个调用本工具
            - 如果任务需要登录、点击、输入、滚动、翻页或其它页面交互，不要使用本工具，应改用浏览器工具
            - 通常应先用 `search_web` 找到候选结果，再对其中最相关的单个链接调用本工具
        """,
        parameters={
            "url": {
                "type": "string",
                "description": "要读取的单个网页完整 URL，必须包含协议前缀，例如 https://example.com/article 。不要传站点首页用于批量发现链接。"
            },
            "max_chars": {
                "type": "integer",
                "description": "(可选)该单个页面正文的最大返回字符数。仅限制当前这一次页面读取，不代表批量抓取。"
            }
        },
        required=["url"]
    )
    async def fetch_page(
            self,
            url: str,
            max_chars: Optional[int] = None,
    ) -> ToolResult[FetchedPage]:
        return await self._fetch_page_via_searxng(
            url=url,
            max_chars=max_chars,
        )
