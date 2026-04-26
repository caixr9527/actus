#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace research capability。"""

from typing import Optional

from app.domain.external import Sandbox
from app.domain.models import FetchedPage, SearchResultItem, SearchResults, ToolResult
from app.domain.services.tools.base import BaseTool, tool
from ..service import WorkspaceRuntimeService


class WorkspaceResearchCapability(BaseTool):
    name: str = "search"

    def __init__(
            self,
            sandbox: Sandbox,
            workspace_runtime_service: WorkspaceRuntimeService,
    ) -> None:
        super().__init__()
        self._sandbox = sandbox
        self._workspace_runtime_service = workspace_runtime_service

    async def _record_search_results(
            self,
            *,
            query: str,
            search_results: SearchResults,
    ) -> None:
        await self._workspace_runtime_service.record_search_results(
            query=query,
            candidate_links=[
                {
                    "title": str(item.title or "").strip(),
                    "url": str(item.url or "").strip(),
                    "snippet": str(item.snippet or "").strip(),
                }
                for item in list(search_results.results or [])
                if str(item.url or "").strip()
            ],
        )

    async def _record_fetched_page(self, page: FetchedPage) -> None:
        await self._workspace_runtime_service.record_fetched_page_summary(
            page_summary={
                "title": str(page.title or "").strip(),
                "url": str(page.final_url or page.url or "").strip(),
                "excerpt": str(page.excerpt or page.content or "").strip()[:240],
            }
        )

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
        sandbox_result = await self._sandbox.search(
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
        search_results = self._to_search_results(payload=sandbox_result.data, time_range=time_range)
        await self._record_search_results(query=query, search_results=search_results)
        return ToolResult(
            success=True,
            message=sandbox_result.message,
            data=search_results,
        )

    async def _fetch_page_via_searxng(
            self,
            url: str,
            max_chars: Optional[int] = None,
    ) -> ToolResult[FetchedPage]:
        sandbox_result = await self._sandbox.fetch_searxng_page(
            url=url,
            max_chars=max_chars,
        )
        if not sandbox_result.success:
            return ToolResult(
                success=False,
                message=sandbox_result.message,
                data=None,
            )
        fetched_page = self._to_fetched_page(payload=sandbox_result.data, max_chars=max_chars)
        await self._record_fetched_page(fetched_page)
        return ToolResult(
            success=True,
            message=sandbox_result.message,
            data=fetched_page,
        )

    @tool(
        name="search_web",
        description="""
            当你需要回答有关当前事件、网页信息检索、课程推荐、资料收集等问题时非常有用。
            输入应该是一条原子性的自然语言检索请求。

            注意事项：
            - 使用“单目标、单主题”的自然语言表达（问句或陈述句均可），清楚表达当前要解决的一个问题
            - 不要把输入压缩成碎片化词串；避免只写名词堆叠、标签堆叠或多个断开的短语
            - 禁止关键词式输入示例：
              `上海 旅游 攻略 2天 预算3000`、
              `2026 五一 杭州 周边 亲子 自驾 推荐`、
              `Python LangGraph human_wait 错误 修复`、
              `新能源 补贴 政策 最新`、
              `AI编程助手 IDE 支持 对比`、
              `上海 上牌 流程 新能源 2026`、
              `项目管理 工具 看板 协作 推荐`
            - 正确示例：`当前主流 AI 编程助手及其支持的 IDE`、`新能源汽车上牌流程的最新要求`、`LangGraph human-in-the-loop 的实现方式`
            - 渐进收敛原则：仅在结果不够准确时，再逐轮增加一个筛选条件，避免一次加入多个条件导致召回下降
            - 若用户任务包含多个子问题，必须拆成多次搜索；每次都用一句自然语言表达一个可验证的问题，按结果逐步推进
            - 该工具用于“找候选来源和摘要”，不是直接读取网页全文；需要核验正文时再对单个链接调用 `fetch_page`
            - 当结果为空、相关性差、中文结果不足或时效性不理想时，可显式指定 `engines`（优先 `bing`、`baidu`），并配合 `language`、`categories`、`time_range` 收敛范围
        """,
        parameters={
            "query": {
                "type": "string",
                "description": "一条原子性的自然语言检索请求，只描述当前要验证的单一主题；若结果不够再逐轮补充一个条件。"
                               "表达形式可用问句或陈述句。"
                               "正确示例：'当前主流 AI 编程助手及其支持的 IDE'、'主流项目管理工具的看板协作能力'、'新能源汽车上牌流程的最新要求'。"
                               "禁止示例：'上海 旅游 攻略 2天 预算3000'、'AI编程助手 IDE 支持 对比'、'上海 上牌 流程 新能源 2026'。"
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
