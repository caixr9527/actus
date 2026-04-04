#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/3 23:37
@Author : caixiaorong01@outlook.com
@File   : searxng.py
"""
import asyncio
import json
import logging
import os
from typing import Optional, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from app.interfaces.errors import BadRequestException, AppException
from app.models import SearXNGFetchPageResult, SearXNGSearchItem, SearXNGSearchResult, SearXNGStatusResult

logger = logging.getLogger(__name__)


class SearXNGService:
    """SearXNG 搜索服务"""

    def __init__(self) -> None:
        self.base_url = os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8082").rstrip("/")
        self.timeout_seconds = int(os.getenv("SEARXNG_TIMEOUT_SECONDS", "30"))
        self.crawl4ai_cdp_url = os.getenv("CRAWL4AI_CDP_URL", "http://127.0.0.1:9222")

    def _build_url(self, path: str, params: Optional[dict[str, Any]] = None) -> str:
        """拼接请求地址并过滤空参数。"""
        normalized_path = path if path.startswith("/") else f"/{path}"
        if not params:
            return f"{self.base_url}{normalized_path}"
        clean_params = {
            key: value
            for key, value in params.items()
            if value is not None and value != ""
        }
        if not clean_params:
            return f"{self.base_url}{normalized_path}"
        return f"{self.base_url}{normalized_path}?{urlencode(clean_params, doseq=True)}"

    async def _request(self, path: str, params: Optional[dict[str, Any]] = None) -> tuple[int, dict[str, str], str]:
        """通过标准库请求容器内的 SearXNG 服务。"""
        url = self._build_url(path=path, params=params)
        logger.info(f"请求SearXNG服务: {url}")

        def sync_request() -> tuple[int, dict[str, str], str]:
            request = Request(
                url=url,
                headers={
                    "Accept": "application/json, text/html;q=0.9",
                    "User-Agent": "ActusSandbox/1.0",
                },
                method="GET",
            )
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
                return response.status, dict(response.headers.items()), body

        try:
            return await asyncio.to_thread(sync_request)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.error(f"SearXNG请求失败, 状态码: {exc.code}, 响应: {detail}")
            raise BadRequestException(f"SearXNG请求失败: HTTP {exc.code}")
        except URLError as exc:
            logger.error(f"SearXNG连接失败: {exc}")
            raise AppException(f"SearXNG连接失败: {exc}")
        except Exception as exc:
            logger.error(f"SearXNG调用异常: {exc}")
            raise AppException(f"SearXNG调用异常: {exc}")

    @staticmethod
    def _normalize_page_url(url: str) -> str:
        normalized_url = str(url or "").strip()
        if not normalized_url:
            raise BadRequestException("页面地址不能为空")

        parsed = urlparse(normalized_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BadRequestException("页面地址必须是有效的 http 或 https URL")

        return normalized_url

    @staticmethod
    def _pick_header_value(headers: Optional[dict[str, Any]], key: str) -> Optional[str]:
        if not headers:
            return None
        for header_key, header_value in headers.items():
            if str(header_key).lower() == key.lower():
                return str(header_value)
        return None

    @staticmethod
    def _extract_crawl_content(result: Any) -> str:
        markdown = getattr(result, "markdown", None)
        if isinstance(markdown, str):
            return markdown.strip()

        if markdown is not None:
            for attr_name in ("fit_markdown", "raw_markdown", "markdown_with_citations"):
                value = getattr(markdown, attr_name, None)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        extracted_content = getattr(result, "extracted_content", None)
        if isinstance(extracted_content, str):
            return extracted_content.strip()

        return ""

    def _build_crawl4ai_browser_config(self, browser_config_cls: Any) -> Any:
        return browser_config_cls(
            browser_mode="custom",
            cdp_url=self.crawl4ai_cdp_url,
            headless=True,
            text_mode=True,
            verbose=False,
        )

    async def get_status(self) -> SearXNGStatusResult:
        """检查 SearXNG 服务是否可访问。"""
        status_code, headers, _ = await self._request(path="/")
        return SearXNGStatusResult(
            base_url=self.base_url,
            available=200 <= status_code < 400,
            status_code=status_code,
            content_type=headers.get("Content-Type"),
        )

    async def search(
            self,
            query: str,
            categories: Optional[str] = None,
            engines: Optional[str] = None,
            language: Optional[str] = None,
            page: Optional[int] = None,
            time_range: Optional[str] = None,
            safesearch: Optional[int] = None,
    ) -> SearXNGSearchResult:
        """调用 SearXNG JSON API 执行搜索。"""
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise BadRequestException("搜索词不能为空")

        _, _, body = await self._request(
            path="/search",
            params={
                "q": normalized_query,
                "categories": categories,
                "engines": engines,
                "language": language,
                "page": page,
                "time_range": time_range,
                "safesearch": safesearch,
                "format": "json",
            },
        )

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.error(f"SearXNG返回非JSON响应: {exc}")
            raise AppException(f"SearXNG返回非JSON响应: {exc}")

        raw_results = payload.get("results") or []
        search_results = [
            SearXNGSearchItem(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                content=str(item.get("content") or ""),
                engine=item.get("engine"),
                category=item.get("category"),
                score=item.get("score"),
                published_date=item.get("publishedDate"),
                thumbnail=item.get("thumbnail"),
            )
            for item in raw_results
            if isinstance(item, dict)
        ]

        return SearXNGSearchResult(
            query=str(payload.get("query") or normalized_query),
            number_of_results=len(search_results),
            results=search_results,
            suggestions=[str(item) for item in (payload.get("suggestions") or [])],
            answers=[str(item) for item in (payload.get("answers") or [])],
            corrections=[str(item) for item in (payload.get("corrections") or [])],
            infoboxes=[item for item in (payload.get("infoboxes") or []) if isinstance(item, dict)],
            unresponsive_engines=[str(item) for item in (payload.get("unresponsive_engines") or [])],
        )

    async def fetch_page(self, url: str, max_chars: Optional[int] = 20000) -> SearXNGFetchPageResult:
        """使用 crawl4ai 读取单个页面内容。"""
        normalized_url = self._normalize_page_url(url)
        if max_chars is not None and max_chars <= 0:
            raise BadRequestException("max_chars 必须大于 0")

        browser_config = self._build_crawl4ai_browser_config(BrowserConfig)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=self.timeout_seconds * 1000,
        )

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=normalized_url, config=run_config)
        except BadRequestException:
            raise
        except Exception as exc:
            logger.error(f"crawl4ai读取页面异常: {exc}")
            raise AppException(f"页面读取失败: {exc}")

        if not getattr(result, "success", False):
            error_message = str(getattr(result, "error_message", None) or "页面读取失败")
            status_code = getattr(result, "status_code", None)
            if isinstance(status_code, int) and 400 <= status_code < 500:
                raise BadRequestException(error_message)
            raise AppException(error_message)

        full_content = self._extract_crawl_content(result)
        if not full_content:
            raise AppException("页面读取成功，但未提取到正文内容")

        truncated = max_chars is not None and len(full_content) > max_chars
        content = full_content[:max_chars] if truncated and max_chars is not None else full_content
        metadata = getattr(result, "metadata", None) or {}
        title = str(metadata.get("title") or "") if isinstance(metadata, dict) else ""
        response_headers = getattr(result, "response_headers", None)
        content_type = self._pick_header_value(response_headers, "Content-Type")
        status_code = getattr(result, "status_code", None)

        return SearXNGFetchPageResult(
            url=normalized_url,
            final_url=str(getattr(result, "url", None) or normalized_url),
            status_code=int(status_code) if isinstance(status_code, int) else 200,
            content_type=content_type,
            title=title,
            content=content,
            excerpt=full_content[:280],
            content_length=len(full_content),
            truncated=truncated,
        )
