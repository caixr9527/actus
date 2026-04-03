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
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.interfaces.errors import BadRequestException, AppException
from app.models import SearXNGSearchItem, SearXNGSearchResult, SearXNGStatusResult

logger = logging.getLogger(__name__)


class SearXNGService:
    """SearXNG 搜索服务"""

    def __init__(self) -> None:
        self.base_url = os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8082").rstrip("/")
        self.timeout_seconds = int(os.getenv("SEARXNG_TIMEOUT_SECONDS", "30"))

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
    def _normalize_number_of_results(value: Any) -> int:
        """兼容 SearXNG 返回的不同结果数字格式。"""
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
            number_of_results=self._normalize_number_of_results(payload.get("number_of_results")),
            results=search_results,
            suggestions=[str(item) for item in (payload.get("suggestions") or [])],
            answers=[str(item) for item in (payload.get("answers") or [])],
            corrections=[str(item) for item in (payload.get("corrections") or [])],
            infoboxes=[item for item in (payload.get("infoboxes") or []) if isinstance(item, dict)],
            unresponsive_engines=[str(item) for item in (payload.get("unresponsive_engines") or [])],
        )
