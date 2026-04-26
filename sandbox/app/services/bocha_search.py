#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/18
@File   : bocha_search.py
"""
import asyncio
import json
import logging
import os
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.interfaces.errors import AppException, BadRequestException
from app.models import SearXNGSearchItem, SearXNGSearchResult

logger = logging.getLogger(__name__)


class BochaSearchService:
    """博查 web-search 服务（对外返回 SearXNG 兼容结构）。"""

    def __init__(self) -> None:
        self.base_url = os.getenv("BOCHA_BASE_URL", "https://api.bocha.cn").rstrip("/")
        self.api_key = str(os.getenv("BOCHA_API_KEY", "")).strip()
        self.default_count = int(os.getenv("BOCHA_DEFAULT_COUNT", "10"))

    @staticmethod
    def _pick_result_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload.get("data")
        if isinstance(data, dict):
            if isinstance(data.get("results"), list):
                return [item for item in data["results"] if isinstance(item, dict)]
            web_pages = data.get("webPages")
            if isinstance(web_pages, dict) and isinstance(web_pages.get("value"), list):
                return [item for item in web_pages["value"] if isinstance(item, dict)]
        if isinstance(payload.get("results"), list):
            return [item for item in payload["results"] if isinstance(item, dict)]
        web_pages = payload.get("webPages")
        if isinstance(web_pages, dict) and isinstance(web_pages.get("value"), list):
            return [item for item in web_pages["value"] if isinstance(item, dict)]
        return []

    @staticmethod
    def _extract_query(payload: dict[str, Any], fallback: str) -> str:
        data = payload.get("data")
        if isinstance(data, dict):
            query_context = data.get("queryContext")
            if isinstance(query_context, dict):
                original_query = str(query_context.get("originalQuery") or "").strip()
                if original_query:
                    return original_query
            data_query = str(data.get("query") or "").strip()
            if data_query:
                return data_query
        top_query = str(payload.get("query") or "").strip()
        return top_query or fallback

    @staticmethod
    def _build_search_item(item: dict[str, Any]) -> Optional[SearXNGSearchItem]:
        url = str(item.get("url")).strip()
        if not url:
            return None
        title = str(item.get("name")).strip()
        content = str(item.get("summary") or item.get("snippet") or "").strip()
        published_date = str(
            item.get("publishedDate")
            or item.get("datePublished")
            or item.get("dateLastCrawled")
            or ""
        ).strip() or None
        return SearXNGSearchItem(
            title=title,
            url=url,
            content=content,
            engine="bocha",
            published_date=published_date,
            score=1.0,
        )

    async def search(
            self,
            query: str,
    ) -> SearXNGSearchResult:

        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise BadRequestException("搜索词不能为空")
        if not self.api_key:
            raise AppException("BOCHA_API_KEY 未配置")

        request_body = {
            "query": normalized_query,
            "freshness": "noLimit",
            "summary": True,
            "count": max(1, min(self.default_count, 50)),
        }
        request_url = f"{self.base_url}/v1/web-search"
        logger.info("请求Bocha搜索: %s", request_url)

        def sync_request() -> str:
            request = Request(
                url=request_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                method="POST",
                data=json.dumps(request_body).encode("utf-8"),
            )
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")

        try:
            body = await asyncio.to_thread(sync_request)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.error("Bocha搜索请求失败, status=%s, detail=%s", exc.code, detail)
            raise BadRequestException(f"Bocha搜索请求失败: HTTP {exc.code}")
        except URLError as exc:
            logger.error("Bocha搜索连接失败: %s", exc)
            raise AppException(f"Bocha搜索连接失败: {exc}")
        except Exception as exc:
            logger.error("Bocha搜索调用异常: %s", exc)
            raise AppException(f"Bocha搜索调用异常: {exc}")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.error("Bocha返回非JSON响应: %s", exc)
            raise AppException(f"Bocha返回非JSON响应: {exc}")

        raw_results = self._pick_result_list(payload)
        results: list[SearXNGSearchItem] = []
        for raw_item in raw_results:
            search_item = self._build_search_item(raw_item)
            if search_item is not None:
                results.append(search_item)

        return SearXNGSearchResult(
            query=self._extract_query(payload=payload, fallback=normalized_query),
            number_of_results=len(results),
            results=results,
            suggestions=[],
            answers=[],
            corrections=[],
            infoboxes=[],
            unresponsive_engines=[],
        )
