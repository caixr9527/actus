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
from urllib.parse import urlencode, urlparse, urlsplit, urlunsplit, parse_qsl
from urllib.request import Request, urlopen

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from app.interfaces.errors import BadRequestException, AppException
from app.models import SearXNGFetchPageResult, SearXNGSearchItem, SearXNGSearchResult, SearXNGStatusResult
from app.services.search_quality_policy import SearchQualityPolicy, get_search_quality_policy

logger = logging.getLogger(__name__)


class SearXNGService:
    """SearXNG 搜索服务"""

    def __init__(self) -> None:
        self.base_url = os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8082").rstrip("/")
        self.timeout_seconds = int(os.getenv("SEARXNG_TIMEOUT_SECONDS", "30"))
        self.crawl4ai_cdp_url = os.getenv("CRAWL4AI_CDP_URL", "http://127.0.0.1:9222")
        self.quality_policy = get_search_quality_policy()

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
        parsed_cdp_url = urlparse(self.crawl4ai_cdp_url)
        cdp_host = str(parsed_cdp_url.hostname or "127.0.0.1").strip() or "127.0.0.1"
        cdp_port = int(parsed_cdp_url.port or 9222)
        return browser_config_cls(
            browser_mode="builtin",
            host=cdp_host,
            debugging_port=cdp_port,
            headless=True,
            text_mode=True,
            verbose=False,
        )

    @staticmethod
    def _get_crawl4ai_components() -> tuple[Any, Any, Any, Any]:
        return AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

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
        search_results, quality_stats = self._quality_filter_and_rank_results(
            raw_results=raw_results,
            query=normalized_query,
            policy=self.quality_policy,
        )
        top_domains = self._collect_top_domains(search_results, limit=5)
        logger.info(
            "SearXNG搜索质控完成: query=%s raw=%s filtered=%s deduped=%s final=%s dropped=%s top_domains=%s",
            normalized_query,
            int(quality_stats.get("raw_count", 0)),
            int(quality_stats.get("filtered_count", 0)),
            int(quality_stats.get("deduped_count", 0)),
            len(search_results),
            dict(quality_stats.get("drop_reason_stats") or {}),
            top_domains,
        )

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

    def _quality_filter_and_rank_results(
            self,
            *,
            raw_results: Any,
            query: str,
            policy: SearchQualityPolicy,
    ) -> tuple[list[SearXNGSearchItem], dict[str, Any]]:
        query_tokens = self._tokenize_query(query, policy=policy)
        raw_count = len(raw_results) if isinstance(raw_results, list) else 0
        if not isinstance(raw_results, list):
            return [], {
                "raw_count": raw_count,
                "filtered_count": 0,
                "deduped_count": 0,
                "drop_reason_stats": {},
            }

        drop_reason_stats: dict[str, int] = {}
        filtered_entries: list[dict[str, Any]] = []
        dedupe_keys: set[str] = set()

        for item in raw_results:
            if not isinstance(item, dict):
                self._inc_drop_reason(drop_reason_stats, "invalid_item")
                continue
            normalized_url = self._normalize_search_result_url(item.get("url"))
            if not normalized_url:
                self._inc_drop_reason(drop_reason_stats, "invalid_url")
                continue

            blocked_reason = self._resolve_intermediate_url_reason(normalized_url, policy=policy)
            if blocked_reason:
                self._inc_drop_reason(drop_reason_stats, blocked_reason)
                continue

            dedupe_key = self._build_canonical_url_key(normalized_url, policy=policy)
            if not dedupe_key:
                self._inc_drop_reason(drop_reason_stats, "invalid_canonical")
                continue
            if dedupe_key in dedupe_keys:
                self._inc_drop_reason(drop_reason_stats, "duplicate_url")
                continue
            dedupe_keys.add(dedupe_key)

            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            if len(title) == 0 and len(content) == 0:
                self._inc_drop_reason(drop_reason_stats, "empty_text")
                continue

            domain = self._extract_domain(normalized_url)
            quality_score = self._compute_search_item_score(
                query_tokens=query_tokens,
                title=title,
                content=content,
                domain=domain,
                engine=item.get("engine"),
                origin_score=item.get("score"),
                policy=policy,
            )
            filtered_entries.append(
                {
                    "title": title,
                    "url": normalized_url,
                    "content": content,
                    "engine": item.get("engine"),
                    "category": item.get("category"),
                    "score": item.get("score"),
                    "published_date": item.get("publishedDate"),
                    "thumbnail": item.get("thumbnail"),
                    "quality_score": quality_score,
                    "domain": domain,
                }
            )

        filtered_entries.sort(
            key=lambda row: (
                float(row.get("quality_score") or 0.0),
                len(str(row.get("title") or "")),
                len(str(row.get("content") or "")),
            ),
            reverse=True,
        )
        reranked_entries = self._rerank_for_domain_diversity(filtered_entries, policy=policy)
        normalized_results = [
            SearXNGSearchItem(
                title=str(entry.get("title") or ""),
                url=str(entry.get("url") or ""),
                content=str(entry.get("content") or ""),
                engine=entry.get("engine"),
                category=entry.get("category"),
                score=entry.get("score"),
                published_date=entry.get("published_date"),
                thumbnail=entry.get("thumbnail"),
            )
            for entry in reranked_entries
        ]
        return normalized_results, {
            "raw_count": raw_count,
            "filtered_count": len(filtered_entries),
            "deduped_count": len(dedupe_keys),
            "drop_reason_stats": drop_reason_stats,
        }

    @staticmethod
    def _inc_drop_reason(stats: dict[str, int], reason: str) -> None:
        normalized_reason = str(reason or "").strip() or "unknown"
        stats[normalized_reason] = int(stats.get(normalized_reason, 0)) + 1

    @staticmethod
    def _normalize_search_result_url(raw_url: Any) -> str:
        normalized = str(raw_url or "").strip()
        if not normalized:
            return ""
        try:
            parsed = urlsplit(normalized)
        except Exception:
            return ""
        scheme = str(parsed.scheme or "").strip().lower()
        netloc = str(parsed.netloc or "").strip().lower()
        if scheme not in {"http", "https"} or not netloc:
            return ""
        path = str(parsed.path or "").strip()
        query = str(parsed.query or "").strip()
        normalized_url = urlunsplit((scheme, netloc, path, query, ""))
        return normalized_url.strip()

    @staticmethod
    def _build_canonical_url_key(normalized_url: str, *, policy: SearchQualityPolicy) -> str:
        try:
            parsed = urlsplit(normalized_url)
        except Exception:
            return ""
        if not parsed.netloc:
            return ""
        path = str(parsed.path or "").strip()
        if path == "/":
            path = ""
        elif path.endswith("/"):
            path = path.rstrip("/")
        query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
        filtered_query_pairs = []
        for key, value in query_pairs:
            normalized_key = str(key or "").strip().lower()
            if not normalized_key:
                continue
            if normalized_key.startswith("utm_") or normalized_key in policy.tracking_query_keys:
                continue
            filtered_query_pairs.append((normalized_key, str(value or "").strip()))
        filtered_query_pairs.sort()
        canonical_query = urlencode(filtered_query_pairs, doseq=True)
        return urlunsplit(
            (
                str(parsed.scheme or "").strip().lower() or "https",
                str(parsed.netloc or "").strip().lower(),
                path,
                canonical_query,
                "",
            )
        )

    @staticmethod
    def _extract_domain(normalized_url: str) -> str:
        try:
            parsed = urlsplit(normalized_url)
        except Exception:
            return ""
        return str(parsed.netloc or "").strip().lower()

    @staticmethod
    def _resolve_intermediate_url_reason(normalized_url: str, *, policy: SearchQualityPolicy) -> str:
        try:
            parsed = urlsplit(normalized_url)
        except Exception:
            return "invalid_url"
        host = str(parsed.netloc or "").strip().lower()
        path = str(parsed.path or "").strip().lower()
        query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
        query_keys = {str(key or "").strip().lower() for key, _ in query_pairs}
        if host in policy.intermediate_hosts:
            return "intermediate_host"
        is_search_engine_host = any(host.endswith(suffix) for suffix in policy.search_engine_host_suffixes)
        has_search_path = any(path.startswith(prefix) for prefix in policy.search_endpoint_path_prefixes)
        has_search_query_key = bool(policy.search_query_keys & query_keys)
        if is_search_engine_host and has_search_path and has_search_query_key:
            return "search_engine_intermediate"
        if has_search_path and has_search_query_key:
            return "generic_search_page"
        return ""

    @staticmethod
    def _tokenize_query(query: str, *, policy: SearchQualityPolicy) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for raw_token in policy.query_token_pattern.split(str(query or "").strip().lower()):
            token = str(raw_token or "").strip()
            if len(token) < 2 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

    @staticmethod
    def _compute_search_item_score(
            *,
            query_tokens: list[str],
            title: str,
            content: str,
            domain: str,
            engine: Any,
            origin_score: Any,
            policy: SearchQualityPolicy,
    ) -> float:
        lower_title = str(title or "").strip().lower()
        lower_content = str(content or "").strip().lower()
        text_blob = " ".join([lower_title, lower_content]).strip()
        token_hits = 0
        for token in query_tokens:
            if token in text_blob:
                token_hits += 1
        token_recall = (float(token_hits) / float(len(query_tokens))) if len(query_tokens) > 0 else 0.0
        score = token_recall * policy.token_recall_weight
        if len(title) >= 8:
            score += policy.title_length_bonus
        if len(content) >= 20:
            score += policy.content_length_bonus
        if len(query_tokens) > 0 and token_hits == 0:
            score -= policy.zero_token_penalty
        if len(content) < 8:
            score -= policy.short_content_penalty
        if any(domain.endswith(suffix) for suffix in policy.high_quality_host_suffixes):
            score += policy.high_quality_suffix_bonus
        normalized_engine = str(engine or "").strip().lower()
        if normalized_engine in policy.engine_bonus_names:
            score += policy.engine_bonus_score
        try:
            numeric_origin_score = float(origin_score) if origin_score is not None else 0.0
        except Exception:
            numeric_origin_score = 0.0
        if numeric_origin_score > 0:
            score += min(numeric_origin_score / policy.origin_score_divisor, policy.origin_score_max_bonus)
        return round(score, 4)

    def _rerank_for_domain_diversity(
            self,
            entries: list[dict[str, Any]],
            *,
            policy: SearchQualityPolicy,
    ) -> list[dict[str, Any]]:
        if len(entries) <= 1:
            return list(entries)
        first_pass: list[dict[str, Any]] = []
        second_pass: list[dict[str, Any]] = []
        seen_domains: set[str] = set()
        for entry in entries:
            domain = str(entry.get("domain") or "").strip().lower()
            if domain and domain not in seen_domains:
                seen_domains.add(domain)
                first_pass.append(entry)
                continue
            second_pass.append(entry)
        mixed = first_pass + second_pass
        return self._enforce_domain_cap_in_top_window(
            mixed,
            top_window=policy.domain_diversity_top_window,
            per_domain_cap=policy.domain_cap_in_top_window,
        )

    @staticmethod
    def _enforce_domain_cap_in_top_window(
            entries: list[dict[str, Any]],
            *,
            top_window: int,
            per_domain_cap: int,
    ) -> list[dict[str, Any]]:
        if len(entries) <= 1 or top_window <= 0 or per_domain_cap <= 0:
            return list(entries)
        output: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        top_domain_counter: dict[str, int] = {}
        for entry in entries:
            domain = str(entry.get("domain") or "").strip().lower()
            if len(output) < top_window and domain:
                used_count = int(top_domain_counter.get(domain, 0))
                if used_count >= per_domain_cap:
                    overflow.append(entry)
                    continue
                top_domain_counter[domain] = used_count + 1
            output.append(entry)
        output.extend(overflow)
        return output

    def _collect_top_domains(self, results: list[SearXNGSearchItem], *, limit: int) -> list[str]:
        domains: list[str] = []
        for item in list(results or []):
            domain = self._extract_domain(str(item.url or ""))
            if not domain or domain in domains:
                continue
            domains.append(domain)
            if len(domains) >= limit:
                break
        return domains

    async def fetch_page(self, url: str, max_chars: Optional[int] = 20000) -> SearXNGFetchPageResult:
        """使用 crawl4ai 读取单个页面内容。"""
        normalized_url = self._normalize_page_url(url)
        if max_chars is not None and max_chars <= 0:
            raise BadRequestException("max_chars 必须大于 0")

        crawler_cls, browser_config_cls, run_config_cls, cache_mode_cls = self._get_crawl4ai_components()
        browser_config = self._build_crawl4ai_browser_config(browser_config_cls)
        run_config = run_config_cls(
            cache_mode=cache_mode_cls.BYPASS,
            page_timeout=self.timeout_seconds * 1000,
        )

        try:
            async with crawler_cls(config=browser_config) as crawler:
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
