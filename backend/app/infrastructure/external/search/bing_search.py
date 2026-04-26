#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/30 15:27
@Author : caixiaorong01@outlook.com
@File   : bing_search.py
"""
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from app.domain.external import SearchEngine
from app.domain.models import SearchResultItem, SearchResults, ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _QueryLocale:
    """按查询语种推导的搜索地域参数。"""

    mkt: str
    cc: str
    set_lang: str
    accept_language: str


@dataclass(frozen=True)
class _BingSearchPage:
    """单次 Bing 请求返回的数据快照。"""

    query: str
    soup: BeautifulSoup
    total_results: int


class BingSearchEngine(SearchEngine):
    """Bing 搜索引擎实现。"""

    _MAX_RESULTS = 10
    _MAX_QUERY_VARIANTS = 3

    _TIME_QFT_MAPPING: dict[str, str] = {
        "past_hour": "+filterui:age-lt60",
        "past_day": "+filterui:age-lt1440",
        "past_week": "+filterui:age-lt10080",
        "past_month": "+filterui:age-lt43200",
        "past_year": "+filterui:age-lt525600",
    }

    _TRUSTED_DOMAIN_SUFFIXES: tuple[str, ...] = (
        ".gov.cn",
        ".edu.cn",
        ".org",
    )

    def __init__(self) -> None:
        """构造函数，初始化搜索基础配置。"""
        self.base_url = "https://www.bing.com/search"
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        self.cookies = httpx.Cookies()

    @staticmethod
    def _resolve_query_locale(query: str) -> _QueryLocale:
        """根据 query 语种选择地域和语言参数。"""
        has_cjk = re.search(r"[\u4e00-\u9fff]", query) is not None
        if has_cjk:
            return _QueryLocale(
                mkt="zh-CN",
                cc="CN",
                set_lang="zh-Hans",
                accept_language="zh-CN,zh;q=0.9,en;q=0.6",
            )

        return _QueryLocale(
            mkt="en-US",
            cc="US",
            set_lang="en-US",
            accept_language="en-US,en;q=0.8",
        )

    @classmethod
    def _build_date_filter_params(cls, date_range: Optional[str]) -> dict[str, str]:
        """构建时间过滤参数，避免传入预编码值导致双重编码。"""
        if not date_range or date_range == "all":
            return {}

        qft = cls._TIME_QFT_MAPPING.get(date_range)
        if qft is None:
            logger.warning("Bing搜索收到未知日期范围[%s]，按 all 处理", date_range)
            return {}

        params: dict[str, str] = {"qft": qft}
        if date_range == "past_year":
            # 对一年范围，补充 custom day range 过滤提升命中稳定性。
            days_since_epoch = int(time.time() / (24 * 60 * 60))
            params["filters"] = f'ex1:"ez5_{days_since_epoch - 365}_{days_since_epoch}"'
        return params

    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        """提取 query 关键词用于相关性判断。"""
        tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}", query.lower())
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            normalized = token.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    @classmethod
    def _build_query_variants(cls, query: str) -> list[str]:
        """构造通用 query 变体，避免单路召回偏差。"""
        normalized_query = re.sub(r"\s+", " ", str(query or "")).strip()
        if not normalized_query:
            return [""]

        variants: list[str] = [normalized_query]
        tokens = cls._tokenize_query(normalized_query)

        # 变体 1：短语搜索，提升实体完整命中概率。
        if len(normalized_query) >= 6 and (" " in normalized_query or len(tokens) >= 2):
            variants.append(f'"{normalized_query}"')

        # 变体 2：关键词压缩，降低长自然语言噪声。
        if len(tokens) >= 2:
            variants.append(" ".join(tokens[:6]))

        uniq_variants: list[str] = []
        seen: set[str] = set()
        for item in variants:
            candidate = item.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            uniq_variants.append(candidate)

        return uniq_variants[: cls._MAX_QUERY_VARIANTS]

    @staticmethod
    def _query_coverage_bonus(item: SearchResultItem, query_tokens: list[str]) -> float:
        """计算结果对 query token 的覆盖分。"""
        if len(query_tokens) == 0:
            return 0.0

        title_lower = item.title.lower()
        snippet_lower = item.snippet.lower()
        matched_tokens: set[str] = set()

        for token in query_tokens:
            if token in title_lower or token in snippet_lower:
                matched_tokens.add(token)

        return float(len(matched_tokens))

    def _is_low_quality_results(self, items: list[SearchResultItem], query_tokens: list[str]) -> bool:
        """判断当前结果质量是否偏低，低质量时触发变体补检索。"""
        if len(items) == 0:
            return True

        top_items = items[: min(5, len(items))]
        coverage_scores = [self._query_coverage_bonus(item, query_tokens=query_tokens) for item in top_items]
        avg_coverage = sum(coverage_scores) / len(coverage_scores)
        max_coverage = max(coverage_scores) if coverage_scores else 0.0

        domains = [urlparse(item.url).netloc.lower() for item in top_items if item.url]
        domain_ratio = 1.0
        if len(domains) > 0:
            domain_ratio = max(domains.count(domain) for domain in set(domains)) / len(domains)

        # 质量判定策略：命中几乎为 0 或单域名垄断且关键词覆盖低，判定为低质量。
        return max_coverage <= 0.0 or avg_coverage < 0.8 or (domain_ratio >= 0.8 and avg_coverage < 1.2)

    @classmethod
    def _normalize_result_url(cls, raw_url: str) -> str:
        """规范化结果 URL，过滤无效链接并清理锚点。"""
        url = str(raw_url or "").strip()
        if not url:
            return ""

        if url.startswith("//"):
            url = f"https:{url}"
        elif url.startswith("/"):
            url = f"https://www.bing.com{url}"

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""

        # 兼容部分 Bing 跳转链接，优先尝试解析真实目标。
        if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/"):
            query_values = parse_qs(parsed.query)
            for key in ("u", "url", "r"):
                candidate_values = query_values.get(key) or []
                for candidate in candidate_values:
                    decoded = unquote(str(candidate or "")).strip()
                    if decoded.startswith("http://") or decoded.startswith("https://"):
                        candidate_parsed = urlparse(decoded)
                        if candidate_parsed.netloc:
                            parsed = candidate_parsed
                            break
                else:
                    continue
                break

        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))

    @staticmethod
    def _dedupe_key(url: str) -> str:
        """构建 URL 去重键（忽略 query/fragment）。"""
        parsed = urlparse(url)
        normalized_path = parsed.path.rstrip("/") or "/"
        return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), normalized_path, "", "", ""))

    @staticmethod
    def _extract_title_and_url(item) -> tuple[str, str]:
        """抽取搜索结果标题与链接。"""
        title, url = "", ""

        title_tag = item.find("h2")
        if title_tag is not None:
            a_tag = title_tag.find("a")
            if a_tag is not None:
                title = a_tag.get_text(" ", strip=True)
                url = str(a_tag.get("href") or "")

        if title:
            return title, url

        # 回退策略：取第一个可读链接。
        for a_tag in item.find_all("a"):
            text = a_tag.get_text(" ", strip=True)
            href = str(a_tag.get("href") or "")
            if len(text) >= 3 and href:
                return text, href

        return "", ""

    @staticmethod
    def _extract_snippet(item, title: str) -> str:
        """抽取搜索摘要，优先使用 b_caption 段落。"""
        selectors = (
            "div.b_caption p",
            "p.b_lineclamp3",
            "p.b_paractl",
            "div.b_caption",
        )
        for selector in selectors:
            node = item.select_one(selector)
            if node is None:
                continue
            text = node.get_text(" ", strip=True)
            if text and text != title:
                return re.sub(r"\s+", " ", text)

        for p_tag in item.find_all("p"):
            text = p_tag.get_text(" ", strip=True)
            if len(text) >= 20 and text != title:
                return re.sub(r"\s+", " ", text)

        return ""

    @classmethod
    def _score_result(cls, item: SearchResultItem, query_tokens: list[str], date_range: Optional[str]) -> float:
        """结果轻量打分：关键词命中 + 可信域名 + 时间特征。"""
        title_lower = item.title.lower()
        snippet_lower = item.snippet.lower()
        score = 0.0

        for token in query_tokens[:10]:
            if token in title_lower:
                score += 3.0
            if token in snippet_lower:
                score += 1.0

        netloc = urlparse(item.url).netloc.lower()
        if netloc.endswith(cls._TRUSTED_DOMAIN_SUFFIXES):
            score += 1.0

        if date_range and date_range != "all":
            if re.search(r"刚刚|分钟前|小时前|天前|today|hour|day", item.snippet, flags=re.IGNORECASE):
                score += 0.8

        return score

    @staticmethod
    def _extract_total_results(soup: BeautifulSoup) -> int:
        """提取搜索结果总数（中英文页面兼容）。"""
        patterns = (
            r"([\d,]+)\s*results",
            r"约\s*([\d,]+)\s*条结果",
            r"([\d,]+)\s*条结果",
        )

        candidate_nodes = [
            *soup.find_all(string=True),
            *soup.find_all(["span", "div", "p"], class_=re.compile(r"sb_count|b_focusTextMedium")),
        ]

        for node in candidate_nodes:
            text = node if isinstance(node, str) else node.get_text(" ", strip=True)
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    try:
                        return int(match.group(1).replace(",", ""))
                    except Exception:
                        continue
        return 0

    def _parse_search_results(self, soup: BeautifulSoup) -> list[SearchResultItem]:
        """解析并去重单页搜索结果（保留页面原始顺序）。"""
        result_items = soup.find_all("li", class_="b_algo")
        parsed_results: list[SearchResultItem] = []
        seen_keys: set[str] = set()

        for item in result_items:
            try:
                title, raw_url = self._extract_title_and_url(item)
                if not title:
                    continue

                normalized_url = self._normalize_result_url(raw_url)
                if not normalized_url:
                    continue

                dedupe_key = self._dedupe_key(normalized_url)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

                snippet = self._extract_snippet(item, title)
                parsed_results.append(
                    SearchResultItem(
                        title=title,
                        url=normalized_url,
                        snippet=snippet,
                    )
                )
            except Exception as e:
                logger.warning("Bing搜索结果解析失败: %s", e)
                continue

        return parsed_results[: self._MAX_RESULTS]

    async def _search_once(
            self,
            *,
            client: httpx.AsyncClient,
            query: str,
            date_range: Optional[str],
            locale: _QueryLocale,
    ) -> _BingSearchPage:
        """执行单次 Bing 请求并返回页面快照。"""
        params = {
            "q": query,
            "mkt": locale.mkt,
            "cc": locale.cc,
            "setlang": locale.set_lang,
        }
        params.update(self._build_date_filter_params(date_range=date_range))

        logger.debug("Bing搜索请求: query=%s, date_range=%s, params=%s", query, date_range, params)
        response = await client.get(self.base_url, params=params)
        response.raise_for_status()
        self.cookies.update(response.cookies)

        soup = BeautifulSoup(response.text, "html.parser")
        total_results = self._extract_total_results(soup)
        return _BingSearchPage(query=query, soup=soup, total_results=total_results)

    async def invoke(self, query: str, date_range: Optional[str] = None) -> ToolResult[SearchResults]:
        """调用 Bing 搜索并返回结构化结果。"""
        locale = self._resolve_query_locale(query)
        headers = dict(self.headers)
        headers["Accept-Language"] = locale.accept_language

        query_variants = self._build_query_variants(query)
        base_query_tokens = self._tokenize_query(query)

        # 聚合多路召回结果，避免单次召回偏差。
        ranked_items: dict[str, tuple[float, int, SearchResultItem]] = {}
        max_total_results = 0

        try:
            async with httpx.AsyncClient(
                    headers=headers,
                    cookies=self.cookies,
                    timeout=60,
                    follow_redirects=True,
            ) as client:
                should_expand_variants = True
                for variant_index, variant_query in enumerate(query_variants):
                    if variant_index > 0 and not should_expand_variants:
                        break

                    page = await self._search_once(
                        client=client,
                        query=variant_query,
                        date_range=date_range,
                        locale=locale,
                    )
                    max_total_results = max(max_total_results, page.total_results)
                    page_items = self._parse_search_results(page.soup)

                    # 首轮质量可接受时，避免额外请求放大时延。
                    if variant_index == 0:
                        should_expand_variants = self._is_low_quality_results(
                            page_items,
                            query_tokens=base_query_tokens,
                        )

                    variant_tokens = self._tokenize_query(variant_query)
                    for rank_index, item in enumerate(page_items):
                        primary_score = self._score_result(item, query_tokens=base_query_tokens, date_range=date_range)
                        variant_bonus = self._query_coverage_bonus(item, query_tokens=variant_tokens) * 0.6
                        attempt_bonus = max(0.8 - 0.25 * variant_index, 0.0)
                        rank_penalty = rank_index * 0.02
                        final_score = primary_score + variant_bonus + attempt_bonus - rank_penalty

                        dedupe_key = self._dedupe_key(item.url)
                        current = ranked_items.get(dedupe_key)
                        if current is None or final_score > current[0]:
                            ranked_items[dedupe_key] = (final_score, rank_index, item)

                sorted_items = sorted(
                    ranked_items.values(),
                    key=lambda payload: (-payload[0], payload[1]),
                )
                merged_results = [payload[2] for payload in sorted_items[: self._MAX_RESULTS]]
                quality_is_low = self._is_low_quality_results(
                    merged_results,
                    query_tokens=base_query_tokens,
                )

                logger.debug(
                    "Bing搜索解析完成: query=%s, variants=%s, merged_results=%s, total_results=%s, low_quality=%s",
                    query,
                    len(query_variants),
                    len(merged_results),
                    max_total_results,
                    quality_is_low,
                )

                if quality_is_low:
                    logger.warning(
                        "Bing搜索结果低相关，拒绝返回低质量结果: query=%s, variants=%s",
                        query,
                        query_variants,
                    )
                    return ToolResult(
                        success=False,
                        message="搜索结果相关性较低，请尝试更具体关键词或切换搜索源",
                        data=SearchResults(
                            query=query,
                            date_range=date_range,
                            total_results=0,
                            results=[],
                        ),
                    )

                return ToolResult(
                    success=True,
                    data=SearchResults(
                        query=query,
                        date_range=date_range,
                        total_results=max_total_results,
                        results=merged_results,
                    ),
                )
        except Exception as e:
            logger.error("Bing搜索出错: %s", e)
            return ToolResult(
                success=False,
                message=f"Bing搜索出错: {e}",
                data=SearchResults(
                    query=query,
                    date_range=date_range,
                    total_results=0,
                    results=[],
                ),
            )
