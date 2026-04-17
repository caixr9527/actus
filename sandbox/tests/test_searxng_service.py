import json
import unittest
from os import environ
from types import SimpleNamespace
from unittest.mock import patch

from app.interfaces.errors import AppException, BadRequestException
from app.services.searxng import SearXNGService
from app.services.search_quality_policy import get_search_quality_policy


class _FakeResponse:
    def __init__(
            self,
            status: int,
            body: str,
            headers: dict[str, str] | None = None,
            final_url: str = "https://example.com/final",
    ) -> None:
        self.status = status
        self._body = body.encode("utf-8")
        self.headers = headers or {"Content-Type": "application/json"}
        self._final_url = final_url

    def read(self) -> bytes:
        return self._body

    def geturl(self) -> str:
        return self._final_url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


class _FakeBrowserConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeCrawlerRunConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeCacheMode:
    BYPASS = "bypass"


class _FakeAsyncWebCrawler:
    result = None
    init_config = None
    arun_url = None
    arun_config = None

    def __init__(self, config=None) -> None:
        _FakeAsyncWebCrawler.init_config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    async def arun(self, url: str, config=None):
        _FakeAsyncWebCrawler.arun_url = url
        _FakeAsyncWebCrawler.arun_config = config
        return _FakeAsyncWebCrawler.result


class SearXNGServiceTestCase(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        for key in (
            "SEARXNG_SEARCH_INTERMEDIATE_HOSTS",
            "SEARXNG_SEARCH_ENGINE_BONUS_NAMES",
            "SEARXNG_SEARCH_DOMAIN_DIVERSITY_TOP_WINDOW",
            "SEARXNG_SEARCH_DOMAIN_CAP_IN_TOP_WINDOW",
        ):
            environ.pop(key, None)
        get_search_quality_policy.cache_clear()


    async def test_search_should_parse_json_payload(self) -> None:
        payload = {
            "query": "openai",
            "number_of_results": 1234,
            "results": [
                {
                    "title": "OpenAI",
                    "url": "https://openai.com",
                    "content": "AI research and products",
                    "engine": "bing",
                    "category": "general",
                    "score": 12.3,
                    "publishedDate": "2026-04-03",
                }
            ],
            "suggestions": ["open ai"],
            "answers": ["OpenAI is an AI company."],
            "corrections": ["openai"],
            "infoboxes": [{"content": "info"}],
            "unresponsive_engines": ["duckduckgo"],
        }
        service = SearXNGService()

        with patch("app.services.searxng.urlopen", return_value=_FakeResponse(status=200, body=json.dumps(payload))):
            result = await service.search(query="openai", page=2, safesearch=1)

        self.assertEqual(result.query, "openai")
        self.assertEqual(result.number_of_results, 1)
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0].url, "https://openai.com")
        self.assertEqual(result.suggestions, ["open ai"])
        self.assertEqual(result.unresponsive_engines, ["duckduckgo"])

    async def test_search_should_filter_intermediate_urls_and_dedupe_tracking_variants(self) -> None:
        payload = {
            "query": "上海 周末 自驾游",
            "results": [
                {
                    "title": "百度中间页",
                    "url": "https://mbd.baidu.com/newspage/data/dtlandingsuper?nid=123",
                    "content": "中间页",
                    "engine": "baidu",
                },
                {
                    "title": "候选A-1",
                    "url": "https://example.com/guide?source=bing&utm_source=test",
                    "content": "上海周末自驾游攻略",
                    "engine": "bing",
                },
                {
                    "title": "候选A-2",
                    "url": "https://example.com/guide?source=baidu",
                    "content": "上海周末自驾游攻略重复",
                    "engine": "baidu",
                },
                {
                    "title": "候选B",
                    "url": "https://travel.example.org/plan",
                    "content": "2天1夜行程",
                    "engine": "bing",
                },
            ],
        }
        service = SearXNGService()

        with patch("app.services.searxng.urlopen", return_value=_FakeResponse(status=200, body=json.dumps(payload))):
            result = await service.search(query="上海 周末 自驾游")

        self.assertEqual(len(result.results), 2)
        urls = [item.url for item in result.results]
        self.assertTrue(all("mbd.baidu.com" not in item for item in urls))
        self.assertEqual(len([item for item in urls if "example.com/guide" in item]), 1)

    async def test_search_policy_should_support_env_override_for_intermediate_hosts(self) -> None:
        environ["SEARXNG_SEARCH_INTERMEDIATE_HOSTS"] = "redirect.example.com"
        get_search_quality_policy.cache_clear()
        payload = {
            "query": "openai release notes",
            "results": [
                {
                    "title": "中间页",
                    "url": "https://redirect.example.com/jump?id=1",
                    "content": "redirect",
                    "engine": "bing",
                },
                {
                    "title": "目标页",
                    "url": "https://docs.example.com/notes",
                    "content": "release notes",
                    "engine": "bing",
                },
            ],
        }

        service = SearXNGService()
        with patch("app.services.searxng.urlopen", return_value=_FakeResponse(status=200, body=json.dumps(payload))):
            result = await service.search(query="openai release notes")

        urls = [item.url for item in result.results]
        self.assertEqual(urls, ["https://docs.example.com/notes"])

    async def test_search_should_limit_same_domain_in_top_window(self) -> None:
        payload = {
            "query": "上海 周末 自驾游",
            "results": [
                {
                    "title": f"同域{i}",
                    "url": f"https://same.example.com/article-{i}",
                    "content": "上海 周末 自驾游 推荐",
                    "engine": "bing",
                }
                for i in range(5)
            ] + [
                {
                    "title": "异域1",
                    "url": "https://other.example.org/a",
                    "content": "上海 周末 自驾游 推荐",
                    "engine": "baidu",
                },
                {
                    "title": "异域2",
                    "url": "https://another.example.net/b",
                    "content": "上海 周末 自驾游 推荐",
                    "engine": "baidu",
                },
                {
                    "title": "异域3",
                    "url": "https://city.example.cn/c",
                    "content": "上海 周末 自驾游 推荐",
                    "engine": "bing",
                },
                {
                    "title": "异域4",
                    "url": "https://news.example.com/d",
                    "content": "上海 周末 自驾游 推荐",
                    "engine": "bing",
                },
                {
                    "title": "异域5",
                    "url": "https://tips.example.io/e",
                    "content": "上海 周末 自驾游 推荐",
                    "engine": "baidu",
                },
                {
                    "title": "异域6",
                    "url": "https://travel.example.co/f",
                    "content": "上海 周末 自驾游 推荐",
                    "engine": "baidu",
                },
            ],
        }
        service = SearXNGService()

        with patch("app.services.searxng.urlopen", return_value=_FakeResponse(status=200, body=json.dumps(payload))):
            result = await service.search(query="上海 周末 自驾游")

        top_urls = [item.url for item in result.results[:8]]
        same_domain_count = len([item for item in top_urls if "same.example.com" in item])
        self.assertLessEqual(same_domain_count, 2)

    async def test_search_should_tokenize_chinese_natural_language_query_for_scoring(self) -> None:
        payload = {
            "query": "AI编程助手有哪些",
            "results": [
                {
                    "title": "AI编程助手综述",
                    "url": "https://example.com/a",
                    "content": "主流AI编程助手盘点",
                    "engine": "bing",
                },
                {
                    "title": "无关结果",
                    "url": "https://example.com/b",
                    "content": "完全无关文本",
                    "engine": "bing",
                },
            ],
        }
        service = SearXNGService()
        with patch("app.services.searxng.urlopen", return_value=_FakeResponse(status=200, body=json.dumps(payload))):
            result = await service.search(query="AI编程助手有哪些")
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.results[0].url, "https://example.com/a")

    async def test_search_should_limit_query_tokens_when_chinese_query_is_long(self) -> None:
        long_query = "这是一条用于测试中文自然语言搜索查询分词上限的非常长的句子，包含多个主题片段和描述性文本"
        payload = {
            "query": long_query,
            "results": [
                {
                    "title": "测试结果",
                    "url": "https://example.com/a",
                    "content": "测试内容",
                    "engine": "bing",
                }
            ],
        }
        service = SearXNGService()
        with patch("app.services.searxng.urlopen", return_value=_FakeResponse(status=200, body=json.dumps(payload))):
            _ = await service.search(query=long_query)
        tokens = service._tokenize_query(long_query, policy=service.quality_policy)
        self.assertLessEqual(len(tokens), 48)

    async def test_search_should_reject_empty_query(self) -> None:
        service = SearXNGService()

        with self.assertRaises(BadRequestException):
            await service.search(query="   ")

    async def test_get_status_should_return_available_when_service_responds(self) -> None:
        service = SearXNGService()

        with patch(
                "app.services.searxng.urlopen",
                return_value=_FakeResponse(
                    status=200,
                    body="<html></html>",
                    headers={"Content-Type": "text/html; charset=utf-8"},
                ),
        ):
            result = await service.get_status()

        self.assertTrue(result.available)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content_type, "text/html; charset=utf-8")

    async def test_fetch_page_should_extract_title_and_text(self) -> None:
        service = SearXNGService()
        _FakeAsyncWebCrawler.result = SimpleNamespace(
            success=True,
            url="https://example.com/final",
            status_code=200,
            response_headers={"Content-Type": "text/html; charset=utf-8"},
            metadata={"title": "Example Page"},
            markdown=SimpleNamespace(
                fit_markdown="Hello world.\n\nSecond line.",
                raw_markdown="ignored raw markdown",
            ),
            extracted_content=None,
        )

        with patch.object(
                SearXNGService,
                "_get_crawl4ai_components",
                return_value=(
                    _FakeAsyncWebCrawler,
                    _FakeBrowserConfig,
                    _FakeCrawlerRunConfig,
                    _FakeCacheMode,
                ),
        ):
            result = await service.fetch_page(url="https://example.com/article", max_chars=12)

        self.assertEqual(result.url, "https://example.com/article")
        self.assertEqual(result.final_url, "https://example.com/final")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content_type, "text/html; charset=utf-8")
        self.assertEqual(result.title, "Example Page")
        self.assertEqual(result.content, "Hello world.")
        self.assertEqual(result.excerpt, "Hello world.\n\nSecond line.")
        self.assertEqual(result.content_length, len("Hello world.\n\nSecond line."))
        self.assertTrue(result.truncated)
        self.assertEqual(_FakeAsyncWebCrawler.arun_url, "https://example.com/article")
        self.assertEqual(_FakeAsyncWebCrawler.init_config.kwargs["browser_mode"], "builtin")
        self.assertEqual(_FakeAsyncWebCrawler.init_config.kwargs["host"], "127.0.0.1")
        self.assertEqual(_FakeAsyncWebCrawler.init_config.kwargs["debugging_port"], 9222)
        self.assertEqual(_FakeAsyncWebCrawler.arun_config.kwargs["cache_mode"], "bypass")
        self.assertEqual(_FakeAsyncWebCrawler.arun_config.kwargs["page_timeout"], service.timeout_seconds * 1000)

    async def test_fetch_page_should_raise_when_crawl_failed(self) -> None:
        service = SearXNGService()
        _FakeAsyncWebCrawler.result = SimpleNamespace(
            success=False,
            status_code=500,
            error_message="crawl failed",
        )

        with patch.object(
                SearXNGService,
                "_get_crawl4ai_components",
                return_value=(
                    _FakeAsyncWebCrawler,
                    _FakeBrowserConfig,
                    _FakeCrawlerRunConfig,
                    _FakeCacheMode,
                ),
        ):
            with self.assertRaises(AppException):
                await service.fetch_page(url="https://example.com/file.pdf")
