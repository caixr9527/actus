import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.interfaces.errors import AppException, BadRequestException
from app.services.searxng import SearXNGService


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
