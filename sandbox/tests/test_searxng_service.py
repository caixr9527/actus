import json
import unittest
from unittest.mock import patch

from app.interfaces.errors import BadRequestException
from app.services.searxng import SearXNGService


class _FakeResponse:
    def __init__(self, status: int, body: str, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self._body = body.encode("utf-8")
        self.headers = headers or {"Content-Type": "application/json"}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


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
        self.assertEqual(result.number_of_results, 1234)
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
