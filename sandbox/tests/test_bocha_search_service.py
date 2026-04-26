import json
import unittest
from os import environ
from urllib.error import HTTPError
from unittest.mock import patch

from app.interfaces.errors import AppException, BadRequestException
from app.services.bocha_search import BochaSearchService


class _FakeResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


class _FakeHTTPError(HTTPError):
    def __init__(self, code: int, body: str = "") -> None:
        super().__init__(url="https://api.bochaai.com/v1/web-search", code=code, msg="error", hdrs=None, fp=None)
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body


class BochaSearchServiceTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._old_env = {
            "BOCHA_API_KEY": environ.get("BOCHA_API_KEY"),
            "BOCHA_BASE_URL": environ.get("BOCHA_BASE_URL"),
            "BOCHA_TIMEOUT_SECONDS": environ.get("BOCHA_TIMEOUT_SECONDS"),
            "BOCHA_DEFAULT_COUNT": environ.get("BOCHA_DEFAULT_COUNT"),
        }
        environ["BOCHA_API_KEY"] = "test-key"
        environ["BOCHA_BASE_URL"] = "https://api.bochaai.com"
        environ["BOCHA_TIMEOUT_SECONDS"] = "30"
        environ["BOCHA_DEFAULT_COUNT"] = "10"

    def tearDown(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                environ.pop(key, None)
            else:
                environ[key] = value

    async def test_search_should_parse_results_and_keep_shape(self) -> None:
        payload = {
            "data": {
                "queryContext": {"originalQuery": "openai"},
                "webPages": {
                    "value": [
                        {
                            "name": "OpenAI",
                            "url": "https://openai.com",
                            "snippet": "AI research and products",
                            "datePublished": "2026-04-18",
                        }
                    ]
                },
            }
        }
        service = BochaSearchService()
        with patch("app.services.bocha_search.urlopen", return_value=_FakeResponse(status=200, body=json.dumps(payload))):
            result = await service.search(query="openai")

        self.assertEqual(result.query, "openai")
        self.assertEqual(result.number_of_results, 1)
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0].url, "https://openai.com")
        self.assertEqual(result.results[0].engine, "bocha")

    async def test_search_should_reject_empty_query(self) -> None:
        service = BochaSearchService()
        with self.assertRaises(BadRequestException):
            await service.search(query="   ")

    async def test_search_should_fail_when_api_key_missing(self) -> None:
        environ["BOCHA_API_KEY"] = ""
        service = BochaSearchService()
        with self.assertRaises(AppException):
            await service.search(query="openai")

    async def test_search_should_raise_bad_request_on_http_error(self) -> None:
        service = BochaSearchService()
        with patch("app.services.bocha_search.urlopen", side_effect=_FakeHTTPError(code=401, body="unauthorized")):
            with self.assertRaises(BadRequestException):
                await service.search(query="openai")

    async def test_search_should_keep_result_without_url_as_none_string(self) -> None:
        payload = {
            "data": {
                "query": "openai",
                "results": [
                    {
                        "name": "No URL Result",
                        "summary": "missing url",
                    },
                    {
                        "name": "OpenAI",
                        "url": "https://openai.com",
                        "summary": "AI research and products",
                    },
                ],
            }
        }
        service = BochaSearchService()
        with patch("app.services.bocha_search.urlopen", return_value=_FakeResponse(status=200, body=json.dumps(payload))):
            result = await service.search(query="openai")

        # 以当前业务代码为准：缺失 url 会被 str(None) 处理为 "None"，不会被过滤。
        self.assertEqual(result.number_of_results, 2)
        self.assertEqual(result.results[0].url, "None")
        self.assertEqual(result.results[1].url, "https://openai.com")
