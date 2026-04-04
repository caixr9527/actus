import unittest
from unittest.mock import AsyncMock

from app.interfaces.endpoints.searxng import fetch_page, get_status, search
from app.interfaces.schemas import SearXNGFetchPageRequest, SearXNGSearchRequest
from app.models import SearXNGFetchPageResult, SearXNGStatusResult, SearXNGSearchResult, SearXNGSearchItem


class SearXNGEndpointTestCase(unittest.IsolatedAsyncioTestCase):

    async def test_status_route_should_return_service_status(self) -> None:
        fake_service = AsyncMock()
        fake_service.get_status.return_value = SearXNGStatusResult(
            base_url="http://127.0.0.1:8082",
            available=True,
            status_code=200,
            content_type="text/html; charset=utf-8",
        )

        response = await get_status(searxng_service=fake_service)

        self.assertEqual(response.code, 200)
        self.assertTrue(response.data.available)
        self.assertEqual(response.data.status_code, 200)

    async def test_search_route_should_return_search_results(self) -> None:
        fake_service = AsyncMock()
        fake_service.search.return_value = SearXNGSearchResult(
            query="openai",
            number_of_results=1,
            results=[
                SearXNGSearchItem(
                    title="OpenAI",
                    url="https://openai.com",
                    content="AI research and products",
                )
            ],
        )

        response = await search(
            request=SearXNGSearchRequest(query="openai", page=1),
            searxng_service=fake_service,
        )

        self.assertEqual(response.code, 200)
        self.assertEqual(response.data.query, "openai")
        self.assertEqual(len(response.data.results), 1)
        fake_service.search.assert_awaited_once()

    async def test_fetch_page_route_should_return_page_content(self) -> None:
        fake_service = AsyncMock()
        fake_service.fetch_page.return_value = SearXNGFetchPageResult(
            url="https://example.com/article",
            final_url="https://example.com/article",
            status_code=200,
            content_type="text/html; charset=utf-8",
            title="Example Page",
            content="Hello world.",
            excerpt="Hello world.",
            content_length=12,
            truncated=False,
        )

        response = await fetch_page(
            request=SearXNGFetchPageRequest(url="https://example.com/article", max_chars=5000),
            searxng_service=fake_service,
        )

        self.assertEqual(response.code, 200)
        self.assertEqual(response.data.title, "Example Page")
        self.assertEqual(response.data.content, "Hello world.")
        fake_service.fetch_page.assert_awaited_once_with(
            url="https://example.com/article",
            max_chars=5000,
        )
