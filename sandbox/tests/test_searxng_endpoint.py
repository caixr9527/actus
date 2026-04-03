import unittest
from unittest.mock import AsyncMock

from app.interfaces.endpoints.searxng import get_status, search
from app.interfaces.schemas import SearXNGSearchRequest
from app.models import SearXNGStatusResult, SearXNGSearchResult, SearXNGSearchItem


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
