import asyncio

from app.domain.models import FetchedPage, ToolResult
from app.domain.services.workspace_runtime.capabilities import WorkspaceResearchCapability


class _FakeWorkspaceRuntimeService:
    async def record_search_results(self, *, query: str, candidate_links: list[dict]):
        return None

    async def record_fetched_page_summary(self, *, page_summary: dict):
        return None


class _FakeSandbox:
    async def search_searxng(
            self,
            query: str,
            categories: str | None = None,
            engines: str | None = None,
            language: str | None = None,
            page: int | None = None,
            time_range: str | None = None,
            safesearch: int | None = None,
            **kwargs,
    ) -> ToolResult[dict]:
        return ToolResult(
            success=True,
            message=time_range,
            data={
                "query": query,
                "number_of_results": 1,
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "example",
                        "content": "example snippet",
                    }
                ],
            },
        )

    async def search(self, **kwargs):
        return await self.search_searxng(**kwargs)

    async def fetch_searxng_page(
            self,
            url: str,
            max_chars: int | None = None,
            **kwargs,
    ) -> ToolResult[dict]:
        return ToolResult(
            success=True,
            message="ok",
            data={
                "url": url,
                "final_url": url,
                "status_code": 200,
                "content_type": "text/html; charset=utf-8",
                "title": "example title",
                "content": "example page content",
                "excerpt": "example page content",
                "content_length": 20,
                "truncated": False,
                "max_chars": max_chars,
            },
        )


def test_search_tool_registers_search_web() -> None:
    tool = WorkspaceResearchCapability(
        sandbox=_FakeSandbox(),
        workspace_runtime_service=_FakeWorkspaceRuntimeService(),
    )
    assert tool.has_tool("search_web") is True
    assert tool.has_tool("fetch_page") is True
    assert tool.has_tool("search_searxng") is False

    tools = tool.get_tools()
    assert len(tools) > 0
    assert any(item["function"]["name"] == "search_web" for item in tools)
    assert any(item["function"]["name"] == "fetch_page" for item in tools)
    assert all(item["function"]["name"] != "search_searxng" for item in tools)


def test_search_tool_invoke_search_web_should_delegate_to_searxng() -> None:
    tool = WorkspaceResearchCapability(
        sandbox=_FakeSandbox(),
        workspace_runtime_service=_FakeWorkspaceRuntimeService(),
    )
    result = asyncio.run(tool.invoke(
        "search_web",
        query="openai",
        language="zh-CN",
        page=2,
        time_range="month",
        safesearch=1,
    ))

    assert result.success is True
    assert result.data is not None
    assert result.data.query == "openai"
    assert result.data.date_range == "month"
    assert result.data.results[0].snippet == "example snippet"
    assert result.data.total_results == 1


def test_search_tool_invoke_fetch_page_should_delegate_to_sandbox() -> None:
    tool = WorkspaceResearchCapability(
        sandbox=_FakeSandbox(),
        workspace_runtime_service=_FakeWorkspaceRuntimeService(),
    )
    result = asyncio.run(tool.invoke(
        "fetch_page",
        url="https://example.com/article",
        max_chars=5000,
    ))

    assert result.success is True
    assert result.data is not None
    assert isinstance(result.data, FetchedPage)
    assert result.data.url == "https://example.com/article"
    assert result.data.title == "example title"
    assert result.data.content == "example page content"
    assert result.data.max_chars == 5000
