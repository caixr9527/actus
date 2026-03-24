import asyncio

from app.domain.models import ToolResult, SearchResults, SearchResultItem
from app.domain.services.tools.search import SearchTool


class _FakeSearchEngine:
    async def invoke(self, query: str, date_range: str | None = None) -> ToolResult[SearchResults]:
        return ToolResult(
            success=True,
            data=SearchResults(
                query=query,
                date_range=date_range,
                total_results=1,
                results=[
                    SearchResultItem(
                        url="https://example.com",
                        title="example",
                        snippet="example snippet",
                    )
                ],
            ),
        )


def test_search_tool_registers_search_web() -> None:
    tool = SearchTool(search_engine=_FakeSearchEngine())
    assert tool.has_tool("search_web") is True

    tools = tool.get_tools()
    assert len(tools) > 0
    assert any(item["function"]["name"] == "search_web" for item in tools)


def test_search_tool_invoke_search_web_success() -> None:
    tool = SearchTool(search_engine=_FakeSearchEngine())
    result = asyncio.run(tool.invoke("search_web", query="openai", data_range="past_day"))

    assert result.success is True
    assert result.data is not None
    assert result.data.query == "openai"
    assert result.data.date_range == "past_day"
