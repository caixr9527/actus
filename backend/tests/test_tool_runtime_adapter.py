import asyncio

from app.domain.models import (
    SearchResultItem,
    SearchResults,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
)
from app.domain.services.tools import CapabilityBuildContext, CapabilityRegistry, ToolRuntimeAdapter, ToolRuntimeEventHooks


def test_capability_registry_default_v1_should_build_expected_local_tools() -> None:
    registry = CapabilityRegistry.default_v1()
    tools = registry.build_tools(
        capability_ids=ToolRuntimeAdapter.DEFAULT_LOCAL_CAPABILITIES,
        context=CapabilityBuildContext(
            sandbox=object(),
            browser=object(),
            search_engine=object(),
        ),
    )

    assert any(tool.has_tool("read_file") for tool in tools)
    assert any(tool.has_tool("shell_execute") for tool in tools)
    assert any(tool.has_tool("browser_view") for tool in tools)
    assert any(tool.has_tool("search_web") for tool in tools)
    assert any(tool.has_tool("message_ask_user") for tool in tools)


def test_tool_runtime_adapter_should_enrich_search_event() -> None:
    adapter = ToolRuntimeAdapter()
    event = ToolEvent(
        tool_name="search",
        function_name="search_web",
        function_args={"query": "openai"},
        function_result=ToolResult[SearchResults](
            success=True,
            data=SearchResults(
                query="openai",
                results=[SearchResultItem(url="https://example.com", title="example")],
            ),
        ),
        status=ToolEventStatus.CALLED,
    )

    async def _get_browser_screenshot() -> str:
        return "unused"

    async def _read_shell_output(_session_id: str) -> ToolResult:
        return ToolResult(success=True, data={})

    async def _read_file_content(_filepath: str) -> ToolResult:
        return ToolResult(success=True, data={})

    async def _sync_file_to_storage(_filepath: str) -> None:
        return None

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(
                get_browser_screenshot=_get_browser_screenshot,
                read_shell_output=_read_shell_output,
                read_file_content=_read_file_content,
                sync_file_to_storage=_sync_file_to_storage,
            ),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert len(event.tool_content.results) == 1
    assert event.tool_content.results[0].url == "https://example.com"


def test_tool_runtime_adapter_should_enrich_file_event_and_sync_storage() -> None:
    adapter = ToolRuntimeAdapter()
    event = ToolEvent(
        tool_name="file",
        function_name="read_file",
        function_args={"filepath": "/tmp/a.txt"},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
    )
    synced_paths: list[str] = []

    async def _get_browser_screenshot() -> str:
        return "unused"

    async def _read_shell_output(_session_id: str) -> ToolResult:
        return ToolResult(success=True, data={})

    async def _read_file_content(_filepath: str) -> ToolResult:
        return ToolResult(success=True, data={"content": "hello world"})

    async def _sync_file_to_storage(filepath: str) -> None:
        synced_paths.append(filepath)
        return None

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(
                get_browser_screenshot=_get_browser_screenshot,
                read_shell_output=_read_shell_output,
                read_file_content=_read_file_content,
                sync_file_to_storage=_sync_file_to_storage,
            ),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert event.tool_content.content == "hello world"
    assert synced_paths == ["/tmp/a.txt"]
