import asyncio

from app.domain.models import (
    BrowserLinkMatchResult,
    BrowserPageStructuredResult,
    BrowserPageType,
    FetchPageToolContent,
    FetchedPage,
    MCPConfig,
    MCPServerConfig,
    MCPTransport,
    SearchResultItem,
    SearchResults,
    ToolDiagnosticContent,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
)
from app.domain.services.tools import (
    BaseTool,
    CapabilityBuildContext,
    CapabilityRegistry,
    MCPCapabilityAdapter,
    MessageTool,
    ToolRuntimeAdapter,
    ToolRuntimeEventHooks,
)
from app.domain.services.workspace_runtime.capabilities import (
    WorkspaceBrowserCapability,
    WorkspaceFileCapability,
    WorkspaceResearchCapability,
    WorkspaceShellCapability,
)


class _FakeWorkspaceRuntimeService:
    session_id = "session-1"


def test_capability_registry_default_v1_should_build_expected_local_tools() -> None:
    registry = CapabilityRegistry.default_v1()
    tools = registry.build_tools(
        capability_ids=ToolRuntimeAdapter.DEFAULT_LOCAL_CAPABILITIES,
        context=CapabilityBuildContext(
            sandbox=object(),
            browser=object(),
            search_engine=object(),
            workspace_runtime_service=_FakeWorkspaceRuntimeService(),
        ),
    )

    assert any(tool.has_tool("read_file") for tool in tools)
    assert any(tool.has_tool("shell_execute") for tool in tools)
    assert any(tool.has_tool("browser_view") for tool in tools)
    assert any(tool.has_tool("search_web") for tool in tools)
    assert any(tool.has_tool("message_ask_user") for tool in tools)
    assert any(isinstance(tool, WorkspaceBrowserCapability) for tool in tools)
    assert any(isinstance(tool, WorkspaceFileCapability) for tool in tools)
    assert any(isinstance(tool, WorkspaceShellCapability) for tool in tools)
    assert any(isinstance(tool, WorkspaceResearchCapability) for tool in tools)


def test_tool_runtime_adapter_should_enrich_search_event() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
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

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert len(event.tool_content.results) == 1
    assert event.tool_content.results[0].url == "https://example.com"


def test_tool_runtime_adapter_should_enrich_fetch_page_event() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="search",
        function_name="fetch_page",
        function_args={"url": "https://example.com/article", "max_chars": 2000},
        function_result=ToolResult[FetchedPage](
            success=True,
            data=FetchedPage(
                url="https://example.com/article",
                final_url="https://example.com/article",
                status_code=200,
                content_type="text/html",
                title="example title",
                content="example page content",
                excerpt="example page content",
                content_length=20,
                truncated=False,
                max_chars=2000,
            ),
        ),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert isinstance(event.tool_content, FetchPageToolContent)
    assert event.tool_content.title == "example title"
    assert event.tool_content.content == "example page content"
    assert event.tool_content.max_chars == 2000


def test_tool_runtime_adapter_should_render_search_diagnostic_content_when_blocked() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="search",
        function_name="search_web",
        function_args={"query": "漳州周末旅游攻略"},
        function_result=ToolResult(
            success=False,
            message="当前必须先抓取候选页面，再决定是否继续搜索。",
            data={
                "reason_code": "research_route_fetch_required",
                "block_mode": "hard_block_break",
            },
        ),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert isinstance(event.tool_content, ToolDiagnosticContent)
    assert event.tool_content.reason_code == "research_route_fetch_required"
    assert event.tool_content.message == "当前必须先抓取候选页面，再决定是否继续搜索。"
    assert event.tool_content.diagnostic_type == "tool_result_fallback"


def test_tool_runtime_adapter_should_render_fetch_diagnostic_content_when_low_value() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="search",
        function_name="fetch_page",
        function_args={"url": "https://example.com/listing"},
        function_result=ToolResult(
            success=False,
            message="页面已读取，但正文价值不足，未拿到可用于当前研究步骤的有效信息。",
            data={
                "research_diagnosis": {
                    "code": "fetch_low_value",
                    "message": "页面抓取已执行，但正文价值不足，未拿到有效信息。",
                    "candidate_url_count": 5,
                },
            },
        ),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert isinstance(event.tool_content, ToolDiagnosticContent)
    assert event.tool_content.reason_code == "fetch_low_value"
    assert event.tool_content.diagnostic_type == "research_diagnosis"
    assert event.tool_content.details["candidate_url_count"] == 5


def test_tool_runtime_adapter_should_enrich_file_event_and_sync_storage() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="file",
        function_name="read_file",
        function_args={"filepath": "/tmp/a.txt", "max_length": 1000},
        function_result=ToolResult(success=True, data={"content": "hello world"}),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert event.tool_content.content == "hello world"


def test_tool_runtime_adapter_should_sync_storage_only_for_write_file() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="file",
        function_name="write_file",
        function_args={"filepath": "/tmp/a.txt"},
        function_result=ToolResult(success=True, data={"filepath": "/tmp/a.txt"}),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert "/tmp/a.txt" in event.tool_content.content


def test_tool_runtime_adapter_should_enrich_list_files_event_without_filepath() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="file",
        function_name="list_files",
        function_args={"dir_path": "/home/ubuntu"},
        function_result=ToolResult(
            success=True,
            data={
                "dir_path": "/home/ubuntu",
                "files": ["/home/ubuntu/a.md", "/home/ubuntu/b.md"],
            },
        ),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert "目录: /home/ubuntu" in event.tool_content.content
    assert "/home/ubuntu/a.md" in event.tool_content.content
    assert "/home/ubuntu/b.md" in event.tool_content.content


def test_tool_runtime_adapter_should_capture_screenshot_for_key_browser_actions() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    screenshot_calls: list[str] = []

    async def _get_browser_screenshot() -> str:
        screenshot_calls.append("shot")
        return "https://cdn.example.com/browser-shot.png"

    view_event = ToolEvent(
        tool_name="browser",
        function_name="browser_view",
        function_args={},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
    )
    scroll_event = ToolEvent(
        tool_name="browser",
        function_name="browser_scroll_down",
        function_args={},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
    )

    handled_view = asyncio.run(
        adapter.enrich_tool_event(
            event=view_event,
            hooks=ToolRuntimeEventHooks(get_browser_screenshot=_get_browser_screenshot),
        )
    )
    handled_scroll = asyncio.run(
        adapter.enrich_tool_event(
            event=scroll_event,
            hooks=ToolRuntimeEventHooks(get_browser_screenshot=_get_browser_screenshot),
        )
    )

    assert handled_view is True
    assert handled_scroll is True
    assert view_event.tool_content is not None
    assert view_event.tool_content.screenshot == "https://cdn.example.com/browser-shot.png"
    assert scroll_event.tool_content is not None
    assert scroll_event.tool_content.screenshot == ""
    assert screenshot_calls == ["shot"]


def test_tool_runtime_adapter_should_keep_browser_event_usable_without_screenshot_hook() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="browser",
        function_name="browser_view",
        function_args={},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert event.tool_content.screenshot == ""


def test_tool_runtime_adapter_should_enrich_high_level_browser_event() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="browser",
        function_name="browser_read_current_page_structured",
        function_args={},
        function_result=ToolResult(
            success=True,
            data=BrowserPageStructuredResult(
                url="https://example.com/docs/runtime",
                title="Runtime Docs",
                page_type=BrowserPageType.DOCUMENT,
                content_summary="runtime summary",
            ),
        ),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert event.tool_content.url == "https://example.com/docs/runtime"
    assert event.tool_content.page_type == "document"
    assert event.tool_content.structured_page is not None


def test_tool_runtime_adapter_should_expose_browser_degrade_reason() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="browser",
        function_name="browser_extract_main_content",
        function_args={},
        function_result=ToolResult(
            success=False,
            message="提取当前页面正文失败",
            data={
                "degrade_reason": "browser_extract_main_content_failed",
                "page_type": "document",
                "url": "https://example.com/docs/runtime",
                "title": "Runtime Docs",
            },
        ),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert event.tool_content.degrade_reason == "browser_extract_main_content_failed"
    assert event.tool_content.page_type == "document"
    assert event.tool_content.screenshot == ""


def test_tool_runtime_adapter_should_expose_browser_link_match_position() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="browser",
        function_name="browser_find_link_by_text",
        function_args={"text": "Execution Model"},
        function_result=ToolResult(
            success=True,
            data=BrowserLinkMatchResult(
                query="Execution Model",
                matched_text="Execution Model",
                url="https://example.com/docs/execution",
                index=1,
                selector="[data-manus-id='manus-element-1']",
            ),
        ),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert event.tool_content.matched_link_text == "Execution Model"
    assert event.tool_content.matched_link_url == "https://example.com/docs/execution"
    assert event.tool_content.matched_link_index == 1
    assert event.tool_content.matched_link_selector == "[data-manus-id='manus-element-1']"


def test_tool_runtime_adapter_should_keep_shell_event_usable_without_shell_hook() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="shell",
        function_name="shell_execute",
        function_args={},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
    )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert event.tool_content.console == "(No console)"


def test_tool_runtime_adapter_should_enrich_shell_event_from_workspace_observation_hook() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    event = ToolEvent(
        tool_name="shell",
        function_name="read_shell_output",
        function_args={},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
    )

    async def _get_shell_tool_result() -> ToolResult:
        return ToolResult(
            success=True,
            data={
                "output": "pytest -q\n24 passed",
                "console_records": [
                    {
                        "command": "pytest -q",
                        "output": "24 passed",
                    }
                ],
            },
        )

    handled = asyncio.run(
        adapter.enrich_tool_event(
            event=event,
            hooks=ToolRuntimeEventHooks(get_shell_tool_result=_get_shell_tool_result),
        )
    )

    assert handled is True
    assert event.tool_content is not None
    assert event.tool_content.console == [
        {
            "command": "pytest -q",
            "output": "24 passed",
        }
    ]


class _FakeMCPTool(BaseTool):
    name = "mcp"

    def __init__(self, delay_seconds: float = 0.0) -> None:
        super().__init__()
        self._delay_seconds = delay_seconds
        self.invocations: list[tuple[str, dict]] = []

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "mcp_demo_ping",
                    "description": "demo",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

    def has_tool(self, tool_name: str) -> bool:
        return tool_name == "mcp_demo_ping"

    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        self.invocations.append((tool_name, kwargs))
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        return ToolResult(success=True, data="pong")


def test_tool_runtime_adapter_should_build_mcp_capability_tool() -> None:
    adapter = ToolRuntimeAdapter(capability_registry=CapabilityRegistry.default_v1())
    fake_mcp_tool = _FakeMCPTool()

    tools = adapter.build_runtime_tools(
        capability_context=CapabilityBuildContext(
            sandbox=object(),
            browser=object(),
            search_engine=object(),
            workspace_runtime_service=_FakeWorkspaceRuntimeService(),
            mcp_tool=fake_mcp_tool,
            mcp_config=MCPConfig(
                mcpServers={
                    "demo": MCPServerConfig(
                        transport=MCPTransport.STREAMABLE_HTTP,
                        url="https://mcp.example.com",
                        enabled=True,
                    ),
                }
            ),
            user_id="user-1",
        ),
        mcp_tool=fake_mcp_tool,
    )

    assert any(tool.has_tool("mcp_demo_ping") for tool in tools)


def test_message_tool_should_have_stable_tool_name() -> None:
    tool = MessageTool()

    assert tool.name == "message"
    assert tool.has_tool("message_ask_user") is True
    assert tool.has_tool("message_notify_user") is True


def test_mcp_capability_adapter_should_timeout_and_normalize_error() -> None:
    fake_mcp_tool = _FakeMCPTool(delay_seconds=0.05)
    adapter = MCPCapabilityAdapter(
        mcp_tool=fake_mcp_tool,
        invoke_timeout_seconds=0.01,
    )

    result = asyncio.run(adapter.invoke("mcp_demo_ping"))

    assert result.success is False
    assert "超时" in (result.message or "")


def test_mcp_capability_adapter_should_resolve_dash_alias() -> None:
    fake_mcp_tool = _FakeMCPTool()
    adapter = MCPCapabilityAdapter(mcp_tool=fake_mcp_tool)

    # 兼容前端/模型输出中 `-/_` 混用的工具名漂移。
    assert adapter.has_tool("mcp-demo-ping") is True
    result = asyncio.run(adapter.invoke("mcp-demo-ping"))

    assert result.success is True
    assert fake_mcp_tool.invocations == [("mcp_demo_ping", {})]


def test_mcp_capability_adapter_should_block_disabled_server() -> None:
    fake_mcp_tool = _FakeMCPTool()
    adapter = MCPCapabilityAdapter(
        mcp_tool=fake_mcp_tool,
        mcp_config=MCPConfig(
            mcpServers={
                "demo": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    url="https://mcp.example.com",
                    enabled=False,
                )
            }
        ),
    )

    result = asyncio.run(adapter.invoke("mcp_demo_ping"))

    assert result.success is False
    assert "不可用或未启用" in (result.message or "")
