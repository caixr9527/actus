#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21
@Author : caixiaorong01@outlook.com
@File   : runtime_adapter.py
"""
import logging
import json
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from app.domain.models import (
    A2AToolContent,
    BrowserActionableElementsResult,
    BrowserCardExtractionResult,
    BrowserLinkMatchResult,
    BrowserMainContentResult,
    BrowserPageStructuredResult,
    BrowserToolContent,
    FetchPageToolContent,
    FetchedPage,
    MCPToolContent,
    SearchResults,
    SearchToolContent,
    FileToolContent,
    ShellToolContent,
    ToolDiagnosticContent,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
)
from app.domain.models.app_config import A2AConfig, MCPConfig
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.base import BaseTool
from app.domain.services.tools.capability_registry import CapabilityBuildContext, CapabilityRegistry
from app.domain.services.tools.mcp import MCPTool

logger = logging.getLogger(__name__)

FILE_CONTENT_PREVIEW_MAX_CHARS = 2000
BROWSER_SCREENSHOT_FUNCTIONS: tuple[str, ...] = (
    "browser_view",
    "browser_navigate",
    "browser_restart",
    "browser_click",
    "browser_input",
    "browser_press_key",
    "browser_select_option",
)


@dataclass(frozen=True)
class ToolRuntimeEventHooks:
    """工具事件富化阶段依赖的外部钩子。

    设计说明：
    - ToolRuntimeAdapter 位于 domain service 层，不直接依赖具体基础设施实现。
    - 通过 hooks 由调用方按需注入环境能力，实现“逻辑抽离 + 依赖倒置”。
    - 非热路径或当前阶段不需要的能力允许留空，避免调用方传递 no-op 占位函数。
    """

    get_browser_screenshot: Optional[Callable[[], Awaitable[str]]] = None
    get_shell_tool_result: Optional[Callable[[], Awaitable[ToolResult]]] = None
    read_file_content: Optional[Callable[[str], Awaitable[ToolResult]]] = None
    sync_file_to_storage: Optional[Callable[[str], Awaitable[object]]] = None


class ToolRuntimeAdapter:
    """BE-LG-05 工具运行时适配器。

    当前版本职责：
    1. 通过 CapabilityRegistry 构建工具集合（本地能力优先）。
    2. 管理 MCP/A2A 的初始化与清理生命周期（兼容现有链路）。
    3. 统一处理 ToolEvent 的结果富化逻辑（search/browser/shell/file + 兼容 MCP/A2A）。
    """

    # BE-LG-05 本地能力基座；BE-LG-11 起 MCP 以 capability provider 形式接入。
    DEFAULT_LOCAL_CAPABILITIES: tuple[str, ...] = (
        CapabilityRegistry.CAPABILITY_SANDBOX_FILE,
        CapabilityRegistry.CAPABILITY_LOCAL_SHELL,
        CapabilityRegistry.CAPABILITY_SEARCH,
        CapabilityRegistry.CAPABILITY_BROWSER,
        CapabilityRegistry.CAPABILITY_MESSAGE,
    )
    DEFAULT_REMOTE_CAPABILITIES: tuple[str, ...] = (
        CapabilityRegistry.CAPABILITY_MCP,
    )

    def __init__(self, capability_registry: CapabilityRegistry) -> None:
        self._capability_registry = capability_registry

    @staticmethod
    def _render_file_tool_result(event: ToolEvent) -> str:
        """将文件工具结果渲染为可读文本。

        兼容场景：
        - list_files / find_files：展示目录与文件列表；
        - 其他文件工具：优先展示 data，其次展示 message。
        """
        function_result = event.function_result
        if function_result is None:
            return ""

        result_data = function_result.data if hasattr(function_result, "data") else None
        function_name = str(event.function_name or "").strip().lower()

        if function_name == "read_file" and isinstance(result_data, dict):
            content = str(result_data.get("content") or "")
            if content:
                return content[:FILE_CONTENT_PREVIEW_MAX_CHARS]
            return str(function_result.message or "").strip()

        if function_name in {"list_files", "find_files"} and isinstance(result_data, dict):
            raw_files = result_data.get("files")
            dir_path = str(result_data.get("dir_path") or event.function_args.get("dir_path") or "")
            files = [str(item).strip() for item in (raw_files or []) if str(item).strip()]
            if len(files) == 0:
                return f"目录{dir_path}下未找到文件" if dir_path else "未找到文件"
            header = f"目录: {dir_path}\n共 {len(files)} 项" if dir_path else f"共 {len(files)} 项"
            return f"{header}\n" + "\n".join(files)

        if result_data is not None:
            if isinstance(result_data, str):
                return result_data
            return json.dumps(result_data, ensure_ascii=False, indent=2, default=str)

        message = str(function_result.message or "").strip()
        if message:
            return message
        return ""

    @staticmethod
    def _build_search_results_content(event: ToolEvent) -> SearchToolContent:
        function_result = event.function_result
        if function_result is None or function_result.data is None:
            return SearchToolContent(results=[])

        search_results = function_result.data
        if not isinstance(search_results, SearchResults):
            raise TypeError("search_web 的结果必须是 SearchResults")
        return SearchToolContent(results=search_results.results)

    @staticmethod
    def _build_fetch_page_content(event: ToolEvent) -> FetchPageToolContent:
        function_result = event.function_result
        if function_result is None or function_result.data is None:
            return FetchPageToolContent(url="")

        fetched_page = function_result.data
        if not isinstance(fetched_page, FetchedPage):
            raise TypeError("fetch_page 的结果必须是 FetchedPage")
        return FetchPageToolContent.from_fetched_page(fetched_page)

    @staticmethod
    def _build_tool_diagnostic_content(event: ToolEvent) -> ToolDiagnosticContent:
        """构建 search/fetch 的通用诊断卡片。

        使用场景：
        - 执行约束层 block 生成的虚拟失败结果；
        - effects 层生成的 research 诊断降级结果；
        - search/fetch 未命中真实结构化结果类型时的展示兜底。
        """
        function_result = event.function_result
        if function_result is None:
            return ToolDiagnosticContent()

        result_data = function_result.data if isinstance(function_result.data, dict) else {}
        research_diagnosis = result_data.get("research_diagnosis") if isinstance(result_data, dict) else None
        if isinstance(research_diagnosis, dict):
            return ToolDiagnosticContent(
                message=str(function_result.message or "").strip(),
                reason_code=str(research_diagnosis.get("code") or "").strip(),
                diagnostic_type="research_diagnosis",
                details=dict(research_diagnosis),
            )

        return ToolDiagnosticContent(
            message=str(function_result.message or "").strip(),
            reason_code=str(result_data.get("reason_code") or "").strip(),
            diagnostic_type="tool_result_fallback",
            details=dict(result_data),
        )

    @staticmethod
    def _build_browser_content(event: ToolEvent, screenshot: str) -> BrowserToolContent:
        function_result = event.function_result
        result_data = function_result.data if function_result is not None else None
        content = BrowserToolContent(screenshot=screenshot)

        if isinstance(result_data, BrowserPageStructuredResult):
            content.url = result_data.url
            content.title = result_data.title
            content.page_type = result_data.page_type.value
            content.structured_page = result_data
            content.cards = list(result_data.cards or [])
            content.actionable_elements = list(result_data.actionable_elements or [])
            return content

        if isinstance(result_data, BrowserMainContentResult):
            content.url = result_data.url
            content.title = result_data.title
            content.page_type = result_data.page_type.value
            content.main_content = result_data
            return content

        if isinstance(result_data, BrowserCardExtractionResult):
            content.url = result_data.url
            content.title = result_data.title
            content.page_type = result_data.page_type.value
            content.cards = list(result_data.cards or [])
            return content

        if isinstance(result_data, BrowserActionableElementsResult):
            content.url = result_data.url
            content.title = result_data.title
            content.page_type = result_data.page_type.value
            content.actionable_elements = list(result_data.elements or [])
            return content

        if isinstance(result_data, BrowserLinkMatchResult):
            content.url = result_data.url
            content.title = result_data.matched_text
            content.matched_link_text = result_data.matched_text
            content.matched_link_url = result_data.url
            content.matched_link_selector = result_data.selector
            content.matched_link_index = result_data.index
            if result_data.card is not None:
                content.cards = [result_data.card]
            return content

        if isinstance(result_data, dict):
            content.url = str(result_data.get("url") or "").strip()
            content.title = str(result_data.get("title") or "").strip()
            content.page_type = str(result_data.get("page_type") or "").strip()
            content.degrade_reason = str(result_data.get("degrade_reason") or "").strip()
            return content

        return content

    def build_runtime_tools(
            self,
            capability_context: CapabilityBuildContext,
            mcp_tool: Optional[MCPTool] = None,
            mcp_config: Optional[MCPConfig] = None,
            a2a_tool: Optional[A2ATool] = None,
    ) -> List[BaseTool]:
        """构建运行时工具列表。

        规则：
        - 先按能力注册表构建本地工具；
        - 再按上下文追加远端 MCP/A2A 工具，保持统一运行时装配入口。
        """
        context = CapabilityBuildContext(
            sandbox=capability_context.sandbox,
            browser=capability_context.browser,
            search_engine=capability_context.search_engine,
            workspace_runtime_service=capability_context.workspace_runtime_service,
            # 优先使用显式参数，其次回退到上下文已有值。
            mcp_tool=mcp_tool if mcp_tool is not None else capability_context.mcp_tool,
            a2a_tool=a2a_tool if a2a_tool is not None else capability_context.a2a_tool,
            mcp_config=mcp_config if mcp_config is not None else capability_context.mcp_config,
            a2a_config=capability_context.a2a_config,
            user_id=capability_context.user_id,
        )

        tools = self._capability_registry.build_tools(
            capability_ids=self.DEFAULT_LOCAL_CAPABILITIES,
            context=context,
        )

        # MCP provider 升级为 capability provider，通过能力声明接入。
        if context.mcp_tool is not None:
            tools.extend(
                self._capability_registry.build_tools(
                    capability_ids=self.DEFAULT_REMOTE_CAPABILITIES,
                    context=context,
                )
            )

        if a2a_tool is not None:
            tools.append(a2a_tool)
        return tools

    @staticmethod
    async def initialize_remote_tools(
            mcp_tool: Optional[MCPTool],
            mcp_config: Optional[MCPConfig],
            a2a_tool: Optional[A2ATool],
            a2a_config: Optional[A2AConfig],
    ) -> None:
        """初始化远程工具管理器（MCP/A2A）。"""
        if mcp_tool is not None:
            await mcp_tool.initialize(mcp_config)
        if a2a_tool is not None:
            await a2a_tool.initialize(a2a_config)

    @staticmethod
    async def cleanup_remote_tools(
            mcp_tool: Optional[MCPTool],
            a2a_tool: Optional[A2ATool],
    ) -> None:
        """清理远程工具资源，失败不抛异常（best effort）。"""
        try:
            if mcp_tool is not None:
                await mcp_tool.cleanup()
        except Exception as e:
            logger.warning(f"清理MCP工具资源时出错: {e}")

        try:
            manager = getattr(a2a_tool, "manager", None) if a2a_tool is not None else None
            if manager is not None:
                await manager.cleanup()
        except Exception as e:
            logger.warning(f"清理A2A工具资源时出错: {e}")

    async def enrich_tool_event(self, event: ToolEvent, hooks: ToolRuntimeEventHooks) -> bool:
        """对 ToolEvent 进行结果富化。

        返回值：
        - True: 当前事件已被 adapter 处理（含兼容 MCP/A2A）
        - False: 当前事件无需处理或未命中处理分支
        """
        if event.status != ToolEventStatus.CALLED:
            return False

        if event.tool_name == "browser":
            function_name = str(event.function_name or "").strip().lower()
            screenshot = ""
            if hooks.get_browser_screenshot is not None and function_name in BROWSER_SCREENSHOT_FUNCTIONS:
                screenshot = str(await hooks.get_browser_screenshot() or "").strip()
            event.tool_content = self._build_browser_content(event, screenshot)
            return True

        if event.tool_name == "search":
            function_name = str(event.function_name or "").strip().lower()
            if function_name == "search_web":
                function_result = event.function_result
                if (
                        function_result is not None
                        and bool(function_result.success)
                        and isinstance(function_result.data, SearchResults)
                ):
                    event.tool_content = self._build_search_results_content(event)
                else:
                    event.tool_content = self._build_tool_diagnostic_content(event)
                return True
            if function_name == "fetch_page":
                function_result = event.function_result
                if (
                        function_result is not None
                        and bool(function_result.success)
                        and isinstance(function_result.data, FetchedPage)
                ):
                    event.tool_content = self._build_fetch_page_content(event)
                else:
                    event.tool_content = self._build_tool_diagnostic_content(event)
                return True
            return False

        if event.tool_name == "shell":
            if hooks.get_shell_tool_result is None:
                event.tool_content = ShellToolContent(console="(No console)")
                return True

            shell_result = await hooks.get_shell_tool_result()
            shell_payload = shell_result.data or {}
            console_content = (
                shell_payload.get("console_records")
                or shell_payload.get("output")
                or "(No console)"
            )
            event.tool_content = ShellToolContent(
                console=console_content,
            )
            return True

        if event.tool_name == "file":
            rendered_result = self._render_file_tool_result(event)
            event.tool_content = FileToolContent(content=rendered_result or "(No Content)")
            return True

        # 事件富化层保留 MCP/A2A 兼容分支，确保既有前端结果卡片协议不变。
        if event.tool_name in {"mcp", "a2a"}:
            if event.function_result:
                if hasattr(event.function_result, "data") and event.function_result.data:
                    event.tool_content = (
                        MCPToolContent(result=event.function_result.data)
                        if event.tool_name == "mcp"
                        else A2AToolContent(a2a_result=event.function_result.data)
                    )
                    return True

                if hasattr(event.function_result, "success") and event.function_result.success:
                    result_data = (
                        event.function_result.model_dump()
                        if hasattr(event.function_result, "model_dump")
                        else str(event.function_result)
                    )
                    event.tool_content = (
                        MCPToolContent(result=result_data)
                        if event.tool_name == "mcp"
                        else A2AToolContent(a2a_result=result_data)
                    )
                    return True

                event.tool_content = (
                    MCPToolContent(result=str(event.function_result))
                    if event.tool_name == "mcp"
                    else A2AToolContent(a2a_result=str(event.function_result))
                )
                return True

            event.tool_content = (
                MCPToolContent(result="(MCP工具无可用结果)")
                if event.tool_name == "mcp"
                else A2AToolContent(a2a_result="(A2A智能体无可用结果)")
            )
            return True

        return False
