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
    BrowserToolContent,
    MCPToolContent,
    SearchResults,
    SearchToolContent,
    ShellToolContent,
    FileToolContent,
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


@dataclass(frozen=True)
class ToolRuntimeEventHooks:
    """工具事件富化阶段依赖的外部钩子。

    设计说明：
    - ToolRuntimeAdapter 位于 domain service 层，不直接依赖具体基础设施实现。
    - 通过 hooks 由调用方注入环境能力，实现“逻辑抽离 + 依赖倒置”。
    """

    get_browser_screenshot: Callable[[], Awaitable[str]]
    read_shell_output: Callable[[str], Awaitable[ToolResult]]
    read_file_content: Callable[[str], Awaitable[ToolResult]]
    sync_file_to_storage: Callable[[str], Awaitable[object]]


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

    def __init__(self, capability_registry: Optional[CapabilityRegistry] = None) -> None:
        self._capability_registry = capability_registry or CapabilityRegistry.default_v1()

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
            # 优先使用显式参数，其次回退到上下文已有值。
            mcp_tool=mcp_tool if mcp_tool is not None else capability_context.mcp_tool,
            a2a_tool=a2a_tool if a2a_tool is not None else capability_context.a2a_tool,
            mcp_config=mcp_config if mcp_config is not None else capability_context.mcp_config,
            a2a_config=capability_context.a2a_config,
            session_id=capability_context.session_id,
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
            event.tool_content = BrowserToolContent(
                screenshot=await hooks.get_browser_screenshot(),
            )
            return True

        if event.tool_name == "search":
            search_results: Optional[ToolResult[SearchResults]] = event.function_result
            results = []
            if search_results is not None and search_results.data is not None:
                results = search_results.data.results
            event.tool_content = SearchToolContent(results=results)
            return True

        if event.tool_name == "shell":
            session_id = str(event.function_args.get("session_id") or "")
            if not session_id:
                event.tool_content = ShellToolContent(console="(No console)")
                return True

            shell_result = await hooks.read_shell_output(session_id)
            event.tool_content = ShellToolContent(
                console=(shell_result.data or {}).get("console_records", []),
            )
            return True

        if event.tool_name == "file":
            filepath = str(event.function_args.get("filepath") or "")
            if not filepath:
                rendered_result = self._render_file_tool_result(event)
                event.tool_content = FileToolContent(content=rendered_result or "(No Content)")
                return True

            file_read_result = await hooks.read_file_content(filepath)
            file_content = (file_read_result.data or {}).get("content", "")
            event.tool_content = FileToolContent(content=file_content)
            await hooks.sync_file_to_storage(filepath)
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
