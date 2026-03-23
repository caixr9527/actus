#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21
@Author : caixiaorong01@outlook.com
@File   : capability_registry.py
"""
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from app.domain.models.app_config import A2AConfig, MCPConfig
from app.domain.external import Browser, Sandbox, SearchEngine
from app.domain.services.tools.base import BaseTool
from app.domain.services.tools.browser import BrowserTool
from app.domain.services.tools.file import FileTool
from app.domain.services.tools.message import MessageTool
from app.domain.services.tools.mcp_capability_adapter import MCPCapabilityAdapter
from app.domain.services.tools.search import SearchTool
from app.domain.services.tools.shell import ShellTool


@dataclass(frozen=True)
class CapabilityBuildContext:
    """构建能力工具时所需的运行时上下文。

    说明：
    - 能力注册表只负责“如何构建工具对象”，不直接执行工具调用。
    - 这里保留 Browser/Sandbox/SearchEngine 三类基础依赖，覆盖 BE-LG-05 首批本地能力。
    """

    sandbox: Sandbox
    browser: Browser
    search_engine: SearchEngine
    mcp_tool: Optional[BaseTool] = None
    a2a_tool: Optional[BaseTool] = None
    mcp_config: Optional[MCPConfig] = None
    a2a_config: Optional[A2AConfig] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


CapabilityToolFactory = Callable[[CapabilityBuildContext], BaseTool]


@dataclass(frozen=True)
class CapabilityDefinition:
    """能力定义。

    字段约束：
    - capability_id: 能力唯一标识，用于编排层声明依赖能力。
    - tool_family: 该能力构建出的 BaseTool.name，用于去重和审计。
    - factory: 基于运行时上下文构建工具实例。
    """

    capability_id: str
    description: str
    tool_family: str
    factory: CapabilityToolFactory


class CapabilityRegistry:
    """能力注册表（BE-LG-05 V1）。

    职责：
    1. 统一注册能力定义，避免工具初始化散落在多个调用点。
    2. 根据能力ID批量构建工具对象，为 Flow/RunEngine 提供稳定入口。
    3. 在构建阶段按 tool_family 去重，避免同一工具族被重复注入。
    """

    # BE-LG-05 首批能力ID。
    CAPABILITY_LOCAL_SHELL = "local_shell"
    CAPABILITY_SEARCH = "search"
    CAPABILITY_BROWSER = "browser"
    CAPABILITY_SANDBOX_FILE = "sandbox_file"
    CAPABILITY_MESSAGE = "message"
    CAPABILITY_MCP = "mcp"

    def __init__(self) -> None:
        self._definitions: Dict[str, CapabilityDefinition] = {}

    def register(self, definition: CapabilityDefinition) -> None:
        """注册单个能力定义。"""
        if definition.capability_id in self._definitions:
            raise ValueError(f"能力[{definition.capability_id}]已存在，禁止重复注册")
        self._definitions[definition.capability_id] = definition

    def register_many(self, definitions: Iterable[CapabilityDefinition]) -> None:
        """批量注册能力定义。"""
        for definition in definitions:
            self.register(definition)

    def get(self, capability_id: str) -> CapabilityDefinition:
        """按能力ID获取能力定义。"""
        definition = self._definitions.get(capability_id)
        if definition is None:
            raise ValueError(f"能力[{capability_id}]不存在，请检查能力注册表配置")
        return definition

    def list_capabilities(self) -> List[str]:
        """返回已注册能力ID列表。"""
        return list(self._definitions.keys())

    def build_tools(
            self,
            capability_ids: Iterable[str],
            context: CapabilityBuildContext,
    ) -> List[BaseTool]:
        """按能力ID构建工具列表。

        说明：
        - 构建顺序遵循 capability_ids 的输入顺序；
        - 同一 tool_family 只保留首个实例，避免重复工具定义污染模型工具清单。
        """
        built_tools: List[BaseTool] = []
        seen_tool_families: set[str] = set()

        for capability_id in capability_ids:
            definition = self.get(capability_id)
            if definition.tool_family in seen_tool_families:
                continue
            tool = definition.factory(context)
            built_tools.append(tool)
            seen_tool_families.add(definition.tool_family)

        return built_tools

    @classmethod
    def default_v1(cls) -> "CapabilityRegistry":
        """构建 BE-LG-05 V1 默认注册表。"""
        registry = cls()
        registry.register_many(
            [
                CapabilityDefinition(
                    capability_id=cls.CAPABILITY_LOCAL_SHELL,
                    description="本地Shell执行能力",
                    tool_family="shell",
                    factory=lambda context: ShellTool(sandbox=context.sandbox),
                ),
                CapabilityDefinition(
                    capability_id=cls.CAPABILITY_SEARCH,
                    description="网络搜索能力",
                    tool_family="search",
                    factory=lambda context: SearchTool(search_engine=context.search_engine),
                ),
                CapabilityDefinition(
                    capability_id=cls.CAPABILITY_BROWSER,
                    description="浏览器交互能力",
                    tool_family="browser",
                    factory=lambda context: BrowserTool(browser=context.browser),
                ),
                CapabilityDefinition(
                    capability_id=cls.CAPABILITY_SANDBOX_FILE,
                    description="沙箱文件读写能力",
                    tool_family="file",
                    factory=lambda context: FileTool(sandbox=context.sandbox),
                ),
                CapabilityDefinition(
                    capability_id=cls.CAPABILITY_MESSAGE,
                    description="用户通知/提问能力",
                    tool_family="message",
                    factory=lambda _context: MessageTool(),
                ),
                CapabilityDefinition(
                    capability_id=cls.CAPABILITY_MCP,
                    description="MCP能力适配器（统一超时/审计/错误语义）",
                    tool_family="mcp",
                    factory=lambda context: MCPCapabilityAdapter(
                        mcp_tool=context.mcp_tool,
                        mcp_config=context.mcp_config,
                        session_id=context.session_id,
                        user_id=context.user_id,
                    ),
                ),
            ]
        )
        return registry
