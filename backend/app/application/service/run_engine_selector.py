#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : run_engine_selector.py
"""
import logging
from typing import Callable

from app.domain.external import LLM, JSONParser, Browser, Sandbox, SearchEngine, FileStorage
from app.domain.models import AgentConfig, MCPConfig
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine
from app.domain.services.tools import MCPTool, A2ATool, ToolRuntimeAdapter, CapabilityBuildContext
from app.infrastructure.runtime import LangGraphRunEngine
from core.config import get_settings

logger = logging.getLogger(__name__)


def build_run_engine(
        llm: LLM,
        agent_config: AgentConfig,
        session_id: str,
        file_storage: FileStorage,
        uow_factory: Callable[[], IUnitOfWork],
        json_parser: JSONParser,
        browser: Browser,
        sandbox: Sandbox,
        search_engine: SearchEngine,
        mcp_tool: MCPTool,
        a2a_tool: A2ATool,
        mcp_config: MCPConfig | None = None,
        user_id: str | None = None,
        tool_runtime_adapter: ToolRuntimeAdapter | None = None,
) -> RunEngine:
    """根据配置选择运行时引擎（BE-LG-12 起仅支持 LangGraph）。"""
    settings = get_settings()
    engine_kind = settings.agent_runtime_engine.strip().lower()

    if engine_kind != "langgraph":
        raise ValueError(f"不支持的运行时引擎配置: {engine_kind}，当前仅支持 langgraph")

    logger.info("启用 LangGraph 运行时引擎")
    runtime_adapter = tool_runtime_adapter or ToolRuntimeAdapter()
    runtime_tools = runtime_adapter.build_runtime_tools(
        capability_context=CapabilityBuildContext(
            sandbox=sandbox,
            browser=browser,
            search_engine=search_engine,
            mcp_tool=mcp_tool,
            a2a_tool=a2a_tool,
            mcp_config=mcp_config,
            session_id=session_id,
            user_id=user_id,
        ),
        mcp_tool=mcp_tool,
        mcp_config=mcp_config,
        a2a_tool=a2a_tool,
    )
    # LangGraph 单步骤工具循环设置上限，避免单次 step 在高 max_iterations 配置下耗时失控。
    max_tool_iterations = max(1, min(int(agent_config.max_iterations), 20))
    return LangGraphRunEngine(
        session_id=session_id,
        llm=llm,
        file_storage=file_storage,
        user_id=user_id,
        uow_factory=uow_factory,
        runtime_tools=runtime_tools,
        max_tool_iterations=max_tool_iterations,
    )
