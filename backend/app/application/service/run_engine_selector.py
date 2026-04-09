#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : run_engine_selector.py
"""
import logging
from typing import Callable, Dict

from app.domain.external import LLM, JSONParser, Browser, Sandbox, SearchEngine, FileStorage
from app.domain.models import AgentConfig, MCPConfig
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine
from app.domain.services.runtime.stage_llm import REQUIRED_STAGE_LLM_NAMES
from app.domain.services.tools import MCPTool, A2ATool, ToolRuntimeAdapter, CapabilityBuildContext
from app.infrastructure.runtime import LangGraphRunEngine, get_langgraph_checkpointer
from core.config import get_settings

logger = logging.getLogger(__name__)


def _maybe_clone_llm_for_stage(
        llm: LLM,
        *,
        model_name: str | None,
        max_tokens: int | None,
) -> LLM | None:
    clone_with_overrides = getattr(llm, "clone_with_overrides", None)
    if not callable(clone_with_overrides):
        return None
    return clone_with_overrides(
        model_name=model_name,
        max_tokens=max_tokens,
    )


def _build_stage_llms(llm: LLM) -> Dict[str, LLM]:
    stage_specs: Dict[str, Dict[str, str | int | None]] = {
        # todo 临时使用 LLM 模型名称及max_tokens，后续会支持自定义，不要对这处进行修改【重点】
        "router": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 2048),
        },
        "planner": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 4096),
        },
        "executor": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 8192),
        },
        "replan": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 4096),
        },
        "summary": {
            "model_name": llm.model_name,
            "max_tokens": getattr(llm, "max_tokens", 4096),
        },
    }
    stage_llms: Dict[str, LLM] = {}
    for stage_name in REQUIRED_STAGE_LLM_NAMES:
        spec = stage_specs[stage_name]
        stage_llm = _maybe_clone_llm_for_stage(
            llm,
            model_name=spec["model_name"],
            max_tokens=spec["max_tokens"],
        )
        stage_llms[stage_name] = stage_llm or llm
    return stage_llms


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
    # P0: 单步骤工具循环先做硬上限收口，避免错误回路被配置值无限放大。
    max_tool_iterations = max(1, min(int(agent_config.max_iterations), 20))
    return LangGraphRunEngine(
        session_id=session_id,
        stage_llms=_build_stage_llms(llm),
        file_storage=file_storage,
        user_id=user_id,
        uow_factory=uow_factory,
        runtime_tools=runtime_tools,
        max_tool_iterations=max_tool_iterations,
        checkpointer=get_langgraph_checkpointer().get_checkpointer(),
    )
