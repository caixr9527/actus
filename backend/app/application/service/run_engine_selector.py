#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : run_engine_selector.py
"""
import logging
from typing import Callable

from app.domain.external import LLM, JSONParser, Browser, Sandbox, SearchEngine
from app.domain.models import AgentConfig
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine, LegacyPlannerReActRunEngine
from app.domain.services.tools import MCPTool, A2ATool
from app.infrastructure.runtime import LangGraphRunEngine
from core.config import get_settings

logger = logging.getLogger(__name__)


def build_run_engine(
        llm: LLM,
        agent_config: AgentConfig,
        session_id: str,
        uow_factory: Callable[[], IUnitOfWork],
        json_parser: JSONParser,
        browser: Browser,
        sandbox: Sandbox,
        search_engine: SearchEngine,
        mcp_tool: MCPTool,
        a2a_tool: A2ATool,
) -> RunEngine:
    """根据配置选择运行时引擎，LangGraph POC 不可用时自动回退旧实现。"""
    settings = get_settings()
    engine_kind = settings.agent_runtime_engine.strip().lower()

    if engine_kind == "langgraph_poc":
        try:
            logger.info("启用 LangGraph POC 运行时引擎")
            return LangGraphRunEngine(session_id=session_id, llm=llm)
        except Exception as e:
            logger.warning(f"LangGraph POC 初始化失败，回退 legacy planner-react: {e}")

    return LegacyPlannerReActRunEngine(
        llm=llm,
        agent_config=agent_config,
        session_id=session_id,
        uow_factory=uow_factory,
        json_parser=json_parser,
        browser=browser,
        sandbox=sandbox,
        search_engine=search_engine,
        mcp_tool=mcp_tool,
        a2a_tool=a2a_tool,
    )
