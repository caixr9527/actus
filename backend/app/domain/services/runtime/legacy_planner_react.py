#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : legacy_planner_react.py
"""
from typing import Callable, AsyncGenerator

from app.domain.external import Sandbox, Browser, SearchEngine, LLM, JSONParser
from app.domain.models import Message, BaseEvent, AgentConfig
from app.domain.repositories import IUnitOfWork
from app.domain.services.flows import PlannerReActFlow
from app.domain.services.runtime.run_engine import RunEngine
from app.domain.services.tools import MCPTool, A2ATool


class LegacyPlannerReActRunEngine(RunEngine):
    """旧版 planner-react 流程适配器，作为默认运行时实现。"""

    def __init__(
            self,
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
    ) -> None:
        self._flow = PlannerReActFlow(
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

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        async for event in self._flow.invoke(message):
            yield event
