#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : langgraph_run_engine.py
"""
from typing import AsyncGenerator

from app.domain.external import LLM
from app.domain.models import BaseEvent, Message
from app.domain.services.runtime import RunEngine
from app.infrastructure.runtime.langgraph_graphs import build_planner_react_poc_graph


class LangGraphRunEngine(RunEngine):
    """基于 LangGraph 的最小 POC 运行时引擎。"""

    def __init__(self, session_id: str, llm: LLM) -> None:
        self._session_id = session_id
        self._graph = build_planner_react_poc_graph(llm=llm)

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        state = await self._graph.ainvoke(
            {
                "session_id": self._session_id,
                "user_message": message.message,
                "emitted_events": [],
            },
            config={"configurable": {"thread_id": self._session_id}},
        )
        for event in state.get("emitted_events", []):
            yield event
