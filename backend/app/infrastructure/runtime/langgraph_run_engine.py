#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : langgraph_run_engine.py
"""
import logging
from typing import AsyncGenerator, Callable, Optional

from app.domain.external import LLM
from app.domain.models import BaseEvent, Message
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine
from app.infrastructure.runtime.checkpoint_store_adapter import CheckpointStoreAdapter
from app.infrastructure.runtime.langgraph_graphs import build_planner_react_poc_graph

logger = logging.getLogger(__name__)


class LangGraphRunEngine(RunEngine):
    """基于 LangGraph 的最小 POC 运行时引擎。"""

    def __init__(
            self,
            session_id: str,
            llm: LLM,
            uow_factory: Optional[Callable[[], IUnitOfWork]] = None,
    ) -> None:
        self._session_id = session_id
        self._graph = build_planner_react_poc_graph(llm=llm)
        self._checkpoint_adapter = (
            CheckpointStoreAdapter(session_id=session_id, uow_factory=uow_factory)
            if uow_factory is not None
            else None
        )

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        invoke_config = {"configurable": {"thread_id": self._session_id}}
        run_id = None
        if self._checkpoint_adapter is not None:
            try:
                invoke_config, run_id = await self._checkpoint_adapter.resolve_invoke_config()
            except Exception as e:
                # 配置解析失败不阻断主流程，降级到 session 级 thread。
                logger.warning("会话[%s]解析checkpoint配置失败，回退默认thread配置: %s", self._session_id, e)

        try:
            state = await self._graph.ainvoke(
                {
                    "session_id": self._session_id,
                    "user_message": message.message,
                    "emitted_events": [],
                },
                config=invoke_config,
            )
        finally:
            if self._checkpoint_adapter is not None:
                # 无论图执行成功或失败，都尝试同步最新 checkpoint 引用，保证恢复点尽量前移。
                await self._checkpoint_adapter.sync_latest_checkpoint_ref(
                    run_id=run_id,
                    checkpointer=getattr(self._graph, "checkpointer", None),
                    invoke_config=invoke_config,
                )

        for event in state.get("emitted_events", []):
            yield event
