#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : langgraph_run_engine.py
"""
import logging
from typing import AsyncGenerator, Callable, Dict, Optional

from app.domain.external import LLM
from app.domain.models import BaseEvent, Message
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine
from app.domain.services.runtime.langgraph_state import GraphStateContractMapper, PlannerReActPOCState
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
        self._uow_factory = uow_factory
        self._graph = build_planner_react_poc_graph(llm=llm)
        self._checkpoint_adapter = (
            CheckpointStoreAdapter(session_id=session_id, uow_factory=uow_factory)
            if uow_factory is not None
            else None
        )

    async def _build_graph_input_state(
            self,
            message: Message,
            run_id: Optional[str],
            invoke_config: Dict[str, Dict[str, str]],
    ) -> PlannerReActPOCState:
        """构建 BE-LG-04 契约化 graph input state。"""
        if self._uow_factory is None:
            return {
                "session_id": self._session_id,
                "user_message": message.message,
                "emitted_events": [],
            }

        configurable = invoke_config.get("configurable") or {}
        thread_id = str(configurable.get("thread_id") or self._session_id)
        checkpoint_namespace = str(configurable.get("checkpoint_ns") or "")
        checkpoint_id_raw = configurable.get("checkpoint_id")
        checkpoint_id = str(checkpoint_id_raw) if checkpoint_id_raw else None

        try:
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id=self._session_id)
                if session is None:
                    raise ValueError(f"会话[{self._session_id}]不存在，无法构建Graph初始状态")

                resolved_run_id = run_id or session.current_run_id
                run = (
                    await uow.workflow_run.get_by_id(resolved_run_id)
                    if resolved_run_id
                    else None
                )

                return GraphStateContractMapper.build_initial_state(
                    session=session,
                    run=run,
                    user_message=message.message,
                    thread_id=thread_id,
                    checkpoint_namespace=checkpoint_namespace,
                    checkpoint_id=checkpoint_id,
                )
        except Exception as e:
            # 状态构建失败时必须降级，不能阻断主链路执行。
            logger.warning("会话[%s]构建Graph初始状态失败，回退最小输入: %s", self._session_id, e)
            return {
                "session_id": self._session_id,
                "run_id": run_id,
                "thread_id": thread_id,
                "checkpoint_ref_namespace": checkpoint_namespace,
                "checkpoint_ref_id": checkpoint_id,
                "user_message": message.message,
                "emitted_events": [],
            }

    async def _sync_graph_state_contract(
            self,
            run_id: Optional[str],
            state: Optional[PlannerReActPOCState],
    ) -> None:
        """回写 BE-LG-04 graph state contract 到 runtime_metadata。"""
        if self._uow_factory is None or state is None:
            return

        resolved_run_id = run_id or state.get("run_id")
        if not resolved_run_id:
            return

        runtime_metadata = GraphStateContractMapper.build_runtime_metadata(state)
        try:
            async with self._uow_factory() as uow:
                run = await uow.workflow_run.get_by_id(resolved_run_id)
                if run is None:
                    logger.warning("运行[%s]不存在，跳过graph_state_contract回写", resolved_run_id)
                    return
                await uow.workflow_run.update_runtime_metadata(
                    run_id=resolved_run_id,
                    runtime_metadata=runtime_metadata,
                    current_step_id=state.get("current_step_id"),
                )
        except Exception as e:
            logger.warning("会话[%s]回写graph_state_contract失败: %s", self._session_id, e)

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        invoke_config = {"configurable": {"thread_id": self._session_id}}
        run_id = None
        if self._checkpoint_adapter is not None:
            try:
                invoke_config, run_id = await self._checkpoint_adapter.resolve_invoke_config()
            except Exception as e:
                # 配置解析失败不阻断主流程，降级到 session 级 thread。
                logger.warning("会话[%s]解析checkpoint配置失败，回退默认thread配置: %s", self._session_id, e)

        graph_input_state = await self._build_graph_input_state(
            message=message,
            run_id=run_id,
            invoke_config=invoke_config,
        )
        state: Optional[PlannerReActPOCState] = None
        try:
            state = await self._graph.ainvoke(
                graph_input_state,
                config=invoke_config,
            )
            state = GraphStateContractMapper.apply_emitted_events(state=state)
        finally:
            if self._checkpoint_adapter is not None:
                # 无论图执行成功或失败，都尝试同步最新 checkpoint 引用，保证恢复点尽量前移。
                await self._checkpoint_adapter.sync_latest_checkpoint_ref(
                    run_id=run_id,
                    checkpointer=getattr(self._graph, "checkpointer", None),
                    invoke_config=invoke_config,
                )
            # graph 执行失败时，回写当前已知状态（至少包含本次输入上下文）。
            await self._sync_graph_state_contract(
                run_id=run_id,
                state=state or graph_input_state,
            )

        for event in (state or {}).get("emitted_events", []):
            yield event
