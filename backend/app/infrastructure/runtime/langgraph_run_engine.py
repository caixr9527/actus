#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : langgraph_run_engine.py
"""
import logging
import asyncio
import json
from contextlib import suppress
from typing import Any, AsyncGenerator, Callable, Dict, Optional, List

from app.domain.external import LLM
from app.domain.models import BaseEvent, Message
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine
from app.domain.services.tools import BaseTool
from app.domain.services.runtime.langgraph_state import GraphStateContractMapper, PlannerReActLangGraphState
from app.infrastructure.runtime.checkpoint_store_adapter import CheckpointStoreAdapter
from app.infrastructure.runtime.langgraph_graphs import (
    bind_live_event_sink,
    build_planner_react_langgraph_graph,
    unbind_live_event_sink,
)

logger = logging.getLogger(__name__)


class _EventDeduplicator:
    """基于 event_id + payload signature 的事件去重器。"""

    def __init__(self) -> None:
        self._event_ids: set[str] = set()
        self._event_signatures: set[str] = set()

    @staticmethod
    def _build_signature(event: BaseEvent) -> str:
        payload = event.model_dump(mode="json")
        payload.pop("id", None)
        return json.dumps(
            {
                "event_type": event.type,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def should_emit(self, event: BaseEvent) -> bool:
        event_id = str(event.id or "")
        event_signature = self._build_signature(event)
        if event_id and event_id in self._event_ids:
            return False
        if event_signature in self._event_signatures:
            return False
        if event_id:
            self._event_ids.add(event_id)
        self._event_signatures.add(event_signature)
        return True


class LangGraphRunEngine(RunEngine):
    """基于 LangGraph 的运行时引擎。"""

    def __init__(
            self,
            session_id: str,
            llm: LLM,
            uow_factory: Optional[Callable[[], IUnitOfWork]] = None,
            runtime_tools: Optional[List[BaseTool]] = None,
            max_tool_iterations: Optional[int] = None,
    ) -> None:
        self._session_id = session_id
        self._uow_factory = uow_factory
        if runtime_tools is None and max_tool_iterations is None:
            self._graph = build_planner_react_langgraph_graph(llm=llm)
        else:
            try:
                self._graph = build_planner_react_langgraph_graph(
                    llm=llm,
                    runtime_tools=runtime_tools,
                    max_tool_iterations=max_tool_iterations or 5,
                )
            except TypeError:
                # 兼容测试/历史 monkeypatch 场景：回退到仅传 llm 的构图签名。
                self._graph = build_planner_react_langgraph_graph(llm=llm)
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
    ) -> PlannerReActLangGraphState:
        """构建 BE-LG-04 契约化 graph input state。"""
        if self._uow_factory is None:
            return {
                "session_id": self._session_id,
                "user_message": message.message,
                "input_parts": message.input_envelope.model_dump(mode="json").get("parts", []),
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
                    input_parts=message.input_envelope.model_dump(mode="json").get("parts", []),
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
                "input_parts": message.input_envelope.model_dump(mode="json").get("parts", []),
                "emitted_events": [],
            }

    async def _sync_graph_state_contract(
            self,
            run_id: Optional[str],
            state: Optional[PlannerReActLangGraphState],
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
        state: Optional[PlannerReActLangGraphState] = None
        deduplicator = _EventDeduplicator()
        live_event_queue: "asyncio.Queue[BaseEvent]" = asyncio.Queue()

        async def _enqueue_live_event(event: BaseEvent) -> None:
            await live_event_queue.put(event)

        sink_token = bind_live_event_sink(_enqueue_live_event)
        graph_task: asyncio.Task | None = None
        try:
            graph_task = asyncio.create_task(
                self._graph.ainvoke(
                    graph_input_state,
                    config=invoke_config,
                )
            )
            async for live_event in self._forward_live_events(
                    graph_task=graph_task,
                    live_event_queue=live_event_queue,
                    deduplicator=deduplicator,
            ):
                yield live_event

            state = await graph_task
            state = GraphStateContractMapper.apply_emitted_events(state=state)
        finally:
            unbind_live_event_sink(sink_token)
            if graph_task is not None and not graph_task.done():
                graph_task.cancel()
                with suppress(Exception):
                    await graph_task

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
            if deduplicator.should_emit(event):
                yield event.model_copy(deep=True)

    async def _forward_live_events(
            self,
            *,
            graph_task: "asyncio.Task[Any]",
            live_event_queue: "asyncio.Queue[BaseEvent]",
            deduplicator: _EventDeduplicator,
    ) -> AsyncGenerator[BaseEvent, None]:
        """边执行边转发 LangGraph 节点实时事件。"""
        while True:
            if graph_task.done() and live_event_queue.empty():
                break
            try:
                event = await asyncio.wait_for(live_event_queue.get(), timeout=0.1)
            except TimeoutError:
                continue

            if deduplicator.should_emit(event):
                # 对外输出副本，避免下游写入 event.id 反向污染 graph state 中的同一对象。
                yield event.model_copy(deep=True)
