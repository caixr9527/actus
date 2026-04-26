#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 图装配入口。"""

import logging
from typing import Any, List, Optional

from langgraph.graph import END, START, StateGraph

from app.domain.external import LLM
from app.domain.repositories import LongTermMemoryRepository
from app.domain.services.memory_consolidation import MemoryConsolidationService
from app.domain.services.runtime import SkillGraphRuntime
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.stage_llm import ensure_required_stage_llms
from app.domain.services.tools import BaseTool
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.graphs.skills.registry import build_default_skill_graph_registry
from .nodes import (
    atomic_action_node,
    consolidate_memory_node,
    create_or_reuse_plan_node,
    direct_answer_node,
    direct_wait_node,
    entry_router_node,
    execute_step_node,
    finalize_node,
    guard_step_reuse_node,
    recall_memory_context_node,
    replan_node,
    summarize_node,
    wait_for_human_node,
)
from .routing import (
    route_after_entry,
    route_after_execute,
    route_after_guard,
    route_after_plan,
    route_after_replan,
    route_after_wait,
)

logger = logging.getLogger(__name__)


def build_planner_react_langgraph_graph(
        stage_llms: dict[str, LLM],
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
        checkpointer: Optional[Any] = None,
        long_term_memory_repository: Optional[LongTermMemoryRepository] = None,
        memory_consolidation_service: Optional[MemoryConsolidationService] = None,
        *,
        runtime_context_service: RuntimeContextService,
) -> Any:
    """构建 LangGraph Planner-ReAct V1 图。"""
    if runtime_context_service is None:
        raise ValueError("runtime_context_service 不能为空")
    stage_llm_map = ensure_required_stage_llms(stage_llms)

    skill_runtime: Optional[SkillGraphRuntime] = None
    try:
        skill_runtime = build_default_skill_graph_registry().create_runtime(
            llm=stage_llm_map["executor"]
        )
    except Exception as e:
        log_runtime(
            logger,
            logging.WARNING,
            "技能注册表初始化失败",
            error=str(e),
        )

    async def _create_plan_with_llm(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await create_or_reuse_plan_node(
            state,
            stage_llm_map["planner"],
            runtime_context_service=runtime_context_service,
        )

    async def _entry_router(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await entry_router_node(state)

    async def _direct_answer(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await direct_answer_node(
            state,
            stage_llm_map["router"],
            runtime_context_service=runtime_context_service,
        )

    async def _direct_wait(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await direct_wait_node(state)

    async def _atomic_action(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await atomic_action_node(state)

    async def _recall_memory_context(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await recall_memory_context_node(
            state,
            long_term_memory_repository=long_term_memory_repository,
        )

    async def _execute_step_with_llm(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await execute_step_node(
            state,
            stage_llm_map["executor"],
            skill_runtime=skill_runtime,
            runtime_tools=runtime_tools,
            runtime_context_service=runtime_context_service,
            max_tool_iterations=max_tool_iterations,
        )

    async def _replan_with_llm(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await replan_node(
            state,
            stage_llm_map["replan"],
            runtime_context_service=runtime_context_service,
        )

    async def _summarize_with_llm(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await summarize_node(
            state,
            stage_llm_map["summary"],
            runtime_context_service=runtime_context_service,
        )

    async def _consolidate_memory(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await consolidate_memory_node(
            state,
            long_term_memory_repository=long_term_memory_repository,
            memory_consolidation_service=memory_consolidation_service,
        )

    graph = StateGraph(PlannerReActLangGraphState)
    graph.add_node("entry_router", _entry_router)
    graph.add_node("direct_answer", _direct_answer)
    graph.add_node("direct_wait", _direct_wait)
    graph.add_node("atomic_action", _atomic_action)
    graph.add_node("recall_memory_context", _recall_memory_context)
    graph.add_node("create_plan_or_reuse", _create_plan_with_llm)
    graph.add_node("guard_step_reuse", guard_step_reuse_node)
    graph.add_node("execute_step", _execute_step_with_llm)
    graph.add_node("wait_for_human", wait_for_human_node)
    graph.add_node("replan", _replan_with_llm)
    graph.add_node("summarize", _summarize_with_llm)
    graph.add_node("consolidate_memory", _consolidate_memory)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "entry_router")
    graph.add_conditional_edges(
        "entry_router",
        route_after_entry,
        {
            "direct_answer": "direct_answer",
            "direct_wait": "direct_wait",
            "atomic_action": "atomic_action",
            "create_plan_or_reuse": "create_plan_or_reuse",
            "recall_memory_context": "recall_memory_context",
        },
    )
    graph.add_edge("recall_memory_context", "create_plan_or_reuse")
    graph.add_edge("direct_answer", "consolidate_memory")
    graph.add_edge("direct_wait", "wait_for_human")
    graph.add_edge("atomic_action", "guard_step_reuse")
    graph.add_conditional_edges(
        "create_plan_or_reuse",
        route_after_plan,
        {
            "guard_step_reuse": "guard_step_reuse",
            "summarize": "summarize",
            "consolidate_memory": "consolidate_memory",
        },
    )
    graph.add_conditional_edges(
        "guard_step_reuse",
        route_after_guard,
        {
            "guard_step_reuse": "guard_step_reuse",
            "execute_step": "execute_step",
            "replan": "replan",
            "summarize": "summarize",
        },
    )
    graph.add_conditional_edges(
        "execute_step",
        route_after_execute,
        {
            "wait_for_human": "wait_for_human",
            "guard_step_reuse": "guard_step_reuse",
            "replan": "replan",
            "summarize": "summarize",
        },
    )
    graph.add_conditional_edges(
        "wait_for_human",
        route_after_wait,
        {
            "guard_step_reuse": "guard_step_reuse",
            "replan": "replan",
            "summarize": "summarize",
        },
    )
    graph.add_conditional_edges(
        "replan",
        route_after_replan,
        {
            "guard_step_reuse": "guard_step_reuse",
            "summarize": "summarize",
            "consolidate_memory": "consolidate_memory",
        },
    )
    graph.add_edge("summarize", "consolidate_memory")
    graph.add_edge("consolidate_memory", "finalize")
    graph.add_edge("finalize", END)
    log_runtime(
        logger,
        logging.INFO,
        "流程图编译完成",
        runtime_tool_count=len(list(runtime_tools or [])),
        stage_llm_count=len(stage_llm_map),
        max_tool_iterations=max_tool_iterations,
        has_checkpointer=checkpointer is not None,
        has_skill_runtime=skill_runtime is not None,
    )
    if checkpointer is None:
        return graph.compile()
    return graph.compile(checkpointer=checkpointer)
