#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 图装配入口。"""

import logging
from typing import Any, List, Optional

from app.domain.external import LLM
from app.domain.services.runtime import SkillGraphRuntime
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph_graphs.skill_subgraphs import build_default_skill_graph_registry

from .nodes import (
    create_or_reuse_plan_node,
    execute_step_node,
    finalize_node,
    replan_node,
    summarize_node,
)
from .routing import route_after_plan, route_after_replan

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - 依赖缺失时的保护逻辑
    StateGraph = None
    START = "__start__"
    END = "__end__"
    InMemorySaver = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = e


def build_planner_react_langgraph_graph(
        llm: LLM,
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
) -> Any:
    """构建 LangGraph Planner-ReAct V1 图。"""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(f"LangGraph 未安装，无法构建图: {LANGGRAPH_IMPORT_ERROR}")

    skill_runtime: Optional[SkillGraphRuntime] = None
    try:
        skill_runtime = build_default_skill_graph_registry().create_runtime(llm=llm)
    except Exception as e:
        logger.warning("初始化默认 Skill 注册表失败，继续使用无 Skill 模式: %s", e)

    async def _create_plan_with_llm(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await create_or_reuse_plan_node(state, llm)

    async def _execute_step_with_llm(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await execute_step_node(
            state,
            llm,
            skill_runtime=skill_runtime,
            runtime_tools=runtime_tools,
            max_tool_iterations=max_tool_iterations,
        )

    async def _replan_with_llm(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await replan_node(state, llm)

    async def _summarize_with_llm(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        return await summarize_node(state, llm)

    graph = StateGraph(PlannerReActLangGraphState)
    graph.add_node("create_plan_or_reuse", _create_plan_with_llm)
    graph.add_node("execute_step", _execute_step_with_llm)
    graph.add_node("replan", _replan_with_llm)
    graph.add_node("summarize", _summarize_with_llm)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "create_plan_or_reuse")
    graph.add_conditional_edges(
        "create_plan_or_reuse",
        route_after_plan,
        {
            "execute_step": "execute_step",
            "summarize": "summarize",
            "finalize": "finalize",
        },
    )
    graph.add_edge("execute_step", "replan")
    graph.add_conditional_edges(
        "replan",
        route_after_replan,
        {
            "execute_step": "execute_step",
            "summarize": "summarize",
            "finalize": "finalize",
        },
    )
    graph.add_edge("summarize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=InMemorySaver())
