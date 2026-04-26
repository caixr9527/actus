#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangGraph Skill 子图注册表。"""
from typing import Any, List, TypedDict

from pydantic import BaseModel, Field

from app.domain.external import LLM
from app.domain.services.prompts import EXECUTION_PROMPT
from app.domain.services.runtime.normalizers import normalize_execution_response
from app.domain.services.runtime.skill_graph_registry import (
    SkillDefinition,
    SkillGraphRegistry,
    SkillRuntimePolicy,
)
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    format_attachments_for_prompt,
    normalize_attachments,
    safe_parse_json,
)

try:
    from langgraph.graph import StateGraph, START, END

    _LANGGRAPH_SKILL_AVAILABLE = True
    _LANGGRAPH_SKILL_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - 依赖缺失时保护
    StateGraph = None
    START = "__start__"
    END = "__end__"
    _LANGGRAPH_SKILL_AVAILABLE = False
    _LANGGRAPH_SKILL_IMPORT_ERROR = e


class PlannerExecuteStepSkillInput(BaseModel):
    """Planner-ReAct 步骤执行 Skill 输入。"""

    session_id: str
    user_message: str
    step_description: str
    language: str = "zh"
    attachments: List[str] = Field(default_factory=list)


class PlannerExecuteStepSkillOutput(BaseModel):
    """Planner-ReAct 步骤执行 Skill 输出。"""

    success: bool = True
    result: str = ""
    attachments: List[str] = Field(default_factory=list)


class PlannerExecuteStepSkillState(TypedDict, total=False):
    session_id: str
    user_message: str
    step_description: str
    language: str
    attachments: List[str]
    success: bool
    result: str


# 包重构边界说明：此模块承接旧 `skill_subgraphs.py` 职责，统一收口到 `graphs/skills/registry.py`。
async def _execute_step_skill_node(
        state: PlannerExecuteStepSkillState,
        llm: LLM,
) -> PlannerExecuteStepSkillState:
    prompt = EXECUTION_PROMPT.format(
        message=state.get("user_message", ""),
        attachments=format_attachments_for_prompt(list(state.get("attachments") or [])),
        language=str(state.get("language") or "zh"),
        step=str(state.get("step_description") or ""),
    )
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))
    parsed_execution = normalize_execution_response(parsed)

    step_description = str(state.get("step_description") or "")
    return {
        **state,
        "success": bool(parsed_execution.get("success", True)),
        "result": str(parsed_execution.get("summary") or f"已完成步骤：{step_description}"),
        "attachments": normalize_attachments(parsed_execution.get("attachments")),
    }


def build_planner_execute_step_skill_graph(llm: LLM) -> Any:
    """构建 Planner-ReAct 步骤执行 Skill 子图。"""
    if not _LANGGRAPH_SKILL_AVAILABLE:
        raise RuntimeError(f"LangGraph 未安装，无法构建 Skill 子图: {_LANGGRAPH_SKILL_IMPORT_ERROR}")

    async def _execute_with_llm(state: PlannerExecuteStepSkillState) -> PlannerExecuteStepSkillState:
        return await _execute_step_skill_node(state=state, llm=llm)

    graph = StateGraph(PlannerExecuteStepSkillState)
    graph.add_node("execute_step_skill", _execute_with_llm)
    graph.add_edge(START, "execute_step_skill")
    graph.add_edge("execute_step_skill", END)
    return graph.compile()


def build_default_skill_graph_registry() -> SkillGraphRegistry:
    """构建 BE-LG-10 默认 Skill 注册表。"""
    registry = SkillGraphRegistry()
    registry.register(
        SkillDefinition(
            skill_id="planner_react.execute_step",
            version="1.0.0",
            description="Planner-ReAct 步骤执行子图（输入步骤描述，输出步骤结果）",
            input_schema=PlannerExecuteStepSkillInput,
            output_schema=PlannerExecuteStepSkillOutput,
            runtime_policy=SkillRuntimePolicy(
                capability_dependencies=(),
                tool_strategy="none",
                max_iterations=1,
            ),
            graph_factory=build_planner_execute_step_skill_graph,
        )
    )
    return registry
