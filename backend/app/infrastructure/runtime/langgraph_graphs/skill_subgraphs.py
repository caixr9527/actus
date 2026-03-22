#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/22
@Author : caixiaorong01@outlook.com
@File   : skill_subgraphs.py
"""
import json
import logging
from typing import Any, Dict, List, TypedDict

from pydantic import BaseModel, Field

from app.domain.external import LLM
from app.domain.services.prompts import EXECUTION_PROMPT
from app.domain.services.runtime.skill_graph_registry import (
    SkillDefinition,
    SkillGraphRegistry,
    SkillRuntimePolicy,
)

logger = logging.getLogger(__name__)

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


def _safe_parse_json(content: str | None) -> Dict[str, Any]:
    if not content:
        return {}
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        logger.warning("Skill子图解析JSON失败，使用回退逻辑")
        return {}


def _normalize_attachments(raw_attachments: Any) -> List[str]:
    if isinstance(raw_attachments, str):
        return [raw_attachments] if raw_attachments.strip() else []
    if isinstance(raw_attachments, list):
        return [str(item) for item in raw_attachments if str(item).strip()]
    return []


def _format_attachments_for_prompt(attachments: List[str]) -> str:
    if not attachments:
        return "无"
    return "\n".join(f"- {item}" for item in attachments)


async def _execute_step_skill_node(
        state: PlannerExecuteStepSkillState,
        llm: LLM,
) -> PlannerExecuteStepSkillState:
    prompt = EXECUTION_PROMPT.format(
        message=state.get("user_message", ""),
        attachments=_format_attachments_for_prompt(list(state.get("attachments") or [])),
        language=str(state.get("language") or "zh"),
        step=str(state.get("step_description") or ""),
    )
    llm_message = await llm.invoke(messages=[{"role": "user", "content": prompt}], tools=[])
    parsed = _safe_parse_json(llm_message.get("content"))

    step_description = str(state.get("step_description") or "")
    return {
        **state,
        "success": bool(parsed.get("success", True)),
        "result": str(parsed.get("result") or f"已完成步骤：{step_description}"),
        "attachments": _normalize_attachments(parsed.get("attachments")),
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
