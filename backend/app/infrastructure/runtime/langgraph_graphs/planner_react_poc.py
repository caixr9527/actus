#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : planner_react_poc.py
"""
import json
import logging
import uuid
from typing import Any, Dict

from app.domain.external import LLM
from app.domain.models import (
    Plan,
    Step,
    ExecutionStatus,
    TitleEvent,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
    MessageEvent,
    DoneEvent,
)
from app.domain.services.runtime.langgraph_events import append_events
from app.domain.services.runtime.langgraph_state import PlannerReActPOCState

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - 依赖缺失时的保护逻辑
    StateGraph = None
    START = "__start__"
    END = "__end__"
    InMemorySaver = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = e


def _safe_parse_json(content: str | None) -> Dict[str, Any]:
    if not content:
        return {}
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        logger.warning("LangGraph POC 解析JSON失败，使用回退逻辑")
        return {}


async def _create_plan_node(state: PlannerReActPOCState, llm: LLM) -> PlannerReActPOCState:
    user_message = state.get("user_message", "").strip()
    prompt = (
        "请将用户任务拆成最小POC计划，并返回JSON。"
        "格式：{\"title\": \"任务标题\", \"steps\": [\"步骤1\", \"步骤2\"]}。"
        "如果任务无法拆分，也至少返回一个步骤。"
        f"\n用户任务：{user_message}"
    )
    llm_message = await llm.invoke(messages=[{"role": "user", "content": prompt}], tools=[])
    parsed = _safe_parse_json(llm_message.get("content"))
    title = str(parsed.get("title") or "LangGraph POC 任务")
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) == 0:
        raw_steps = [user_message or "处理用户任务"]

    steps = [
        Step(
            id=str(uuid.uuid4()),
            description=str(item).strip() or "执行任务步骤",
            status=ExecutionStatus.PENDING,
        )
        for item in raw_steps
    ]
    plan = Plan(
        title=title,
        goal=user_message,
        language="zh",
        message=user_message,
        steps=steps,
        status=ExecutionStatus.PENDING,
    )
    return {
        **state,
        "plan": plan,
        "emitted_events": append_events(
            state.get("emitted_events"),
            TitleEvent(title=title),
            PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.CREATED),
        ),
    }


async def _execute_step_node(state: PlannerReActPOCState, llm: LLM) -> PlannerReActPOCState:
    plan = state.get("plan")
    if plan is None:
        return state

    step = plan.get_next_step()
    if step is None:
        return state

    step.status = ExecutionStatus.RUNNING
    started_event = StepEvent(
        step=step.model_copy(deep=True),
        status=StepEventStatus.STARTED,
    )

    prompt = (
        "请执行以下任务步骤，并返回JSON。"
        "格式：{\"success\": true, \"result\": \"执行结果\"}"
        f"\n任务步骤：{step.description}"
        f"\n用户原始需求：{state.get('user_message', '')}"
    )
    llm_message = await llm.invoke(messages=[{"role": "user", "content": prompt}], tools=[])
    parsed = _safe_parse_json(llm_message.get("content"))
    step.success = bool(parsed.get("success", True))
    step.result = str(parsed.get("result") or f"已完成步骤：{step.description}")
    step.status = ExecutionStatus.COMPLETED if step.success else ExecutionStatus.FAILED

    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=StepEventStatus.COMPLETED if step.success else StepEventStatus.FAILED,
    )
    assistant_message_event = MessageEvent(role="assistant", message=step.result or "")

    return {
        **state,
        "plan": plan,
        "final_message": step.result or "",
        "emitted_events": append_events(
            state.get("emitted_events"),
            started_event,
            completed_event,
            assistant_message_event,
        ),
    }


async def _finalize_node(state: PlannerReActPOCState) -> PlannerReActPOCState:
    plan = state.get("plan")
    if plan is not None:
        plan.status = ExecutionStatus.COMPLETED

    final_events = []
    if plan is not None:
        final_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.COMPLETED))
    final_events.append(DoneEvent())

    return {
        **state,
        "plan": plan,
        "emitted_events": append_events(state.get("emitted_events"), *final_events),
    }


def build_planner_react_poc_graph(llm: LLM) -> Any:
    """构建 LangGraph POC 图。"""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(f"LangGraph 未安装，无法构建 POC 图: {LANGGRAPH_IMPORT_ERROR}")

    graph = StateGraph(PlannerReActPOCState)
    graph.add_node("create_plan", lambda state: _create_plan_node(state, llm))
    graph.add_node("execute_step", lambda state: _execute_step_node(state, llm))
    graph.add_node("finalize", _finalize_node)
    graph.add_edge(START, "create_plan")
    graph.add_edge("create_plan", "execute_step")
    graph.add_edge("execute_step", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=InMemorySaver())
