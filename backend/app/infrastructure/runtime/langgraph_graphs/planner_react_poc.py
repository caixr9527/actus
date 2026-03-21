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
from typing import Any, Dict, List, Literal, Optional

from app.domain.external import LLM
from app.domain.models import (
    File,
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
from app.domain.services.prompts import CREATE_PLAN_PROMPT, EXECUTION_PROMPT, SUMMARIZE_PROMPT, UPDATE_PLAN_PROMPT
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


def _normalize_attachments(raw_attachments: Any) -> List[str]:
    if isinstance(raw_attachments, str):
        return [raw_attachments]
    if isinstance(raw_attachments, list):
        return [str(item) for item in raw_attachments if str(item).strip()]
    return []


def _build_step_from_payload(payload: Any, fallback_index: int) -> Step:
    if isinstance(payload, dict):
        step_id = str(payload.get("id") or str(uuid.uuid4()))
        description = str(payload.get("description") or f"步骤{fallback_index + 1}")
        return Step(
            id=step_id,
            description=description,
            status=ExecutionStatus.PENDING,
        )

    return Step(
        id=str(uuid.uuid4()),
        description=str(payload).strip() or f"步骤{fallback_index + 1}",
        status=ExecutionStatus.PENDING,
    )


def _route_after_plan(state: PlannerReActPOCState) -> Literal["execute_step", "summarize"]:
    plan = state.get("plan")
    if plan is None:
        return "summarize"
    if state.get("execution_count", 0) >= state.get("max_execution_steps", 20):
        logger.warning("LangGraph V1 执行次数达到上限，提前进入总结阶段")
        return "summarize"
    return "execute_step" if plan.get_next_step() is not None else "summarize"


def _route_after_replan(state: PlannerReActPOCState) -> Literal["execute_step", "summarize"]:
    return _route_after_plan(state)


async def _create_or_reuse_plan_node(state: PlannerReActPOCState, llm: LLM) -> PlannerReActPOCState:
    """创建计划或复用已恢复计划。"""
    plan = state.get("plan")
    if plan is not None and len(plan.steps) > 0 and not plan.done:
        next_step = plan.get_next_step()
        return {
            **state,
            "current_step_id": next_step.id if next_step is not None else None,
        }

    user_message = state.get("user_message", "").strip()
    prompt = CREATE_PLAN_PROMPT.format(
        message=user_message,
        attachments="",
    )
    llm_message = await llm.invoke(messages=[{"role": "user", "content": prompt}], tools=[])
    parsed = _safe_parse_json(llm_message.get("content"))

    title = str(parsed.get("title") or "LangGraph 任务")
    language = str(parsed.get("language") or "zh")
    goal = str(parsed.get("goal") or user_message)
    planner_message = str(parsed.get("message") or user_message or "已生成任务计划")
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) == 0:
        raw_steps = [user_message or "处理用户任务"]

    steps = [_build_step_from_payload(item, index) for index, item in enumerate(raw_steps)]
    plan = Plan(
        title=title,
        goal=goal,
        language=language,
        message=planner_message,
        steps=steps,
        status=ExecutionStatus.PENDING,
    )
    next_step = plan.get_next_step()
    return {
        **state,
        "plan": plan,
        "current_step_id": next_step.id if next_step is not None else None,
        "emitted_events": append_events(
            state.get("emitted_events"),
            TitleEvent(title=title),
            MessageEvent(role="assistant", message=planner_message),
            PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.CREATED),
        ),
    }


async def _execute_step_node(state: PlannerReActPOCState, llm: LLM) -> PlannerReActPOCState:
    """执行单个步骤，完成后交给 replan 节点更新后续计划。"""
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

    prompt = EXECUTION_PROMPT.format(
        message=state.get("user_message", ""),
        attachments="",
        language=plan.language or "zh",
        step=step.description,
    )
    llm_message = await llm.invoke(messages=[{"role": "user", "content": prompt}], tools=[])
    parsed = _safe_parse_json(llm_message.get("content"))
    step.success = bool(parsed.get("success", True))
    step.result = str(parsed.get("result") or f"已完成步骤：{step.description}")
    step.attachments = _normalize_attachments(parsed.get("attachments"))
    step.status = ExecutionStatus.COMPLETED if step.success else ExecutionStatus.FAILED

    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=StepEventStatus.COMPLETED if step.success else StepEventStatus.FAILED,
    )
    events = [started_event, completed_event]
    if step.result:
        events.append(MessageEvent(role="assistant", message=step.result))
    next_step = plan.get_next_step()

    return {
        **state,
        "plan": plan,
        "last_executed_step": step.model_copy(deep=True),
        "execution_count": int(state.get("execution_count", 0)) + 1,
        "current_step_id": next_step.id if next_step is not None else None,
        "final_message": step.result or "",
        "emitted_events": append_events(
            state.get("emitted_events"),
            *events,
        ),
    }


async def _replan_node(state: PlannerReActPOCState, llm: LLM) -> PlannerReActPOCState:
    """根据最新步骤执行结果更新后续未完成步骤。"""
    plan = state.get("plan")
    last_step = state.get("last_executed_step")
    if plan is None or last_step is None:
        return state

    prompt = UPDATE_PLAN_PROMPT.format(
        step=last_step.model_dump_json(),
        plan=plan.model_dump_json(),
    )
    llm_message = await llm.invoke(messages=[{"role": "user", "content": prompt}], tools=[])
    parsed = _safe_parse_json(llm_message.get("content"))

    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list):
        return state

    new_steps = [_build_step_from_payload(item, index) for index, item in enumerate(raw_steps)]
    first_pending_index: Optional[int] = None
    for index, current_step in enumerate(plan.steps):
        if not current_step.done:
            first_pending_index = index
            break

    if first_pending_index is not None:
        updated_steps = plan.steps[:first_pending_index]
        updated_steps.extend(new_steps)
        plan.steps = updated_steps

    next_step = plan.get_next_step()
    return {
        **state,
        "plan": plan,
        "current_step_id": next_step.id if next_step is not None else None,
        "emitted_events": append_events(
            state.get("emitted_events"),
            PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.UPDATED),
        ),
    }


async def _summarize_node(state: PlannerReActPOCState, llm: LLM) -> PlannerReActPOCState:
    """在所有步骤完成后汇总结果。"""
    plan = state.get("plan")
    prompt = SUMMARIZE_PROMPT
    llm_message = await llm.invoke(messages=[{"role": "user", "content": prompt}], tools=[])
    parsed = _safe_parse_json(llm_message.get("content"))
    summary_message = str(parsed.get("message") or state.get("final_message") or "任务已完成")
    attachment_paths = _normalize_attachments(parsed.get("attachments"))
    attachments = [File(filepath=filepath) for filepath in attachment_paths]

    final_events: List[Any] = [
        MessageEvent(role="assistant", message=summary_message, attachments=attachments),
    ]
    if plan is not None:
        plan.status = ExecutionStatus.COMPLETED
        final_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.COMPLETED))

    return {
        **state,
        "plan": plan,
        "current_step_id": None,
        "final_message": summary_message,
        "emitted_events": append_events(state.get("emitted_events"), *final_events),
    }


async def _finalize_node(state: PlannerReActPOCState) -> PlannerReActPOCState:
    """结束节点，追加 done 事件。"""
    events = list(state.get("emitted_events") or [])
    if events and isinstance(events[-1], DoneEvent):
        return state

    return {
        **state,
        "emitted_events": append_events(state.get("emitted_events"), DoneEvent()),
    }


def build_planner_react_poc_graph(llm: LLM) -> Any:
    """构建 LangGraph Planner-ReAct V1 图（沿用 POC 编译入口）。"""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(f"LangGraph 未安装，无法构建 POC 图: {LANGGRAPH_IMPORT_ERROR}")

    # 显式 async wrapper，避免 lambda 返回 coroutine 导致节点返回值非法。
    async def _create_plan_with_llm(state: PlannerReActPOCState) -> PlannerReActPOCState:
        return await _create_or_reuse_plan_node(state, llm)

    async def _execute_step_with_llm(state: PlannerReActPOCState) -> PlannerReActPOCState:
        return await _execute_step_node(state, llm)

    async def _replan_with_llm(state: PlannerReActPOCState) -> PlannerReActPOCState:
        return await _replan_node(state, llm)

    async def _summarize_with_llm(state: PlannerReActPOCState) -> PlannerReActPOCState:
        return await _summarize_node(state, llm)

    graph = StateGraph(PlannerReActPOCState)
    graph.add_node("create_plan_or_reuse", _create_plan_with_llm)
    graph.add_node("execute_step", _execute_step_with_llm)
    graph.add_node("replan", _replan_with_llm)
    graph.add_node("summarize", _summarize_with_llm)
    graph.add_node("finalize", _finalize_node)
    graph.add_edge(START, "create_plan_or_reuse")
    graph.add_conditional_edges(
        "create_plan_or_reuse",
        _route_after_plan,
        {
            "execute_step": "execute_step",
            "summarize": "summarize",
        },
    )
    graph.add_edge("execute_step", "replan")
    graph.add_conditional_edges(
        "replan",
        _route_after_replan,
        {
            "execute_step": "execute_step",
            "summarize": "summarize",
        },
    )
    graph.add_edge("summarize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=InMemorySaver())
