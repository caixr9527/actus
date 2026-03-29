#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 节点实现。"""
import json
import logging
from typing import Any, Dict, List, Optional

from app.domain.external import LLM
from app.domain.models import (
    DoneEvent,
    ExecutionStatus,
    File,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
    TitleEvent,
    ToolEvent,
)
from app.domain.services.prompts import CREATE_PLAN_PROMPT, EXECUTION_PROMPT, UPDATE_PLAN_PROMPT, SYSTEM_PROMPT, \
    PLANNER_SYSTEM_PROMPT, SUMMARIZE_PROMPT
from app.domain.services.runtime import SkillGraphRuntime
from app.domain.services.runtime.langgraph_events import append_events
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.tools import BaseTool
from .live_events import emit_live_events
from .parsers import (
    build_fallback_plan_title,
    format_attachments_for_prompt,
    build_step_from_payload,
    merge_attachment_paths,
    normalize_attachments,
    safe_parse_json,
)
from .tools import execute_step_with_prompt

logger = logging.getLogger(__name__)

PLANNER_EXECUTE_STEP_SKILL_ID = "planner_react.execute_step"


async def _build_message(llm: LLM, user_message_prompt: str, input_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if llm.multimodal and input_parts is not None and len(input_parts) > 0:
        multiplexed_message = await llm.format_multiplexed_message(input_parts)
        user_content = [
            {"type": "text", "text": user_message_prompt},
            *multiplexed_message,
        ]
    else:
        user_content = [{"type": "text", "text": user_message_prompt}]
    return user_content


async def create_or_reuse_plan_node(state: PlannerReActLangGraphState, llm: LLM) -> PlannerReActLangGraphState:
    """创建计划或复用已有计划。"""
    plan = state.get("plan")
    if plan is not None and len(plan.steps) > 0 and not plan.done:
        next_step = plan.get_next_step()
        return {
            **state,
            "current_step_id": next_step.id if next_step is not None else None,
        }

    user_message = state.get("user_message", "").strip()

    input_parts = list(state.get("input_parts") or [])
    attachments = [part.get("sandbox_filepath") for part in input_parts]
    user_message_prompt = CREATE_PLAN_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
    )

    user_content = await _build_message(llm, user_message_prompt, input_parts)

    logger.info("planner 计划")
    llm_message = await llm.invoke(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))

    title = str(parsed.get("title") or build_fallback_plan_title(user_message))
    language = str(parsed.get("language") or "zh")
    goal = str(parsed.get("goal") or user_message)
    planner_message = str(parsed.get("message") or user_message or "已生成任务计划")
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or raw_steps is None or len(raw_steps) == 0:
        plan = Plan(
            title=title,
            goal=goal,
            language=language,
            message=planner_message,
            steps=[],
            status=ExecutionStatus.COMPLETED,
        )
        planner_events: List[Any] = [
            TitleEvent(title=title),
            MessageEvent(role="assistant", message=planner_message)
        ]
        await emit_live_events(*planner_events)
        return {
            **state,
            "plan": plan,
            "current_step_id": None,
            "final_message": planner_message,
            "graph_metadata": {
                **dict(state.get("graph_metadata") or {}),
            },
            "emitted_events": append_events(state.get("emitted_events"), *planner_events),
        }
    else:
        steps = [build_step_from_payload(item, index) for index, item in enumerate(raw_steps)]
        plan = Plan(
            title=title,
            goal=goal,
            language=language,
            message=planner_message,
            steps=steps,
            status=ExecutionStatus.PENDING,
        )
        next_step = plan.get_next_step()

        planner_events: List[Any] = [
            TitleEvent(title=title),
            PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.CREATED),
            MessageEvent(role="assistant", message=planner_message)
        ]
        await emit_live_events(*planner_events)

        return {
            **state,
            "plan": plan,
            "current_step_id": next_step.id if next_step is not None else None,
            "graph_metadata": {
                **dict(state.get("graph_metadata") or {}),
            },
            "emitted_events": append_events(state.get("emitted_events"), *planner_events),
        }


async def execute_step_node(
        state: PlannerReActLangGraphState,
        llm: LLM,
        skill_runtime: Optional[SkillGraphRuntime] = None,
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
) -> PlannerReActLangGraphState:
    """执行单个步骤，完成后交给 replan 节点更新后续计划。"""
    plan = state.get("plan")
    if plan is None:
        return state

    step = plan.get_next_step()
    if step is None:
        return state

    step.status = ExecutionStatus.RUNNING
    started_event = StepEvent(step=step.model_copy(deep=True), status=StepEventStatus.STARTED)
    await emit_live_events(started_event)

    user_message = str(state.get("user_message", ""))
    language = plan.language or "zh"
    input_parts = list(state.get("input_parts") or [])
    attachments = [part.get("sandbox_filepath") for part in input_parts]

    user_message_prompt = EXECUTION_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
        language=language,
        step=step.description,
    )
    user_content = await _build_message(llm, user_message_prompt, input_parts)

    llm_message: Optional[Dict[str, Any]] = None
    tool_events: List[ToolEvent] = []

    # 若运行时已注入工具能力，则优先走“提示词 + 工具循环”路径。
    if runtime_tools:
        llm_message, tool_events = await execute_step_with_prompt(
            llm=llm,
            step=step,
            runtime_tools=runtime_tools,
            max_tool_iterations=max_tool_iterations,
            on_tool_event=emit_live_events,
            user_content=user_content,
        )

    # 无工具能力时保持原 BE-LG-10 skill 路径。
    if llm_message is None and skill_runtime is not None:
        try:
            skill_result = await skill_runtime.execute_skill(
                skill_id=PLANNER_EXECUTE_STEP_SKILL_ID,
                payload={
                    "session_id": str(state.get("session_id") or ""),
                    "user_message": user_message,
                    "step_description": step.description,
                    "language": language,
                    "attachments": attachments,
                },
            )
            execution_payload = {
                "success": bool(getattr(skill_result, "success", True)),
                "result": str(getattr(skill_result, "result", "") or f"已完成步骤：{step.description}"),
                "attachments": normalize_attachments(getattr(skill_result, "attachments", [])),
            }
        except Exception as e:
            logger.warning("执行步骤 Skill 运行失败，回退默认执行链路: %s", e)

    step.success = bool(llm_message.get("success", True))
    step.result = str(llm_message.get("result"))
    model_attachment_paths = normalize_attachments(llm_message.get("attachments"))
    step.attachments = model_attachment_paths
    step.status = ExecutionStatus.COMPLETED if step.success else ExecutionStatus.FAILED

    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=step.status,
    )
    final_step_events: List[Any] = [completed_event]
    # if step.result:
    #     final_step_events.append(
    #         MessageEvent(
    #             role="assistant",
    #             message=step.result,
    #             attachments=[File(filepath=filepath) for filepath in step.attachments],
    #         )
    #     )

    await emit_live_events(*final_step_events)

    events: List[Any] = [started_event, *tool_events, *final_step_events]
    next_step = plan.get_next_step()
    graph_metadata = dict(state.get("graph_metadata") or {})
    return {
        **state,
        "plan": plan,
        "last_executed_step": step.model_copy(deep=True),
        "execution_count": int(state.get("execution_count", 0)) + 1,
        "current_step_id": next_step.id if next_step is not None else None,
        "graph_metadata": graph_metadata,
        "final_message": step.result or "",
        "emitted_events": append_events(state.get("emitted_events"), *events),
    }


async def replan_node(state: PlannerReActLangGraphState, llm: LLM) -> PlannerReActLangGraphState:
    """根据最新步骤执行结果更新后续未完成步骤。"""
    plan = state.get("plan")
    last_step = state.get("last_executed_step")
    if plan is None or last_step is None:
        return state

    prompt = UPDATE_PLAN_PROMPT.format(step=last_step.model_dump_json(), plan=plan.model_dump_json())
    logger.info("replan 计划")
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))

    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list):
        return state

    new_steps = [build_step_from_payload(item, index) for index, item in enumerate(raw_steps)]
    first_pending_index: Optional[int] = None
    for index, current_step in enumerate(plan.steps):
        if not current_step.done:
            first_pending_index = index
            break

    if first_pending_index is not None:
        updated_steps = plan.steps[:first_pending_index]
        # 对new_steps中对步骤进行修正
        if len(new_steps) > 0 and updated_steps[0].id == new_steps[0].id:
            logger.warning("修正步骤id")
            # 修正步骤id
            max_step = max(updated_steps, key=lambda step: int(step.id))
            for new_step in new_steps:
                new_step.id = str(int(max_step.id) + 1)
                max_step = new_step
        updated_steps.extend(new_steps)
        plan.steps = updated_steps

    next_step = plan.get_next_step()
    updated_event = PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.UPDATED)
    await emit_live_events(updated_event)
    return {
        **state,
        "plan": plan,
        "current_step_id": next_step.id if next_step is not None else None,
        "emitted_events": append_events(state.get("emitted_events"), updated_event),
    }


async def summarize_node(state: PlannerReActLangGraphState, llm: LLM) -> PlannerReActLangGraphState:
    """在所有步骤完成后汇总结果。"""
    plan = state.get("plan")
    if plan is None:
        return state
    plan_snapshot = plan.model_dump(mode="json") if plan is not None else {}
    final_message = str(state.get("final_message") or "")
    user_message = str(state.get("user_message") or "")
    execution_count = int(state.get("execution_count") or 0)
    summarize_prompt = SUMMARIZE_PROMPT.format(user_message=user_message,
                                               execution_count=execution_count,
                                               final_message=final_message,
                                               plan_snapshot=json.dumps(plan_snapshot, ensure_ascii=False))
    logger.info("总结计划")
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": summarize_prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))
    summary_message = str(parsed.get("message") or "")
    # 附件处理
    summary_attachment_paths = normalize_attachments(parsed.get("attachments"))
    summary_attachment_paths = [File(filepath=filepath) for filepath in
                                merge_attachment_paths(summary_attachment_paths)]

    final_events: List[Any] = [
        MessageEvent(role="assistant", message=summary_message, attachments=summary_attachment_paths)]

    plan.status = ExecutionStatus.COMPLETED
    final_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.COMPLETED))

    await emit_live_events(*final_events)
    return {
        **state,
        "plan": plan,
        "current_step_id": None,
        "final_message": summary_message,
        "emitted_events": append_events(state.get("emitted_events"), *final_events),
    }


async def finalize_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """结束节点，追加 done 事件。"""
    events = list(state.get("emitted_events") or [])
    if events and isinstance(events[-1], DoneEvent):
        return state

    done_event = DoneEvent()
    await emit_live_events(done_event)
    return {
        **state,
        "emitted_events": append_events(state.get("emitted_events"), done_event),
    }
