#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 节点实现。"""

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
    Step,
    StepEvent,
    StepEventStatus,
    TitleEvent,
    ToolEvent,
)
from app.domain.services.prompts import CREATE_PLAN_PROMPT, EXECUTION_PROMPT, UPDATE_PLAN_PROMPT
from app.domain.services.runtime import SkillGraphRuntime
from app.domain.services.runtime.langgraph_events import append_events
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.tools import BaseTool

from .live_events import emit_live_events
from .parsers import (
    build_fallback_plan_title,
    format_attachments_for_prompt,
    build_simple_greeting_reply,
    build_step_from_payload,
    build_summarize_prompt,
    collect_message_attachment_paths,
    collect_plan_attachment_paths,
    extract_write_file_paths_from_tool_events,
    get_last_assistant_message_event,
    is_simple_greeting_message,
    merge_attachment_paths,
    normalize_attachments,
    resolve_model_input_policy,
    safe_parse_json,
    should_accept_summary_message,
    should_emit_planner_message,
)
from .tools import build_execution_prompt, execute_step_with_prompt

logger = logging.getLogger(__name__)

PLANNER_EXECUTE_STEP_SKILL_ID = "planner_react.execute_step"


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
    if is_simple_greeting_message(user_message):
        greeting_reply = build_simple_greeting_reply(user_message=user_message)
        greeting_event = MessageEvent(role="assistant", message=greeting_reply)
        await emit_live_events(greeting_event)
        return {
            **state,
            "plan": None,
            "current_step_id": None,
            "final_message": greeting_reply,
            "emitted_events": append_events(state.get("emitted_events"), greeting_event),
        }

    input_parts = list(state.get("input_parts") or [])
    input_policy = resolve_model_input_policy(
        llm=llm,
        input_parts=input_parts,
    )
    prompt = CREATE_PLAN_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(input_policy["text_attachment_paths"]),
    )
    if bool(input_policy.get("inline_text_from_attachments")):
        prompt = (
            f"{prompt}\n\n"
            "注意：你已经拿到文本附件正文，视为存在有效附件；不要声称“未检测到附件”，也不要调用 read_file。"
        )
    planner_user_content: Any
    if len(input_policy["native_user_content_parts"]) > 0:
        planner_user_content = [
            {"type": "text", "text": prompt},
            *list(input_policy["native_user_content_parts"]),
        ]
    else:
        planner_user_content = prompt
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": planner_user_content}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))

    title = str(parsed.get("title") or build_fallback_plan_title(user_message))
    language = str(parsed.get("language") or "zh")
    goal = str(parsed.get("goal") or user_message)
    planner_message = str(parsed.get("message") or user_message or "已生成任务计划")
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) == 0:
        raw_steps = [user_message or "处理用户任务"]

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

    planner_events: List[Any] = [TitleEvent(title=title)]
    if should_emit_planner_message(
            user_message=user_message,
            planner_message=planner_message,
            steps=steps,
    ):
        planner_events.append(MessageEvent(role="assistant", message=planner_message))
    planner_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.CREATED))
    await emit_live_events(*planner_events)

    return {
        **state,
        "plan": plan,
        "current_step_id": next_step.id if next_step is not None else None,
        "graph_metadata": {
            **dict(state.get("graph_metadata") or {}),
            "input_policy": {
                "multimodal": input_policy["multimodal"],
                "supported": input_policy["supported"],
                "unsupported_parts": input_policy["unsupported_parts"],
            },
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
    input_policy = resolve_model_input_policy(
        llm=llm,
        input_parts=input_parts,
    )
    attachments = list(input_policy["text_attachment_paths"])
    native_user_content_parts = list(input_policy["native_user_content_parts"])
    has_inline_text_attachments = bool(
        input_policy.get("inline_text_from_attachments")
    ) or any(
        isinstance(part, dict) and str(part.get("type") or "").strip().lower() == "text"
        for part in native_user_content_parts
    )
    execution_prompt = build_execution_prompt(
        execution_prompt_template=EXECUTION_PROMPT,
        user_message=user_message,
        step_description=step.description,
        language=language,
        attachments=attachments,
    )
    if has_inline_text_attachments:
        execution_prompt = (
            f"{execution_prompt}\n\n"
            "注意：文本附件正文已在输入中提供，请优先直接基于输入内容完成任务，不要调用 read_file。"
        )

    execution_payload: Optional[Dict[str, Any]] = None
    tool_events: List[ToolEvent] = []

    # 若运行时已注入工具能力，则优先走“提示词 + 工具循环”路径。
    if runtime_tools:
        execution_payload, tool_events = await execute_step_with_prompt(
            llm=llm,
            execution_prompt=execution_prompt,
            step=step,
            runtime_tools=runtime_tools,
            max_tool_iterations=max_tool_iterations,
            on_tool_event=emit_live_events,
            extra_user_content_parts=native_user_content_parts,
            disallowed_function_names=["read_file"] if has_inline_text_attachments else [],
        )

    # 无工具能力时保持原 BE-LG-10 skill 路径。
    if execution_payload is None and skill_runtime is not None:
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

    if execution_payload is None:
        execution_payload, tool_events = await execute_step_with_prompt(
            llm=llm,
            execution_prompt=execution_prompt,
            step=step,
            on_tool_event=emit_live_events,
            extra_user_content_parts=native_user_content_parts,
            disallowed_function_names=["read_file"] if has_inline_text_attachments else [],
        )

    step.success = bool(execution_payload.get("success", True))
    step.result = str(execution_payload.get("result") or f"已完成步骤：{step.description}")
    model_attachment_paths = normalize_attachments(execution_payload.get("attachments"))
    inferred_attachment_paths = extract_write_file_paths_from_tool_events(tool_events)
    # 执行结果未显式给出附件时，兜底从 write_file 工具结果提取产物路径。
    step.attachments = merge_attachment_paths(model_attachment_paths, inferred_attachment_paths)
    step.status = ExecutionStatus.COMPLETED if step.success else ExecutionStatus.FAILED

    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=StepEventStatus.COMPLETED if step.success else StepEventStatus.FAILED,
    )
    final_step_events: List[Any] = [completed_event]
    if step.result:
        final_step_events.append(
            MessageEvent(
                role="assistant",
                message=step.result,
                attachments=[File(filepath=filepath) for filepath in step.attachments],
            )
        )

    await emit_live_events(*final_step_events)

    events: List[Any] = [started_event, *tool_events, *final_step_events]
    next_step = plan.get_next_step()
    graph_metadata = dict(state.get("graph_metadata") or {})
    graph_metadata["input_policy"] = {
        "multimodal": input_policy["multimodal"],
        "supported": input_policy["supported"],
        "unsupported_parts": input_policy["unsupported_parts"],
    }
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
    prompt = build_summarize_prompt(state)
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = safe_parse_json(llm_message.get("content"))
    fallback_message = str(state.get("final_message") or "任务已完成")
    candidate_summary_message = str(parsed.get("message") or "")
    accepted_candidate = should_accept_summary_message(
        state=state,
        candidate_message=candidate_summary_message,
        fallback_message=fallback_message,
    )
    summary_message = candidate_summary_message if accepted_candidate else fallback_message
    summary_attachment_paths = normalize_attachments(parsed.get("attachments")) if accepted_candidate else []

    # summarize 未返回附件时，兜底使用步骤执行阶段已产出的附件。
    attachment_paths = merge_attachment_paths(summary_attachment_paths, collect_plan_attachment_paths(plan))
    attachments = [File(filepath=filepath) for filepath in attachment_paths]

    previous_assistant_event = get_last_assistant_message_event(list(state.get("emitted_events") or []))
    previous_assistant_message = str(previous_assistant_event.message or "").strip() if previous_assistant_event else ""
    previous_assistant_attachment_paths = collect_message_attachment_paths(previous_assistant_event)

    final_events: List[Any] = []
    # summarize 与上一条 assistant 在“文本 + 附件”都一致时，跳过重复 message。
    if not (
            summary_message.strip()
            and previous_assistant_message
            and summary_message.strip() == previous_assistant_message
            and attachment_paths == previous_assistant_attachment_paths
    ):
        final_events.append(MessageEvent(role="assistant", message=summary_message, attachments=attachments))

    if plan is not None:
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
