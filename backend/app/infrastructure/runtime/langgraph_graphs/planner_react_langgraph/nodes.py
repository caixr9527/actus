#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 节点实现。"""
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.domain.external import LLM
from app.domain.models import (
    DoneEvent,
    ExecutionStatus,
    File,
    LongTermMemory,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
    TitleEvent,
    ToolEvent,
)
from app.domain.repositories import LongTermMemoryRepository
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
MESSAGE_WINDOW_MAX_ITEMS = 100


def _ensure_working_memory(state: PlannerReActLangGraphState) -> Dict[str, Any]:
    working_memory = dict(state.get("working_memory") or {})
    working_memory.setdefault("goal", "")
    working_memory.setdefault("constraints", [])
    working_memory.setdefault("decisions", [])
    working_memory.setdefault("open_questions", [])
    working_memory.setdefault("user_preferences", {})
    working_memory.setdefault("facts_in_session", [])
    return working_memory


def _append_unique_text_item(items: List[Any], value: str) -> List[str]:
    normalized_items = [str(item).strip() for item in items if str(item).strip()]
    normalized_value = str(value).strip()
    if normalized_value and normalized_value not in normalized_items:
        normalized_items.append(normalized_value)
    return normalized_items


def _build_memory_context_version(state: PlannerReActLangGraphState) -> str:
    return ":".join(
        [
            str(state.get("thread_id") or ""),
            str(int(state.get("execution_count") or 0)),
            str(len(state.get("message_window") or [])),
            str(len(state.get("retrieved_memories") or [])),
        ]
    )


def _build_memory_query(state: PlannerReActLangGraphState) -> str:
    working_memory = _ensure_working_memory(state)
    parts = [
        str(state.get("user_message") or "").strip(),
        str(state.get("conversation_summary") or "").strip(),
        str(working_memory.get("goal") or "").strip(),
    ]
    return " | ".join([item for item in parts if item][:3])


def _build_memory_namespace_prefixes(state: PlannerReActLangGraphState) -> List[str]:
    prefixes: List[str] = []
    user_id = str(state.get("user_id") or "").strip()
    session_id = str(state.get("session_id") or "").strip()
    if user_id:
        prefixes.append(f"user/{user_id}/")
    if session_id:
        prefixes.append(f"session/{session_id}/")
    prefixes.append("agent/planner_react/")
    return prefixes


def _build_memory_dedupe_key(*, namespace: str, memory_type: str, content: Dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "namespace": namespace,
            "memory_type": memory_type,
            "content": content,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_memory_fact_items(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    normalized_items: List[str] = []
    for item in raw:
        normalized_text = str(item or "").strip()
        if normalized_text and normalized_text not in normalized_items:
            normalized_items.append(normalized_text)
    return normalized_items


def _normalize_memory_preferences(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    normalized_preferences: Dict[str, Any] = {}
    for key, value in raw.items():
        normalized_key = str(key or "").strip()
        if not normalized_key:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized_preferences[normalized_key] = value
            continue
        if value is None:
            continue
        normalized_value = str(value).strip()
        if normalized_value:
            normalized_preferences[normalized_key] = normalized_value
    return normalized_preferences


def _build_outcome_fact_text(
        state: PlannerReActLangGraphState,
        summary_message: str,
) -> str:
    normalized_summary = str(summary_message or "").strip()
    if normalized_summary:
        return normalized_summary

    working_memory = _ensure_working_memory(state)
    decisions = [str(item or "").strip() for item in list(working_memory.get("decisions") or []) if str(item or "").strip()]
    if len(decisions) > 0:
        return decisions[-1]

    last_step = state.get("last_executed_step")
    last_step_result = str(getattr(last_step, "result", "") or "").strip()
    return last_step_result


def _build_outcome_memory_candidate(
        state: PlannerReActLangGraphState,
        summary_message: str,
) -> Optional[Dict[str, Any]]:
    outcome_text = _build_outcome_fact_text(state, summary_message)
    if not outcome_text:
        return None

    session_id = str(state.get("session_id") or "").strip()
    run_id = str(state.get("run_id") or "").strip()
    thread_id = str(state.get("thread_id") or "").strip()
    namespace = f"session/{session_id}/fact"
    content = {
        "text": outcome_text[:2000],
        "source_kind": "task_outcome",
    }
    return {
        "namespace": namespace,
        "memory_type": "fact",
        "summary": outcome_text[:120],
        "content": content,
        "tags": ["task_outcome"],
        "source": {
            "session_id": session_id,
            "run_id": run_id,
            "thread_id": thread_id,
            "stage": "summarize",
        },
        "confidence": 0.5,
        "dedupe_key": _build_memory_dedupe_key(
            namespace=namespace,
            memory_type="fact",
            content=content,
        ),
    }


def _merge_memory_candidates(
        current_candidates: List[Dict[str, Any]],
        new_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in [*current_candidates, *new_candidates]:
        if not isinstance(item, dict):
            continue
        dedupe_key = str(item.get("dedupe_key") or item.get("id") or "").strip()
        if not dedupe_key:
            dedupe_key = hashlib.sha1(
                json.dumps(item, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        merged.append(item)
    return merged


def _build_memory_candidates(state: PlannerReActLangGraphState) -> List[Dict[str, Any]]:
    working_memory = _ensure_working_memory(state)
    user_id = str(state.get("user_id") or "").strip()
    session_id = str(state.get("session_id") or "").strip()
    run_id = str(state.get("run_id") or "").strip()
    thread_id = str(state.get("thread_id") or "").strip()
    candidates: List[Dict[str, Any]] = []

    user_preferences = dict(working_memory.get("user_preferences") or {})
    if user_preferences:
        namespace = f"user/{user_id}/profile" if user_id else f"session/{session_id}/profile"
        candidates.append(
            {
                "namespace": namespace,
                "memory_type": "profile",
                "summary": "用户偏好",
                "content": user_preferences,
                "tags": list(user_preferences.keys())[:5],
                "source": {
                    "session_id": session_id,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "stage": "summarize",
                },
                "confidence": 0.8,
                "dedupe_key": _build_memory_dedupe_key(
                    namespace=namespace,
                    memory_type="profile",
                    content=user_preferences,
                ),
            }
        )

    for fact in list(working_memory.get("facts_in_session") or []):
        normalized_fact = str(fact or "").strip()
        if not normalized_fact:
            continue
        namespace = f"session/{session_id}/fact"
        content = {"text": normalized_fact}
        candidates.append(
            {
                "namespace": namespace,
                "memory_type": "fact",
                "summary": normalized_fact[:120],
                "content": content,
                "tags": ["fact"],
                "source": {
                    "session_id": session_id,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "stage": "summarize",
                },
                "confidence": 0.6,
                "dedupe_key": _build_memory_dedupe_key(
                    namespace=namespace,
                    memory_type="fact",
                    content=content,
                ),
            }
        )

    return candidates


def _build_model_memory_candidates(
        state: PlannerReActLangGraphState,
        raw_candidates: Any,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_candidates, list):
        return []

    session_id = str(state.get("session_id") or "").strip()
    run_id = str(state.get("run_id") or "").strip()
    thread_id = str(state.get("thread_id") or "").strip()
    user_id = str(state.get("user_id") or "").strip()
    normalized_candidates: List[Dict[str, Any]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        memory_type = str(item.get("memory_type") or "fact").strip().lower()
        if memory_type not in {"profile", "fact", "instruction"}:
            continue
        summary = str(item.get("summary") or "").strip()
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        if not summary and not content:
            continue
        namespace = str(item.get("namespace") or "").strip()
        if not namespace:
            if memory_type == "profile":
                namespace = f"user/{user_id}/profile" if user_id else f"session/{session_id}/profile"
            elif memory_type == "instruction":
                namespace = "agent/planner_react/instruction"
            else:
                namespace = f"session/{session_id}/fact"
        tags = [str(tag).strip() for tag in list(item.get("tags") or []) if str(tag).strip()]
        confidence_raw = item.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else 0.6
        except Exception:
            confidence = 0.6
        normalized_candidates.append(
            {
                "namespace": namespace,
                "memory_type": memory_type,
                "summary": summary[:120],
                "content": content,
                "tags": tags[:8],
                "source": {
                    "session_id": session_id,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "stage": "summarize",
                    "source_type": "llm_extract",
                },
                "confidence": max(0.0, min(confidence, 1.0)),
                "dedupe_key": _build_memory_dedupe_key(
                    namespace=namespace,
                    memory_type=memory_type,
                    content=content or {"summary": summary[:120]},
                ),
            }
        )
    return normalized_candidates


def _append_message_window_entry(
        message_window: List[Dict[str, Any]],
        *,
        role: str,
        message: str,
        attachments: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    # 标准化消息内容和附件路径，去除首尾空白字符
    normalized_message = str(message or "").strip()
    normalized_attachments = [str(item).strip() for item in list(attachments or []) if str(item).strip()]

    # 如果消息和附件均为空，则直接返回原始消息窗口，不做任何修改
    if not normalized_message and len(normalized_attachments) == 0:
        return list(message_window)

    # 构建新的消息条目
    next_entry = {
        "role": role,
        "message": message,
        "attachment_paths": normalized_attachments,
    }

    # 创建消息窗口的副本以避免直接修改原列表
    updated_window = list(message_window)

    # 检查是否与最后一条消息完全重复（角色、内容、附件均一致），若是则避免重复添加
    if updated_window:
        latest_entry = dict(updated_window[-1])
        if (
                str(latest_entry.get("role") or "") == role
                and str(latest_entry.get("message") or "") == str(message or "")
                and list(latest_entry.get("attachment_paths") or []) == normalized_attachments
        ):
            return updated_window

    # 将新条目添加到消息窗口末尾
    updated_window.append(next_entry)
    return updated_window


def _compact_message_window(message_window: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(message_window) <= MESSAGE_WINDOW_MAX_ITEMS:
        return list(message_window)
    return list(message_window[-MESSAGE_WINDOW_MAX_ITEMS:])


def _build_conversation_summary(state: PlannerReActLangGraphState) -> str:
    # 获取之前的对话摘要
    previous_summary = str(state.get("conversation_summary") or "").strip()
    # 确保工作记忆存在并初始化默认值
    working_memory = _ensure_working_memory(state)
    # 获取当前计划对象
    plan = state.get("plan")
    # 获取步骤状态列表
    step_states = list(state.get("step_states") or [])
    # 统计已完成的步骤数量
    completed_steps = sum(1 for item in step_states if str(item.get("status") or "") == ExecutionStatus.COMPLETED.value)
    # 计算总步骤数：优先使用 step_states 长度，若为空则使用 plan.steps 长度
    total_steps = len(step_states) if len(step_states) > 0 else len(getattr(plan, "steps", []) or [])
    parts: List[str] = []
    # 如果存在之前的摘要，则添加到部分列表中
    if previous_summary:
        parts.append(previous_summary)

    # 构建目标字符串：优先从工作记忆获取，其次从 plan 获取，最后从用户消息获取
    goal = str(working_memory.get("goal") or getattr(plan, "goal", "") or state.get("user_message") or "").strip()
    if goal:
        parts.append(f"目标:{goal}")

    # 如果总步骤数大于 0，添加进度信息
    if total_steps > 0:
        parts.append(f"进度:{completed_steps}/{total_steps}")

    # 获取最终消息并截断至 120 字符
    final_message = str(state.get("final_message") or "").strip()
    if final_message:
        parts.append(f"结果:{final_message[:120]}")

    # 返回最近 3 个部分，用 " | " 连接
    return " | ".join(parts[-3:])


def _extract_profile_preferences(retrieved_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    preferences: Dict[str, Any] = {}
    for item in retrieved_memories:
        if not isinstance(item, dict):
            continue
        if str(item.get("memory_type") or "").strip().lower() != "profile":
            continue
        content = item.get("content")
        if isinstance(content, dict):
            preferences.update(content)
    return preferences


async def _build_message(llm: LLM, user_message_prompt: str, input_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if getattr(llm, "multimodal", False) and input_parts is not None and len(input_parts) > 0:
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
        working_memory = _ensure_working_memory(state)
        if not str(working_memory.get("goal") or "").strip():
            working_memory["goal"] = str(plan.goal or state.get("user_message") or "")
        planner_local_memory = dict(state.get("planner_local_memory") or {})
        planner_local_memory["plan_brief"] = str(plan.message or plan.title or "")
        return {
            **state,
            "working_memory": working_memory,
            "planner_local_memory": planner_local_memory,
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
    working_memory = _ensure_working_memory(state)
    working_memory["goal"] = goal
    planner_local_memory = dict(state.get("planner_local_memory") or {})
    planner_local_memory["plan_brief"] = planner_message or title
    planner_local_memory["plan_assumptions"] = _append_unique_text_item(
        list(planner_local_memory.get("plan_assumptions") or []),
        str(parsed.get("assumption") or ""),
    )
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
            "working_memory": working_memory,
            "planner_local_memory": planner_local_memory,
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
            "working_memory": working_memory,
            "planner_local_memory": planner_local_memory,
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
    working_memory = _ensure_working_memory(state)
    working_memory["decisions"] = _append_unique_text_item(
        list(working_memory.get("decisions") or []),
        str(step.result or ""),
    )
    step_local_memory = dict(state.get("step_local_memory") or {})
    step_local_memory["current_step_id"] = str(step.id)
    step_local_memory["observation_summary"] = str(step.result or "")
    step_local_memory["pending_findings"] = list(step.attachments or [])
    return {
        **state,
        "plan": plan,
        "last_executed_step": step.model_copy(deep=True),
        "execution_count": int(state.get("execution_count", 0)) + 1,
        "current_step_id": next_step.id if next_step is not None else None,
        "working_memory": working_memory,
        "step_local_memory": step_local_memory,
        "graph_metadata": graph_metadata,
        "final_message": step.result or "",
        "emitted_events": append_events(state.get("emitted_events"), *events),
    }


async def recall_memory_context_node(
        state: PlannerReActLangGraphState,
        long_term_memory_repository: Optional[LongTermMemoryRepository] = None,
) -> PlannerReActLangGraphState:
    """统一整理线程级短期记忆，为后续 planner/react 节点提供稳定输入。"""
    plan = state.get("plan")
    working_memory = _ensure_working_memory(state)
    if not str(working_memory.get("goal") or "").strip():
        working_memory["goal"] = str(getattr(plan, "goal", "") or state.get("user_message") or "")

    retrieved_memories = list(state.get("retrieved_memories") or [])
    if long_term_memory_repository is not None:
        try:
            recalled_memories = await long_term_memory_repository.search(
                namespace_prefixes=_build_memory_namespace_prefixes(state),
                query=_build_memory_query(state),
                limit=8,
            )
            retrieved_memories = [memory.model_dump(mode="json") for memory in recalled_memories]
        except Exception as e:
            logger.warning("长期记忆召回失败，回退已有线程态记忆快照: %s", e)
    if not dict(working_memory.get("user_preferences") or {}):
        working_memory["user_preferences"] = _extract_profile_preferences(retrieved_memories)

    graph_metadata = dict(state.get("graph_metadata") or {})
    graph_metadata["memory_recall_prepared_at"] = datetime.now().isoformat()
    graph_metadata["memory_recall_count"] = len(retrieved_memories)
    graph_metadata["memory_recall_namespaces"] = _build_memory_namespace_prefixes(state)

    return {
        **state,
        "working_memory": working_memory,
        "retrieved_memories": retrieved_memories,
        "planner_local_memory": dict(state.get("planner_local_memory") or {}),
        "step_local_memory": dict(state.get("step_local_memory") or {}),
        "summary_local_memory": dict(state.get("summary_local_memory") or {}),
        "memory_context_version": _build_memory_context_version(state),
        "graph_metadata": graph_metadata,
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
    planner_local_memory = dict(state.get("planner_local_memory") or {})
    planner_local_memory["replan_rationale"] = str(last_step.result or "")
    return {
        **state,
        "plan": plan,
        "planner_local_memory": planner_local_memory,
        "step_local_memory": {},
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
    extracted_facts = _normalize_memory_fact_items(parsed.get("facts_in_session"))
    extracted_preferences = _normalize_memory_preferences(parsed.get("user_preferences"))
    model_memory_candidates = _build_model_memory_candidates(
        state=state,
        raw_candidates=parsed.get("memory_candidates"),
    )
    # 附件处理
    summary_attachment_refs = merge_attachment_paths(normalize_attachments(parsed.get("attachments")))
    summary_attachment_paths = [File(filepath=filepath) for filepath in summary_attachment_refs]

    final_events: List[Any] = [
        MessageEvent(role="assistant", message=summary_message, attachments=summary_attachment_paths)]

    plan.status = ExecutionStatus.COMPLETED
    final_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.COMPLETED))

    await emit_live_events(*final_events)
    working_memory = _ensure_working_memory(state)
    working_memory["facts_in_session"] = list(working_memory.get("facts_in_session") or [])
    for fact in extracted_facts:
        working_memory["facts_in_session"] = _append_unique_text_item(
            list(working_memory.get("facts_in_session") or []),
            fact,
        )
    if extracted_preferences:
        merged_preferences = dict(working_memory.get("user_preferences") or {})
        merged_preferences.update(extracted_preferences)
        working_memory["user_preferences"] = merged_preferences

    summary_local_memory = dict(state.get("summary_local_memory") or {})
    summary_local_memory["answer_outline"] = summary_message
    summary_local_memory["selected_artifacts"] = summary_attachment_refs
    next_state_for_memory: PlannerReActLangGraphState = {
        **state,
        "working_memory": working_memory,
        "final_message": summary_message,
    }
    memory_candidates = _build_memory_candidates(next_state_for_memory)
    if len(memory_candidates) == 0:
        outcome_candidate = _build_outcome_memory_candidate(
            next_state_for_memory,
            summary_message=summary_message,
        )
        if outcome_candidate is not None:
            memory_candidates = [outcome_candidate]
    memory_candidates = _merge_memory_candidates(memory_candidates, model_memory_candidates)
    if len(model_memory_candidates) > 0:
        summary_local_memory["memory_candidates_reason"] = "基于总结阶段的结构化提炼生成长期记忆候选"
    elif extracted_facts or extracted_preferences:
        summary_local_memory["memory_candidates_reason"] = "从总结阶段提炼出的稳定事实与偏好生成长期记忆候选"
    elif len(memory_candidates) > 0:
        summary_local_memory["memory_candidates_reason"] = "使用本轮任务结果生成保守的会话级长期记忆候选"
    else:
        summary_local_memory["memory_candidates_reason"] = ""
    return {
        **state,
        "plan": plan,
        "current_step_id": None,
        "final_message": summary_message,
        "working_memory": working_memory,
        "pending_memory_writes": _merge_memory_candidates(
            list(state.get("pending_memory_writes") or []),
            memory_candidates,
        ),
        "summary_local_memory": summary_local_memory,
        "emitted_events": append_events(state.get("emitted_events"), *final_events),
    }


async def consolidate_memory_node(
        state: PlannerReActLangGraphState,
        long_term_memory_repository: Optional[LongTermMemoryRepository] = None,
) -> PlannerReActLangGraphState:
    """统一收敛线程级短期记忆，压缩消息窗口并记录压缩元数据。"""
    # 获取并初始化摘要本地记忆
    summary_local_memory = dict(state.get("summary_local_memory") or {})
    
    # 将最终消息和选中的附件添加到消息窗口中
    message_window = _append_message_window_entry(
        list(state.get("message_window") or []),
        role="assistant",
        message=str(state.get("final_message") or ""),
        attachments=list(summary_local_memory.get("selected_artifacts") or []),
    )
    
    # 压缩消息窗口，防止超出最大长度限制
    compacted_message_window = _compact_message_window(message_window)
    
    # 更新图元数据，记录记忆压缩相关信息
    graph_metadata = dict(state.get("graph_metadata") or {})
    graph_metadata["memory_compacted"] = True
    graph_metadata["memory_last_compaction_at"] = datetime.now().isoformat()
    graph_metadata["memory_message_window_size"] = len(compacted_message_window)

    # 处理待写入的长期记忆候选项
    pending_memory_writes = list(state.get("pending_memory_writes") or [])
    remaining_memory_writes: List[Dict[str, Any]] = []
    persisted_memory_ids: List[str] = []
    
    if long_term_memory_repository is None:
        # 若未提供长期记忆仓库，则跳过写入，保留候选项供后续重试
        graph_metadata["memory_write_skipped"] = len(pending_memory_writes) > 0
        remaining_memory_writes = pending_memory_writes
    else:
        # 遍历所有待写入的记忆候选项，尝试持久化
        for item in pending_memory_writes:
            try:
                memory = LongTermMemory.model_validate(item)
                persisted_memory = await long_term_memory_repository.upsert(memory)
                persisted_memory_ids.append(persisted_memory.id)
            except Exception as e:
                logger.warning("长期记忆写入失败，保留候选待后续重试：%s", e)
                if isinstance(item, dict):
                    remaining_memory_writes.append(item)
        graph_metadata["memory_write_count"] = len(persisted_memory_ids)
        graph_metadata["memory_write_ids"] = persisted_memory_ids

    # 构建下一个状态对象，更新消息窗口、对话摘要、清理临时记忆并保留未成功写入的记忆候选
    next_state: PlannerReActLangGraphState = {
        **state,
        "message_window": compacted_message_window,
        "conversation_summary": _build_conversation_summary(
            {
                **state,
                "message_window": compacted_message_window,
            }
        ),
        "pending_memory_writes": remaining_memory_writes,
        "planner_local_memory": {},
        "step_local_memory": {},
        "summary_local_memory": {},
        "graph_metadata": graph_metadata,
    }
    return next_state


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
