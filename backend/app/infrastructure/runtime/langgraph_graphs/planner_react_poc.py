#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : planner_react_poc.py
"""
import json
import logging
import re
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


_SIMPLE_GREETING_NORMALIZED_SET = {
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "在吗",
    "hi",
    "hello",
    "hey",
}


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


def _normalize_user_text(text: str) -> str:
    """归一化用户消息文本，便于做轻量意图判断。"""
    return re.sub(r"[\s\W_]+", "", text.strip().lower(), flags=re.UNICODE)


def _is_simple_greeting_message(user_message: str) -> bool:
    """识别纯问候类输入，避免进入完整 planner/step 流程。"""
    normalized = _normalize_user_text(user_message)
    return normalized in _SIMPLE_GREETING_NORMALIZED_SET


def _build_simple_greeting_reply(user_message: str) -> str:
    """根据问候语种返回简洁回复。"""
    has_ascii_letters = re.search(r"[a-z]", user_message, flags=re.IGNORECASE) is not None
    has_cjk = re.search(r"[\u4e00-\u9fff]", user_message) is not None
    if has_ascii_letters and not has_cjk:
        return "Hello! I'm your assistant, happy to help."
    return "你好！我是助手，很高兴为您服务。"


def _get_last_assistant_message(events: List[Any]) -> str:
    """获取当前事件列表中最后一条 assistant message 文本。"""
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.role == "assistant":
            return str(event.message or "")
    return ""


def _build_summarize_prompt(state: PlannerReActPOCState) -> str:
    """构建带上下文的总结提示词，降低无关总结风险。"""
    plan = state.get("plan")
    plan_snapshot = plan.model_dump(mode="json") if plan is not None else {}
    final_message = str(state.get("final_message") or "")
    user_message = str(state.get("user_message") or "")
    execution_count = int(state.get("execution_count") or 0)

    return (
        f"{SUMMARIZE_PROMPT}\n\n"
        "请严格基于以下运行上下文输出总结，禁止引入上下文之外的场景或数据：\n"
        f"- 用户原始消息: {user_message}\n"
        f"- 执行轮次: {execution_count}\n"
        f"- 最近一步结果: {final_message}\n"
        f"- 计划快照(JSON): {json.dumps(plan_snapshot, ensure_ascii=False)}\n"
    )


def _should_accept_summary_message(
        state: PlannerReActPOCState,
        candidate_message: str,
        fallback_message: str,
) -> bool:
    """判断是否接受模型总结文本，避免简单任务被无关长文覆盖。"""
    candidate = candidate_message.strip()
    if not candidate:
        return False

    # 单步任务通常没有必要产出超长总结；若出现明显异常长文，回退到步骤结果。
    execution_count = int(state.get("execution_count") or 0)
    if execution_count <= 1 and fallback_message.strip():
        # 单步任务优先与执行结果保持一致；若总结与步骤结果无明显关联，则拒绝覆盖。
        fallback = fallback_message.strip()
        if fallback not in candidate and candidate not in fallback:
            logger.warning("LangGraph summarize 与单步执行结果无关联，回退到步骤结果")
            return False

        max_allowed_length = len(fallback_message.strip()) * 3 + 120
        if len(candidate) > max_allowed_length:
            logger.warning("LangGraph summarize 产出异常长文本，回退到步骤结果以避免无关回复")
            return False

    return True


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


def _should_emit_planner_message(user_message: str, planner_message: str, steps: List[Step]) -> bool:
    """判断是否需要对外输出规划阶段消息，避免简单问候场景的回显噪音。"""
    normalized_planner_message = planner_message.strip()
    if not normalized_planner_message:
        return False

    normalized_user_message = user_message.strip()
    # 规划消息与用户输入完全一致时，通常是模型回显，不需要额外展示。
    if normalized_user_message and normalized_planner_message == normalized_user_message:
        return False

    # 单步任务下若规划消息仅重复步骤描述，同样不输出，避免和步骤结果形成“多条问候”。
    if len(steps) == 1:
        first_step_description = str(steps[0].description or "").strip()
        if first_step_description and normalized_planner_message == first_step_description:
            return False

    return True


def _route_after_plan(state: PlannerReActPOCState) -> Literal["execute_step", "summarize", "finalize"]:
    plan = state.get("plan")
    if plan is None:
        # 无 plan 且已有最终回复时，直接结束，避免无意义 summarize。
        return "finalize" if str(state.get("final_message") or "").strip() else "summarize"
    if state.get("execution_count", 0) >= state.get("max_execution_steps", 20):
        logger.warning("LangGraph V1 执行次数达到上限，提前进入总结阶段")
        return "summarize"
    return "execute_step" if plan.get_next_step() is not None else "summarize"


def _route_after_replan(state: PlannerReActPOCState) -> Literal["execute_step", "summarize", "finalize"]:
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
    if _is_simple_greeting_message(user_message):
        # 纯问候直接回复并结束，不生成 plan/step，避免前端出现无意义步骤卡片。
        greeting_reply = _build_simple_greeting_reply(user_message=user_message)
        return {
            **state,
            "plan": None,
            "current_step_id": None,
            "final_message": greeting_reply,
            "emitted_events": append_events(
                state.get("emitted_events"),
                MessageEvent(role="assistant", message=greeting_reply),
            ),
        }

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
    planner_events: List[Any] = [TitleEvent(title=title)]
    if _should_emit_planner_message(
            user_message=user_message,
            planner_message=planner_message,
            steps=steps,
    ):
        planner_events.append(MessageEvent(role="assistant", message=planner_message))
    planner_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.CREATED))

    return {
        **state,
        "plan": plan,
        "current_step_id": next_step.id if next_step is not None else None,
        "emitted_events": append_events(
            state.get("emitted_events"),
            *planner_events,
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
    prompt = _build_summarize_prompt(state)
    llm_message = await llm.invoke(messages=[{"role": "user", "content": prompt}], tools=[])
    parsed = _safe_parse_json(llm_message.get("content"))
    fallback_message = str(state.get("final_message") or "任务已完成")
    candidate_summary_message = str(parsed.get("message") or "")
    accepted_candidate = _should_accept_summary_message(
        state=state,
        candidate_message=candidate_summary_message,
        fallback_message=fallback_message,
    )
    summary_message = candidate_summary_message if accepted_candidate else fallback_message
    attachment_paths = _normalize_attachments(parsed.get("attachments")) if accepted_candidate else []
    attachments = [File(filepath=filepath) for filepath in attachment_paths]
    previous_assistant_message = _get_last_assistant_message(list(state.get("emitted_events") or []))

    final_events: List[Any] = []
    # 单步问候等轻量场景下，如果 summarize 回退文本与上一次 assistant 输出一致且无附件，
    # 不再重复发一条 message，避免前端看到“最终回复重复”。
    if not (
            summary_message.strip()
            and previous_assistant_message.strip()
            and summary_message.strip() == previous_assistant_message.strip()
            and len(attachments) == 0
    ):
        final_events.append(MessageEvent(role="assistant", message=summary_message, attachments=attachments))

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
            "finalize": "finalize",
        },
    )
    graph.add_edge("execute_step", "replan")
    graph.add_conditional_edges(
        "replan",
        _route_after_replan,
        {
            "execute_step": "execute_step",
            "summarize": "summarize",
            "finalize": "finalize",
        },
    )
    graph.add_edge("summarize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=InMemorySaver())
