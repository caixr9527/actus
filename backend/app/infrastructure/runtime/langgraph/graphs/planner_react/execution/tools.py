#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 工具调用循环与仲裁逻辑。"""

import json
import logging
from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.domain.external import LLM
from app.domain.models import (
    Step,
    ToolEvent,
)
from app.domain.services.prompts import SYSTEM_PROMPT, REACT_SYSTEM_PROMPT
from app.domain.services.runtime.contracts.langgraph_settings import (
    ASK_USER_FUNCTION_NAME,
    BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
    NOTIFY_USER_FUNCTION_NAME,
)
from app.domain.services.runtime.contracts.runtime_logging import elapsed_ms, log_runtime, now_perf
from app.domain.services.runtime.normalizers import (
    normalize_url_value,
    normalize_file_path_list,
    normalize_execution_response,
)
from app.domain.services.tools import BaseTool
from app.domain.services.workspace_runtime.policies import (
    build_loop_break_payload as _build_loop_break_payload,
    build_tool_feedback_content as _build_tool_feedback_content,
    analyze_text_intent as _analyze_text_intent,
    build_log_text_preview as _build_log_text_preview,
    build_step_candidate_text as _build_step_candidate_text,
    build_browser_preferred_function_names as _build_browser_preferred_function_names,
    normalize_execution_payload as _normalize_execution_payload,
)
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    normalize_attachments,
    safe_parse_json,
)
from .execution_context import build_execution_context
from .execution_state import ExecutionState
from .iteration_context import build_iteration_context
from ..convergence.general_convergence import GeneralConvergenceJudge
from ..convergence.judge import ConvergenceJudge
from ..convergence.research_convergence import ResearchConvergenceJudge
from ..convergence.web_reading_convergence import WebReadingConvergenceJudge
from ..policy_engine.engine import ToolPolicyEngine
from ..tool_runtime.tool_effects import build_interrupt_payload, extract_interrupt_request
from ..tool_runtime.tool_events import (
    ToolEventDispatcher,
    bind_tool_name,
    build_called_event,
    build_calling_event,
    build_tool_call_lifecycle,
    build_tool_feedback_message,
)
from ..tool_runtime.tool_schema import extract_function_name

logger = logging.getLogger(__name__)


def has_available_file_context(
        *,
        user_message: str,
        attachment_paths: Optional[List[str]] = None,
        artifact_paths: Optional[List[str]] = None,
) -> bool:
    """统一判断当前步骤是否已经具备真实可读的文件上下文。"""
    normalized_paths = normalize_file_path_list([*(attachment_paths or []), *(artifact_paths or [])])
    if len(normalized_paths) > 0:
        return True
    signals = _analyze_text_intent(user_message)
    return bool(signals["has_absolute_path"])


def _build_read_only_intent_text(
        *,
        step: Step,
        user_message: str,
        attachment_paths: Optional[List[str]],
        artifact_paths: Optional[List[str]],
) -> str:
    """只读治理只看业务语义输入，禁止把执行提示词正文混入判定。"""
    candidate_parts: List[str] = []
    normalized_user_message = str(user_message or "").strip()
    if normalized_user_message:
        candidate_parts.append(normalized_user_message)
    step_candidate_text = _build_step_candidate_text(step)
    if step_candidate_text:
        candidate_parts.append(step_candidate_text)
    file_context_paths = normalize_file_path_list([*(attachment_paths or []), *(artifact_paths or [])])
    if file_context_paths:
        candidate_parts.append("已知文件上下文: " + " ".join(file_context_paths))
    return "\n".join(candidate_parts).strip()


def collect_available_tools(runtime_tools: Optional[List[BaseTool]]) -> List[Dict[str, Any]]:
    """收集当前步骤可用的工具 schema 列表。"""
    available_tools: List[Dict[str, Any]] = []
    for tool in list(runtime_tools or []):
        try:
            available_tools.extend(tool.get_tools())
        except Exception as e:
            log_runtime(
                logger,
                logging.WARNING,
                "工具Schema读取失败",
                tool_name=getattr(tool, "name", "unknown"),
                error=str(e),
            )

    def _tool_priority(tool_schema: Dict[str, Any]) -> Tuple[int, str]:
        function_name = str(
            (tool_schema.get("function") or {}).get("name")
            if isinstance(tool_schema, dict)
            else ""
        ).strip().lower()
        # 搜索类优先，浏览器类后置：多数检索任务先走 search 更快。
        if "search" in function_name:
            return 0, function_name
        if function_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES:
            return 10, function_name
        if function_name.startswith("browser_"):
            return 80, function_name
        return 20, function_name

    available_tools.sort(key=_tool_priority)
    return available_tools


def _has_pending_research_candidate_urls(execution_state: ExecutionState) -> bool:
    fetched_keys = {
        normalize_url_value(url, drop_query=True)
        for url in list(execution_state.research_fetched_urls or [])
        if normalize_url_value(url, drop_query=True)
    }
    failed_keys = set(list(execution_state.research_failed_fetch_url_keys or set()))
    for raw_url in list(execution_state.research_candidate_urls or []):
        pending_key = normalize_url_value(raw_url, drop_query=True)
        if not pending_key:
            continue
        if pending_key in fetched_keys or pending_key in failed_keys:
            continue
        return True
    return False


def _should_prefer_fetch_page_for_research(execution_state: ExecutionState) -> bool:
    """判断 research 多工具候选中是否应优先执行 fetch_page。

    search_web 返回的 snippet 是研究链路的一等证据；只有摘要证据不足且仍有未读取候选链接时，
    才在同轮多工具调用中偏向 fetch_page，避免无条件进入页面抓取。
    """
    if bool(execution_state.research_snippet_sufficient):
        return False
    search_quality = dict(execution_state.last_search_evidence_quality or {})
    need_fetch = bool(search_quality.get("need_fetch"))
    return need_fetch and _has_pending_research_candidate_urls(execution_state)


def _tool_call_priority(
        function_name: str,
        *,
        preferred_function_names: Optional[Tuple[str, ...]] = None,
) -> int:
    normalized_name = function_name.strip().lower()
    preferred_names = tuple(
        str(item or "").strip().lower()
        for item in tuple(preferred_function_names or ())
        if str(item or "").strip()
    )
    if normalized_name in preferred_names:
        return -50 + preferred_names.index(normalized_name)
    if "search" in normalized_name:
        return 0
    if normalized_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES:
        return 10
    if normalized_name == NOTIFY_USER_FUNCTION_NAME:
        return 90
    if normalized_name == ASK_USER_FUNCTION_NAME:
        return 95
    if normalized_name.startswith("browser_"):
        return 80
    return 20


def pick_preferred_tool_call(
        tool_calls: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        *,
        preferred_function_names: Optional[Tuple[str, ...]] = None,
) -> Optional[Dict[str, Any]]:
    """从同轮多个 tool_call 中挑选本轮优先执行的候选。"""
    if len(tool_calls) == 0:
        return None
    if len(tool_calls) == 1:
        return tool_calls[0] if isinstance(tool_calls[0], dict) else None

    available_function_names: set[str] = set()
    for tool_schema in available_tools:
        function_name = extract_function_name(tool_schema)
        if function_name:
            available_function_names.add(function_name)

    ranked_candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for index, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        function_name = str(function.get("name") or "").strip()
        if not function_name:
            continue

        normalized_name = function_name.lower()
        priority = _tool_call_priority(
            function_name,
            preferred_function_names=preferred_function_names,
        )
        if normalized_name not in available_function_names:
            priority += 1000
        ranked_candidates.append((priority, index, raw_call))

    if len(ranked_candidates) == 0:
        return None

    ranked_candidates.sort(key=lambda item: (item[0], item[1]))
    selected_call = ranked_candidates[0][2]
    selected_function = str((selected_call.get("function") or {}).get("name") or "")
    log_runtime(
        logger,
        logging.INFO,
        "工具候选仲裁完成",
        candidate_count=len(tool_calls),
        selected_function=selected_function,
    )
    return selected_call


def _resolve_tool_by_function_name(function_name: str, runtime_tools: Optional[List[BaseTool]]) -> Optional[BaseTool]:
    for tool in list(runtime_tools or []):
        try:
            if tool.has_tool(function_name):
                return tool
        except Exception:
            continue
    return None


def _parse_tool_call_args(raw_arguments: Any) -> Dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            log_runtime(
                logger,
                logging.WARNING,
                "工具参数解析失败",
                raw_length=len(raw_arguments),
            )
            return {}
    return {}


def _build_assistant_tool_call_message(
        llm_message: Dict[str, Any],
        *,
        selected_tool_call: Dict[str, Any],
) -> Dict[str, Any]:
    """回放 assistant tool-call 消息时保留所有未知字段，避免推理态信息被裁掉。"""
    assistant_message = deepcopy(llm_message if isinstance(llm_message, dict) else {})
    assistant_message["role"] = "assistant"
    assistant_message["tool_calls"] = [deepcopy(selected_tool_call)]
    if "content" not in assistant_message:
        assistant_message["content"] = llm_message.get("content")
    return assistant_message


def _build_tool_feedback_content_with_runtime_progress(
        *,
        function_name: str,
        tool_result: Any,
        execution_context: Any,
        execution_state: ExecutionState,
) -> str:
    """构造带 research 运行进度的工具反馈消息。

    业务含义：
    - 普通 tool feedback 只回放“这一次工具调用返回了什么”；
    - 但 research 链路的下一轮决策，往往还依赖“当前步骤累计已经搜到什么、缺什么、该不该转 fetch_page”；
    - 因此这里会把 runtime_recent_action 中沉淀的 research 进度状态，回灌到 tool feedback 里，
      让下一轮 LLM 在同一步内显式继承阶段性研究进展，而不是只看到单次工具结果。

    生效范围：
    - 仅在 research_route_enabled 打开时启用；
    - 仅对 `search_web` / `fetch_page` 两类 research 主工具追加进度字段；
    - 其他工具保持原始反馈，避免把 research 噪音扩散到无关模式。
    """
    feedback_content = _build_tool_feedback_content(function_name, tool_result)
    if not bool(getattr(execution_context, "research_route_enabled", False)):
        return feedback_content
    normalized_function_name = str(function_name or "").strip().lower()
    if normalized_function_name not in {"search_web", "fetch_page"}:
        return feedback_content
    parsed_feedback = safe_parse_json(feedback_content)
    if not isinstance(parsed_feedback, dict):
        # 只有结构化 JSON 反馈才能安全追加 research 进度字段；
        # 非结构化文本保持原样返回，避免污染工具反馈合同。
        return feedback_content
    research_progress = dict(execution_state.runtime_recent_action.get("research_progress") or {})
    if len(research_progress) == 0:
        # 当前步骤还没有累计出可复用 research 进度时，不强行附加空状态。
        return feedback_content
    parsed_feedback["research_progress"] = research_progress
    research_diagnosis = dict(execution_state.runtime_recent_action.get("research_diagnosis") or {})
    if research_diagnosis:
        # 诊断信息描述“当前证据为什么足够/不足”，供下一轮模型纠偏。
        parsed_feedback["research_diagnosis"] = research_diagnosis
    search_evidence_quality = dict(execution_state.last_search_evidence_quality or {})
    if search_evidence_quality:
        # 搜索质量评估帮助模型判断当前 snippet 是否已够用，还是应继续抓页补强。
        parsed_feedback["search_evidence_quality"] = search_evidence_quality
    if execution_state.research_recommended_fetch_urls:
        # 推荐抓取链接是 research 路由器给出的下一步候选，减少无效重复搜索。
        parsed_feedback["recommended_fetch_urls"] = list(execution_state.research_recommended_fetch_urls)
    search_evidence_summaries = _build_search_evidence_feedback_summaries(execution_state)
    if search_evidence_summaries:
        # 最近搜索证据摘要作为同一步内的显式证据基底，避免模型忽略已拿到的 snippet。
        parsed_feedback["search_evidence_summaries"] = search_evidence_summaries
    missing_signals = list(research_progress.get("missing_signals") or [])
    if missing_signals:
        # next_action_hint 是给下一轮模型的明确动作提示，优先补齐尚未覆盖的事实维度。
        parsed_feedback["next_action_hint"] = (
                "请优先补齐缺失信息后再继续检索："
                + "；".join([str(item) for item in missing_signals if str(item).strip()])
        )
    return json.dumps(parsed_feedback, ensure_ascii=False)


def _build_search_evidence_feedback_summaries(execution_state: ExecutionState) -> List[Dict[str, str]]:
    """构造给下一轮模型消费的搜索摘要证据，避免模型忽略 search_web snippet。"""
    summaries: List[Dict[str, str]] = []
    for item in list(execution_state.research_search_evidence_items or [])[:5]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not url and not snippet:
            continue
        summaries.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )
    return summaries


def _seed_execution_state_from_initial_recent_action(execution_state: ExecutionState) -> None:
    """把跨步骤投影证据灌入单步执行态，供约束层在首轮工具调用前消费。"""
    recent_action = dict(execution_state.runtime_recent_action or {})
    research_progress = dict(recent_action.get("research_progress") or {})
    search_items: List[Dict[str, Any]] = []
    for raw_items in (
            recent_action.get("search_evidence_summaries"), research_progress.get("search_evidence_summaries")):
        for item in list(raw_items or []):
            if isinstance(item, dict):
                search_items.append(item)
    for item in search_items:
        url = str(item.get("url") or "").strip()
        if url and url not in execution_state.research_candidate_urls:
            execution_state.research_candidate_urls.append(url)
        if item not in execution_state.research_search_evidence_items:
            execution_state.research_search_evidence_items.append(item)
    if search_items:
        execution_state.research_search_ready = True

    web_items: List[Dict[str, Any]] = []
    for raw_items in (
            recent_action.get("web_reading_evidence_summaries"),
            research_progress.get("web_reading_evidence_summaries"),
    ):
        for item in list(raw_items or []):
            if isinstance(item, dict):
                web_items.append(item)
    for item in web_items:
        if item not in execution_state.web_reading_evidence_items:
            execution_state.web_reading_evidence_items.append(item)
    if web_items:
        execution_state.research_fetch_completed = True


async def execute_step_with_prompt(
        *,
        llm: LLM,
        step: Step,
        runtime_tools: Optional[List[BaseTool]],
        max_tool_iterations: int = 5,
        task_mode: str = "general",
        on_tool_event: Optional[Callable[[ToolEvent], Optional[Awaitable[None]]]] = None,
        user_content: Optional[List[Dict[str, Any]]] = None,
        user_message: str = "",
        attachment_paths: Optional[List[str]] = None,
        artifact_paths: Optional[List[str]] = None,
        has_available_file_context: bool = False,
        initial_runtime_recent_action: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[ToolEvent]]:
    """执行单步任务，支持“模型决策 -> 调工具 -> 回传模型”的最小循环。

    实现语义：
    - 这是 planner-react 单步执行的主编排函数；
    - 内部只负责驱动 LLM、policy engine、tool events、收敛判断，不直接承载约束细节；
    - 所有执行前约束都下沉到 `ToolPolicyEngine -> ConstraintEngine`，所有状态写回都下沉到 effects 域。
    """
    started_at = now_perf()
    # P3 重构：统一事件投递入口，tools 主流程不再内联事件分发细节。
    event_dispatcher = ToolEventDispatcher(
        logger=logger,
        on_tool_event=on_tool_event,
    )
    policy_engine = ToolPolicyEngine(logger=logger)
    convergence_judge = ConvergenceJudge()
    general_convergence_judge = GeneralConvergenceJudge()
    research_convergence_judge = ResearchConvergenceJudge()
    web_reading_convergence_judge = WebReadingConvergenceJudge()
    step_file_context: Dict[str, Any] = {
        "called_functions": set(),
    }

    available_tools = collect_available_tools(runtime_tools)
    available_function_names: set[str] = set()
    for tool_schema in available_tools:
        function_name = extract_function_name(tool_schema)
        if function_name:
            available_function_names.add(function_name)
    # P3 重构：上下文构建统一沉淀到独立模块，tools.py 仅保留编排职责。
    execution_context = build_execution_context(
        step=step,
        task_mode=task_mode,
        max_tool_iterations=max_tool_iterations,
        user_content=user_content,
        has_available_file_context=has_available_file_context,
        available_tools=available_tools,
        available_function_names=available_function_names,
        user_message_text=user_message,
        read_only_intent_text=_build_read_only_intent_text(
            step=step,
            user_message=user_message,
            attachment_paths=attachment_paths,
            artifact_paths=artifact_paths,
        ),
    )
    log_runtime(
        logger,
        logging.INFO,
        "执行上下文构建完成",
        step_id=str(step.id or ""),
        task_mode=task_mode,
        available_tool_count=len(available_tools),
        blocked_tool_count=len(execution_context.blocked_function_names),
        requested_max_tool_iterations=execution_context.requested_max_tool_iterations,
        max_tool_iterations=execution_context.effective_max_tool_iterations,
    )

    if task_mode == "human_wait" and ASK_USER_FUNCTION_NAME not in available_function_names:
        log_runtime(
            logger,
            logging.WARNING,
            "等待步骤缺少 ask_user 工具",
            step_id=str(step.id or ""),
        )
        return _build_loop_break_payload(
            step=step,
            blocker="当前等待步骤缺少可用的 message_ask_user 工具，无法向用户发起确认或选择。",
            next_hint="请先为当前运行时注入 message_ask_user 工具，再重新执行该等待步骤。",
        ), event_dispatcher.emitted_events
    log_runtime(
        logger,
        logging.INFO,
        "工具执行循环准备完成",
        step_id=str(step.id or ""),
        step_title=str(step.title or step.description or ""),
        task_mode=task_mode,
        available_tool_count=len(available_tools),
        blocked_tool_count=len(execution_context.blocked_function_names),
        requested_max_tool_iterations=execution_context.requested_max_tool_iterations,
        max_tool_iterations=execution_context.effective_max_tool_iterations,
    )

    if len(available_tools) == 0:
        log_runtime(
            logger,
            logging.INFO,
            "当前步骤无可用工具",
            step_id=str(step.id or ""),
        )
        llm_started_at = now_perf()
        llm_message = await llm.invoke(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + REACT_SYSTEM_PROMPT},
                {"role": "user", "content": execution_context.normalized_user_content}
            ],
            tools=[],
            response_format={"type": "json_object"},
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        parsed = safe_parse_json(llm_message.get("content"))
        parsed_execution = normalize_execution_response(parsed)
        has_summary = bool(str(parsed_execution.get("summary") or "").strip())
        has_structured_progress = bool(
            normalize_attachments(parsed.get("attachments"))
            or list(parsed_execution.get("blockers") or [])
            or list(parsed_execution.get("facts_learned") or [])
            or list(parsed_execution.get("open_questions") or [])
        )
        has_explicit_success = isinstance(parsed, dict) and "success" in parsed
        inferred_success = bool(parsed.get("success")) if has_explicit_success else bool(
            has_summary or has_structured_progress)
        log_runtime(
            logger,
            logging.INFO,
            "无工具模式执行完成",
            step_id=str(step.id or ""),
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
            attachment_count=len(normalize_attachments(parsed.get("attachments"))),
        )
        if not inferred_success and not has_summary and not has_structured_progress:
            return _build_loop_break_payload(
                step=step,
                blocker="当前步骤无可用工具，且模型未返回可交付结果。",
                next_hint="请补充更明确的执行指令，或启用对应工具后重试。",
            ), event_dispatcher.emitted_events
        return _normalize_execution_payload(
            {
                **parsed,
                "success": inferred_success,
            },
            default_summary=(
                f"步骤执行完成：{step.description}"
                if inferred_success
                else f"步骤暂未完成：{step.description}"
            ),
        ), event_dispatcher.emitted_events

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + REACT_SYSTEM_PROMPT},
        {"role": "user", "content": execution_context.normalized_user_content}
    ]

    execution_state = ExecutionState(runtime_recent_action=dict(initial_runtime_recent_action or {}))
    _seed_execution_state_from_initial_recent_action(execution_state)

    for index in range(execution_context.effective_max_tool_iterations):
        # P3 重构：单轮工具白名单/黑名单计算下沉，主循环只保留调度职责。
        iteration_context = build_iteration_context(
            task_mode=task_mode,
            execution_context=execution_context,
            execution_state=execution_state,
        )
        log_runtime(
            logger,
            logging.INFO,
            "开始工具决策轮次",
            step_id=str(step.id or ""),
            iteration=index,
            task_mode=task_mode,
            available_tool_count=len(iteration_context.iteration_tools),
        )
        llm_started_at = now_perf()
        execution_state.llm_message = await llm.invoke(
            messages=messages,
            tools=iteration_context.iteration_tools,
            tool_choice="auto",
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        tool_calls = execution_state.llm_message.get("tool_calls") or []
        if len(tool_calls) == 0:
            no_tool_call_result = policy_engine.finalize_no_tool_call(
                step=step,
                task_mode=task_mode,
                llm_message=execution_state.llm_message,
                llm_cost_ms=llm_cost_ms,
                started_at=started_at,
                iteration=index,
                runtime_recent_action=execution_state.runtime_recent_action,
            )
            if no_tool_call_result.action == "retry":
                # 这里的 retry 不是“利用本轮结果推进下一轮”，而只是“丢弃本轮结果后原地重试”
                # - payload 不参与后续流程
                # - parsed 也不会写回主循环
                # - 下一轮基本还是基于旧状态重新问模型
                # 所以这里的内容可以做反馈。
                continue
            return no_tool_call_result.payload or {}, event_dispatcher.emitted_events

        selected_tool_call = pick_preferred_tool_call(
            tool_calls=[item for item in tool_calls if isinstance(item, dict)],
            available_tools=iteration_context.iteration_tools,
            preferred_function_names=(
                ("fetch_page",)
                if execution_context.research_route_enabled
                   and not execution_state.research_fetch_completed
                   and _should_prefer_fetch_page_for_research(execution_state)
                else _build_browser_preferred_function_names(
                    task_mode=task_mode,
                    available_function_names=execution_context.available_function_names,
                    browser_page_type=execution_state.browser_page_type,
                    browser_structured_ready=execution_state.browser_structured_ready,
                    browser_main_content_ready=execution_state.browser_main_content_ready,
                    browser_cards_ready=execution_state.browser_cards_ready,
                    browser_link_match_ready=execution_state.browser_link_match_ready,
                    browser_actionables_ready=execution_state.browser_actionables_ready,
                    failed_high_level_functions=iteration_context.failed_browser_high_level_functions,
                )
            ),
        )
        if selected_tool_call is None:
            # 缺少受控反馈
            continue

        lifecycle = build_tool_call_lifecycle(
            selected_tool_call=selected_tool_call,
            parse_tool_call_args=_parse_tool_call_args,
        )
        if lifecycle is None:
            # 缺少受控反馈
            continue
        requested_function_name = str(lifecycle.function_name or "")
        matched_tool = _resolve_tool_by_function_name(function_name=lifecycle.function_name,
                                                      runtime_tools=runtime_tools)
        policy_result = await policy_engine.evaluate_tool_call(
            step=step,
            task_mode=task_mode,
            function_name=lifecycle.function_name,
            normalized_function_name=lifecycle.normalized_function_name,
            function_args=lifecycle.function_args,
            matched_tool=matched_tool,
            runtime_tools=runtime_tools,
            browser_route_state_key=iteration_context.browser_route_state_key,
            iteration_blocked_function_names=iteration_context.iteration_blocked_function_names,
            execution_context=execution_context,
            execution_state=execution_state,
            started_at=started_at,
        )
        log_runtime(
            logger,
            logging.INFO,
            "工具策略结果已回传主循环",
            step_id=str(step.id or ""),
            iteration=index,
            function_name=str(lifecycle.function_name or ""),
            requested_function_name=requested_function_name,
            final_function_name=str(policy_result.final_function_name or ""),
            loop_break_reason=str(policy_result.loop_break_reason or ""),
            tool_elapsed_ms=policy_result.tool_cost_ms,
        )
        lifecycle.function_name = str(policy_result.final_function_name or lifecycle.function_name or "")
        lifecycle.normalized_function_name = str(
            policy_result.final_normalized_function_name or lifecycle.normalized_function_name or ""
        ).strip().lower()
        lifecycle.function_args = dict(
            policy_result.executed_function_args or policy_result.final_function_args or lifecycle.function_args or {})
        matched_tool = policy_result.final_matched_tool
        lifecycle.tool_name = str(
            policy_result.final_tool_name or (matched_tool.name if matched_tool is not None else ""))

        selected_tool_call = {
            **dict(selected_tool_call),
            "function": {
                **dict(selected_tool_call.get("function") or {}),
                "name": lifecycle.function_name,
                "arguments": json.dumps(lifecycle.function_args, ensure_ascii=False),
            },
        }
        rewrite_reason = str(policy_result.rewrite_reason or "")
        if rewrite_reason:
            log_runtime(
                logger,
                logging.INFO,
                "研究链路工具自动纠偏",
                step_id=str(step.id or ""),
                iteration=index,
                rewrite_reason=rewrite_reason,
                **dict(policy_result.rewrite_metadata or {}),
            )
        messages.append(
            _build_assistant_tool_call_message(
                execution_state.llm_message,
                selected_tool_call=selected_tool_call,
            )
        )

        log_runtime(
            logger,
            logging.INFO,
            "已选择工具调用",
            step_id=str(step.id or ""),
            iteration=index,
            tool_call_id=lifecycle.tool_call_id,
            function_name=lifecycle.function_name,
            requested_function_name=requested_function_name,
            executed_function_name=lifecycle.function_name,
            task_mode=task_mode,
            same_tool_repeat_count=execution_state.same_tool_repeat_count,
            arg_keys=sorted(lifecycle.function_args.keys()),
            search_query_preview=(
                _build_log_text_preview(lifecycle.function_args.get("query"), max_chars=100)
                if lifecycle.normalized_function_name == "search_web"
                else ""
            ),
            rewrite_reason=rewrite_reason,
        )

        bind_tool_name(lifecycle, matched_tool)
        await event_dispatcher.emit(build_calling_event(lifecycle))
        tool_result = policy_result.tool_result
        loop_break_reason = policy_result.loop_break_reason
        tool_cost_ms = policy_result.tool_cost_ms

        await event_dispatcher.emit(build_called_event(lifecycle, tool_result))
        interrupt_request = extract_interrupt_request(tool_result)
        log_runtime(
            logger,
            logging.INFO,
            "工具调用完成",
            step_id=str(step.id or ""),
            tool_call_id=lifecycle.tool_call_id,
            function_name=lifecycle.function_name,
            requested_function_name=requested_function_name,
            executed_function_name=lifecycle.function_name,
            tool_name=lifecycle.tool_name,
            success=bool(tool_result.success),
            has_interrupt=bool(interrupt_request),
            loop_break_reason=loop_break_reason or "",
            browser_no_progress_count=execution_state.browser_no_progress_count,
            consecutive_failure_count=execution_state.consecutive_failure_count,
            research_candidate_url_count=len(execution_state.research_candidate_urls),
            research_fetched_url_count=len(execution_state.research_fetched_urls),
            research_candidate_domain_count=len(execution_state.research_candidate_domains),
            research_fetched_domain_count=len(execution_state.research_fetched_domains),
            research_coverage_score=execution_state.research_coverage_score,
            llm_elapsed_ms=llm_cost_ms,
            tool_elapsed_ms=tool_cost_ms if matched_tool is not None else 0,
            elapsed_ms=elapsed_ms(started_at),
        )
        if interrupt_request is not None:
            log_runtime(
                logger,
                logging.INFO,
                "工具请求进入等待",
                step_id=str(step.id or ""),
                tool_call_id=lifecycle.tool_call_id,
                function_name=lifecycle.function_name,
                interrupt_kind=str(interrupt_request.get("kind") or ""),
                elapsed_ms=elapsed_ms(started_at),
            )
            return build_interrupt_payload(
                interrupt_request=interrupt_request,
                runtime_recent_action=execution_state.runtime_recent_action,
            ), event_dispatcher.emitted_events
        messages.append(
            build_tool_feedback_message(
                lifecycle=lifecycle,
                tool_result=tool_result,
                feedback_content_builder=lambda function_name, current_tool_result: (
                    _build_tool_feedback_content_with_runtime_progress(
                        function_name=function_name,
                        tool_result=current_tool_result,
                        execution_context=execution_context,
                        execution_state=execution_state,
                    )
                ),
            )
        )
        convergence_result = policy_engine.evaluate_iteration_convergence(
            loop_break_reason=loop_break_reason or "",
            step=step,
            tool_result=tool_result,
            execution_state=execution_state,
        )
        if convergence_result.should_break and convergence_result.payload is not None:
            return convergence_result.payload, event_dispatcher.emitted_events
        research_convergence_result = research_convergence_judge.evaluate_after_iteration(
            step=step,
            task_mode=task_mode,
            recent_function_name=lifecycle.normalized_function_name,
            execution_state=execution_state,
        )
        if research_convergence_result.should_break and research_convergence_result.payload is not None:
            log_runtime(
                logger,
                logging.INFO,
                "研究证据满足，提前收敛步骤",
                step_id=str(step.id or ""),
                task_mode=task_mode,
                reason_code=research_convergence_result.reason_code,
                iteration=index,
            )
            return research_convergence_result.payload, event_dispatcher.emitted_events
        web_reading_convergence_result = web_reading_convergence_judge.evaluate_after_iteration(
            step=step,
            task_mode=task_mode,
            recent_function_name=lifecycle.normalized_function_name,
            execution_state=execution_state,
        )
        if web_reading_convergence_result.should_break and web_reading_convergence_result.payload is not None:
            log_runtime(
                logger,
                logging.INFO,
                "网页阅读证据满足，提前收敛步骤",
                step_id=str(step.id or ""),
                task_mode=task_mode,
                reason_code=web_reading_convergence_result.reason_code,
                iteration=index,
            )
            return web_reading_convergence_result.payload, event_dispatcher.emitted_events
        progress_result = convergence_judge.evaluate_file_processing_progress(
            step=step,
            task_mode=task_mode,
            recent_function_name=lifecycle.normalized_function_name,
            function_args=lifecycle.function_args,
            tool_result_data=tool_result.data,
            tool_result_success=bool(tool_result.success),
            step_file_context=step_file_context,
            runtime_recent_action=execution_state.runtime_recent_action,
        )
        if progress_result.should_break and progress_result.payload is not None:
            log_runtime(
                logger,
                logging.INFO,
                "关键事实满足，提前收敛步骤",
                step_id=str(step.id or ""),
                task_mode=task_mode,
                reason_code=progress_result.reason_code,
                iteration=index,
            )
            return progress_result.payload, event_dispatcher.emitted_events
        general_convergence_result = general_convergence_judge.evaluate_after_iteration(
            step=step,
            task_mode=task_mode,
            runtime_recent_action=execution_state.runtime_recent_action,
            iteration=index,
        )
        if general_convergence_result.should_break and general_convergence_result.payload is not None:
            log_runtime(
                logger,
                logging.INFO,
                "general 文件观察事实满足，提前收敛步骤",
                step_id=str(step.id or ""),
                task_mode=task_mode,
                reason_code=general_convergence_result.reason_code,
                iteration=index,
            )
            return general_convergence_result.payload, event_dispatcher.emitted_events

    return policy_engine.finalize_max_iterations(
        step=step,
        task_mode=task_mode,
        llm_message=execution_state.llm_message,
        started_at=started_at,
        requested_max_tool_iterations=execution_context.requested_max_tool_iterations,
        iteration_count=execution_context.effective_max_tool_iterations,
        runtime_recent_action=execution_state.runtime_recent_action,
        step_file_context=step_file_context,
    ), event_dispatcher.emitted_events
