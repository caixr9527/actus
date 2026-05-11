#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 工具调用循环与仲裁逻辑。"""

import json
import logging
from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from app.domain.external import LLM
from app.domain.models import (
    Step,
    StepArtifactPolicy,
    StepOutputMode,
    ToolEvent,
    ToolResult,
)
from app.domain.models.evidence import EvidenceResolvedStatus
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.prompts import SYSTEM_PROMPT, REACT_SYSTEM_PROMPT
from app.domain.services.runtime.contracts.langgraph_settings import (
    ASK_USER_FUNCTION_NAME,
    BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
    NOTIFY_USER_FUNCTION_NAME,
)
from app.domain.services.runtime.contracts.runtime_logging import elapsed_ms, log_runtime, now_perf
from app.domain.services.runtime.contracts.evidence_ledger_contract import RuntimeEvidenceContextResult
from app.domain.services.runtime.normalizers import (
    normalize_url_value,
    normalize_file_path_list,
    normalize_execution_response,
    normalize_controlled_value,
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
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.contracts import IterationConvergenceContext
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.engine import ConvergenceEngine
from app.infrastructure.runtime.langgraph.graphs.planner_react.loop_breaks import build_loop_break_result
from app.infrastructure.runtime.langgraph.graphs.planner_react.policy_engine.engine import ToolPolicyEngine
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_effects import build_interrupt_payload, \
    extract_interrupt_request
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_events import (
    ToolEventDispatcher,
    bind_tool_name,
    build_called_event,
    build_calling_event,
    build_tool_call_lifecycle,
    build_tool_feedback_message,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_schema import extract_function_name
from .execution_context import build_execution_context
from .execution_state import ExecutionState
from .iteration_context import build_iteration_context

logger = logging.getLogger(__name__)

FILE_WRITE_RESULT_FUNCTION_NAMES = {"write_file", "replace_in_file"}


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
    """只读治理只看当前步骤语义和已知文件上下文，禁止多步骤用户原文污染当前 step。"""
    candidate_parts: List[str] = []
    _ = user_message
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


def _sandbox_profile_blocks_file_step(
        *,
        task_mode: str,
        available_function_names: set[str],
        sandbox_capability_profile: Optional[Dict[str, Any]],
) -> bool:
    if str(task_mode or "").strip().lower() != "file_processing":
        return False
    profile = dict(sandbox_capability_profile or {})
    if not profile:
        return False
    if not bool(profile.get("sandbox_profile_stale")):
        return False
    required_any = {"read_file", "write_file", "replace_in_file", "list_files", "shell_execute"}
    return len(required_any.intersection(set(available_function_names or set()))) == 0


def _step_requires_file_write_result(step: Step) -> bool:
    """结构化合同要求文件产物时，必须由真实写工具或后续 artifact index 证明。"""
    output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
    artifact_policy = normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy)
    return (
        output_mode == StepOutputMode.FILE.value
        or artifact_policy == StepArtifactPolicy.REQUIRE_FILE_OUTPUT.value
    )


def _collect_function_names_from_tool_schemas(tool_schemas: List[Dict[str, Any]]) -> set[str]:
    function_names: set[str] = set()
    for tool_schema in list(tool_schemas or []):
        function_name = extract_function_name(tool_schema)
        if function_name:
            function_names.add(str(function_name or "").strip().lower())
    return function_names


def _build_file_write_tool_unavailable_payload(
        *,
        step: Step,
        runtime_recent_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _build_loop_break_payload(
        step=step,
        blocker="当前步骤要求产出文件，但执行轮次没有可用的 write_file 或 replace_in_file 工具，已拒绝模型自报文件成功。",
        next_hint="请恢复文件写入工具能力，或重新规划为不要求文件产出的步骤。",
        runtime_recent_action={
            **dict(runtime_recent_action or {}),
            "reason_code": "file_write_tool_unavailable",
        },
    )


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

    search_web 返回的 snippet 只是候选来源信号；只有候选摘要不足且仍有未读取候选链接时，
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
        # 最近搜索候选摘要只帮助模型选择下一步来源，不是 Evidence Ledger evidence，也不参与完成判定。
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
    """构造给下一轮模型消费的候选来源摘要，不作为 guard/completion 的证据来源。"""
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
        sandbox_capability_profile: Optional[Dict[str, Any]] = None,
        runtime_evidence_context: Optional[RuntimeEvidenceContextResult] = None,
        has_previous_completed_steps: bool = False,
        previous_completed_step_task_modes: Optional[Dict[str, str]] = None,
        evidence_result_handle_resolver: Any = None,
        evidence_resolution_state: Optional[Dict[str, Any]] = None,
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
    convergence_engine = ConvergenceEngine(logger=logger)
    step_file_context: Dict[str, Any] = {
        "called_functions": set(),
    }

    available_tools = collect_available_tools(runtime_tools)
    available_function_names: set[str] = set()
    for tool_schema in available_tools:
        function_name = extract_function_name(tool_schema)
        if function_name:
            available_function_names.add(function_name)
    if _sandbox_profile_blocks_file_step(
            task_mode=task_mode,
            available_function_names=available_function_names,
            sandbox_capability_profile=sandbox_capability_profile,
    ):
        log_runtime(
            logger,
            logging.ERROR,
            "sandbox_runtime_tool_snapshot_invalid",
            step_id=str(step.id or ""),
            task_mode=task_mode,
            available_tool_count=len(available_tools),
            reason_code="sandbox_runtime_tool_snapshot_invalid",
        )
        return _build_loop_break_payload(
            step=step,
            blocker="当前 sandbox capability profile 已过期，且文件/命令运行时工具不可用，已停止文件类步骤执行。",
            next_hint="请刷新 sandbox capability profile 并确认文件读取工具可用后再继续。",
        ), event_dispatcher.emitted_events
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
        available_function_names=sorted(available_function_names),
        blocked_tool_count=len(execution_context.blocked_function_names),
        blocked_function_names=sorted(execution_context.blocked_function_names),
        read_only_file_blocked_function_names=sorted(execution_context.read_only_file_blocked_function_names),
        file_write_intent_blocked_function_names=sorted(execution_context.file_write_intent_blocked_function_names),
        artifact_policy_blocked_function_names=sorted(execution_context.artifact_policy_blocked_function_names),
        file_processing_shell_blocked_function_names=sorted(execution_context.file_processing_shell_blocked_function_names),
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

    initial_execution_state = ExecutionState(runtime_recent_action=dict(initial_runtime_recent_action or {}))
    _seed_execution_state_from_initial_recent_action(initial_execution_state)
    initial_iteration_context = build_iteration_context(
        task_mode=task_mode,
        execution_context=execution_context,
        execution_state=initial_execution_state,
    )
    initial_iteration_function_names = _collect_function_names_from_tool_schemas(
        initial_iteration_context.iteration_tools,
    )
    if (
            _step_requires_file_write_result(step)
            and len(FILE_WRITE_RESULT_FUNCTION_NAMES.intersection(initial_iteration_function_names)) == 0
    ):
        log_runtime(
            logger,
            logging.ERROR,
            "文件产出步骤缺少可用写工具",
            step_id=str(step.id or ""),
            task_mode=task_mode,
            reason_code="file_write_tool_unavailable",
            available_function_names=sorted(available_function_names),
            blocked_function_names=sorted(execution_context.blocked_function_names),
            initial_iteration_function_names=sorted(initial_iteration_function_names),
        )
        return _build_file_write_tool_unavailable_payload(
            step=step,
            runtime_recent_action=initial_execution_state.runtime_recent_action,
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
        initial_iteration_function_names=sorted(initial_iteration_function_names),
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

    execution_state = initial_execution_state

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
            step_id=str(step.id or ""),
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
            evidence_reuse_snapshot=(
                runtime_evidence_context.evidence_reuse_snapshot
                if runtime_evidence_context is not None
                else None
            ),
            has_previous_completed_steps=(
                bool(has_previous_completed_steps or runtime_evidence_context.has_previous_completed_steps)
                if runtime_evidence_context is not None
                else bool(has_previous_completed_steps)
            ),
            previous_completed_step_task_modes=dict(previous_completed_step_task_modes or {}),
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
        if loop_break_reason == "virtual_success_pending_resolution":
            resolved_message = await _resolve_pending_evidence_reuse_before_emit(
                llm_message={
                    "success": bool(tool_result.success),
                    "message": str(tool_result.message or ""),
                    "loop_break_reason": loop_break_reason,
                    "data": dict(tool_result.data or {}),
                },
                runtime_evidence_context=runtime_evidence_context,
                evidence_result_handle_resolver=evidence_result_handle_resolver,
                state=evidence_resolution_state or {},
            )
            tool_result = ToolResult(
                success=bool(resolved_message.get("success", False)),
                message=str(resolved_message.get("summary") or resolved_message.get("message") or tool_result.message),
                data=dict(resolved_message.get("data") or {}),
            )
            loop_break_reason = str(resolved_message.get("loop_break_reason") or loop_break_reason)

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
        if _is_valid_web_reading_fetch_result(
                task_mode=task_mode,
                function_name=lifecycle.normalized_function_name,
                tool_result=tool_result,
        ):
            log_runtime(
                logger,
                logging.INFO,
                "page_evidence_pending_gate",
                step_id=str(step.id or ""),
                function_name=lifecycle.function_name,
                reason_code="page_evidence_pending_gate",
            )
            return {
                "success": False,
                "summary": "页面抓取已完成，等待 evidence gate 校验页面证据",
                "loop_break_reason": "page_evidence_pending_gate",
                "data": {
                    "reason_code": "page_evidence_pending_gate",
                },
            }, event_dispatcher.emitted_events
        loop_break_payload = build_loop_break_result(
            loop_break_reason=loop_break_reason or "",
            step=step,
            tool_result=tool_result,
            runtime_recent_action=execution_state.runtime_recent_action,
        )
        if loop_break_payload is not None:
            return loop_break_payload, event_dispatcher.emitted_events
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
        convergence_result = convergence_engine.evaluate_after_iteration(
            context=IterationConvergenceContext(
                step=step,
                task_mode=task_mode,
                iteration=index,
                recent_function_name=lifecycle.normalized_function_name,
                function_args=lifecycle.function_args,
                tool_result=tool_result,
                loop_break_reason=loop_break_reason or "",
                execution_state=execution_state,
                step_file_context=step_file_context,
            ),
        )
        if convergence_result.should_break and convergence_result.payload is not None:
            return convergence_result.payload, event_dispatcher.emitted_events
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


def _is_valid_web_reading_fetch_result(
        *,
        task_mode: str,
        function_name: str,
        tool_result: ToolResult,
) -> bool:
    if str(task_mode or "").strip().lower() != "web_reading":
        return False
    if str(function_name or "").strip().lower() != "fetch_page":
        return False
    if not bool(tool_result.success):
        return False
    data = _normalize_tool_result_data(tool_result.data)
    if data.get("result_handle_resolved") is True:
        return False
    has_content = bool(str(data.get("content") or data.get("excerpt") or "").strip())
    is_truncated = bool(data.get("is_truncated") or data.get("truncated"))
    return has_content and not is_truncated


def _normalize_tool_result_data(value: Any) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    return {}


async def _resolve_pending_evidence_reuse_before_emit(
        *,
        llm_message: Dict[str, Any],
        runtime_evidence_context: RuntimeEvidenceContextResult | None,
        evidence_result_handle_resolver: Any,
        state: Dict[str, Any],
) -> Dict[str, Any]:
    if evidence_result_handle_resolver is None or runtime_evidence_context is None:
        return llm_message
    data = llm_message.get("data")
    if not isinstance(data, dict):
        return llm_message
    result_handle_id = str(data.get("result_handle_id") or "").strip()
    handle = runtime_evidence_context.result_handle_index.get(result_handle_id)
    current_step_id = str(runtime_evidence_context.current_step_id or state.get("current_step_id") or "").strip()
    if handle is None:
        _log_pending_evidence_resolution(
            event_name="evidence_result_handle_resolve_failed",
            state=state,
            current_step_id=current_step_id,
            result_handle_id=result_handle_id,
            read_strategy="",
            status="missing",
            reason_code="result_handle_missing",
        )
        return {
            **llm_message,
            "success": False,
            "loop_break_reason": "result_handle_missing",
            "data": {**data, "reason_code": "result_handle_missing"},
        }
    scope = _build_resolver_scope(state=state, current_step_id=current_step_id)
    if scope is None:
        _log_pending_evidence_resolution(
            event_name="evidence_result_handle_resolve_failed",
            state=state,
            current_step_id=current_step_id,
            result_handle_id=result_handle_id,
            read_strategy=handle.read_strategy.value,
            status="scope_missing",
            reason_code="evidence_scope_missing",
        )
        return {
            **llm_message,
            "success": False,
            "loop_break_reason": "result_handle_resolve_failed",
            "data": {**data, "reason_code": "evidence_scope_missing"},
        }
    _log_pending_evidence_resolution(
        event_name="evidence_result_handle_resolution_started",
        state=state,
        current_step_id=str(scope.current_step_id or ""),
        result_handle_id=result_handle_id,
        read_strategy=handle.read_strategy.value,
        status="started",
        reason_code="pending_resolution",
    )
    resolved = await evidence_result_handle_resolver.resolve(scope=scope, handle=handle)
    if resolved.status != EvidenceResolvedStatus.RESOLVED:
        _log_pending_evidence_resolution(
            event_name=(
                "evidence_result_handle_stale"
                if resolved.status == EvidenceResolvedStatus.STALE
                else "evidence_result_handle_resolve_failed"
            ),
            state=state,
            current_step_id=str(scope.current_step_id or ""),
            result_handle_id=result_handle_id,
            read_strategy=handle.read_strategy.value,
            status=resolved.status.value,
            reason_code=resolved.reason_code,
        )
        return {
            **llm_message,
            "success": False,
            "loop_break_reason": "result_handle_resolve_failed",
            "data": {
                **data,
                "result_handle_resolved": False,
                "resolved_status": resolved.status.value,
                "reason_code": resolved.reason_code,
            },
        }
    _log_pending_evidence_resolution(
        event_name="evidence_result_handle_resolved",
        state=state,
        current_step_id=str(scope.current_step_id or ""),
        result_handle_id=result_handle_id,
        read_strategy=handle.read_strategy.value,
        status=resolved.status.value,
        reason_code=resolved.reason_code or "resolved",
    )
    _log_pending_evidence_resolution(
        event_name="evidence_reuse_virtual_tool_result_returned",
        state=state,
        current_step_id=str(scope.current_step_id or ""),
        result_handle_id=result_handle_id,
        read_strategy=handle.read_strategy.value,
        status="resolved",
        reason_code="evidence_reuse_allowed",
    )
    return {
        **llm_message,
        "success": True,
        "summary": str(resolved.summary or data.get("reuse_summary") or "已复用前序证据结果"),
        "loop_break_reason": "evidence_reuse_allowed",
        "data": {
            **data,
            "duplicate_decision": "reuse_existing_evidence",
            "result_handle_resolved": True,
            "resolved_result": resolved.model_dump(mode="json"),
        },
    }


def _build_resolver_scope(
        *,
        state: Dict[str, Any],
        current_step_id: str,
) -> AccessScopeResult | None:
    user_id = str(state.get("user_id") or "").strip()
    session_id = str(state.get("session_id") or "").strip()
    workspace_id = str(state.get("workspace_id") or "").strip()
    run_id = str(state.get("run_id") or "").strip()
    if not user_id or not session_id or not workspace_id or not run_id:
        return None
    return AccessScopeResult(
        tenant_id=user_id,
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        run_id=run_id,
        current_step_id=current_step_id or str(state.get("current_step_id") or "").strip() or None,
    )


def _log_pending_evidence_resolution(
        *,
        event_name: str,
        state: Dict[str, Any],
        current_step_id: str,
        result_handle_id: str,
        read_strategy: str,
        status: str,
        reason_code: str | None,
) -> None:
    log_runtime(
        logger,
        logging.INFO,
        event_name,
        state={**state, "current_step_id": current_step_id or state.get("current_step_id")},
        user_id=str(state.get("user_id") or ""),
        workspace_id=str(state.get("workspace_id") or ""),
        result_handle_id=result_handle_id,
        read_strategy=read_strategy,
        status=status,
        reason_code=str(reason_code or ""),
    )
