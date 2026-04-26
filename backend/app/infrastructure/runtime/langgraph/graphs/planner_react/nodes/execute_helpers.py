#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""execute_step_node 专项 helper。

本模块承接执行节点的稳定职责：
1. 执行前输入装配：把 state/control/step 转成执行器可消费的输入；
2. 执行结果归一：把执行器返回的原始消息与 tool event 收口成统一结果；
3. interrupt / completed 分支的状态写回数据装配。

注意：
- 本模块不改变 wait / replan / summary 的业务语义；
- 本模块不直接决定 graph node 的流转，只给 execute_step_node 提供可复用的阶段 helper。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.domain.external import LLM
from app.domain.models import (
    ExecutionStatus,
    Plan,
    Step,
    StepEvent,
    StepOutcome,
    ToolEvent,
)
from app.domain.services.prompts import EXECUTION_PROMPT
from app.domain.services.runtime.langgraph_events import append_events
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import (
    normalize_execution_response,
    normalize_file_path_list,
    normalize_step_result_text,
    normalize_text_list,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.entry import EntryContract, EntryRoute
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    format_attachments_for_prompt,
    normalize_attachments,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.tools import (
    has_available_file_context,
)
from .confirmed_facts import _build_step_execution_text, _normalize_confirmed_fact_map
from .control_state import get_entry_contract_payload, replace_control_metadata
from .delivery_helpers import (
    _collect_available_file_context_refs,
    _infer_step_attachment_delivery_preference,
    _merge_step_outcome_into_working_memory,
)
from .prompt_context_helpers import (
    _append_prompt_context_to_prompt,
    _build_prompt_context_packet_async,
    _extract_prompt_context_state_updates,
)
from .wait_helpers import _build_step_label
from .working_memory import _ensure_working_memory
from ..parsers import extract_write_file_paths_from_tool_events, merge_attachment_paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecuteStepPreparedInput:
    """执行阶段输入装配结果。

    业务含义：
    - 这是 execute_step_node 真正调用执行器前的完整输入快照；
    - 节点壳只负责决定是否执行，具体 prompt/context/附件装配由这里统一产出。
    """

    user_message: str
    task_mode: str
    working_memory: Dict[str, Any]
    confirmed_fact_keys: List[str]
    input_parts: List[Dict[str, Any]]
    attachments: List[str]
    available_file_context_refs: List[str]
    available_file_context: bool
    execute_context_updates: Dict[str, Any]
    initial_runtime_recent_action: Dict[str, Any]
    user_content: List[Dict[str, Any]]


@dataclass(frozen=True)
class ExecuteStepNormalizedRuntimeResult:
    """执行器原始输出的统一归一结果。

    业务含义：
    - `raw_message` 保留执行器原始返回，供后续 interrupt/completed 分支继续消费；
    - `runtime_recent_action` 是 execute 节点后续写回上下文的依据；
    - `normalized_execution` 只在非 interrupt 完成分支使用，用来构造 `StepOutcome`。
    """

    raw_message: Dict[str, Any]
    runtime_recent_action: Dict[str, Any]
    normalized_execution: Dict[str, Any]


@dataclass(frozen=True)
class ExecuteStepInterruptTransition:
    """interrupt 分支的状态写回结果。

    业务含义：
    - 本结构只描述 execute 节点在进入 wait 前要写回的 state updates；
    - 节点壳继续负责事件归并与 `_reduce_state_with_events` 返回，不把 graph 流转语义下沉到 helper。
    """

    updates: Dict[str, Any]


@dataclass(frozen=True)
class ExecuteStepCompletedTransition:
    """completed 分支的结果落账与 state 写回结果。

    字段说明：
    - `updates`: execute_step_node 最终写回 graph state 的增量；
    - `events`: 当前步骤需要落账的 started/tool/completed 事件；
    - `next_step_id`: 本轮完成后下一步的标识，供节点路由继续推进；
    - 计数字段仅用于日志与测试断言，不参与业务判定。
    """

    updates: Dict[str, Any]
    events: List[Any]
    next_step_id: Optional[str]
    step_success: bool
    step_summary: str
    artifact_count: int
    blocker_count: int
    open_question_count: int


def _tool_family(function_name: str) -> str:
    normalized_name = str(function_name or "").strip().lower()
    if not normalized_name:
        return ""
    if normalized_name.startswith("browser_"):
        return "browser"
    if normalized_name in {"search_web", "fetch_page"}:
        return "web"
    if normalized_name in {"read_file", "write_file", "list_files"}:
        return "file"
    if normalized_name.startswith("shell_"):
        return "shell"
    if normalized_name.startswith("message_") or normalized_name in {"ask_user", "notify_user"}:
        return "human"
    return normalized_name.split("_", 1)[0]


def _called_tool_function_names(tool_events: List[ToolEvent]) -> List[str]:
    function_names: List[str] = []
    for event in tool_events:
        event_status = getattr(event, "status", "")
        normalized_status = str(getattr(event_status, "value", event_status) or "")
        if normalized_status != "called":
            continue
        function_name = str(event.function_name or "").strip()
        if function_name:
            function_names.append(function_name)
    return function_names


def _build_entry_upgrade_payload(
        *,
        control: Dict[str, Any],
        reason_code: str,
        evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    contract_payload = get_entry_contract_payload(control)
    if not contract_payload:
        return None
    contract = EntryContract.model_validate(contract_payload)
    if contract.route != EntryRoute.ATOMIC_ACTION or not contract.upgrade_policy.allow_upgrade:
        return None
    return {
        "reason_code": reason_code,
        "source_route": EntryRoute.ATOMIC_ACTION.value,
        "target_route": EntryRoute.PLANNED_TASK.value,
        "evidence": evidence,
    }


def _summarize_entry_upgrade_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """压缩升级证据日志，保留定位信息，避免直接打印文件路径或问题原文。"""
    summarized: Dict[str, Any] = {}
    called_functions = evidence.get("called_functions")
    if isinstance(called_functions, list):
        summarized["called_functions"] = [str(item) for item in called_functions]
        summarized["called_function_count"] = len(called_functions)
    tool_families = evidence.get("tool_families")
    if isinstance(tool_families, list):
        summarized["tool_families"] = [str(item) for item in tool_families]
    if "called_function_count" in evidence:
        summarized["called_function_count"] = int(evidence.get("called_function_count") or 0)
    if "max_tool_calls_before_upgrade" in evidence:
        summarized["max_tool_calls_before_upgrade"] = int(evidence.get("max_tool_calls_before_upgrade") or 0)
    produced_artifacts = evidence.get("produced_artifacts")
    if isinstance(produced_artifacts, list):
        summarized["produced_artifact_count"] = len(produced_artifacts)
    open_questions = evidence.get("open_questions")
    if isinstance(open_questions, list):
        summarized["open_question_count"] = len(open_questions)
    interrupt_kind = str(evidence.get("interrupt_kind") or "").strip()
    if interrupt_kind:
        summarized["interrupt_kind"] = interrupt_kind
    return summarized


def _write_entry_upgrade(
        *,
        updated_control: Dict[str, Any],
        entry_upgrade: Optional[Dict[str, Any]],
        state: PlannerReActLangGraphState,
) -> None:
    """写入 atomic_action 升级信号，并记录可追溯日志。"""
    if entry_upgrade is None:
        return
    updated_control["entry_upgrade"] = entry_upgrade
    log_runtime(
        logger,
        logging.INFO,
        "入口原子动作升级信号写入",
        state=state,
        upgrade_reason_code=str(entry_upgrade.get("reason_code") or ""),
        source_route=str(entry_upgrade.get("source_route") or ""),
        target_route=str(entry_upgrade.get("target_route") or ""),
        upgrade_evidence=_summarize_entry_upgrade_evidence(
            entry_upgrade.get("evidence") if isinstance(entry_upgrade.get("evidence"), dict) else {}
        ),
    )


def _resolve_completed_entry_upgrade(
        *,
        control: Dict[str, Any],
        tool_events: List[ToolEvent],
        produced_artifacts: List[str],
        open_questions: List[str],
        normalized_execution: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    contract_payload = get_entry_contract_payload(control)
    if not contract_payload:
        return None
    contract = EntryContract.model_validate(contract_payload)
    if contract.route != EntryRoute.ATOMIC_ACTION or not contract.upgrade_policy.allow_upgrade:
        return None

    called_functions = _called_tool_function_names(tool_events)
    if (
            contract.upgrade_policy.upgrade_on_user_confirmation_required
            and any(function_name in {"message_ask_user", "ask_user"} for function_name in called_functions)
    ):
        return _build_entry_upgrade_payload(
            control=control,
            reason_code="user_confirmation_requires_planner",
            evidence={
                "called_functions": called_functions,
            },
        )
    tool_families = [
        family
        for family in [_tool_family(function_name) for function_name in called_functions]
        if family
    ]
    unique_tool_families = sorted(set(tool_families))
    if (
            contract.upgrade_policy.upgrade_on_second_tool_family
            and len(unique_tool_families) >= 2
    ):
        return _build_entry_upgrade_payload(
            control=control,
            reason_code="second_tool_family_requires_planner",
            evidence={
                "tool_families": unique_tool_families,
                "called_functions": called_functions,
            },
        )
    if len(called_functions) > contract.upgrade_policy.max_tool_calls_before_upgrade:
        return _build_entry_upgrade_payload(
            control=control,
            reason_code="tool_budget_exceeded_requires_planner",
            evidence={
                "called_function_count": len(called_functions),
                "max_tool_calls_before_upgrade": contract.upgrade_policy.max_tool_calls_before_upgrade,
                "called_functions": called_functions,
            },
        )
    if (
            contract.upgrade_policy.upgrade_on_file_output_required
            and produced_artifacts
    ):
        return _build_entry_upgrade_payload(
            control=control,
            reason_code="file_output_requires_planner",
            evidence={
                "produced_artifacts": produced_artifacts,
            },
        )
    if (
            contract.upgrade_policy.upgrade_on_open_questions
            and open_questions
            and bool(normalized_execution.get("success", True))
    ):
        return _build_entry_upgrade_payload(
            control=control,
            reason_code="open_questions_require_planner",
            evidence={
                "open_questions": open_questions,
            },
        )
    return None


async def _build_message(
        llm: LLM,
        user_message_prompt: str,
        input_parts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """按多模态能力统一构造执行节点 user content。"""
    if getattr(llm, "multimodal", False) and len(input_parts) > 0:
        multiplexed_message = await llm.format_multiplexed_message(input_parts)
        return [
            {"type": "text", "text": user_message_prompt},
            *multiplexed_message,
        ]
    return [{"type": "text", "text": user_message_prompt}]


def collect_recent_search_queries_from_tool_events(tool_events: List[ToolEvent]) -> List[str]:
    """从当前步骤 tool event 中提取搜索词，供 runtime_recent_action 回写。"""
    queries: List[str] = []
    for event in tool_events:
        if str(event.function_name or "").strip().lower() != "search_web":
            continue
        query = str((event.function_args or {}).get("query") or (event.function_args or {}).get("q") or "").strip()
        if not query or query in queries:
            continue
        queries.append(query)
    return queries


async def prepare_execute_step_input(
        *,
        state: PlannerReActLangGraphState,
        step: Step,
        llm: LLM,
        runtime_context_service: RuntimeContextService,
        task_mode: str,
        user_message: str,
) -> ExecuteStepPreparedInput:
    """装配执行阶段输入，不处理 graph state 写回。

    这里统一负责三类输入：
    - prompt 主体：用户消息、步骤描述、附件提示；
    - 执行上下文：workspace/context service 输出的阶段上下文；
    - 执行约束辅助信号：当前可用文件上下文、recent action 初始快照。

    `initial_runtime_recent_action` 保留“执行前”上下文快照，
    便于后续工具执行和完成写回在同一份基线之上累积结果，避免阶段信息丢失。
    """
    plan = state.get("plan")
    language = getattr(plan, "language", None) or "zh"
    working_memory = _ensure_working_memory(state)
    confirmed_fact_keys = sorted(
        list(
            _normalize_confirmed_fact_map(working_memory.get("confirmed_facts")).keys()
        )
    )
    step_execution_text = _build_step_execution_text(
        step,
        working_memory=working_memory,
    )
    input_parts = list(state.get("input_parts") or [])
    attachments = [str(part.get("sandbox_filepath") or "") for part in input_parts if part.get("sandbox_filepath")]
    available_file_context_refs = _collect_available_file_context_refs(state)
    available_file_context = has_available_file_context(
        user_message=user_message,
        attachment_paths=attachments,
        artifact_paths=available_file_context_refs,
    )
    user_message_prompt = EXECUTION_PROMPT.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
        language=language,
        step=step_execution_text,
    )
    execute_context_packet = await _build_prompt_context_packet_async(
        stage="execute",
        state=state,
        runtime_context_service=runtime_context_service,
        step=step,
        task_mode=task_mode,
    )
    execute_context_updates = _extract_prompt_context_state_updates(
        runtime_context_service=runtime_context_service,
        context_packet=execute_context_packet,
    )
    initial_runtime_recent_action = runtime_context_service.normalize_runtime_recent_action(
        execute_context_packet.get("recent_action_digest")
    )
    user_message_prompt = _append_prompt_context_to_prompt(user_message_prompt, execute_context_packet)
    user_content = await _build_message(llm, user_message_prompt, input_parts)
    return ExecuteStepPreparedInput(
        user_message=user_message,
        task_mode=task_mode,
        working_memory=working_memory,
        confirmed_fact_keys=confirmed_fact_keys,
        input_parts=input_parts,
        attachments=attachments,
        available_file_context_refs=available_file_context_refs,
        available_file_context=available_file_context,
        execute_context_updates=execute_context_updates,
        initial_runtime_recent_action=initial_runtime_recent_action,
        user_content=user_content,
    )


def build_timeout_execution_message(*, step: Step, timeout_seconds: int) -> Dict[str, Any]:
    """构造执行超时的标准失败消息。"""
    step_label = _build_step_label(step)
    return {
        "success": False,
        "summary": f"步骤执行超时：{step_label}",
        "attachments": [],
        "blockers": [f"当前步骤超过 {timeout_seconds} 秒未完成"],
        "next_hint": "请缩小当前步骤范围后重试",
    }


def build_empty_execution_message(*, step: Step) -> Dict[str, Any]:
    """构造执行器未返回内容时的标准失败消息。"""
    step_label = _build_step_label(step)
    return {
        "success": False,
        "summary": f"步骤执行失败：{step_label}",
        "attachments": [],
    }


def normalize_execute_runtime_result(
        *,
        llm_message: Optional[Dict[str, Any]],
        tool_events: List[ToolEvent],
        runtime_context_service: RuntimeContextService,
) -> ExecuteStepNormalizedRuntimeResult:
    """归一执行器输出，供 execute_step_node 后续分支消费。"""
    raw_message = dict(llm_message or {})
    runtime_recent_action = runtime_context_service.normalize_runtime_recent_action(
        raw_message.get("runtime_recent_action")
    )
    recent_search_queries = collect_recent_search_queries_from_tool_events(tool_events)
    if recent_search_queries:
        runtime_recent_action["recent_search_queries"] = recent_search_queries
    return ExecuteStepNormalizedRuntimeResult(
        raw_message=raw_message,
        runtime_recent_action=runtime_recent_action,
        normalized_execution=normalize_execution_response(raw_message),
    )


async def build_execute_interrupt_transition(
        *,
        state: PlannerReActLangGraphState,
        plan: Plan,
        step: Step,
        control: Dict[str, Any],
        runtime_context_service: RuntimeContextService,
        task_mode: str,
        execute_context_updates: Dict[str, Any],
        runtime_recent_action: Dict[str, Any],
        interrupt_request: Dict[str, Any],
        user_message: str,
) -> ExecuteStepInterruptTransition:
    """构造 execute 节点 interrupt 分支的 state updates。"""
    next_state = {
        **state,
        **runtime_context_service.merge_runtime_recent_action(
            state_updates=execute_context_updates,
            task_mode=task_mode,
            runtime_recent_action=runtime_recent_action,
        ),
        "plan": plan,
        "current_step_id": step.id,
        "pending_interrupt": interrupt_request,
    }
    interrupt_context_packet = await _build_prompt_context_packet_async(
        stage="execute",
        state=next_state,
        runtime_context_service=runtime_context_service,
        step=step,
        task_mode=task_mode,
    )
    interrupt_context_updates = runtime_context_service.merge_runtime_recent_action(
        state_updates=_extract_prompt_context_state_updates(
            runtime_context_service=runtime_context_service,
            context_packet=interrupt_context_packet,
        ),
        task_mode=task_mode,
        runtime_recent_action=runtime_recent_action,
    )
    updated_control = dict(control)
    updated_control["step_reuse_hit"] = False
    if interrupt_request:
        entry_upgrade = _build_entry_upgrade_payload(
            control=control,
            reason_code="user_confirmation_requires_planner",
            evidence={
                "interrupt_kind": str(interrupt_request.get("kind") or ""),
            },
        )
        _write_entry_upgrade(
            updated_control=updated_control,
            entry_upgrade=entry_upgrade,
            state=state,
        )
    return ExecuteStepInterruptTransition(
        updates={
            "plan": plan,
            **interrupt_context_updates,
            "current_step_id": step.id,
            "user_message": user_message,
            "graph_metadata": replace_control_metadata(state, updated_control),
            "pending_interrupt": interrupt_request,
        }
    )


async def build_execute_completed_transition(
        *,
        state: PlannerReActLangGraphState,
        plan: Plan,
        step: Step,
        control: Dict[str, Any],
        runtime_context_service: RuntimeContextService,
        task_mode: str,
        execute_context_updates: Dict[str, Any],
        runtime_recent_action: Dict[str, Any],
        normalized_execution: Dict[str, Any],
        started_event: Any,
        tool_events: List[ToolEvent],
        user_message: str,
        working_memory: Dict[str, Any],
        is_entry_wait_execute_step: bool,
) -> ExecuteStepCompletedTransition:
    """构造 execute 节点 completed 分支的结果落账与状态写回。"""
    step_success = bool(normalized_execution.get("success", True))
    step_label = _build_step_label(step)
    step_summary = normalize_step_result_text(
        normalized_execution.get("summary"),
        fallback=f"步骤执行完成：{step_label}" if step_success else f"步骤执行失败：{step_label}",
    )
    step_deliver_result_as_attachment = _infer_step_attachment_delivery_preference(
        user_message=user_message,
        normalized_execution=normalized_execution,
    )
    model_attachment_paths = normalize_attachments(normalized_execution.get("attachments"))
    tool_attachment_paths = extract_write_file_paths_from_tool_events(tool_events)
    step_attachment_paths = normalize_file_path_list(
        merge_attachment_paths(model_attachment_paths, tool_attachment_paths),
    )
    open_questions = normalize_text_list(normalized_execution.get("open_questions"))
    step.outcome = StepOutcome(
        done=step_success,
        summary=step_summary,
        produced_artifacts=step_attachment_paths,
        blockers=normalize_text_list(normalized_execution.get("blockers")),
        facts_learned=normalize_text_list(normalized_execution.get("facts_learned")),
        open_questions=open_questions,
        deliver_result_as_attachment=step_deliver_result_as_attachment,
        next_hint=normalize_step_result_text(normalized_execution.get("next_hint")),
    )
    step.status = ExecutionStatus.COMPLETED if step_success else ExecutionStatus.FAILED
    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=step.status,
    )
    events: List[Any] = [started_event, *tool_events, completed_event]
    next_step = plan.get_next_step()
    updated_control = dict(control)
    updated_control["step_reuse_hit"] = False
    entry_upgrade = _resolve_completed_entry_upgrade(
        control=control,
        tool_events=tool_events,
        produced_artifacts=step_attachment_paths,
        open_questions=open_questions,
        normalized_execution=normalized_execution,
    )
    _write_entry_upgrade(
        updated_control=updated_control,
        entry_upgrade=entry_upgrade,
        state=state,
    )
    updated_working_memory = _merge_step_outcome_into_working_memory(
        working_memory,
        step=step,
    )
    completed_context_packet = await _build_prompt_context_packet_async(
        stage="execute",
        state={
            **state,
            **runtime_context_service.merge_runtime_recent_action(
                state_updates=execute_context_updates,
                task_mode=task_mode,
                runtime_recent_action=runtime_recent_action,
            ),
            "plan": plan,
            "last_executed_step": step.model_copy(deep=True),
            "working_memory": updated_working_memory,
            "pending_interrupt": {},
            "emitted_events": append_events(state.get("emitted_events"), *events),
        },
        runtime_context_service=runtime_context_service,
        step=step,
        task_mode=task_mode,
    )
    completed_context_updates = runtime_context_service.merge_runtime_recent_action(
        state_updates=_extract_prompt_context_state_updates(
            runtime_context_service=runtime_context_service,
            context_packet=completed_context_packet,
        ),
        task_mode=task_mode,
        runtime_recent_action=runtime_recent_action,
    )
    next_step_id = str(next_step.id or "") if next_step is not None else ""
    return ExecuteStepCompletedTransition(
        updates={
            "plan": plan,
            **completed_context_updates,
            "last_executed_step": step.model_copy(deep=True),
            "execution_count": int(state.get("execution_count", 0)) + 1,
            "current_step_id": next_step.id if next_step is not None else None,
            "user_message": user_message,
            "working_memory": updated_working_memory,
            "graph_metadata": replace_control_metadata(state, updated_control),
            # Step 语义已收紧为执行摘要，不再覆盖最终正文；但 final_message 状态键仍需稳定保留。
            "final_message": str(state.get("final_message") or ""),
            "selected_artifacts": list(state.get("selected_artifacts") or []),
            "pending_interrupt": {},
        },
        events=events,
        next_step_id=next_step_id or None,
        step_success=step_success,
        step_summary=step_summary,
        artifact_count=len(step_attachment_paths),
        blocker_count=len(list(step.outcome.blockers or [])),
        open_question_count=len(list(step.outcome.open_questions or [])),
    )
