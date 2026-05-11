#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具循环收口与统一返回。"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.domain.models import Step, StepArtifactPolicy, StepOutputMode
from app.domain.services.runtime.contracts.runtime_logging import elapsed_ms, log_runtime
from app.domain.services.runtime.normalizers import normalize_controlled_value, normalize_execution_response
from app.domain.services.workspace_runtime.policies import (
    build_human_wait_missing_interrupt_payload as _build_human_wait_missing_interrupt_payload,
    build_loop_break_payload as _build_loop_break_payload,
    normalize_execution_payload as _normalize_execution_payload,
)
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    extract_text_outside_json_blocks,
    normalize_attachments,
    safe_parse_json,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence import WebReadingConvergenceJudge
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.contracts import (
    MaxIterationConvergenceContext,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.engine import ConvergenceEngine


@dataclass(slots=True)
class NoToolCallFinalizationResult:
    """无工具调用场景的统一收口结果。

    - `action=return` 表示当前轮可以直接收口；
    - `action=retry` 表示当前轮结果不足，外层工具循环应继续尝试；
    - `payload` 是最终返回给执行节点的标准化执行结果。
    """

    action: str
    payload: Optional[Dict[str, Any]] = None
    parsed: Dict[str, Any] = field(default_factory=dict)


def finalize_no_tool_call(
        *,
        logger: logging.Logger,
        step: Step,
        task_mode: str,
        llm_message: Dict[str, Any],
        llm_cost_ms: int,
        started_at: float,
        iteration: int,
        runtime_recent_action: Optional[Dict[str, Any]] = None,
) -> NoToolCallFinalizationResult:
    raw_llm_content = str(llm_message.get("content") or "")
    parsed = safe_parse_json(raw_llm_content)
    extra_response_text = extract_text_outside_json_blocks(raw_llm_content)
    parsed_execution = normalize_execution_response(parsed)
    if task_mode == "human_wait":
        log_runtime(
            logger,
            logging.WARNING,
            "等待步骤未发起 interrupt，已拒绝直接完成",
            step_id=str(step.id or ""),
            iteration=iteration,
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
        return NoToolCallFinalizationResult(
            action="return",
            payload=_build_human_wait_missing_interrupt_payload(
                step,
                runtime_recent_action=runtime_recent_action,
            ),
            parsed=parsed,
        )

    has_summary = bool(str(parsed_execution.get("summary") or "").strip())
    has_structured_progress = bool(
        normalize_attachments(parsed_execution.get("attachments"))
        or list(parsed_execution.get("blockers") or [])
        or list(parsed_execution.get("facts_learned") or [])
        or list(parsed_execution.get("open_questions") or [])
    )
    # P3-1A 收敛修复：无显式 success 字段时不再盲目按 True 处理，避免“空响应被判成功”。
    has_explicit_success = isinstance(parsed, dict) and "success" in parsed
    inferred_success = bool(parsed.get("success")) if has_explicit_success else bool(has_summary or has_structured_progress)
    if inferred_success and not has_summary and not has_structured_progress:
        log_runtime(
            logger,
            logging.WARNING,
            "模型结果为空，准备重试",
            step_id=str(step.id or ""),
            iteration=iteration,
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
        return NoToolCallFinalizationResult(action="retry", parsed=parsed)
    if inferred_success and not WebReadingConvergenceJudge.should_allow_model_only_success(
            task_mode=task_mode,
            runtime_recent_action=runtime_recent_action,
    ):
        log_runtime(
            logger,
            logging.WARNING,
            "网页阅读步骤未满足页面证据合同，拒绝无工具直接完成",
            step_id=str(step.id or ""),
            iteration=iteration,
            task_mode=task_mode,
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
        return NoToolCallFinalizationResult(action="retry", parsed=parsed)

    protected_file_payload = _build_successful_file_tool_payload(
        step=step,
        runtime_recent_action=runtime_recent_action,
    )
    file_write_result_required = _step_requires_file_write_result(step)
    protected_write_payload = (
        protected_file_payload
        if _is_successful_write_tool_payload(protected_file_payload)
        else None
    )
    if inferred_success and file_write_result_required and protected_write_payload is None:
        log_runtime(
            logger,
            logging.WARNING,
            "文件产出步骤拒绝无工具伪成功",
            step_id=str(step.id or ""),
            iteration=iteration,
            reason_code="file_output_requires_successful_write_tool",
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
        return NoToolCallFinalizationResult(
            action="return",
            payload=_build_loop_break_payload(
                step=step,
                blocker="当前步骤要求产出文件，但本轮没有成功的 write_file 或 replace_in_file 工具事实，已拒绝模型自报文件成功。",
                next_hint="请调用文件写入工具完成产物落地；如果不需要文件产物，应重新规划为 output_mode=none。",
                runtime_recent_action={
                    **dict(runtime_recent_action or {}),
                    "reason_code": "file_output_requires_successful_write_tool",
                },
            ),
            parsed=parsed,
        )
    if not inferred_success and file_write_result_required and protected_write_payload is None:
        log_runtime(
            logger,
            logging.WARNING,
            "文件产出步骤缺少成功写工具事实",
            step_id=str(step.id or ""),
            iteration=iteration,
            reason_code="file_output_requires_successful_write_tool",
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
        return NoToolCallFinalizationResult(
            action="return",
            payload=_build_loop_break_payload(
                step=step,
                blocker="当前步骤要求产出文件，但本轮没有成功的 write_file 或 replace_in_file 工具事实，无法确认文件已落地。",
                next_hint="请调用文件写入工具完成产物落地；如果不需要文件产物，应重新规划为 output_mode=none。",
                runtime_recent_action={
                    **dict(runtime_recent_action or {}),
                    "reason_code": "file_output_requires_successful_write_tool",
                },
            ),
            parsed=parsed,
        )
    if file_write_result_required:
        protected_file_payload = protected_write_payload
    if protected_file_payload is not None and not inferred_success:
        log_runtime(
            logger,
            logging.INFO,
            "已有成功文件工具事实，保护步骤成功收敛",
            step_id=str(step.id or ""),
            iteration=iteration,
            reason_code="successful_file_tool_result_preserved",
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
        )
        return NoToolCallFinalizationResult(
            action="return",
            payload=protected_file_payload,
            parsed=parsed,
        )

    log_runtime(
        logger,
        logging.INFO,
        "未调用工具直接完成当前轮次",
        step_id=str(step.id or ""),
        iteration=iteration,
        success=inferred_success,
        has_explicit_success=has_explicit_success,
        attachment_count=len(normalize_attachments(parsed.get("attachments"))),
        llm_elapsed_ms=llm_cost_ms,
        elapsed_ms=elapsed_ms(started_at),
    )
    normalized_payload = _normalize_execution_payload(
        {
            **parsed,
            "success": inferred_success,
            "runtime_recent_action": runtime_recent_action or {},
        },
        default_summary=(
            f"步骤执行完成：{step.description}"
            if inferred_success
            else f"步骤暂未完成：{step.description}"
        ),
    )
    return NoToolCallFinalizationResult(
        action="return",
        payload=normalized_payload,
        parsed=parsed,
    )


def _step_requires_file_write_result(step: Step) -> bool:
    output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
    artifact_policy = normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy)
    return (
        output_mode == StepOutputMode.FILE.value
        or artifact_policy == StepArtifactPolicy.REQUIRE_FILE_OUTPUT.value
    )


def _is_successful_write_tool_payload(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    recent_action = payload.get("runtime_recent_action")
    if not isinstance(recent_action, dict):
        return False
    convergence = recent_action.get("file_tool_success_convergence")
    if not isinstance(convergence, dict):
        return False
    function_name = str(convergence.get("function_name") or "").strip().lower()
    return function_name in {"write_file", "replace_in_file"}


def _build_successful_file_tool_payload(
        *,
        step: Step,
        runtime_recent_action: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """已有成功文件工具结果时，禁止后续自然语言解析失败反向覆盖成功事实。"""
    recent_action = dict(runtime_recent_action or {})
    last_success = dict(recent_action.get("last_successful_tool_call") or {})
    function_name = str(last_success.get("function_name") or "").strip().lower()
    if function_name not in {"list_files", "find_files", "read_file", "search_in_file", "write_file", "replace_in_file"}:
        return None
    data = last_success.get("data")
    if not isinstance(data, dict):
        data = {}
    evidence_lines = _build_file_tool_evidence_lines(function_name=function_name, data=data)
    if len(evidence_lines) == 0:
        message = str(last_success.get("message") or "").strip()
        if message:
            evidence_lines = [message]
    if len(evidence_lines) == 0:
        return None
    runtime_action = dict(recent_action)
    runtime_action["file_tool_success_convergence"] = {
        "reason_code": "successful_file_tool_result_preserved",
        "function_name": function_name,
    }
    summary = f"当前步骤已基于成功文件工具结果完成：{step.description}"
    attachments: List[str] = []
    if function_name in {"write_file", "replace_in_file"}:
        filepath = str(data.get("filepath") or data.get("path") or "").strip()
        if filepath:
            attachments.append(filepath)
    result = "\n".join(evidence_lines)
    return {
        "success": True,
        "summary": summary,
        "result": result,
        "attachments": attachments,
        "blockers": [],
        "facts_learned": evidence_lines,
        "open_questions": [],
        "next_hint": "",
        "runtime_recent_action": runtime_action,
    }


def _build_file_tool_evidence_lines(*, function_name: str, data: Dict[str, Any]) -> List[str]:
    evidence_lines: List[str] = []
    if function_name == "list_files":
        dir_path = str(data.get("dir_path") or "").strip()
        if dir_path:
            evidence_lines.append(f"当前目录：{dir_path}")
        file_names = _extract_file_names(data.get("files"))
        if file_names:
            evidence_lines.append("文件列表：" + "、".join(file_names))
    elif function_name == "find_files":
        dir_path = str(data.get("dir_path") or "").strip()
        if dir_path:
            evidence_lines.append(f"搜索目录：{dir_path}")
        file_names = _extract_file_names(data.get("files"))
        if file_names:
            evidence_lines.append("匹配文件：" + "、".join(file_names))
    elif function_name == "read_file":
        filepath = str(data.get("filepath") or data.get("path") or "").strip()
        content = str(data.get("content") or "")
        if filepath:
            evidence_lines.append(f"已读取文件：{filepath}")
        if content:
            evidence_lines.append(f"读取内容长度：{len(content)} 字符")
    elif function_name == "search_in_file":
        filepath = str(data.get("filepath") or data.get("path") or "").strip()
        matches = data.get("matches")
        if filepath:
            evidence_lines.append(f"已搜索文件：{filepath}")
        if isinstance(matches, list):
            evidence_lines.append(f"匹配数量：{len(matches)}")
    elif function_name in {"write_file", "replace_in_file"}:
        filepath = str(data.get("filepath") or data.get("path") or "").strip()
        if filepath:
            evidence_lines.append(f"已写入文件：{filepath}")
    return _dedupe_non_empty(evidence_lines)


def _extract_file_names(raw_files: Any) -> List[str]:
    file_names: List[str] = []
    if not isinstance(raw_files, list):
        return file_names
    for item in raw_files:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("filename") or item.get("path") or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            file_names.append(name)
    return _dedupe_non_empty(file_names)


def _dedupe_non_empty(values: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def finalize_max_iterations(
        *,
        logger: logging.Logger,
        step: Step,
        task_mode: str,
        llm_message: Dict[str, Any],
        started_at: float,
        requested_max_tool_iterations: int,
        iteration_count: int,
        runtime_recent_action: Optional[Dict[str, Any]] = None,
        step_file_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """处理达到最大工具轮次后的统一收口。

    处理顺序：
    1. 先按 research/web_reading/general/common 四层收敛器尝试复用现有证据；
    2. 全部失败后再返回明确未完成的 loop break 结果；
    3. 本函数只决定“如何收口”，不负责继续发起新一轮工具调用。
    """
    parsed = safe_parse_json(llm_message.get("content"))
    if task_mode == "human_wait":
        log_runtime(
            logger,
            logging.WARNING,
            "等待步骤达到最大轮次仍未进入等待",
            step_id=str(step.id or ""),
            requested_max_tool_iterations=requested_max_tool_iterations,
            iteration_count=iteration_count,
            elapsed_ms=elapsed_ms(started_at),
        )
        return _build_human_wait_missing_interrupt_payload(
            step,
            runtime_recent_action=runtime_recent_action,
        )

    log_runtime(
        logger,
        logging.INFO,
        "达到最大工具轮次，步骤判定未完成",
        step_id=str(step.id or ""),
        requested_max_tool_iterations=requested_max_tool_iterations,
        iteration_count=iteration_count,
        task_mode=task_mode,
        loop_break_reason="max_tool_iterations",
        attachment_count=len(normalize_attachments(parsed.get("attachments"))),
        elapsed_ms=elapsed_ms(started_at),
    )
    convergence_decision = ConvergenceEngine(logger=logger).evaluate_max_iteration(
        context=MaxIterationConvergenceContext(
            step=step,
            task_mode=task_mode,
            requested_max_tool_iterations=requested_max_tool_iterations,
            iteration_count=iteration_count,
            runtime_recent_action=runtime_recent_action,
            step_file_context=dict(step_file_context or {}),
        ),
        started_at=started_at,
    )
    if convergence_decision.should_break and convergence_decision.payload is not None:
        return convergence_decision.payload
    # P3-1A 收敛修复：max_tool_iterations 到达后一律按未完成收敛，不再返回 success=true。
    return _build_loop_break_payload(
        step=step,
        blocker="达到最大工具调用轮次，当前步骤仍未形成可交付结果。",
        next_hint=_build_max_iteration_next_hint(
            task_mode=task_mode,
            runtime_recent_action=runtime_recent_action,
        ),
        runtime_recent_action=runtime_recent_action,
    )


def _build_max_iteration_next_hint(
        *,
        task_mode: str,
        runtime_recent_action: Optional[Dict[str, Any]],
) -> str:
    """为最大轮次收口补充下一轮操作建议。"""
    default_hint = "请缩小步骤范围、改用更合适工具，或让模型直接给出当前可确认的结果。"
    if task_mode not in {"research", "web_reading"}:
        return default_hint
    recent_action = dict(runtime_recent_action or {})
    research_progress = dict(recent_action.get("research_progress") or {})
    missing_signals = [str(item).strip() for item in list(research_progress.get("missing_signals") or []) if
                       str(item).strip()]
    if len(missing_signals) == 0:
        return default_hint
    extra_hint = ""
    if bool(research_progress.get("is_low_recall")):
        extra_hint = " 请先用单主题自然语言短句检索，必要时逐轮仅增加一个筛选条件。"
    return (
            "请先补齐研究缺口后再继续："
            + "；".join(missing_signals[:2])
            + "。若当前证据已足够，请直接基于已抓取内容输出结果。"
            + extra_hint
    )
