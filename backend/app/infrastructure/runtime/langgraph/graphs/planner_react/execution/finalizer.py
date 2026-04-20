#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具循环收口与统一返回。"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.domain.models import Step
from app.domain.services.runtime.contracts.runtime_logging import elapsed_ms, log_runtime
from app.domain.services.runtime.normalizers import normalize_execution_response
from app.domain.services.workspace_runtime.policies import (
    build_human_wait_missing_interrupt_payload as _build_human_wait_missing_interrupt_payload,
    build_loop_break_payload as _build_loop_break_payload,
    normalize_execution_payload as _normalize_execution_payload,
)
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import (
    normalize_attachments,
    safe_parse_json,
)
from ..convergence.judge import ConvergenceJudge


@dataclass(slots=True)
class NoToolCallFinalizationResult:
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
    parsed = safe_parse_json(llm_message.get("content"))
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
    has_delivery_text = bool(str(parsed_execution.get("delivery_text") or "").strip())
    # P3-1A 收敛修复：无显式 success 字段时不再盲目按 True 处理，避免“空响应被判成功”。
    has_explicit_success = isinstance(parsed, dict) and "success" in parsed
    inferred_success = bool(parsed.get("success")) if has_explicit_success else bool(has_summary or has_delivery_text)
    if inferred_success and not has_summary and not has_delivery_text:
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
    return NoToolCallFinalizationResult(
        action="return",
        payload=_normalize_execution_payload(
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
        ),
        parsed=parsed,
    )


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
    # P3 解耦：先尝试按关键事实收敛，避免 file_processing final inline 在事实已满足时误判失败。
    converged_payload = ConvergenceJudge.build_max_iteration_convergence_payload(
        step=step,
        task_mode=task_mode,
        runtime_recent_action=runtime_recent_action,
        step_file_context=dict(step_file_context or {}),
    )
    if converged_payload is not None:
        log_runtime(
            logger,
            logging.INFO,
            "达到最大工具轮次但关键事实已满足，按成功收敛",
            step_id=str(step.id or ""),
            task_mode=task_mode,
            requested_max_tool_iterations=requested_max_tool_iterations,
            iteration_count=iteration_count,
            elapsed_ms=elapsed_ms(started_at),
        )
        return converged_payload
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
