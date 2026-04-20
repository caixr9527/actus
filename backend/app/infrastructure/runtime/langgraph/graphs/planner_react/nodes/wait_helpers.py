#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层等待恢复 helper。

本模块只负责 human_wait/direct_wait 的恢复消息归一、分支判断、恢复结果构造，
以及恢复后补出的执行步骤文案构造，不决定 wait 节点的整体路由。
"""

from typing import Any, Dict, Literal, Optional

from app.domain.models import (
    ExecutionStatus,
    Step,
    StepDeliveryContextState,
    StepOutcome,
    StepTaskModeHint,
    normalize_wait_payload,
    resolve_wait_resume_message,
)
from app.domain.services.runtime.normalizers import normalize_controlled_value
from app.domain.services.workspace_runtime.policies import classify_confirmed_user_task_mode


def _build_post_wait_execute_step(
        *,
        original_user_message: str,
        resumed_message: str,
        waiting_step: Optional[Step],
) -> Step:
    """为“只等待用户补充信息”的计划补一条真实执行步骤，避免恢复后掉回 replan。"""
    execution_message = str(original_user_message or "").strip()
    normalized_resume_message = str(resumed_message or "").strip()
    title = execution_message[:40] if execution_message else "根据用户补充信息继续执行"
    description = execution_message or "根据用户最新补充的信息继续执行原始任务"
    waiting_step_mode = normalize_controlled_value(
        getattr(waiting_step, "task_mode_hint", None),
        StepTaskModeHint,
    )
    task_mode = _infer_post_wait_task_mode(
        original_user_message=execution_message,
        resumed_message=normalized_resume_message,
        waiting_step_mode=waiting_step_mode,
    )
    return Step(
        title=title,
        description=description,
        task_mode_hint=task_mode,
        output_mode="inline",
        artifact_policy="default",
        delivery_role="final",
        delivery_context_state=_resolve_direct_delivery_context_state(task_mode),
        status=ExecutionStatus.PENDING,
    )

def _infer_post_wait_task_mode(
        *,
        original_user_message: str,
        resumed_message: str,
        waiting_step_mode: Optional[str],
) -> str:
    """恢复后补出的真实执行步骤应继承原任务语义，而不是沿用 human_wait。"""
    if waiting_step_mode and waiting_step_mode != StepTaskModeHint.HUMAN_WAIT.value:
        return waiting_step_mode
    candidate_message = str(original_user_message or "").strip() or str(resumed_message or "").strip()
    inferred_mode = classify_confirmed_user_task_mode(candidate_message)
    if inferred_mode == StepTaskModeHint.HUMAN_WAIT.value:
        return StepTaskModeHint.GENERAL.value
    return inferred_mode or StepTaskModeHint.GENERAL.value

def _resolve_direct_delivery_context_state(task_mode: str) -> str:
    """直达路径下，只有纯 general 任务才可直接组织最终正文，其余模式需先准备上下文。"""
    normalized_task_mode = normalize_controlled_value(task_mode, StepTaskModeHint)
    if normalized_task_mode == StepTaskModeHint.GENERAL.value:
        return StepDeliveryContextState.READY.value
    return StepDeliveryContextState.NEEDS_PREPARATION.value

def _normalize_interrupt_request(raw: Any) -> Dict[str, Any]:
    return normalize_wait_payload(raw)

def _resume_value_to_message(payload: Dict[str, Any], value: Any) -> str:
    return resolve_wait_resume_message(payload, value)

def _build_step_label(step: Step, default: str = "当前步骤") -> str:
    return str(step.title or step.description or default).strip() or default

def _build_wait_resume_step_summary(step: Step, resumed_message: str) -> str:
    step_label = _build_step_label(step)
    normalized_message = str(resumed_message or "").strip()
    if normalized_message:
        return f"{step_label}已收到用户回复：{normalized_message}"
    return f"{step_label}已完成用户交互"

def _build_wait_cancel_step_summary(step: Step, resumed_message: str) -> str:
    step_label = _build_step_label(step)
    normalized_message = str(resumed_message or "").strip()
    if normalized_message:
        return f"{step_label}已被用户取消：{normalized_message}"
    return f"{step_label}已被用户取消，等待重新规划"

def _resolve_wait_resume_branch(
        payload: Dict[str, Any],
        resume_value: Any,
) -> Literal["confirm_continue", "confirm_cancel", "select", "input_text"]:
    """显式区分不同等待态恢复分支，避免不同 UI 交互共用一条模糊路径。"""
    kind = str(payload.get("kind") or "").strip()
    if kind == "confirm":
        if resume_value == payload.get("cancel_resume_value"):
            return "confirm_cancel"
        return "confirm_continue"
    if kind == "select":
        return "select"
    return "input_text"

def _build_wait_resume_outcome(
        step: Step,
        *,
        branch: Literal["confirm_continue", "select", "input_text"],
        resumed_message: str,
) -> StepOutcome:
    """按恢复分支生成步骤结果，保证确认/选择/文本输入语义清晰。"""
    step_label = _build_step_label(step)
    normalized_message = str(resumed_message or "").strip()

    if branch == "confirm_continue":
        if normalized_message:
            summary = f"{step_label}已确认继续：{normalized_message}"
        else:
            summary = f"{step_label}已确认继续执行"
    elif branch == "select":
        if normalized_message:
            summary = f"{step_label}已收到用户选择：{normalized_message}"
        else:
            summary = f"{step_label}已完成用户选择"
    else:
        if normalized_message:
            summary = f"{step_label}已收到用户输入：{normalized_message}"
        else:
            summary = _build_wait_resume_step_summary(step, resumed_message)

    return StepOutcome(
        done=True,
        summary=summary,
    )
