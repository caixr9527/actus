#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""步骤结构化契约编译策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from app.domain.models import (
    Step,
    StepArtifactPolicy,
    StepDeliveryContextState,
    StepDeliveryRole,
    StepOutputMode,
    StepTaskModeHint,
)
from app.domain.services.runtime.normalizers import normalize_controlled_value
from .task_mode_policy import analyze_text_intent, build_step_candidate_text


@dataclass(frozen=True)
class StepContractCompilationIssue:
    """步骤契约编译问题。"""

    step_id: str
    issue_code: str
    issue_message: str


def compile_step_contracts(
    *,
    steps: List[Step],
    user_message: str,
) -> Tuple[List[Step], List[StepContractCompilationIssue], int]:
    """编译步骤结构化契约；返回编译后的步骤、不可纠偏问题与纠偏次数。"""
    compiled_steps: List[Step] = []
    issues: List[StepContractCompilationIssue] = []
    corrected_count = 0

    for step in list(steps or []):
        compiled_step, step_issues, step_corrected = _compile_single_step_contract(
            step=step,
            user_message=user_message,
        )
        compiled_step = _normalize_step_contract_enum_fields(compiled_step)
        compiled_steps.append(compiled_step)
        issues.extend(step_issues)
        corrected_count += step_corrected

    return compiled_steps, issues, corrected_count


def collect_step_contract_hard_issues(*, steps: List[Step]) -> List[StepContractCompilationIssue]:
    """收集不可自动纠偏的硬错误。"""
    issues: List[StepContractCompilationIssue] = []
    for step in list(steps or []):
        step_id = str(getattr(step, "id", "") or "")
        output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode) or ""
        artifact_policy = normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy) or ""
        delivery_role = normalize_controlled_value(getattr(step, "delivery_role", None), StepDeliveryRole) or ""
        task_mode = normalize_controlled_value(getattr(step, "task_mode_hint", None), StepTaskModeHint) or ""

        if output_mode == StepOutputMode.FILE.value and artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT.value:
            issues.append(
                StepContractCompilationIssue(
                    step_id=step_id,
                    issue_code="output_artifact_conflict",
                    issue_message="output_mode=file 与 artifact_policy=forbid_file_output 互斥。",
                )
            )
        if task_mode == StepTaskModeHint.HUMAN_WAIT.value and delivery_role == StepDeliveryRole.FINAL.value:
            issues.append(
                StepContractCompilationIssue(
                    step_id=step_id,
                    issue_code="human_wait_final_conflict",
                    issue_message="human_wait 步骤不能承担最终交付角色。",
                )
            )
    return issues


def _compile_single_step_contract(
    *,
    step: Step,
    user_message: str,
) -> Tuple[Step, List[StepContractCompilationIssue], int]:
    compiled_step = step.model_copy(deep=True)
    issues: List[StepContractCompilationIssue] = []
    corrected_count = 0

    candidate_text = build_step_candidate_text(compiled_step)
    signals = analyze_text_intent(candidate_text)
    step_requests_write_action = bool(signals.get("has_write_action_signal"))
    _ = user_message  # P3-一次性收口：保留函数签名，当前编译策略仅依赖步骤语义。

    task_mode = normalize_controlled_value(getattr(compiled_step, "task_mode_hint", None), StepTaskModeHint) or ""
    output_mode = normalize_controlled_value(getattr(compiled_step, "output_mode", None), StepOutputMode) or ""
    artifact_policy = normalize_controlled_value(getattr(compiled_step, "artifact_policy", None), StepArtifactPolicy) or ""
    delivery_role = normalize_controlled_value(getattr(compiled_step, "delivery_role", None), StepDeliveryRole) or ""
    delivery_context_state = normalize_controlled_value(
        getattr(compiled_step, "delivery_context_state", None),
        StepDeliveryContextState,
    ) or ""

    # P3-一次性收口：human_wait 步骤必须是纯等待，不允许附带文件产出或最终正文职责。
    if task_mode == StepTaskModeHint.HUMAN_WAIT.value:
        if output_mode != StepOutputMode.NONE.value:
            compiled_step.output_mode = StepOutputMode.NONE
            corrected_count += 1
        if artifact_policy != StepArtifactPolicy.FORBID_FILE_OUTPUT.value:
            compiled_step.artifact_policy = StepArtifactPolicy.FORBID_FILE_OUTPUT
            corrected_count += 1
        if delivery_role != StepDeliveryRole.NONE.value:
            compiled_step.delivery_role = StepDeliveryRole.NONE
            corrected_count += 1
        if delivery_context_state != StepDeliveryContextState.NONE.value:
            compiled_step.delivery_context_state = StepDeliveryContextState.NONE
            corrected_count += 1
        return compiled_step, issues, corrected_count

    # P3-一次性收口：如果步骤明确写副作用，但策略禁止文件产出，按可执行语义纠偏。
    if artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT.value and step_requests_write_action:
        if task_mode in {
            StepTaskModeHint.FILE_PROCESSING.value,
            StepTaskModeHint.CODING.value,
            StepTaskModeHint.GENERAL.value,
        }:
            compiled_step.artifact_policy = StepArtifactPolicy.ALLOW_FILE_OUTPUT
            corrected_count += 1
            if output_mode in {"", StepOutputMode.NONE.value}:
                compiled_step.output_mode = StepOutputMode.FILE
                corrected_count += 1

    # P3-一次性收口：output_mode=file 与 forbid_file_output 互斥时统一纠偏到 allow。
    output_mode = normalize_controlled_value(getattr(compiled_step, "output_mode", None), StepOutputMode) or ""
    artifact_policy = normalize_controlled_value(getattr(compiled_step, "artifact_policy", None), StepArtifactPolicy) or ""
    if output_mode == StepOutputMode.FILE.value and artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT.value:
        compiled_step.artifact_policy = StepArtifactPolicy.ALLOW_FILE_OUTPUT
        corrected_count += 1

    # P3-一次性收口：最终正文步骤至少需要 inline 输出。
    delivery_role = normalize_controlled_value(getattr(compiled_step, "delivery_role", None), StepDeliveryRole) or ""
    output_mode = normalize_controlled_value(getattr(compiled_step, "output_mode", None), StepOutputMode) or ""
    if delivery_role == StepDeliveryRole.FINAL.value and output_mode != StepOutputMode.INLINE.value:
        compiled_step.output_mode = StepOutputMode.INLINE
        corrected_count += 1

    return compiled_step, issues, corrected_count


def _normalize_step_contract_enum_fields(step: Step) -> Step:
    """P3-一次性收口：编译出口统一规范结构化字段类型为 Enum，禁止字符串回流到持久化层。"""
    normalized_step = step.model_copy(deep=True)
    raw_task_mode_hint = normalize_controlled_value(getattr(normalized_step, "task_mode_hint", None), StepTaskModeHint)
    raw_output_mode = normalize_controlled_value(getattr(normalized_step, "output_mode", None), StepOutputMode)
    raw_artifact_policy = normalize_controlled_value(getattr(normalized_step, "artifact_policy", None), StepArtifactPolicy)
    raw_delivery_role = normalize_controlled_value(getattr(normalized_step, "delivery_role", None), StepDeliveryRole)
    raw_delivery_context_state = normalize_controlled_value(
        getattr(normalized_step, "delivery_context_state", None),
        StepDeliveryContextState,
    )
    normalized_step.task_mode_hint = StepTaskModeHint(raw_task_mode_hint) if raw_task_mode_hint else None
    normalized_step.output_mode = StepOutputMode(raw_output_mode) if raw_output_mode else None
    normalized_step.artifact_policy = StepArtifactPolicy(raw_artifact_policy) if raw_artifact_policy else None
    normalized_step.delivery_role = StepDeliveryRole(raw_delivery_role) if raw_delivery_role else None
    normalized_step.delivery_context_state = (
        StepDeliveryContextState(raw_delivery_context_state)
        if raw_delivery_context_state
        else None
    )
    return normalized_step
