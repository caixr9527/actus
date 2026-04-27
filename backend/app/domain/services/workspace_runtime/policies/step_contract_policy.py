#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""步骤结构化契约编译策略。

本模块负责在 planner / replan 之后，对步骤做一次统一收紧：
- 纠正常见的结构化字段冲突；
- 收紧 human_wait / 文件产出 / 写副作用 等结构化字段；
- 保证进入执行链的步骤至少具备可执行的契约语义。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple

from app.domain.models import (
    Step,
    StepArtifactPolicy,
    StepOutputMode,
    StepTaskModeHint,
)
from app.domain.services.runtime.contracts.langgraph_settings import EXPLICIT_FILE_OUTPUT_REQUEST_PATTERN
from app.domain.services.runtime.normalizers import normalize_controlled_value
from .task_mode_policy import analyze_text_intent, build_step_candidate_text, has_environment_write_intent


FINAL_USER_DELIVERY_STEP_PATTERN = re.compile(
    r"(直接向用户输出|输出给用户|向用户(展示|呈现|交付)|展示给用户|等待用户最终确认|最终(回答|答案|说明|报告|正文|交付|确认)|完整(回答|攻略|方案|报告)|面向用户)",
    re.IGNORECASE,
)
EN_FINAL_USER_DELIVERY_STEP_PATTERN = re.compile(
    r"\b("
    r"deliver|present|show|send|respond|reply"
    r")\b.{0,24}\b("
    r"to the user|final answer|final response|final summary|final report|final delivery"
    r")\b"
    r"|\b("
    r"final answer|final response|final summary|final report|final delivery"
    r")\b.{0,24}\b("
    r"to the user|for the user"
    r")\b",
    re.IGNORECASE,
)
SUMMARY_ONLY_STEP_PATTERN = re.compile(
    r"((基于|根据).{0,12}(已收集|全部已收集|已有|搜索摘要|检索结果).{0,30}(整理|归纳|汇总|总结|撰写|形成|输出).{0,20}(要点|方案|计划|回答|说明|报告|正文|成稿))"
    r"|((整理|归纳|汇总|总结).{0,30}(为|成|出).{0,8}(\d+|[一二三四五六七八九十]+)\s*(条|个).{0,12}(要点|模式|方案|结论|建议))",
    re.IGNORECASE,
)
EN_SUMMARY_ONLY_STEP_PATTERN = re.compile(
    r"\b("
    r"summarize|organize|compile|synthesize|write|draft|compose"
    r")\b.{0,36}\b("
    r"collected|gathered|existing|search results|research findings|all findings"
    r")\b.{0,36}\b("
    r"answer|response|summary|report|recommendations|conclusion"
    r")\b"
    r"|\b("
    r"based on|using"
    r")\b.{0,36}\b("
    r"collected|gathered|existing|search results|research findings|all findings"
    r")\b.{0,36}\b("
    r"summarize|organize|compile|synthesize|write|draft|compose"
    r")\b",
    re.IGNORECASE,
)
SUMMARY_FILE_OUTPUT_STEP_PATTERN = re.compile(
    r"((整理|归纳|汇总|总结).{0,30}(导出|保存|写入|生成).{0,12}(markdown|md|文件|文档|报告))",
    re.IGNORECASE,
)
EN_SUMMARY_FILE_OUTPUT_STEP_PATTERN = re.compile(
    r"\b("
    r"summarize|organize|compile|synthesize"
    r")\b.{0,36}\b("
    r"export|save|write|generate"
    r")\b.{0,20}\b("
    r"markdown|md|file|document|report"
    r")\b",
    re.IGNORECASE,
)
FILE_OUTPUT_ACTION_PATTERN = re.compile(
    r"(生成|写入|保存|导出|创建|产出).{0,24}(markdown|md|文件|文档|报告|report)",
    re.IGNORECASE,
)
EN_FILE_OUTPUT_ACTION_PATTERN = re.compile(
    r"\b(generate|write|save|export|create|produce)\b.{0,24}\b(markdown|md|file|document|report)\b",
    re.IGNORECASE,
)


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
    """收集不可自动纠偏的硬错误。

    当前只保留结构化字段级硬冲突，不在这里对“最后一步是否像总结”做文本语义封杀。
    """
    issues: List[StepContractCompilationIssue] = []
    for step in list(steps or []):
        step_id = str(getattr(step, "id", "") or "")
        output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode) or ""
        artifact_policy = normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy) or ""

        if output_mode == StepOutputMode.FILE.value and artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT.value:
            issues.append(
                StepContractCompilationIssue(
                    step_id=step_id,
                    issue_code="output_artifact_conflict",
                    issue_message="output_mode=file 与 artifact_policy=forbid_file_output 互斥。",
                )
            )
    return issues


def filter_final_delivery_steps(
    *,
    steps: List[Step],
    user_message: str,
) -> Tuple[List[Step], int]:
    """过滤最终交付型步骤。

    业务含义：Planner/Replan 只能产出执行步骤；最终用户正文统一由 summary 阶段组织。
    """
    filtered_steps: List[Step] = []
    dropped_count = 0
    user_requests_file_output = bool(EXPLICIT_FILE_OUTPUT_REQUEST_PATTERN.search(str(user_message or "").strip()))
    for step in list(steps or []):
        if _is_final_delivery_step(step=step, user_requests_file_output=user_requests_file_output):
            dropped_count += 1
            continue
        filtered_steps.append(step)
    return filtered_steps, dropped_count


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
    step_requests_environment_write_action = has_environment_write_intent(signals)
    _ = user_message  # P3-一次性收口：保留函数签名，当前编译策略仅依赖步骤语义。

    task_mode = normalize_controlled_value(getattr(compiled_step, "task_mode_hint", None), StepTaskModeHint) or ""
    output_mode = normalize_controlled_value(getattr(compiled_step, "output_mode", None), StepOutputMode) or ""
    artifact_policy = normalize_controlled_value(getattr(compiled_step, "artifact_policy", None), StepArtifactPolicy) or ""

    # P3-Step纯执行化：human_wait 步骤必须是纯等待，不允许附带文件产出。
    if task_mode == StepTaskModeHint.HUMAN_WAIT.value:
        if output_mode != StepOutputMode.NONE.value:
            compiled_step.output_mode = StepOutputMode.NONE
            corrected_count += 1
        if artifact_policy != StepArtifactPolicy.FORBID_FILE_OUTPUT.value:
            compiled_step.artifact_policy = StepArtifactPolicy.FORBID_FILE_OUTPUT
            corrected_count += 1
        return compiled_step, issues, corrected_count

    # P3-一次性收口：如果步骤明确写副作用，但策略禁止文件产出，按可执行语义纠偏。
    if artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT.value and step_requests_environment_write_action:
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

    # P3-一次性收口：只要步骤语义明确包含创建/写入/修改等写副作用，
    # 就不能继续保留 file_processing/general 的只读执行假设，必须纠偏到 coding。
    if step_requests_environment_write_action and task_mode in {
        StepTaskModeHint.FILE_PROCESSING.value,
        StepTaskModeHint.GENERAL.value,
        "",
    }:
        compiled_step.task_mode_hint = StepTaskModeHint.CODING
        corrected_count += 1

    # P3-一次性收口：output_mode=file 与 forbid_file_output 互斥时统一纠偏到 allow。
    output_mode = normalize_controlled_value(getattr(compiled_step, "output_mode", None), StepOutputMode) or ""
    artifact_policy = normalize_controlled_value(getattr(compiled_step, "artifact_policy", None), StepArtifactPolicy) or ""
    if output_mode == StepOutputMode.FILE.value and artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT.value:
        compiled_step.artifact_policy = StepArtifactPolicy.ALLOW_FILE_OUTPUT
        corrected_count += 1

    return compiled_step, issues, corrected_count


def _is_final_delivery_step(*, step: Step, user_requests_file_output: bool) -> bool:
    candidate_text = build_step_candidate_text(step)
    if not candidate_text:
        return False
    output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode) or ""
    artifact_policy = normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy) or ""
    if (
        user_requests_file_output
        and output_mode == StepOutputMode.FILE.value
        and artifact_policy in {
            StepArtifactPolicy.ALLOW_FILE_OUTPUT.value,
            StepArtifactPolicy.REQUIRE_FILE_OUTPUT.value,
        }
        and (
            FILE_OUTPUT_ACTION_PATTERN.search(candidate_text)
            or EN_FILE_OUTPUT_ACTION_PATTERN.search(candidate_text)
        )
    ):
        return False
    if FINAL_USER_DELIVERY_STEP_PATTERN.search(candidate_text):
        return True
    if EN_FINAL_USER_DELIVERY_STEP_PATTERN.search(candidate_text):
        return True
    if SUMMARY_ONLY_STEP_PATTERN.search(candidate_text):
        return True
    if EN_SUMMARY_ONLY_STEP_PATTERN.search(candidate_text):
        return True
    if (
        not user_requests_file_output
        and output_mode == StepOutputMode.FILE.value
        and SUMMARY_FILE_OUTPUT_STEP_PATTERN.search(candidate_text)
    ):
        return True
    if (
        not user_requests_file_output
        and output_mode == StepOutputMode.FILE.value
        and EN_SUMMARY_FILE_OUTPUT_STEP_PATTERN.search(candidate_text)
    ):
        return True
    return False


def _normalize_step_contract_enum_fields(step: Step) -> Step:
    """P3-一次性收口：编译出口统一规范结构化字段类型为 Enum，禁止字符串回流到持久化层。"""
    normalized_step = step.model_copy(deep=True)
    raw_task_mode_hint = normalize_controlled_value(getattr(normalized_step, "task_mode_hint", None), StepTaskModeHint)
    raw_output_mode = normalize_controlled_value(getattr(normalized_step, "output_mode", None), StepOutputMode)
    raw_artifact_policy = normalize_controlled_value(getattr(normalized_step, "artifact_policy", None), StepArtifactPolicy)
    normalized_step.task_mode_hint = StepTaskModeHint(raw_task_mode_hint) if raw_task_mode_hint else None
    normalized_step.output_mode = StepOutputMode(raw_output_mode) if raw_output_mode else None
    normalized_step.artifact_policy = StepArtifactPolicy(raw_artifact_policy) if raw_artifact_policy else None
    return normalized_step
