#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""步骤结构化契约编译策略。

本模块负责在 planner / replan 之后，对步骤做一次统一收紧：
- 纠正常见的结构化字段冲突；
- 拦截不符合“Step 纯执行化”的步骤语义；
- 区分“执行过程型整理”和“最终用户交付型整理”；
- 保证进入执行链的步骤只承载执行职责，不承载最终用户交付职责。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from app.domain.models import (
    Step,
    StepArtifactPolicy,
    StepOutputMode,
    StepTaskModeHint,
)
from app.domain.services.runtime.normalizers import normalize_controlled_value
from .task_mode_policy import analyze_text_intent, build_step_candidate_text, has_environment_write_intent


@dataclass(frozen=True)
class StepContractCompilationIssue:
    """步骤契约编译问题。"""

    step_id: str
    issue_code: str
    issue_message: str


_ORGANIZATION_MARKER_PATTERN = re.compile(
    r"(整理|总结|归纳|提炼|梳理|汇总|形成要点|输出要点|给出要点|撰写总结|组织成说明|组织内容|组织结构)"
    r"|(\b(summarize|summary|organize findings|write[- ]?up)\b)",
    re.IGNORECASE,
)
_FINAL_DELIVERY_MARKER_PATTERN = re.compile(
    r"(输出给用户|回复用户|返回给用户|面向用户|最终说明|最终答复|最终回答|最终正文|最终结论|最终方案|最终攻略|成稿)"
    r"|(\b(final answer|final response|user-facing|user facing|deliver to user)\b)",
    re.IGNORECASE,
)
_EXECUTION_PROGRESS_MARKER_PATTERN = re.compile(
    r"(导出|保存|写入|生成文件|生成文档|输出文件|读取|检索|抓取|提取|收集|比对|分析|计算|执行|运行|查看|打开|获取|记录|归档|形成结构|生成清单|生成表格)"
    r"|(\b(export|save|write|generate file|read|search|fetch|extract|collect|compare|analyze|compute|execute|run|get|record|catalog|table)\b)",
    re.IGNORECASE,
)


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

        if output_mode == StepOutputMode.FILE.value and artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT.value:
            issues.append(
                StepContractCompilationIssue(
                    step_id=step_id,
                    issue_code="output_artifact_conflict",
                    issue_message="output_mode=file 与 artifact_policy=forbid_file_output 互斥。",
                )
            )
        if is_user_facing_final_delivery_step(step):
            issues.append(
                StepContractCompilationIssue(
                    step_id=step_id,
                    issue_code="summary_only_step_forbidden",
                    issue_message="Step 纯执行化主链不允许规划承担最终用户交付整理职责的步骤。",
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


def _step_candidate_text(step: Step) -> str:
    """统一提取步骤判定文本，避免 planner/replan/execute 各自维护一套拼接逻辑。"""
    return build_step_candidate_text(step)


def step_requests_organizational_work(step: Step) -> bool:
    """判断步骤是否带有整理/归纳语义。

    注意：
    - 这里只判断是否“在做整理”，不代表该步骤一定非法；
    - 非法与否取决于它整理的是中间执行结果，还是最终用户交付内容。
    """
    candidate_text = _step_candidate_text(step)
    if not candidate_text:
        return False
    return bool(_ORGANIZATION_MARKER_PATTERN.search(candidate_text))


def step_has_execution_progress_semantics(step: Step) -> bool:
    """判断步骤是否仍在推进执行，而不是只消费已有结果做最终表述。"""
    candidate_text = _step_candidate_text(step)
    if not candidate_text:
        return False
    return bool(_EXECUTION_PROGRESS_MARKER_PATTERN.search(candidate_text))


def is_user_facing_final_delivery_step(step: Step) -> bool:
    """识别“最终用户交付型整理步骤”。

    判定原则：
    - 允许执行过程型整理：例如整理结构、形成中间清单、生成导出文件；
    - 禁止最终用户交付型整理：例如给用户成稿、输出最终要点、形成最终回答。

    这比简单的“纯整理步骤”更稳：
    - 不会误伤合法的中间归纳/结构化步骤；
    - 也能拦截带有少量分析词、但本质已越界到最终交付的步骤。
    """
    task_mode = normalize_controlled_value(getattr(step, "task_mode_hint", None), StepTaskModeHint) or ""
    if task_mode == StepTaskModeHint.HUMAN_WAIT.value:
        return False
    candidate_text = _step_candidate_text(step)
    if not candidate_text:
        return False
    if not step_requests_organizational_work(step):
        return False
    if _FINAL_DELIVERY_MARKER_PATTERN.search(candidate_text):
        return True
    if step_has_execution_progress_semantics(step):
        return False
    return True
