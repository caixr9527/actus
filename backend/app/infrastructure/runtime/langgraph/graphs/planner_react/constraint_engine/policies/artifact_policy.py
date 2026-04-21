#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束策略：产物与只读语义约束。"""

from __future__ import annotations

from typing import Optional

from ..contracts import ConstraintDecision, ConstraintInput, ConstraintToolResultPayload
from ..reason_codes import (
    REASON_ARTIFACT_POLICY_FILE_OUTPUT_BLOCKED,
    REASON_READ_ONLY_FILE_INTENT_WRITE_BLOCKED,
)


def evaluate_artifact_policy(constraint_input: ConstraintInput) -> Optional[ConstraintDecision]:
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    blocked_names = set(constraint_input.iteration_blocked_function_names or set())
    ctx = constraint_input.execution_context

    if normalized_function_name not in blocked_names:
        return None

    if normalized_function_name in set(ctx.read_only_file_blocked_function_names or set()):
        return _hard_block(
            reason_code=REASON_READ_ONLY_FILE_INTENT_WRITE_BLOCKED,
            message="当前步骤是只读文件请求，请使用 read_file/list_files/find_files/search_in_file，不要写文件、改文件或执行 shell 命令。",
        )

    if normalized_function_name in set(ctx.artifact_policy_blocked_function_names or set()):
        return _hard_block(
            reason_code=REASON_ARTIFACT_POLICY_FILE_OUTPUT_BLOCKED,
            message="当前步骤的结构化产物策略禁止文件产出。请直接返回文本结果，或先通过重规划生成允许文件产出的步骤。",
        )

    return None


def _hard_block(*, reason_code: str, message: str) -> ConstraintDecision:
    return ConstraintDecision(
        action="block",
        reason_code=reason_code,
        block_mode="hard_block_break",
        loop_break_reason=reason_code,
        tool_result_payload=ConstraintToolResultPayload(success=False, message=message),
        message_for_model=message,
    )
