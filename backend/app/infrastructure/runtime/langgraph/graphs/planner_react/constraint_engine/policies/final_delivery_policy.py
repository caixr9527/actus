#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束策略：最终交付阶段约束。

本策略只负责“最终交付阶段是否发生工具漂移”：
- 允许模型基于已有证据直接组织最终答案；
- 阻断 search/shell/file output 等会破坏最终交付语义的工具调用；
- 不负责 task_mode 基础限制，也不处理 human_wait / artifact policy。
"""

from __future__ import annotations

from typing import Optional

from ..contracts import ConstraintDecision, ConstraintInput, ConstraintToolResultPayload
from ..reason_codes import (
    REASON_FINAL_DELIVERY_SEARCH_DRIFT_BLOCKED,
    REASON_FINAL_DELIVERY_SHELL_DRIFT_BLOCKED,
    REASON_FINAL_INLINE_FILE_OUTPUT_BLOCKED,
)


def evaluate_final_delivery_policy(constraint_input: ConstraintInput) -> Optional[ConstraintDecision]:
    """评估最终交付步骤是否被 search/shell/file output 工具带偏。"""
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    blocked_names = set(constraint_input.iteration_blocked_function_names or set())
    ctx = constraint_input.execution_context

    if normalized_function_name not in blocked_names:
        return None

    if normalized_function_name in set(ctx.final_delivery_search_blocked_function_names or set()):
        return _hard_block(
            reason_code=REASON_FINAL_DELIVERY_SEARCH_DRIFT_BLOCKED,
            message="当前步骤负责最终交付正文，请直接基于已知上下文组织答案，不要重新调用 search_web 或 fetch_page。",
        )

    if normalized_function_name in set(ctx.final_inline_file_output_blocked_function_names or set()):
        return _hard_block(
            reason_code=REASON_FINAL_INLINE_FILE_OUTPUT_BLOCKED,
            message="当前步骤负责最终内联交付，且用户未明确要求文件交付。请直接输出最终文本，不要调用 write_file/replace_in_file。",
        )

    if normalized_function_name in set(ctx.final_delivery_shell_blocked_function_names or set()):
        return _hard_block(
            reason_code=REASON_FINAL_DELIVERY_SHELL_DRIFT_BLOCKED,
            message="当前步骤负责最终交付正文，请直接输出最终答案，不要调用 shell 类工具。",
        )

    return None


def _hard_block(*, reason_code: str, message: str) -> ConstraintDecision:
    """构造最终交付阶段的硬阻断结果。"""
    return ConstraintDecision(
        action="block",
        reason_code=reason_code,
        block_mode="hard_block_break",
        loop_break_reason=reason_code,
        tool_result_payload=ConstraintToolResultPayload(success=False, message=message),
        message_for_model=message,
    )
