#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束策略：human_wait 执行前约束。"""

from __future__ import annotations

from typing import Optional

from app.domain.services.runtime.contracts.langgraph_settings import ASK_USER_FUNCTION_NAME
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_context import (
    step_allows_user_wait,
)
from ..contracts import ConstraintDecision, ConstraintInput, ConstraintToolResultPayload
from ..reason_codes import (
    REASON_ASK_USER_NOT_ALLOWED,
    REASON_HUMAN_WAIT_NON_INTERRUPT_TOOL_BLOCKED,
)


def evaluate_human_wait_policy(constraint_input: ConstraintInput) -> Optional[ConstraintDecision]:
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    function_name = str(constraint_input.function_name or "").strip()
    task_mode = str(constraint_input.task_mode or "").strip().lower()

    if task_mode == "human_wait" and function_name != ASK_USER_FUNCTION_NAME:
        if normalized_function_name not in set(constraint_input.iteration_blocked_function_names or set()):
            return None
        message = "当前步骤是等待用户确认/选择的步骤，只允许调用 message_ask_user 发起等待。"
        return ConstraintDecision(
            action="block",
            reason_code=REASON_HUMAN_WAIT_NON_INTERRUPT_TOOL_BLOCKED,
            block_mode="hard_block_break",
            loop_break_reason=REASON_HUMAN_WAIT_NON_INTERRUPT_TOOL_BLOCKED,
            tool_result_payload=ConstraintToolResultPayload(success=False, message=message),
            message_for_model=message,
        )

    if function_name == ASK_USER_FUNCTION_NAME and not step_allows_user_wait(constraint_input.step,
                                                                             constraint_input.function_args):
        message = "当前步骤不允许向用户提问。请先完成当前步骤，只能在明确需要用户确认/选择/输入的步骤中使用该工具。"
        return ConstraintDecision(
            action="block",
            reason_code=REASON_ASK_USER_NOT_ALLOWED,
            block_mode="soft_block_continue",
            loop_break_reason="",
            tool_result_payload=ConstraintToolResultPayload(success=False, message=message),
            message_for_model=message,
        )

    return None
