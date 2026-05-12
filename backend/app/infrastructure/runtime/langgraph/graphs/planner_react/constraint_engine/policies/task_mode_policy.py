#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束策略：task_mode 执行前约束。

本模块只负责工具许可边界，不负责 general 步骤的展示型收敛。
"""

from __future__ import annotations

from typing import Optional

from app.domain.models import BrowserPageType
from app.domain.services.runtime.contracts.langgraph_settings import (
    READ_ONLY_FILE_FUNCTION_NAMES,
    SHELL_AUXILIARY_FUNCTION_NAMES,
)
from app.domain.services.workspace_runtime.policies import (
    build_browser_high_level_retry_block_message as _build_browser_high_level_retry_block_message,
    build_browser_route_block_message as _build_browser_route_block_message,
    build_browser_route_state_key as _build_browser_route_state_key,
    build_listing_click_target_block_message as _build_listing_click_target_block_message,
    coerce_optional_int as _coerce_optional_int,
    is_browser_high_level_temporarily_blocked as _is_browser_high_level_temporarily_blocked,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence import (
    WebReadingConvergenceJudge,
)
from ..contracts import ConstraintDecision, ConstraintInput, ConstraintToolResultPayload
from ..reason_codes import (
    REASON_BROWSER_CLICK_TARGET_BLOCKED,
    REASON_BROWSER_HIGH_LEVEL_RETRY_BLOCKED,
    REASON_BROWSER_ROUTE_BLOCKED,
    REASON_FILE_PROCESSING_SHELL_AUXILIARY_BLOCKED,
    REASON_FILE_PROCESSING_SHELL_EXPLICIT_REQUIRED,
    REASON_FILE_WRITE_INTENT_READ_TOOL_BLOCKED,
    REASON_INVALID_TOOL,
    REASON_RESEARCH_FILE_CONTEXT_REQUIRED,
    REASON_TASK_MODE_TOOL_BLOCKED,
    REASON_WEB_READING_FILE_TOOL_BLOCKED,
)


def evaluate_task_mode_policy(constraint_input: ConstraintInput) -> Optional[ConstraintDecision]:
    """处理 task_mode 的前置工具约束。

    这里只处理任务模式、自身页面证据和浏览器路由相关限制；
    更具体的产物策略 / human_wait 原因码交给各自策略处理，避免职责重叠。
    """
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    function_name = str(constraint_input.function_name or "").strip()
    task_mode = str(constraint_input.task_mode or "").strip().lower()
    blocked_names = set(constraint_input.iteration_blocked_function_names or set())
    ctx = constraint_input.execution_context
    state = constraint_input.execution_state
    web_reading_progress_state = (
        WebReadingConvergenceJudge.get_completion_progress(
            step=constraint_input.step,
            runtime_recent_action=state.runtime_recent_action,
            execution_state=state,
        )
        if task_mode == "web_reading"
        else {}
    )

    if constraint_input.matched_tool is None:
        return ConstraintDecision(
            action="block",
            reason_code=REASON_INVALID_TOOL,
            block_mode="soft_block_continue",
            loop_break_reason="",
            tool_result_payload=ConstraintToolResultPayload(
                success=False,
                message=f"无效工具: {function_name}",
            ),
            message_for_model=f"无效工具: {function_name}",
        )

    # human_wait 场景统一由 human_wait_policy 处理，避免原因码被 task_mode 兜底覆盖。
    if task_mode == "human_wait":
        return None

    if (
            task_mode == "web_reading"
            and normalized_function_name in {"fetch_page", "browser_read_current_page_structured", "browser_extract_main_content"}
            and bool(dict(web_reading_progress_state.get("progress") or {}).get("contract_satisfied"))
    ):
        return _hard_block(
            REASON_TASK_MODE_TOOL_BLOCKED,
            "当前步骤已有结构化 strong 页面证据，禁止继续重复读取页面。",
        )

    if (
            bool(ctx.browser_route_enabled)
            and normalized_function_name == "browser_click"
            and state.browser_page_type in {
        BrowserPageType.LISTING.value,
        BrowserPageType.SEARCH_RESULTS.value,
    }
            and bool(state.browser_link_match_ready)
    ):
        requested_index = _coerce_optional_int(constraint_input.function_args.get("index"))
        coordinate_x = constraint_input.function_args.get("coordinate_x")
        coordinate_y = constraint_input.function_args.get("coordinate_y")
        if (
                coordinate_x is not None
                or coordinate_y is not None
                or requested_index is None
                or state.last_browser_route_index is None
                or requested_index != state.last_browser_route_index
        ):
            return _soft_block(
                reason_code=REASON_BROWSER_CLICK_TARGET_BLOCKED,
                message=_build_listing_click_target_block_message(
                    last_browser_route_index=state.last_browser_route_index,
                    last_browser_route_url=state.last_browser_route_url,
                    last_browser_route_selector=state.last_browser_route_selector,
                ),
            )

    if bool(ctx.browser_route_enabled):
        browser_route_state_key = _build_browser_route_state_key(
            browser_page_type=state.browser_page_type,
            browser_url=state.last_browser_route_url,
            browser_observation_fingerprint=state.last_browser_observation_fingerprint,
        )
        if _is_browser_high_level_temporarily_blocked(
                function_name=normalized_function_name,
                function_args=constraint_input.function_args,
                browser_route_state_key=browser_route_state_key,
                failed_high_level_keys=state.failed_browser_high_level_keys,
        ):
            return _soft_block(
                reason_code=REASON_BROWSER_HIGH_LEVEL_RETRY_BLOCKED,
                message=_build_browser_high_level_retry_block_message(
                    function_name=function_name,
                    function_args=constraint_input.function_args,
                ),
            )

    if normalized_function_name not in blocked_names:
        return None

    if _is_system_rewritten_evidence_verification_call(constraint_input):
        return None

    downstream_owned_blocked_names = set()
    downstream_owned_blocked_names.update(ctx.read_only_file_blocked_function_names or set())
    downstream_owned_blocked_names.update(ctx.artifact_policy_blocked_function_names or set())
    if normalized_function_name in downstream_owned_blocked_names:
        # task_mode_policy 固定前置，但后续 policy 拥有更具体的原因码。
        return None

    if normalized_function_name in set(ctx.research_file_context_blocked_function_names or set()):
        message = "当前步骤属于检索任务，只有在用户消息或附件中出现明确文件路径/文件名时，才能调用文件工具。"
        return _hard_block(REASON_RESEARCH_FILE_CONTEXT_REQUIRED, message)

    if normalized_function_name in set(ctx.file_write_intent_blocked_function_names or set()):
        message = "当前步骤是明确的文件创建/写入任务，请直接使用 write_file 或 replace_in_file，不要先调用只读文件工具。"
        return _hard_block(REASON_FILE_WRITE_INTENT_READ_TOOL_BLOCKED, message)

    if task_mode == "web_reading" and normalized_function_name in READ_ONLY_FILE_FUNCTION_NAMES:
        message = "当前步骤属于网页读取任务，请优先使用 search_web、fetch_page 或浏览器高阶读取工具，不要回退到文件工具。"
        return _hard_block(REASON_WEB_READING_FILE_TOOL_BLOCKED, message)

    if normalized_function_name in set(ctx.file_processing_shell_blocked_function_names or set()):
        if normalized_function_name in set(SHELL_AUXILIARY_FUNCTION_NAMES):
            message = "当前步骤属于文件处理，且没有活跃 shell 命令会话，禁止调用 shell 会话辅助工具。"
            return _hard_block(REASON_FILE_PROCESSING_SHELL_AUXILIARY_BLOCKED, message)
        message = "当前步骤属于文件处理，默认禁止调用 shell_execute。仅在用户明确要求执行命令时才允许。"
        return _hard_block(REASON_FILE_PROCESSING_SHELL_EXPLICIT_REQUIRED, message)

    if bool(ctx.browser_route_enabled) and normalized_function_name.startswith("browser_"):
        message = _build_browser_route_block_message(
            task_mode=task_mode,
            function_name=function_name,
            browser_page_type=state.browser_page_type,
            browser_structured_ready=state.browser_structured_ready,
            browser_cards_ready=state.browser_cards_ready,
            browser_link_match_ready=state.browser_link_match_ready,
            browser_actionables_ready=state.browser_actionables_ready,
            last_browser_route_url=state.last_browser_route_url,
            last_browser_route_selector=state.last_browser_route_selector,
            last_browser_route_index=state.last_browser_route_index,
        )
        return _hard_block(REASON_BROWSER_ROUTE_BLOCKED, message)

    message = f"当前步骤的任务模式 {task_mode} 不允许调用工具: {function_name}"
    return _hard_block(REASON_TASK_MODE_TOOL_BLOCKED, message)


def _is_system_rewritten_evidence_verification_call(constraint_input: ConstraintInput) -> bool:
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    if normalized_function_name not in {"search_web", "fetch_page"}:
        return False
    rewrite_marker = dict(
        (constraint_input.external_signals_snapshot or {}).get("evidence_verification_rewrite") or {}
    )
    if rewrite_marker.get("rewrite_type") != "evidence_verification_audit_metadata":
        return False
    if str(rewrite_marker.get("function_name") or "").strip().lower() != normalized_function_name:
        return False
    function_args = dict(constraint_input.function_args or {})
    reason_code = str(function_args.get("verification_reason_code") or "").strip()
    if not reason_code:
        return False
    if normalized_function_name == "search_web":
        return bool(
            str(function_args.get("query") or "").strip()
            and str(function_args.get("query_hash") or "").strip()
        )
    return bool(
        str(function_args.get("url") or "").strip()
        and (
            str(function_args.get("url_hash") or "").strip()
            or str(function_args.get("fetched_url_hash") or "").strip()
        )
    )


def _hard_block(reason_code: str, message: str) -> ConstraintDecision:
    return ConstraintDecision(
        action="block",
        reason_code=reason_code,
        block_mode="hard_block_break",
        loop_break_reason=reason_code,
        tool_result_payload=ConstraintToolResultPayload(
            success=False,
            message=message,
            data={"reason_code": reason_code},
        ),
        message_for_model=message,
    )


def _soft_block(*, reason_code: str, message: str) -> ConstraintDecision:
    return ConstraintDecision(
        action="block",
        reason_code=reason_code,
        block_mode="soft_block_continue",
        # soft block 不触发收敛，loop_break_reason 为空。
        loop_break_reason="",
        tool_result_payload=ConstraintToolResultPayload(success=False, message=message),
        message_for_model=message,
    )
