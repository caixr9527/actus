#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束策略：task_mode 执行前约束。"""

from __future__ import annotations

import re
from typing import Optional

from app.domain.models import BrowserPageType
from app.domain.services.runtime.contracts.langgraph_settings import READ_ONLY_FILE_FUNCTION_NAMES
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
from app.infrastructure.runtime.langgraph.graphs.planner_react.research.research_diagnosis import (
    get_page_reading_contract_state,
)
from ..contracts import ConstraintDecision, ConstraintInput, ConstraintToolResultPayload
from ..reason_codes import (
    REASON_BROWSER_CLICK_TARGET_BLOCKED,
    REASON_BROWSER_HIGH_LEVEL_RETRY_BLOCKED,
    REASON_BROWSER_ROUTE_BLOCKED,
    REASON_FILE_PROCESSING_SHELL_EXPLICIT_REQUIRED,
    REASON_GENERAL_INLINE_FILE_CONTEXT_REQUIRED,
    REASON_INVALID_TOOL,
    REASON_RESEARCH_FILE_CONTEXT_REQUIRED,
    REASON_TASK_MODE_TOOL_BLOCKED,
    REASON_WEB_READING_FILE_TOOL_BLOCKED,
)

_GENERAL_PAGE_EVIDENCE_PRESENTATION_PATTERN = re.compile(
    r"(整理|总结|归纳|概括|提炼|梳理|列出|筛选|汇总|展示|说明|要点|链接)",
    re.IGNORECASE,
)


def evaluate_task_mode_policy(constraint_input: ConstraintInput) -> Optional[ConstraintDecision]:
    """处理 task_mode 的前置工具约束。

    这里只处理任务模式、自身页面证据和浏览器路由相关限制；
    更具体的最终交付/产物/human_wait 原因码交给各自策略处理，避免职责重叠。
    """
    normalized_function_name = str(constraint_input.normalized_function_name or "").strip().lower()
    function_name = str(constraint_input.function_name or "").strip()
    task_mode = str(constraint_input.task_mode or "").strip().lower()
    blocked_names = set(constraint_input.iteration_blocked_function_names or set())
    ctx = constraint_input.execution_context
    state = constraint_input.execution_state
    page_contract_state = get_page_reading_contract_state(
        runtime_recent_action=state.runtime_recent_action,
        execution_state=state,
    )
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
        if (
                task_mode == "general"
                and normalized_function_name in {"search_web", "fetch_page"}
                and _general_step_should_reuse_page_evidence(
                    constraint_input=constraint_input,
                    page_contract_state=page_contract_state,
                )
        ):
            return _hard_block(
                REASON_TASK_MODE_TOOL_BLOCKED,
                "当前整理步骤已经具备可用网页证据，请直接基于已有页面/摘要信息完成，不要重新发起搜索或抓取。",
            )
        if (
                task_mode == "web_reading"
                and normalized_function_name == "fetch_page"
                and bool(dict(web_reading_progress_state.get("progress") or {}).get("contract_satisfied"))
        ):
            return _hard_block(
                REASON_TASK_MODE_TOOL_BLOCKED,
                "当前网页阅读步骤已经具备强页面证据，请直接基于已有页面内容完成，不要继续重复抓取。",
            )
        return None

    downstream_owned_blocked_names = set()
    downstream_owned_blocked_names.update(ctx.read_only_file_blocked_function_names or set())
    downstream_owned_blocked_names.update(ctx.artifact_policy_blocked_function_names or set())
    downstream_owned_blocked_names.update(ctx.final_delivery_search_blocked_function_names or set())
    downstream_owned_blocked_names.update(ctx.final_delivery_shell_blocked_function_names or set())
    downstream_owned_blocked_names.update(ctx.final_inline_file_output_blocked_function_names or set())
    if normalized_function_name in downstream_owned_blocked_names:
        # task_mode_policy 固定前置，但后续 policy 拥有更具体的产物/最终交付原因码。
        return None

    if normalized_function_name in set(ctx.research_file_context_blocked_function_names or set()):
        message = "当前步骤属于检索任务，只有在用户消息或附件中出现明确文件路径/文件名时，才能调用文件工具。"
        return _hard_block(REASON_RESEARCH_FILE_CONTEXT_REQUIRED, message)

    if task_mode == "web_reading" and normalized_function_name in READ_ONLY_FILE_FUNCTION_NAMES:
        message = "当前步骤属于网页读取任务，请优先使用 search_web、fetch_page 或浏览器高阶读取工具，不要回退到文件工具。"
        return _hard_block(REASON_WEB_READING_FILE_TOOL_BLOCKED, message)

    if normalized_function_name in set(ctx.general_inline_blocked_function_names or set()):
        message = "当前步骤是直接内联展示结果的步骤，且没有可用文件上下文，请直接返回文本结果，不要继续读写文件。"
        return _hard_block(REASON_GENERAL_INLINE_FILE_CONTEXT_REQUIRED, message)

    if normalized_function_name in set(ctx.file_processing_shell_blocked_function_names or set()):
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


def _hard_block(reason_code: str, message: str) -> ConstraintDecision:
    return ConstraintDecision(
        action="block",
        reason_code=reason_code,
        block_mode="hard_block_break",
        loop_break_reason=reason_code,
        tool_result_payload=ConstraintToolResultPayload(success=False, message=message),
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


def _general_step_should_reuse_page_evidence(
        *,
        constraint_input: ConstraintInput,
        page_contract_state: dict[str, object],
) -> bool:
    if not bool(page_contract_state.get("has_any_evidence")):
        return False
    step = constraint_input.step
    step_text = " ".join(
        [
            str(getattr(step, "title", "") or ""),
            str(getattr(step, "description", "") or ""),
            " ".join([str(item or "") for item in list(getattr(step, "success_criteria", []) or [])]),
        ]
    ).strip()
    if not step_text:
        return False
    return bool(_GENERAL_PAGE_EVIDENCE_PRESENTATION_PATTERN.search(step_text))
