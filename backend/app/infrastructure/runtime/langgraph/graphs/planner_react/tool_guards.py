#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具调用前置守卫。"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from app.domain.models import Step, ToolResult
from app.domain.services.tools import BaseTool
from app.domain.services.workspace_runtime.policies import (
    build_browser_route_block_message as _build_browser_route_block_message,
)
from .execution_context import ExecutionContext, step_allows_user_wait
from .execution_state import ExecutionState
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.contracts.langgraph_settings import ASK_USER_FUNCTION_NAME, READ_ONLY_FILE_FUNCTION_NAMES


@dataclass(slots=True)
class GuardDecision:
    should_skip: bool
    loop_break_reason: str = ""
    tool_result: Optional[ToolResult] = None


def evaluate_tool_guard(
    *,
    logger: logging.Logger,
    step: Step,
    task_mode: str,
    function_name: str,
    normalized_function_name: str,
    function_args: Dict[str, Any],
    matched_tool: Optional[BaseTool],
    iteration_blocked_function_names: Set[str],
    ctx: ExecutionContext,
    state: ExecutionState,
) -> GuardDecision:
    if matched_tool is None:
        log_runtime(
            logger,
            logging.WARNING,
            "工具调用无效",
            step_id=str(step.id or ""),
            function_name=function_name,
        )
        return GuardDecision(
            should_skip=True,
            tool_result=ToolResult(success=False, message=f"无效工具: {function_name}"),
        )

    if normalized_function_name in iteration_blocked_function_names:
        if task_mode == "human_wait" and function_name != ASK_USER_FUNCTION_NAME:
            log_runtime(
                logger,
                logging.WARNING,
                "等待步骤已拦截非等待工具",
                step_id=str(step.id or ""),
                function_name=function_name,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="human_wait_non_interrupt_tool_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤是等待用户确认/选择的步骤，只允许调用 message_ask_user 发起等待。",
                ),
            )
        if normalized_function_name in ctx.research_file_context_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "缺少明确文件上下文，已拦截文件工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="research_file_context_required",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤属于检索任务，只有在用户消息或附件中出现明确文件路径/文件名时，才能调用文件工具。",
                ),
            )
        if normalized_function_name in ctx.read_only_file_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "只读步骤已拦截写副作用工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="read_only_file_intent_write_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤是只读文件请求，请使用 read_file/list_files/find_files/search_in_file，不要写文件、改文件或执行 shell 命令。",
                ),
            )
        if task_mode == "web_reading" and normalized_function_name in READ_ONLY_FILE_FUNCTION_NAMES:
            log_runtime(
                logger,
                logging.WARNING,
                "网页读取步骤已拦截文件工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="web_reading_file_tool_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤属于网页读取任务，请优先使用 search_web、fetch_page 或浏览器高阶读取工具，不要回退到文件工具。",
                ),
            )
        if normalized_function_name in ctx.general_inline_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "内联展示步骤缺少文件上下文，已拦截文件工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                output_mode=str(getattr(step, "output_mode", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="general_inline_file_context_required",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤是直接内联展示结果的步骤，且没有可用文件上下文，请直接返回文本结果，不要继续读写文件。",
                ),
            )
        if normalized_function_name in ctx.file_processing_shell_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "文件处理步骤缺少显式命令意图，已拦截 shell 工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="file_processing_shell_explicit_required",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤属于文件处理，默认禁止调用 shell_execute。仅在用户明确要求执行命令时才允许。",
                ),
            )
        if normalized_function_name in ctx.artifact_policy_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "步骤产物策略拦截文件产出工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                artifact_policy=str(getattr(step, "artifact_policy", "") or ""),
                output_mode=str(getattr(step, "output_mode", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="artifact_policy_file_output_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤的结构化产物策略禁止文件产出。请直接返回文本结果，或先通过重规划生成允许文件产出的步骤。",
                ),
            )
        if normalized_function_name in ctx.final_delivery_search_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "最终交付步骤已拦截检索漂移",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                delivery_role=str(getattr(step, "delivery_role", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="final_delivery_search_drift_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤负责最终交付正文，请直接基于已知上下文组织答案，不要重新调用 search_web 或 fetch_page。",
                ),
            )
        if normalized_function_name in ctx.final_inline_file_output_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "最终内联交付步骤已拦截写文件漂移",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                delivery_role=str(getattr(step, "delivery_role", "") or ""),
                output_mode=str(getattr(step, "output_mode", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="final_inline_file_output_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤负责最终内联交付，且用户未明确要求文件交付。请直接输出最终文本，不要调用 write_file/replace_in_file。",
                ),
            )
        if normalized_function_name in ctx.final_delivery_shell_blocked_function_names:
            log_runtime(
                logger,
                logging.WARNING,
                "最终交付步骤已拦截 shell 漂移",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                delivery_role=str(getattr(step, "delivery_role", "") or ""),
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="final_delivery_shell_drift_blocked",
                tool_result=ToolResult(
                    success=False,
                    message="当前步骤负责最终交付正文，请直接输出最终答案，不要调用 shell_execute。",
                ),
            )
        if ctx.browser_route_enabled and normalized_function_name.startswith("browser_"):
            log_runtime(
                logger,
                logging.WARNING,
                "浏览器固定路径拦截工具调用",
                step_id=str(step.id or ""),
                function_name=function_name,
                task_mode=task_mode,
                browser_page_type=state.browser_page_type,
                browser_structured_ready=state.browser_structured_ready,
                browser_cards_ready=state.browser_cards_ready,
                browser_link_match_ready=state.browser_link_match_ready,
                browser_actionables_ready=state.browser_actionables_ready,
            )
            return GuardDecision(
                should_skip=True,
                loop_break_reason="browser_route_blocked",
                tool_result=ToolResult(
                    success=False,
                    message=_build_browser_route_block_message(
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
                    ),
                ),
            )

        log_runtime(
            logger,
            logging.WARNING,
            "任务模式拦截工具调用",
            step_id=str(step.id or ""),
            function_name=function_name,
            task_mode=task_mode,
        )
        return GuardDecision(
            should_skip=True,
            loop_break_reason="task_mode_tool_blocked",
            tool_result=ToolResult(
                success=False,
                message=f"当前步骤的任务模式 {task_mode} 不允许调用工具: {function_name}",
            ),
        )

    if (
        ctx.research_route_enabled
        and normalized_function_name == "fetch_page"
        and not ctx.research_has_explicit_url
        and not state.research_search_ready
    ):
        return GuardDecision(
            should_skip=True,
            loop_break_reason="research_route_search_required",
            tool_result=ToolResult(
                success=False,
                message="当前步骤属于检索/网页阅读任务，请先调用 search_web 获取候选链接，再使用 fetch_page 读取正文。",
            ),
        )
    if (
        ctx.research_route_enabled
        and normalized_function_name == "search_web"
        and ctx.research_has_explicit_url
        and not state.research_fetch_completed
    ):
        return GuardDecision(
            should_skip=True,
            loop_break_reason="research_route_fetch_required",
            tool_result=ToolResult(
                success=False,
                message="当前步骤已提供明确 URL，请直接调用 fetch_page 读取页面正文，不要先重复搜索。",
            ),
        )
    if (
        ctx.research_route_enabled
        and normalized_function_name == "search_web"
        and state.research_search_ready
        and not state.research_fetch_completed
    ):
        candidate_hint = "；".join(state.research_candidate_urls[:3])
        return GuardDecision(
            should_skip=True,
            loop_break_reason="research_route_fetch_required",
            tool_result=ToolResult(
                success=False,
                message=(
                    "已经拿到候选链接，请优先对搜索结果中的 URL 调用 fetch_page 读取正文。"
                    + (f" 可用链接示例: {candidate_hint}" if candidate_hint else "")
                ),
            ),
        )
    if function_name == ASK_USER_FUNCTION_NAME and not step_allows_user_wait(step, function_args):
        log_runtime(
            logger,
            logging.WARNING,
            "提前请求用户交互，已拦截",
            step_id=str(step.id or ""),
            function_name=function_name,
            step_description=str(step.description or ""),
        )
        return GuardDecision(
            should_skip=True,
            tool_result=ToolResult(
                success=False,
                message="当前步骤不允许向用户提问。请先完成当前步骤，只能在明确需要用户确认/选择/输入的步骤中使用该工具。",
            ),
        )

    return GuardDecision(should_skip=False)
