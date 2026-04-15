#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具调用执行与专用分支处理。"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.domain.models import BrowserPageType, Step, ToolResult
from app.domain.services.tools import BaseTool
from app.domain.services.workspace_runtime.policies import (
    build_browser_high_level_retry_block_message as _build_browser_high_level_retry_block_message,
    build_listing_click_target_block_message as _build_listing_click_target_block_message,
    build_search_fingerprint as _build_search_fingerprint,
    build_tool_fingerprint as _build_tool_fingerprint,
    coerce_optional_int as _coerce_optional_int,
    is_browser_high_level_temporarily_blocked as _is_browser_high_level_temporarily_blocked,
)
from .execution_context import ExecutionContext
from .execution_state import ExecutionState
from app.domain.services.runtime.contracts.runtime_logging import elapsed_ms, log_runtime, now_perf
from app.domain.services.runtime.contracts.langgraph_settings import FETCH_REPEAT_LIMIT, REPEAT_TOOL_LIMIT, SEARCH_REPEAT_LIMIT


@dataclass(slots=True)
class ToolExecutionDecision:
    tool_result: ToolResult
    tool_cost_ms: int = 0
    loop_break_reason: str = ""


async def execute_tool_with_policy(
    *,
    logger: logging.Logger,
    step: Step,
    function_name: str,
    normalized_function_name: str,
    function_args: Dict[str, Any],
    matched_tool: BaseTool,
    tool_name: str,
    browser_route_state_key: str,
    ctx: ExecutionContext,
    state: ExecutionState,
    started_at: float,
) -> ToolExecutionDecision:
    # P3 重构：列表页点击命中约束保持原有优先级，避免错误元素点击。
    if (
        ctx.browser_route_enabled
        and normalized_function_name == "browser_click"
        and state.browser_page_type in {
            BrowserPageType.LISTING.value,
            BrowserPageType.SEARCH_RESULTS.value,
        }
        and state.browser_link_match_ready
    ):
        requested_index = _coerce_optional_int(function_args.get("index"))
        coordinate_x = function_args.get("coordinate_x")
        coordinate_y = function_args.get("coordinate_y")
        if (
            coordinate_x is not None
            or coordinate_y is not None
            or requested_index is None
            or state.last_browser_route_index is None
            or requested_index != state.last_browser_route_index
        ):
            log_runtime(
                logger,
                logging.INFO,
                "列表页点击目标与已匹配结果不一致，已拦截",
                step_id=str(step.id or ""),
                function_name=function_name,
                requested_index=requested_index,
                matched_index=state.last_browser_route_index,
                has_coordinate_x=coordinate_x is not None,
                has_coordinate_y=coordinate_y is not None,
            )
            return ToolExecutionDecision(
                loop_break_reason="browser_click_target_blocked",
                tool_result=ToolResult(
                    success=False,
                    message=_build_listing_click_target_block_message(
                        last_browser_route_index=state.last_browser_route_index,
                        last_browser_route_url=state.last_browser_route_url,
                        last_browser_route_selector=state.last_browser_route_selector,
                    ),
                ),
            )
        return await _invoke_tool(
            logger=logger,
            step=step,
            function_name=function_name,
            matched_tool=matched_tool,
            function_args=function_args,
            tool_name=tool_name,
            started_at=started_at,
        )

    if ctx.browser_route_enabled and _is_browser_high_level_temporarily_blocked(
        function_name=normalized_function_name,
        function_args=function_args,
        browser_route_state_key=browser_route_state_key,
        failed_high_level_keys=state.failed_browser_high_level_keys,
    ):
        log_runtime(
            logger,
            logging.INFO,
            "浏览器高阶能力在当前页面状态下暂时封禁",
            step_id=str(step.id or ""),
            function_name=function_name,
            browser_route_state_key=browser_route_state_key,
        )
        return ToolExecutionDecision(
            loop_break_reason="browser_high_level_retry_blocked",
            tool_result=ToolResult(
                success=False,
                message=_build_browser_high_level_retry_block_message(
                    function_name=function_name,
                    function_args=function_args,
                ),
            ),
        )

    if normalized_function_name == "search_web":
        search_fingerprint = _build_search_fingerprint(function_args)
        state.search_repeat_counter[search_fingerprint] = state.search_repeat_counter.get(search_fingerprint, 0) + 1
        if state.search_repeat_counter[search_fingerprint] > SEARCH_REPEAT_LIMIT:
            log_runtime(
                logger,
                logging.WARNING,
                "重复搜索已收敛",
                step_id=str(step.id or ""),
                function_name=function_name,
                search_repeat_count=state.search_repeat_counter[search_fingerprint],
            )
            return ToolExecutionDecision(
                loop_break_reason="search_repeat",
                tool_result=ToolResult(
                    success=False,
                    message="同一搜索查询已重复多次，请改写查询、缩小范围，或改用 fetch_page / 其他工具继续。",
                ),
            )
        invoked = await _invoke_tool(
            logger=logger,
            step=step,
            function_name=function_name,
            matched_tool=matched_tool,
            function_args=function_args,
            tool_name=tool_name,
            started_at=started_at,
        )
        if (not bool(invoked.tool_result.success)) and _is_transient_research_transport_error(invoked.tool_result.message):
            return ToolExecutionDecision(
                tool_cost_ms=invoked.tool_cost_ms,
                loop_break_reason="research_route_transport_error",
                tool_result=ToolResult(
                    success=False,
                    message="检索链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或切换其他路径。",
                ),
            )
        return invoked

    if normalized_function_name == "fetch_page":
        fetch_fingerprint = _build_fetch_fingerprint(function_args)
        if fetch_fingerprint:
            state.fetch_repeat_counter[fetch_fingerprint] = state.fetch_repeat_counter.get(fetch_fingerprint, 0) + 1
        if fetch_fingerprint and state.fetch_repeat_counter[fetch_fingerprint] > FETCH_REPEAT_LIMIT:
            log_runtime(
                logger,
                logging.WARNING,
                "重复抓取同一页面已收敛",
                step_id=str(step.id or ""),
                function_name=function_name,
                fetch_repeat_count=state.fetch_repeat_counter[fetch_fingerprint],
                fetch_url=fetch_fingerprint,
            )
            return ToolExecutionDecision(
                loop_break_reason="research_route_fingerprint_repeat",
                tool_result=ToolResult(
                    success=False,
                    message="同一页面 URL 已重复抓取多次，请切换其他候选链接或结束当前步骤。",
                ),
            )
        invoked = await _invoke_tool(
            logger=logger,
            step=step,
            function_name=function_name,
            matched_tool=matched_tool,
            function_args=function_args,
            tool_name=tool_name,
            started_at=started_at,
        )
        if (not bool(invoked.tool_result.success)) and _is_transient_research_transport_error(invoked.tool_result.message):
            return ToolExecutionDecision(
                tool_cost_ms=invoked.tool_cost_ms,
                loop_break_reason="research_route_transport_error",
                tool_result=ToolResult(
                    success=False,
                    message="页面抓取链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或改用其他来源。",
                ),
            )
        return invoked

    if state.same_tool_repeat_count > REPEAT_TOOL_LIMIT:
        log_runtime(
            logger,
            logging.WARNING,
            "重复工具调用已收敛",
            step_id=str(step.id or ""),
            function_name=function_name,
            same_tool_repeat_count=state.same_tool_repeat_count,
        )
        # P3-一次性收口：如果此前已经有同工具成功结果，直接用该结果收口，避免“已成功却被判失败”。
        repeated_success_result = _build_repeat_success_fallback_result(
            function_name=normalized_function_name,
            function_args=function_args,
            current_tool_fingerprint=state.last_tool_fingerprint,
            last_successful_tool_call=state.last_successful_tool_call,
            last_successful_tool_fingerprint=state.last_successful_tool_fingerprint,
        )
        if repeated_success_result is not None:
            return ToolExecutionDecision(
                loop_break_reason="repeat_tool_call_success_fallback",
                tool_result=repeated_success_result,
            )
        return ToolExecutionDecision(
            loop_break_reason="repeat_tool_call",
            tool_result=ToolResult(
                success=False,
                message="检测到同一工具与相近参数被重复调用，请改用其他工具、调整参数，或结束当前步骤。",
            ),
        )

    return await _invoke_tool(
        logger=logger,
        step=step,
        function_name=function_name,
        matched_tool=matched_tool,
        function_args=function_args,
        tool_name=tool_name,
        started_at=started_at,
    )


async def _invoke_tool(
    *,
    logger: logging.Logger,
    step: Step,
    function_name: str,
    matched_tool: BaseTool,
    function_args: Dict[str, Any],
    tool_name: str,
    started_at: float,
) -> ToolExecutionDecision:
    tool_started_at = now_perf()
    try:
        tool_result = await matched_tool.invoke(function_name, **function_args)
        tool_cost_ms = elapsed_ms(tool_started_at)
        if not isinstance(tool_result, ToolResult):
            tool_result = ToolResult(success=True, data=tool_result)
        return ToolExecutionDecision(tool_result=tool_result, tool_cost_ms=tool_cost_ms)
    except Exception as e:
        tool_cost_ms = elapsed_ms(tool_started_at)
        if _is_transient_research_transport_error(f"{e.__class__.__name__}: {e}") and function_name in {"search_web", "fetch_page"}:
            warning_message = (
                "检索链路瞬时错误，已触发快速收敛"
                if function_name == "search_web"
                else "页面抓取链路瞬时错误，已触发快速收敛"
            )
            log_runtime(
                logger,
                logging.WARNING,
                warning_message,
                step_id=str(step.id or ""),
                function_name=function_name,
                tool_name=tool_name,
                error=f"{e.__class__.__name__}: {e}",
                tool_elapsed_ms=tool_cost_ms,
                elapsed_ms=elapsed_ms(started_at),
            )
            transient_message = (
                "检索链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或切换其他路径。"
                if function_name == "search_web"
                else "页面抓取链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或改用其他来源。"
            )
            return ToolExecutionDecision(
                loop_break_reason="research_route_transport_error",
                tool_cost_ms=tool_cost_ms,
                tool_result=ToolResult(success=False, message=transient_message),
            )

        log_runtime(
            logger,
            logging.ERROR,
            "工具调用失败",
            step_id=str(step.id or ""),
            function_name=function_name,
            tool_name=tool_name,
            error=str(e),
            tool_elapsed_ms=tool_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
            exc_info=True,
        )
        return ToolExecutionDecision(
            tool_cost_ms=tool_cost_ms,
            tool_result=ToolResult(success=False, message=f"调用工具失败: {function_name}"),
        )


def _build_fetch_fingerprint(function_args: Dict[str, Any]) -> str:
    return str(function_args.get("url") or "").strip().lower()


def _is_transient_research_transport_error(raw_error: Any) -> bool:
    message = str(raw_error or "").strip().lower()
    if not message:
        return False
    transient_markers = (
        "remoteprotocolerror",
        "server disconnected",
        "connection reset",
        "connection aborted",
        "readtimeout",
        "connecttimeout",
        "connection error",
        "temporarily unavailable",
        "unexpected eof",
    )
    return any(marker in message for marker in transient_markers)


def _build_repeat_success_fallback_result(
    *,
    function_name: str,
    function_args: Dict[str, Any],
    current_tool_fingerprint: str,
    last_successful_tool_call: Dict[str, Any],
    last_successful_tool_fingerprint: str,
) -> Optional[ToolResult]:
    last_call = dict(last_successful_tool_call or {})
    last_function_name = str(last_call.get("function_name") or "").strip().lower()
    if not last_function_name or last_function_name != str(function_name or "").strip().lower():
        return None
    expected_fingerprint = str(last_successful_tool_fingerprint or "").strip()
    if not expected_fingerprint:
        expected_fingerprint = _build_tool_fingerprint(
            function_name=last_function_name,
            function_args=dict(last_call.get("function_args") or {}),
        )
    observed_fingerprint = str(current_tool_fingerprint or "").strip() or _build_tool_fingerprint(
        function_name=str(function_name or "").strip().lower(),
        function_args=dict(function_args or {}),
    )
    # P3-一次性收口：只有“同工具+同参数”才能复用成功结果，避免跨参数误收敛。
    if observed_fingerprint != expected_fingerprint:
        return None
    # 仅兜底可重复读取类函数，避免把写操作错误复用为成功。
    if last_function_name not in {"read_file", "list_files", "find_files", "search_in_file", "fetch_page"}:
        return None
    return ToolResult(
        success=True,
        message=str(last_call.get("message") or "").strip() or "已基于最近一次成功结果收敛当前步骤。",
        data=last_call.get("data"),
    )
