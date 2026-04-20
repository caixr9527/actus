#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具调用执行与专用分支处理。"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.domain.models import Step, ToolResult
from app.domain.services.runtime.contracts.runtime_logging import elapsed_ms, log_runtime, now_perf
from app.domain.services.tools import BaseTool
from app.domain.services.workspace_runtime.policies import (
    build_tool_fingerprint as _build_tool_fingerprint,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_argument_normalizers import (
    normalize_tool_execution_args,
)


@dataclass(slots=True)
class ToolExecutionDecision:
    """executor 域输出。

    业务含义：
    - `tool_result` 是一次真实工具调用后的原始执行结果；
    - `tool_cost_ms` 记录工具侧耗时；
    - `loop_break_reason` 仅用于 executor 直接识别出的快速收敛场景，例如瞬时传输错误。
    """
    tool_result: ToolResult
    tool_cost_ms: int = 0
    loop_break_reason: str = ""
    executed_function_args: Dict[str, Any] = field(default_factory=dict)


async def execute_tool_with_policy(
        *,
        logger: logging.Logger,
        step: Step,
        function_name: str,
        normalized_function_name: str,
        function_args: Dict[str, Any],
        matched_tool: BaseTool,
        tool_name: str,
        started_at: float,
) -> ToolExecutionDecision:
    """执行真实工具调用，并做 executor 域内的少量后处理。

    当前职责：
    - 对 `search_web` query 做 research 主查询规范化收口，但不做关键词化改写；
    - 将 research 链路的瞬时传输错误统一转换为可收敛失败信号；
    - 不处理约束判断，不写执行状态。
    """
    normalized_function_args = normalize_tool_execution_args(
        normalized_function_name=normalized_function_name,
        function_args=function_args,
    )
    log_runtime(
        logger,
        logging.INFO,
        "开始执行真实工具调用",
        step_id=str(step.id or ""),
        function_name=function_name,
        tool_name=tool_name,
        arg_keys=sorted(dict(normalized_function_args or {}).keys()),
    )
    if normalized_function_name == "search_web":
        invoked = await _invoke_tool(
            logger=logger,
            step=step,
            function_name=function_name,
            matched_tool=matched_tool,
            function_args=normalized_function_args,
            tool_name=tool_name,
            started_at=started_at,
        )
        if (not bool(invoked.tool_result.success)) and _is_transient_research_transport_error(
                invoked.tool_result.message):
            return ToolExecutionDecision(
                tool_cost_ms=invoked.tool_cost_ms,
                loop_break_reason="research_route_transport_error",
                tool_result=ToolResult(
                    success=False,
                    message="检索链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或切换其他路径。",
                ),
                executed_function_args=dict(normalized_function_args or {}),
            )
        return invoked

    if normalized_function_name == "fetch_page":
        invoked = await _invoke_tool(
            logger=logger,
            step=step,
            function_name=function_name,
            matched_tool=matched_tool,
            function_args=normalized_function_args,
            tool_name=tool_name,
            started_at=started_at,
        )
        if (not bool(invoked.tool_result.success)) and _is_transient_research_transport_error(
                invoked.tool_result.message):
            return ToolExecutionDecision(
                tool_cost_ms=invoked.tool_cost_ms,
                loop_break_reason="research_route_transport_error",
                tool_result=ToolResult(
                    success=False,
                    message="页面抓取链路出现瞬时网络抖动（如连接中断），当前步骤已停止重试。请稍后重试或改用其他来源。",
                ),
                executed_function_args=dict(normalized_function_args or {}),
            )
        return invoked

    return await _invoke_tool(
        logger=logger,
        step=step,
        function_name=function_name,
        matched_tool=matched_tool,
        function_args=normalized_function_args,
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
    """调用底层工具实现，并把异常统一转换为 `ToolExecutionDecision`。"""
    tool_started_at = now_perf()
    try:
        tool_result = await matched_tool.invoke(function_name, **function_args)
        tool_cost_ms = elapsed_ms(tool_started_at)
        if not isinstance(tool_result, ToolResult):
            tool_result = ToolResult(success=True, data=tool_result)
        log_runtime(
            logger,
            logging.INFO,
            "真实工具调用成功返回",
            step_id=str(step.id or ""),
            function_name=function_name,
            tool_name=tool_name,
            success=bool(tool_result.success),
            tool_elapsed_ms=tool_cost_ms,
            response_keys=sorted(tool_result.data.keys()) if isinstance(tool_result.data, dict) else [],
        )
        return ToolExecutionDecision(
            tool_result=tool_result,
            tool_cost_ms=tool_cost_ms,
            executed_function_args=dict(function_args or {}),
        )
    except Exception as e:
        tool_cost_ms = elapsed_ms(tool_started_at)
        if _is_transient_research_transport_error(f"{e.__class__.__name__}: {e}") and function_name in {"search_web",
                                                                                                        "fetch_page"}:
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
                executed_function_args=dict(function_args or {}),
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
            executed_function_args=dict(function_args or {}),
        )


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


def build_repeat_success_fallback_result(
        *,
        function_name: str,
        function_args: Dict[str, Any],
        current_tool_fingerprint: str,
        last_successful_tool_call: Dict[str, Any],
        last_successful_tool_fingerprint: str,
) -> Optional[ToolResult]:
    """重复调用命中时，若与最近成功只读调用完全一致，则复用成功结果收敛。"""
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
    if observed_fingerprint != expected_fingerprint:
        return None
    if last_function_name not in {"read_file", "list_files", "find_files", "search_in_file", "fetch_page"}:
        return None
    return ToolResult(
        success=True,
        message=str(last_call.get("message") or "").strip() or "已基于最近一次成功结果收敛当前步骤。",
        data=last_call.get("data"),
    )
