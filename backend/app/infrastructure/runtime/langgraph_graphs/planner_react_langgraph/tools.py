#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 工具调用循环与仲裁逻辑。"""

import inspect
import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.domain.external import LLM
from app.domain.models import Step, ToolEvent, ToolEventStatus, ToolResult
from app.domain.services.prompts import SYSTEM_PROMPT, REACT_SYSTEM_PROMPT
from app.domain.services.tools import BaseTool
from .parsers import normalize_attachments, safe_parse_json
from .runtime_logging import elapsed_ms, log_runtime, now_perf

logger = logging.getLogger(__name__)

WAIT_STEP_KEYWORDS: tuple[str, ...] = (
    "等待",
    "选择",
    "确认",
    "询问",
    "提问",
    "回复",
    "补充",
    "澄清",
    "审批",
    "review",
    "approve",
    "approval",
    "confirm",
    "choose",
    "select",
    "input",
    "answer",
    "human",
)


def collect_available_tools(runtime_tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """收集当前步骤可用的工具 schema 列表。"""
    available_tools: List[Dict[str, Any]] = []
    for tool in runtime_tools:
        try:
            available_tools.extend(tool.get_tools())
        except Exception as e:
            log_runtime(
                logger,
                logging.WARNING,
                "工具Schema读取失败",
                tool_name=getattr(tool, "name", "unknown"),
                error=str(e),
            )

    def _tool_priority(tool_schema: Dict[str, Any]) -> Tuple[int, str]:
        function_name = str(
            (tool_schema.get("function") or {}).get("name")
            if isinstance(tool_schema, dict)
            else ""
        ).strip().lower()
        # 搜索类优先，浏览器类后置：多数检索任务先走 search 更快。
        if "search" in function_name:
            return 0, function_name
        if function_name.startswith("browser_"):
            return 80, function_name
        return 20, function_name

    available_tools.sort(key=_tool_priority)
    return available_tools


def _extract_function_name(tool_schema: Dict[str, Any]) -> str:
    if not isinstance(tool_schema, dict):
        return ""
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip().lower()


def _tool_call_priority(function_name: str) -> int:
    normalized_name = function_name.strip().lower()
    if "search" in normalized_name:
        return 0
    if normalized_name.startswith("browser_"):
        return 80
    return 20


def pick_preferred_tool_call(
        tool_calls: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """从同轮多个 tool_call 中挑选本轮优先执行的候选。"""
    if len(tool_calls) == 0:
        return None
    if len(tool_calls) == 1:
        return tool_calls[0] if isinstance(tool_calls[0], dict) else None

    available_function_names: set[str] = set()
    for tool_schema in available_tools:
        function_name = _extract_function_name(tool_schema)
        if function_name:
            available_function_names.add(function_name)

    ranked_candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for index, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        function_name = str(function.get("name") or "").strip()
        if not function_name:
            continue

        normalized_name = function_name.lower()
        priority = _tool_call_priority(function_name)
        if normalized_name not in available_function_names:
            priority += 1000
        ranked_candidates.append((priority, index, raw_call))

    if len(ranked_candidates) == 0:
        return None

    ranked_candidates.sort(key=lambda item: (item[0], item[1]))
    selected_call = ranked_candidates[0][2]
    selected_function = str((selected_call.get("function") or {}).get("name") or "")
    log_runtime(
        logger,
        logging.INFO,
        "工具候选仲裁完成",
        candidate_count=len(tool_calls),
        selected_function=selected_function,
    )
    return selected_call


def _resolve_tool_by_function_name(function_name: str, runtime_tools: List[BaseTool]) -> Optional[BaseTool]:
    for tool in runtime_tools:
        try:
            if tool.has_tool(function_name):
                return tool
        except Exception:
            continue
    return None


def _parse_tool_call_args(raw_arguments: Any) -> Dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            log_runtime(
                logger,
                logging.WARNING,
                "工具参数解析失败",
                raw_length=len(raw_arguments),
            )
            return {}
    return {}


def _step_allows_user_wait(step: Step, function_args: Dict[str, Any]) -> bool:
    takeover = str(function_args.get("suggest_user_takeover") or "").strip().lower()
    if takeover == "browser":
        return True

    candidate_parts = [
        str(step.title or "").strip(),
        str(step.description or "").strip(),
        *[str(item or "").strip() for item in list(step.success_criteria or [])],
    ]
    candidate_text = " ".join([part for part in candidate_parts if part]).lower()
    if not candidate_text:
        return False
    return any(keyword in candidate_text for keyword in WAIT_STEP_KEYWORDS)


async def execute_step_with_prompt(
        *,
        llm: LLM,
        step: Step,
        runtime_tools: Optional[List[BaseTool]],
        max_tool_iterations: int = 5,
        on_tool_event: Optional[Callable[[ToolEvent], Optional[Awaitable[None]]]],
        user_content: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[ToolEvent]]:
    """执行单步任务，支持“模型决策 -> 调工具 -> 回传模型”的最小循环。"""
    started_at = now_perf()

    emitted_tool_events: List[ToolEvent] = []

    async def _notify_tool_event(event: ToolEvent) -> None:
        try:
            emitted_tool_events.append(event)
            if on_tool_event is None:
                return
            maybe_awaitable = on_tool_event(event)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception as e:
            log_runtime(
                logger,
                logging.WARNING,
                "工具事件投递失败",
                tool_name=event.tool_name,
                function_name=event.function_name,
                status=event.status.value,
                error=str(e),
            )

    available_tools = collect_available_tools(runtime_tools)
    log_runtime(
        logger,
        logging.INFO,
        "工具执行循环准备完成",
        step_id=str(step.id or ""),
        step_title=str(step.title or step.description or ""),
        available_tool_count=len(available_tools),
        max_tool_iterations=max(1, int(max_tool_iterations)),
    )

    if len(available_tools) == 0:
        log_runtime(
            logger,
            logging.INFO,
            "当前步骤无可用工具",
            step_id=str(step.id or ""),
        )
        llm_started_at = now_perf()
        llm_message = await llm.invoke(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + REACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            tools=[],
            response_format={"type": "json_object"},
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        parsed = safe_parse_json(llm_message.get("content"))
        log_runtime(
            logger,
            logging.INFO,
            "无工具模式执行完成",
            step_id=str(step.id or ""),
            llm_elapsed_ms=llm_cost_ms,
            elapsed_ms=elapsed_ms(started_at),
            attachment_count=len(normalize_attachments(parsed.get("attachments"))),
        )
        return {
            "success": bool(parsed.get("success", True)),
            "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
            "attachments": normalize_attachments(parsed.get("attachments")),
        }, []

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + REACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    llm_message: Dict[str, Any] = {}

    for index in range(max(1, int(max_tool_iterations))):
        log_runtime(
            logger,
            logging.INFO,
            "开始工具决策轮次",
            step_id=str(step.id or ""),
            iteration=index,
        )
        llm_started_at = now_perf()
        llm_message = await llm.invoke(
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
        )
        llm_cost_ms = elapsed_ms(llm_started_at)
        tool_calls = llm_message.get("tool_calls") or []
        if len(tool_calls) == 0:
            parsed = safe_parse_json(llm_message.get("content"))
            # 如果模型返回结果为空，则尝试重新执行
            if parsed.get("success", True) and str(parsed.get("result", "")).strip() == '':
                log_runtime(
                    logger,
                    logging.WARNING,
                    "模型结果为空，准备重试",
                    step_id=str(step.id or ""),
                    iteration=index,
                    llm_elapsed_ms=llm_cost_ms,
                    elapsed_ms=elapsed_ms(started_at),
                )
                continue
            log_runtime(
                logger,
                logging.INFO,
                "未调用工具直接完成当前轮次",
                step_id=str(step.id or ""),
                iteration=index,
                success=bool(parsed.get("success", True)),
                attachment_count=len(normalize_attachments(parsed.get("attachments"))),
                llm_elapsed_ms=llm_cost_ms,
                elapsed_ms=elapsed_ms(started_at),
            )
            return {
                "success": bool(parsed.get("success", True)),
                "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
                "attachments": normalize_attachments(parsed.get("attachments")),
            }, emitted_tool_events

        selected_tool_call = pick_preferred_tool_call(
            tool_calls=[item for item in tool_calls if isinstance(item, dict)],
            available_tools=available_tools,
        )
        if selected_tool_call is None:
            continue

        messages.append(
            {
                "role": "assistant",
                "content": llm_message.get("content"),
                "tool_calls": [selected_tool_call],
            }
        )

        function = selected_tool_call.get("function")
        if not isinstance(function, dict):
            continue

        function_name = str(function.get("name") or "").strip()
        if not function_name:
            continue
        tool_call_id = str(selected_tool_call.get("id") or uuid.uuid4())
        function_args = _parse_tool_call_args(function.get("arguments"))
        log_runtime(
            logger,
            logging.INFO,
            "已选择工具调用",
            step_id=str(step.id or ""),
            iteration=index,
            tool_call_id=tool_call_id,
            function_name=function_name,
            arg_keys=sorted(function_args.keys()),
        )

        matched_tool = _resolve_tool_by_function_name(function_name=function_name, runtime_tools=runtime_tools)
        tool_name = matched_tool.name if matched_tool is not None else "unknown"
        tool_cost_ms = 0
        calling_event = ToolEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            function_name=function_name,
            function_args=function_args,
            status=ToolEventStatus.CALLING,
        )
        await _notify_tool_event(calling_event)

        if matched_tool is None:
            log_runtime(
                logger,
                logging.WARNING,
                "工具调用无效",
                step_id=str(step.id or ""),
                function_name=function_name,
            )
            tool_result = ToolResult(success=False, message=f"无效工具: {function_name}")
        elif function_name == "message_ask_user" and not _step_allows_user_wait(step, function_args):
            log_runtime(
                logger,
                logging.WARNING,
                "提前请求用户交互，已拦截",
                step_id=str(step.id or ""),
                function_name=function_name,
                step_description=str(step.description or ""),
            )
            tool_result = ToolResult(
                success=False,
                message="当前步骤不允许向用户提问。请先完成当前步骤，只能在明确需要用户确认/选择/输入的步骤中使用该工具。",
            )
        else:
            try:
                tool_started_at = now_perf()
                tool_result = await matched_tool.invoke(function_name, **function_args)
                tool_cost_ms = elapsed_ms(tool_started_at)
                if not isinstance(tool_result, ToolResult):
                    # 兼容少数工具返回 dict/str 的历史实现，统一收敛为 ToolResult。
                    tool_result = ToolResult(success=True, data=tool_result)
            except Exception as e:
                tool_cost_ms = elapsed_ms(tool_started_at)
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
                tool_result = ToolResult(success=False, message=f"调用工具失败: {function_name}")

        called_event = ToolEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            function_name=function_name,
            function_args=function_args,
            function_result=tool_result,
            status=ToolEventStatus.CALLED,
        )
        await _notify_tool_event(called_event)
        interrupt_request = (
            tool_result.data.get("interrupt")
            if isinstance(tool_result.data, dict)
            else None
        )
        log_runtime(
            logger,
            logging.INFO,
            "工具调用完成",
            step_id=str(step.id or ""),
            tool_call_id=tool_call_id,
            function_name=function_name,
            tool_name=tool_name,
            success=bool(tool_result.success),
            has_interrupt=bool(isinstance(interrupt_request, dict) and interrupt_request),
            llm_elapsed_ms=llm_cost_ms,
            tool_elapsed_ms=tool_cost_ms if matched_tool is not None else 0,
            elapsed_ms=elapsed_ms(started_at),
        )
        if isinstance(interrupt_request, dict) and interrupt_request:
            log_runtime(
                logger,
                logging.INFO,
                "工具请求进入等待",
                step_id=str(step.id or ""),
                tool_call_id=tool_call_id,
                function_name=function_name,
                interrupt_kind=str(interrupt_request.get("kind") or ""),
                elapsed_ms=elapsed_ms(started_at),
            )
            return {
                "success": True,
                "interrupt_request": interrupt_request,
                "result": "",
                "attachments": [],
            }, emitted_tool_events
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "function_name": function_name,
                "content": tool_result.model_dump_json(),
            }
        )

    parsed = safe_parse_json(llm_message.get("content"))
    log_runtime(
        logger,
        logging.INFO,
        "达到最大工具轮次，返回当前结果",
        step_id=str(step.id or ""),
        iteration_count=max(1, int(max_tool_iterations)),
        success=bool(parsed.get("success", True)),
        attachment_count=len(normalize_attachments(parsed.get("attachments"))),
        elapsed_ms=elapsed_ms(started_at),
    )
    return {
        "success": bool(parsed.get("success", True)),
        "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
        "attachments": normalize_attachments(parsed.get("attachments")),
    }, emitted_tool_events
