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
from app.domain.services.tools import BaseTool

from .parsers import format_attachments_for_prompt, normalize_attachments, safe_parse_json

logger = logging.getLogger(__name__)


def collect_available_tools(runtime_tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """收集当前步骤可用的工具 schema 列表。"""
    available_tools: List[Dict[str, Any]] = []
    for tool in runtime_tools:
        try:
            available_tools.extend(tool.get_tools())
        except Exception as e:
            logger.warning("读取工具[%s] schema 失败，已跳过: %s", getattr(tool, "name", "unknown"), e)

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
    logger.info(
        "LangGraph 多工具候选仲裁: total=%s, selected=%s",
        len(tool_calls),
        selected_function,
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
            logger.warning("LangGraph tool_call 参数解析失败，按空参数处理")
            return {}
    return {}


async def execute_step_with_prompt(
        *,
        llm: LLM,
        execution_prompt: str,
        step: Step,
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
        on_tool_event: Optional[Callable[[ToolEvent], Optional[Awaitable[None]]]] = None,
        extra_user_content_parts: Optional[List[Dict[str, Any]]] = None,
        disallowed_function_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[ToolEvent]]:
    """执行单步任务，支持“模型决策 -> 调工具 -> 回传模型”的最小循环。"""

    def _build_user_message() -> Dict[str, Any]:
        native_parts = [
            item
            for item in list(extra_user_content_parts or [])
            if isinstance(item, dict)
        ]
        if len(native_parts) == 0:
            return {"role": "user", "content": execution_prompt}
        return {
            "role": "user",
            "content": [{"type": "text", "text": execution_prompt}, *native_parts],
        }

    user_message = _build_user_message()
    blocked_function_names = {
        str(name or "").strip().lower()
        for name in list(disallowed_function_names or [])
        if str(name or "").strip()
    }

    async def _notify_tool_event(event: ToolEvent) -> None:
        if on_tool_event is None:
            return
        try:
            maybe_awaitable = on_tool_event(event)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception as e:
            logger.warning("LangGraph 投递实时工具事件失败，继续主流程: %s", e)

    if not runtime_tools:
        llm_message = await llm.invoke(
            messages=[user_message],
            tools=[],
            response_format={"type": "json_object"},
        )
        parsed = safe_parse_json(llm_message.get("content"))
        return {
            "success": bool(parsed.get("success", True)),
            "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
            "attachments": normalize_attachments(parsed.get("attachments")),
        }, []

    available_tools = collect_available_tools(runtime_tools)
    if blocked_function_names:
        available_tools = [
            tool_schema
            for tool_schema in available_tools
            if _extract_function_name(tool_schema) not in blocked_function_names
        ]
    if len(available_tools) == 0:
        llm_message = await llm.invoke(
            messages=[user_message],
            tools=[],
            response_format={"type": "json_object"},
        )
        parsed = safe_parse_json(llm_message.get("content"))
        return {
            "success": bool(parsed.get("success", True)),
            "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
            "attachments": normalize_attachments(parsed.get("attachments")),
        }, []

    messages: List[Dict[str, Any]] = [user_message]
    emitted_tool_events: List[ToolEvent] = []
    llm_message: Dict[str, Any] = {}

    for _ in range(max(1, int(max_tool_iterations))):
        llm_message = await llm.invoke(
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
        )
        tool_calls = llm_message.get("tool_calls") or []
        if len(tool_calls) == 0:
            break

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

        matched_tool = _resolve_tool_by_function_name(function_name=function_name, runtime_tools=runtime_tools)
        tool_name = matched_tool.name if matched_tool is not None else "unknown"
        calling_event = ToolEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            function_name=function_name,
            function_args=function_args,
            status=ToolEventStatus.CALLING,
        )
        emitted_tool_events.append(calling_event)
        await _notify_tool_event(calling_event)

        if function_name.strip().lower() in blocked_function_names:
            tool_result = ToolResult(success=False, message=f"工具已禁用: {function_name}")
        elif matched_tool is None:
            tool_result = ToolResult(success=False, message=f"无效工具: {function_name}")
        else:
            try:
                tool_result = await matched_tool.invoke(function_name, **function_args)
                if not isinstance(tool_result, ToolResult):
                    # 兼容少数工具返回 dict/str 的历史实现，统一收敛为 ToolResult。
                    tool_result = ToolResult(success=True, data=tool_result)
            except Exception as e:
                logger.exception("LangGraph 调用工具[%s]失败: %s", function_name, e)
                tool_result = ToolResult(success=False, message=f"调用工具失败: {function_name}")

        called_event = ToolEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            function_name=function_name,
            function_args=function_args,
            function_result=tool_result,
            status=ToolEventStatus.CALLED,
        )
        emitted_tool_events.append(called_event)
        await _notify_tool_event(called_event)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "function_name": function_name,
                "content": tool_result.model_dump_json(),
            }
        )

    parsed = safe_parse_json(llm_message.get("content"))
    return {
        "success": bool(parsed.get("success", True)),
        "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
        "attachments": normalize_attachments(parsed.get("attachments")),
    }, emitted_tool_events


def build_execution_prompt(
        *,
        execution_prompt_template: str,
        user_message: str,
        step_description: str,
        language: str,
        attachments: List[str],
) -> str:
    """基于模板构建单步执行提示词。"""
    return execution_prompt_template.format(
        message=user_message,
        attachments=format_attachments_for_prompt(attachments),
        language=language,
        step=step_description,
    )
