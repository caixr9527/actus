#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：工具调用生命周期与事件投递。"""

import inspect
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.domain.models import ToolEvent, ToolEventStatus, ToolResult
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.tools import BaseTool


@dataclass(slots=True)
class ToolCallLifecycle:
    tool_call_id: str
    tool_call_ref: str
    function_name: str
    normalized_function_name: str
    function_args: Dict[str, Any]
    tool_name: str = "unknown"


def build_tool_call_lifecycle(
        *,
        selected_tool_call: Dict[str, Any],
        parse_tool_call_args: Callable[[Any], Dict[str, Any]],
) -> Optional[ToolCallLifecycle]:
    function = selected_tool_call.get("function")
    if not isinstance(function, dict):
        return None
    function_name = str(function.get("name") or "").strip()
    if not function_name:
        return None
    tool_call_id = str(selected_tool_call.get("id") or selected_tool_call.get("call_id") or uuid.uuid4())
    tool_call_ref = str(selected_tool_call.get("call_id") or tool_call_id)
    function_args = parse_tool_call_args(function.get("arguments"))
    return ToolCallLifecycle(
        tool_call_id=tool_call_id,
        tool_call_ref=tool_call_ref,
        function_name=function_name,
        normalized_function_name=function_name.lower(),
        function_args=function_args,
    )


def bind_tool_name(lifecycle: ToolCallLifecycle, matched_tool: Optional[BaseTool]) -> None:
    lifecycle.tool_name = matched_tool.name if matched_tool is not None else "unknown"


def build_calling_event(lifecycle: ToolCallLifecycle) -> ToolEvent:
    return ToolEvent(
        tool_call_id=lifecycle.tool_call_id,
        tool_name=lifecycle.tool_name,
        function_name=lifecycle.function_name,
        function_args=lifecycle.function_args,
        status=ToolEventStatus.CALLING,
    )


def build_called_event(lifecycle: ToolCallLifecycle, tool_result: ToolResult) -> ToolEvent:
    return ToolEvent(
        tool_call_id=lifecycle.tool_call_id,
        tool_name=lifecycle.tool_name,
        function_name=lifecycle.function_name,
        function_args=lifecycle.function_args,
        function_result=tool_result,
        status=ToolEventStatus.CALLED,
    )


def build_tool_feedback_message(
        *,
        lifecycle: ToolCallLifecycle,
        tool_result: ToolResult,
        feedback_content_builder: Callable[[str, ToolResult], str],
) -> Dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": lifecycle.tool_call_id,
        "call_id": lifecycle.tool_call_ref,
        "function_name": lifecycle.function_name,
        "content": feedback_content_builder(lifecycle.function_name, tool_result),
    }


class ToolEventDispatcher:
    def __init__(
            self,
            *,
            logger: logging.Logger,
            on_tool_event: Optional[Callable[[ToolEvent], Optional[Awaitable[None]]]],
    ) -> None:
        self._logger = logger
        self._on_tool_event = on_tool_event
        self.emitted_events: List[ToolEvent] = []

    async def emit(self, event: ToolEvent) -> None:
        try:
            self.emitted_events.append(event)
            if self._on_tool_event is None:
                return
            maybe_awaitable = self._on_tool_event(event)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception as e:
            log_runtime(
                self._logger,
                logging.WARNING,
                "工具事件投递失败",
                tool_name=event.tool_name,
                function_name=event.function_name,
                status=event.status.value,
                error=str(e),
            )
