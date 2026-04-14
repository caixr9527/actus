#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构辅助模块单测。"""

import logging
import time
import asyncio

from app.domain.models import Step, ToolResult
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution_context import (
    ExecutionContext,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution_state import (
    ExecutionState,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.finalizer import (
    finalize_no_tool_call,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_effects import (
    apply_tool_result_effects,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_events import (
    ToolEventDispatcher,
    bind_tool_name,
    build_called_event,
    build_calling_event,
    build_tool_call_lifecycle,
    build_tool_feedback_message,
)


def test_finalize_no_tool_call_should_retry_empty_payload_and_block_human_wait() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="请等待用户确认后继续")

    retry_result = finalize_no_tool_call(
        logger=logger,
        step=step,
        task_mode="general",
        llm_message={"content": '{"success": true, "summary": "", "delivery_text": ""}'},
        llm_cost_ms=1,
        started_at=time.perf_counter(),
        iteration=0,
        runtime_recent_action={},
    )
    assert retry_result.action == "retry"

    wait_result = finalize_no_tool_call(
        logger=logger,
        step=step,
        task_mode="human_wait",
        llm_message={"content": '{"success": true, "summary": "ok"}'},
        llm_cost_ms=1,
        started_at=time.perf_counter(),
        iteration=0,
        runtime_recent_action={"last_failed_action": {"function_name": "search_web"}},
    )
    assert wait_result.action == "return"
    assert wait_result.payload is not None
    assert wait_result.payload.get("success") is False
    assert isinstance(wait_result.payload.get("blockers"), list)


def test_apply_tool_result_effects_should_update_research_search_state() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索网页并读取正文")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names=set(),
        browser_route_enabled=False,
        blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        requested_max_tool_iterations=3,
        effective_max_tool_iterations=3,
        allow_ask_user=False,
        research_route_enabled=True,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()
    tool_result = ToolResult(
        success=True,
        data={
            "results": [
                {"url": "https://example.com/a"},
                {"url": "https://example.com/a"},
                {"url": "https://example.com/b"},
            ]
        },
    )

    effects_result = apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name="search_web",
        normalized_function_name="search_web",
        function_args={"query": "langgraph"},
        tool_result=tool_result,
        loop_break_reason="",
        browser_route_state_key="state-key",
        execution_context=execution_context,
        execution_state=execution_state,
    )

    assert effects_result.loop_break_reason == ""
    assert execution_state.research_search_ready is True
    assert execution_state.research_candidate_urls == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_tool_call_lifecycle_and_events_should_keep_contract_consistent() -> None:
    lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "tool-id-1",
            "call_id": "tool-ref-1",
            "function": {
                "name": "search_web",
                "arguments": '{"query":"langgraph"}',
            },
        },
        parse_tool_call_args=lambda raw: {"query": "langgraph"} if isinstance(raw, str) else {},
    )
    assert lifecycle is not None
    assert lifecycle.tool_call_id == "tool-id-1"
    assert lifecycle.tool_call_ref == "tool-ref-1"
    assert lifecycle.normalized_function_name == "search_web"

    class _FakeTool:
        name = "search-tool"

    bind_tool_name(lifecycle, _FakeTool())
    calling_event = build_calling_event(lifecycle)
    called_event = build_called_event(lifecycle, ToolResult(success=True, data={"k": "v"}))
    feedback_message = build_tool_feedback_message(
        lifecycle=lifecycle,
        tool_result=ToolResult(success=True, message="ok"),
        feedback_content_builder=lambda fn, result: f"{fn}:{bool(result.success)}",
    )
    assert calling_event.tool_name == "search-tool"
    assert called_event.function_result is not None
    assert feedback_message["tool_call_id"] == "tool-id-1"
    assert feedback_message["function_name"] == "search_web"


def test_tool_event_dispatcher_should_emit_and_callback() -> None:
    logger = logging.getLogger(__name__)
    captured = []

    async def _on_tool_event(event):
        captured.append(event.function_name)

    dispatcher = ToolEventDispatcher(
        logger=logger,
        on_tool_event=_on_tool_event,
    )
    lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "tool-id-2",
            "function": {
                "name": "fetch_page",
                "arguments": {"url": "https://example.com"},
            },
        },
        parse_tool_call_args=lambda raw: raw if isinstance(raw, dict) else {},
    )
    assert lifecycle is not None
    asyncio.run(dispatcher.emit(build_calling_event(lifecycle)))
    assert len(dispatcher.emitted_events) == 1
    assert captured == ["fetch_page"]
