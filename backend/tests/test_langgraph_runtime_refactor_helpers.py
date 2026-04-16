#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构辅助模块单测。"""

import logging
import time
import asyncio
import json
from typing import Any, Dict, List, Optional

from app.domain.models import Step, ToolResult
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution_context import (
    ExecutionContext,
    build_execution_context,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution_state import (
    ExecutionState,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.finalizer import (
    finalize_max_iterations,
    finalize_no_tool_call,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.loop_breaks import (
    build_loop_break_result,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_effects import (
    apply_tool_result_effects,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_guards import (
    evaluate_tool_guard,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_handlers import (
    execute_tool_with_policy,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.judge import (
    ConvergenceJudge,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_events import (
    ToolEventDispatcher,
    bind_tool_name,
    build_called_event,
    build_calling_event,
    build_tool_call_lifecycle,
    build_tool_feedback_message,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tools import execute_step_with_prompt


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


def test_finalize_max_iterations_should_always_return_failure_payload() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="读取并整理文件内容")

    payload = finalize_max_iterations(
        logger=logger,
        step=step,
        task_mode="file_processing",
        llm_message={"content": '{"success": true, "summary": "误报完成"}'},
        started_at=time.perf_counter(),
        requested_max_tool_iterations=3,
        iteration_count=3,
        runtime_recent_action={"last_failed_action": {"function_name": "read_file"}},
    )

    assert payload["success"] is False
    assert payload["summary"] == "当前步骤暂时未能完成：读取并整理文件内容"
    assert payload["blockers"] == ["达到最大工具调用轮次，当前步骤仍未形成可交付结果。"]


def test_finalize_max_iterations_should_converge_success_when_file_facts_ready() -> None:
    logger = logging.getLogger(__name__)
    step = Step(
        description="获取目录文件列表并以文本输出",
        task_mode_hint="file_processing",
        output_mode="inline",
        delivery_role="final",
        artifact_policy="default",
    )
    payload = finalize_max_iterations(
        logger=logger,
        step=step,
        task_mode="file_processing",
        llm_message={"content": '{"success": false, "summary": ""}'},
        started_at=time.perf_counter(),
        requested_max_tool_iterations=6,
        iteration_count=6,
        runtime_recent_action={"last_failed_action": {"function_name": "list_files"}},
        step_file_context={"called_functions": {"list_files", "read_file"}},
    )
    assert payload["success"] is True
    assert "收敛成功" in str(payload["summary"])


def test_finalize_max_iterations_should_include_research_gap_hint() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="调研并给出结论")

    payload = finalize_max_iterations(
        logger=logger,
        step=step,
        task_mode="research",
        llm_message={"content": '{"success": false, "summary": ""}'},
        started_at=time.perf_counter(),
        requested_max_tool_iterations=6,
        iteration_count=6,
        runtime_recent_action={
            "research_progress": {
                "missing_signals": [
                    "至少读取 2 个来源页面正文",
                    "至少覆盖 2 个不同站点来源",
                ]
            }
        },
    )

    assert payload["success"] is False
    assert "研究缺口" in str(payload.get("next_hint") or "")
    assert "至少读取 2 个来源页面正文" in str(payload.get("next_hint") or "")


def test_apply_tool_result_effects_should_update_research_search_state() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索网页并读取正文")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names=set(),
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
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


def test_build_execution_context_should_not_infer_read_only_from_prompt_noise() -> None:
    step = Step(description="生成报告并保存结果")
    prompt_like_user_content = [
        {
            "type": "text",
            "text": (
                "执行指令中包含大量读取规则：读取页面、读取文件；"
                "但这只是系统提示词，不代表用户只读意图。"
            ),
        }
    ]
    ctx = build_execution_context(
        step=step,
        task_mode="general",
        max_tool_iterations=3,
        user_content=prompt_like_user_content,
        read_only_intent_text="继续执行并整理结果",
        has_available_file_context=True,
        available_tools=[],
        available_function_names={"write_file", "replace_in_file", "shell_execute"},
    )
    assert ctx.read_only_file_blocked_function_names == set()
    assert "write_file" not in ctx.blocked_function_names
    assert "replace_in_file" not in ctx.blocked_function_names


def test_build_execution_context_should_block_write_tools_for_explicit_read_only_intent() -> None:
    step = Step(description="读取并展示文件内容")
    ctx = build_execution_context(
        step=step,
        task_mode="general",
        max_tool_iterations=3,
        user_content=[{"type": "text", "text": "任意内容"}],
        read_only_intent_text="读取 /tmp/report.md 并展示内容，不要修改文件",
        has_available_file_context=True,
        available_tools=[],
        available_function_names={"write_file", "replace_in_file", "shell_execute"},
    )
    assert ctx.read_only_file_blocked_function_names == {"write_file", "replace_in_file", "shell_execute"}


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


def test_execute_tool_with_policy_should_use_success_fallback_on_repeat_read_file() -> None:
    class _ReadOnlyTool:
        name = "file"

        async def invoke(self, function_name, **kwargs):
            return ToolResult(success=True, data={"content": "hello"})

    step = Step(
        id="step-1",
        title="读取文件",
        description="读取 /tmp/hello.txt 内容",
    )
    ctx = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取文件内容"}],
        available_tools=[],
        available_function_names={"read_file"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=5,
        effective_max_tool_iterations=5,
        allow_ask_user=False,
        research_route_enabled=False,
        research_has_explicit_url=False,
    )
    state = ExecutionState()
    state.same_tool_repeat_count = 3
    state.last_tool_fingerprint = "same-fingerprint"
    state.last_successful_tool_call = {
        "function_name": "read_file",
        "function_args": {"filepath": "/tmp/hello.txt"},
        "message": "读取成功",
        "data": {"content": "hello"},
    }
    state.last_successful_tool_fingerprint = "same-fingerprint"

    decision = asyncio.run(
        execute_tool_with_policy(
            logger=logging.getLogger(__name__),
            step=step,
            function_name="read_file",
            normalized_function_name="read_file",
            function_args={"filepath": "/tmp/hello.txt"},
            matched_tool=_ReadOnlyTool(),
            tool_name="file",
            browser_route_state_key="",
            ctx=ctx,
            state=state,
            started_at=time.perf_counter(),
        )
    )

    assert decision.loop_break_reason == "repeat_tool_call_success_fallback"
    assert decision.tool_result.success is True
    assert decision.tool_result.data == {"content": "hello"}


def test_execute_tool_with_policy_should_not_use_success_fallback_when_fingerprint_mismatch() -> None:
    class _ReadOnlyTool:
        name = "file"

        async def invoke(self, function_name, **kwargs):
            return ToolResult(success=True, data={"content": "hello"})

    step = Step(
        id="step-1",
        title="读取文件",
        description="读取 /tmp/hello.txt 内容",
    )
    ctx = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取文件内容"}],
        available_tools=[],
        available_function_names={"read_file"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=5,
        effective_max_tool_iterations=5,
        allow_ask_user=False,
        research_route_enabled=False,
        research_has_explicit_url=False,
    )
    state = ExecutionState()
    state.same_tool_repeat_count = 3
    state.last_tool_fingerprint = "fingerprint-b"
    state.last_successful_tool_call = {
        "function_name": "read_file",
        "function_args": {"filepath": "/tmp/hello-a.txt"},
        "message": "读取成功",
        "data": {"content": "hello-a"},
    }
    state.last_successful_tool_fingerprint = "fingerprint-a"

    decision = asyncio.run(
        execute_tool_with_policy(
            logger=logging.getLogger(__name__),
            step=step,
            function_name="read_file",
            normalized_function_name="read_file",
            function_args={"filepath": "/tmp/hello-b.txt"},
            matched_tool=_ReadOnlyTool(),
            tool_name="file",
            browser_route_state_key="",
            ctx=ctx,
            state=state,
            started_at=time.perf_counter(),
        )
    )

    assert decision.loop_break_reason == "repeat_tool_call"
    assert decision.tool_result.success is False


def test_apply_tool_result_effects_should_record_success_feedback_payload_for_fallback_delivery() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="读取并输出文件内容")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取并输出内容"}],
        available_tools=[],
        available_function_names={"read_file"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=3,
        effective_max_tool_iterations=3,
        allow_ask_user=False,
        research_route_enabled=False,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()
    tool_result = ToolResult(
        success=True,
        message="读取成功",
        data={
            "filepath": "/tmp/a.txt",
            "content": "A content",
        },
    )

    _ = apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name="read_file",
        normalized_function_name="read_file",
        function_args={"filepath": "/tmp/a.txt"},
        tool_result=tool_result,
        loop_break_reason="",
        browser_route_state_key="",
        execution_context=execution_context,
        execution_state=execution_state,
    )

    feedback_content = str(execution_state.last_successful_tool_call.get("feedback_content") or "")
    parsed_feedback = json.loads(feedback_content)
    assert parsed_feedback["success"] is True
    assert parsed_feedback["data"]["content"] == "A content"


def test_apply_tool_result_effects_should_record_research_progress_snapshot() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索并读取目的地详情")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web", "fetch_page"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=5,
        effective_max_tool_iterations=5,
        allow_ask_user=False,
        research_route_enabled=True,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()

    _ = apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name="search_web",
        normalized_function_name="search_web",
        function_args={"query": "淀山湖 周末 行程", "language": "zh-CN"},
        tool_result=ToolResult(
            success=True,
            data={
                "query": "淀山湖 周末 行程",
                "results": [
                    {"url": "https://example.com/a"},
                    {"url": "https://travel.example.org/b"},
                ],
            },
        ),
        loop_break_reason="",
        browser_route_state_key="",
        execution_context=execution_context,
        execution_state=execution_state,
    )
    progress_after_search = dict(execution_state.runtime_recent_action.get("research_progress") or {})
    assert progress_after_search.get("query_count") == 1
    assert progress_after_search.get("candidate_url_count") == 2
    assert progress_after_search.get("candidate_domain_count") == 2
    assert progress_after_search.get("fetched_domain_count") == 0
    assert progress_after_search.get("fetch_success_count") == 0

    _ = apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name="fetch_page",
        normalized_function_name="fetch_page",
        function_args={"url": "https://travel.example.org/b"},
        tool_result=ToolResult(
            success=True,
            data={
                "url": "https://travel.example.org/b",
                "final_url": "https://travel.example.org/b",
                "title": "淀山湖攻略",
                "content": "行程与交通说明",
            },
        ),
        loop_break_reason="",
        browser_route_state_key="",
        execution_context=execution_context,
        execution_state=execution_state,
    )
    progress_after_fetch = dict(execution_state.runtime_recent_action.get("research_progress") or {})
    assert progress_after_fetch.get("fetch_success_count") == 1
    assert progress_after_fetch.get("fetched_url_count") == 1
    assert float(progress_after_fetch.get("coverage_score") or 0.0) > 0.0


def test_evaluate_tool_guard_should_block_same_domain_fetch_when_cross_domain_candidate_exists() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索并补齐多来源证据")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web", "fetch_page"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=5,
        effective_max_tool_iterations=5,
        allow_ask_user=False,
        research_route_enabled=True,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()
    execution_state.research_search_ready = True
    execution_state.research_fetch_success_count = 1
    execution_state.research_candidate_urls = [
        "https://same.example.com/a",
        "https://other.example.org/b",
    ]
    execution_state.research_fetched_urls = [
        "https://same.example.com/intro",
    ]
    execution_state.research_fetched_domains = ["same.example.com"]

    guard = evaluate_tool_guard(
        logger=logger,
        step=step,
        task_mode="research",
        function_name="fetch_page",
        normalized_function_name="fetch_page",
        function_args={"url": "https://same.example.com/a"},
        matched_tool=object(),  # type: ignore[arg-type]
        iteration_blocked_function_names=set(),
        ctx=execution_context,
        state=execution_state,
    )
    assert guard.should_skip is True
    assert guard.loop_break_reason == ""
    assert "different" not in str(guard.tool_result.message or "").lower()
    assert "不同站点" in str(guard.tool_result.message or "")


def test_evaluate_tool_guard_should_allow_cross_domain_fetch_when_coverage_not_enough() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索并补齐多来源证据")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web", "fetch_page"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=5,
        effective_max_tool_iterations=5,
        allow_ask_user=False,
        research_route_enabled=True,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()
    execution_state.research_search_ready = True
    execution_state.research_fetch_success_count = 1
    execution_state.research_candidate_urls = [
        "https://same.example.com/a",
        "https://other.example.org/b",
    ]
    execution_state.research_fetched_urls = [
        "https://same.example.com/intro",
    ]
    execution_state.research_fetched_domains = ["same.example.com"]

    guard = evaluate_tool_guard(
        logger=logger,
        step=step,
        task_mode="research",
        function_name="fetch_page",
        normalized_function_name="fetch_page",
        function_args={"url": "https://other.example.org/b"},
        matched_tool=object(),  # type: ignore[arg-type]
        iteration_blocked_function_names=set(),
        ctx=execution_context,
        state=execution_state,
    )
    assert guard.should_skip is False


def test_evaluate_tool_guard_should_block_search_web_when_pending_candidates_and_coverage_not_enough() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索并补齐多来源证据")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web", "fetch_page"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=5,
        effective_max_tool_iterations=5,
        allow_ask_user=False,
        research_route_enabled=True,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()
    execution_state.research_search_ready = True
    execution_state.research_fetch_completed = True
    execution_state.research_fetch_success_count = 1
    execution_state.research_fetched_domains = ["same.example.com"]
    execution_state.research_candidate_urls = [
        "https://same.example.com/a",
        "https://other.example.org/b",
    ]
    execution_state.research_fetched_urls = [
        "https://same.example.com/read",
    ]

    guard = evaluate_tool_guard(
        logger=logger,
        step=step,
        task_mode="research",
        function_name="search_web",
        normalized_function_name="search_web",
        function_args={"query": "重新搜索"},
        matched_tool=object(),  # type: ignore[arg-type]
        iteration_blocked_function_names=set(),
        ctx=execution_context,
        state=execution_state,
    )
    assert guard.should_skip is True
    assert guard.loop_break_reason == "research_route_fetch_required"
    assert "优先对搜索结果中的 URL 调用 fetch_page" in str(guard.tool_result.message or "")


def test_evaluate_tool_guard_should_break_after_repeated_same_domain_fetch_blocks() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索并补齐多来源证据")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web", "fetch_page"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=5,
        effective_max_tool_iterations=5,
        allow_ask_user=False,
        research_route_enabled=True,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()
    execution_state.research_search_ready = True
    execution_state.research_fetch_success_count = 2
    execution_state.research_candidate_urls = [
        "https://same.example.com/a",
        "https://other.example.org/b",
    ]
    execution_state.research_fetched_urls = [
        "https://same.example.com/intro",
    ]
    execution_state.research_fetched_domains = ["same.example.com"]

    for _ in range(2):
        guard = evaluate_tool_guard(
            logger=logger,
            step=step,
            task_mode="research",
            function_name="fetch_page",
            normalized_function_name="fetch_page",
            function_args={"url": "https://same.example.com/a"},
            matched_tool=object(),  # type: ignore[arg-type]
            iteration_blocked_function_names=set(),
            ctx=execution_context,
            state=execution_state,
        )
        assert guard.should_skip is True
        assert guard.loop_break_reason == ""

    final_guard = evaluate_tool_guard(
        logger=logger,
        step=step,
        task_mode="research",
        function_name="fetch_page",
        normalized_function_name="fetch_page",
        function_args={"url": "https://same.example.com/a"},
        matched_tool=object(),  # type: ignore[arg-type]
        iteration_blocked_function_names=set(),
        ctx=execution_context,
        state=execution_state,
    )
    assert final_guard.should_skip is True
    assert final_guard.loop_break_reason == "research_route_cross_domain_fetch_limit"


def test_evaluate_tool_guard_should_reset_cross_domain_repeat_blocks_when_not_blocked() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索并补齐多来源证据")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web", "fetch_page"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=5,
        effective_max_tool_iterations=5,
        allow_ask_user=False,
        research_route_enabled=True,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()
    execution_state.research_search_ready = True
    execution_state.research_fetch_success_count = 2
    execution_state.research_fetched_domains = ["same.example.com"]
    execution_state.research_candidate_urls = [
        "https://same.example.com/a",
        "https://other.example.org/b",
    ]
    execution_state.research_fetched_urls = [
        "https://same.example.com/intro",
    ]
    execution_state.research_cross_domain_repeat_blocks = 2

    guard = evaluate_tool_guard(
        logger=logger,
        step=step,
        task_mode="research",
        function_name="fetch_page",
        normalized_function_name="fetch_page",
        function_args={"url": "https://other.example.org/b"},
        matched_tool=object(),  # type: ignore[arg-type]
        iteration_blocked_function_names=set(),
        ctx=execution_context,
        state=execution_state,
    )
    assert guard.should_skip is False
    assert execution_state.research_cross_domain_repeat_blocks == 0


def test_evaluate_tool_guard_should_treat_query_variant_as_already_fetched_url() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索并补齐多来源证据")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web", "fetch_page"},
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        general_inline_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        final_delivery_search_blocked_function_names=set(),
        final_delivery_shell_blocked_function_names=set(),
        final_inline_file_output_blocked_function_names=set(),
        requested_max_tool_iterations=5,
        effective_max_tool_iterations=5,
        allow_ask_user=False,
        research_route_enabled=True,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()
    execution_state.research_search_ready = True
    execution_state.research_fetch_success_count = 1
    execution_state.research_fetched_domains = ["same.example.com"]
    execution_state.research_candidate_urls = [
        "https://same.example.com/a?source=bing",
        "https://other.example.org/b",
    ]
    execution_state.research_fetched_urls = [
        "https://same.example.com/a?source=baidu",
    ]

    guard = evaluate_tool_guard(
        logger=logger,
        step=step,
        task_mode="research",
        function_name="search_web",
        normalized_function_name="search_web",
        function_args={"query": "重新搜索"},
        matched_tool=object(),  # type: ignore[arg-type]
        iteration_blocked_function_names=set(),
        ctx=execution_context,
        state=execution_state,
    )
    assert guard.should_skip is True
    message = str(guard.tool_result.message or "")
    assert "https://same.example.com/a?source=bing" not in message
    assert "https://other.example.org/b" in message


def test_convergence_judge_should_break_when_file_processing_facts_ready() -> None:
    judge = ConvergenceJudge()
    step = Step(
        id="step-final",
        description="列出目录并文本交付",
        task_mode_hint="file_processing",
        output_mode="inline",
        delivery_role="final",
        artifact_policy="default",
    )
    context = {"called_functions": {"write_file", "list_files"}}
    result = judge.evaluate_file_processing_progress(
        step=step,
        task_mode="file_processing",
        recent_function_name="read_file",
        function_args={"filepath": "/home/ubuntu/workspace/hello.txt"},
        tool_result_data={
            "filepath": "/home/ubuntu/workspace/hello.txt",
            "content": "P3_WORKSPACE_OK",
        },
        tool_result_success=True,
        step_file_context=context,
        runtime_recent_action={},
    )
    assert result.should_break is True
    assert result.payload is not None
    assert result.payload["success"] is True
    assert result.reason_code == "file_processing_facts_ready"
    delivery_text = str(result.payload.get("delivery_text") or "")
    assert "已读取文件" in delivery_text
    assert "/home/ubuntu/workspace/hello.txt" in delivery_text
    assert "P3_WORKSPACE_OK" not in delivery_text
    assert "读取内容长度" in delivery_text


def test_convergence_judge_should_break_when_find_files_and_read_file_facts_ready() -> None:
    judge = ConvergenceJudge()
    step = Step(
        id="step-final-find",
        description="读取课程目录并文本交付",
        task_mode_hint="file_processing",
        output_mode="inline",
        delivery_role="final",
        artifact_policy="default",
    )
    shared_context = {}
    result = judge.evaluate_file_processing_progress(
        step=step,
        task_mode="file_processing",
        recent_function_name="find_files",
        function_args={"dir_path": "/home/ubuntu/workspace"},
        tool_result_data={
            "dir_path": "/home/ubuntu/workspace",
            "files": ["hello.txt", "notes.md"],
        },
        tool_result_success=True,
        step_file_context=shared_context,
        runtime_recent_action={},
    )
    assert result.should_break is False

    result = judge.evaluate_file_processing_progress(
        step=step,
        task_mode="file_processing",
        recent_function_name="read_file",
        function_args={"filepath": "/home/ubuntu/workspace/hello.txt"},
        tool_result_data={
            "filepath": "/home/ubuntu/workspace/hello.txt",
            "content": "HELLO",
        },
        tool_result_success=True,
        step_file_context=shared_context,
        runtime_recent_action={},
    )
    assert result.should_break is True
    assert "目录文件" in str((result.payload or {}).get("delivery_text") or "")


class _FakeNoToolLLM:
    def __init__(self, message: Dict[str, Any]) -> None:
        self._message = message

    async def invoke(self, **kwargs: Any) -> Dict[str, Any]:
        return dict(self._message)


class _FakeToolOnly:
    def __init__(self, schema: Dict[str, Any]) -> None:
        self._schema = schema
        self.name = "fake-tool"

    def get_tools(self) -> List[Dict[str, Any]]:
        return [dict(self._schema)]

    def has_tool(self, function_name: str) -> bool:
        return str(function_name or "").strip().lower() == str(
            (self._schema.get("function") or {}).get("name") or ""
        ).strip().lower()

    async def invoke(self, function_name: str, **kwargs: Any) -> Any:
        return {"ok": True}


def test_execute_step_with_prompt_should_return_loop_break_when_human_wait_missing_ask_user() -> None:
    llm = _FakeNoToolLLM({"content": '{"success": true, "summary": "ok"}'})
    step = Step(description="等待用户确认后继续")
    runtime_tools = [
        _FakeToolOnly(
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    ]

    payload, tool_events = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=step,
            runtime_tools=runtime_tools,  # type: ignore[arg-type]
            task_mode="human_wait",
        )
    )

    assert payload["success"] is False
    blockers = [str(item) for item in list(payload.get("blockers") or [])]
    assert any("缺少可用的 message_ask_user 工具" in item for item in blockers)
    assert tool_events == []


def test_build_loop_break_result_should_append_research_progress_hint() -> None:
    payload = build_loop_break_result(
        loop_break_reason="search_repeat",
        step=Step(description="检索并总结"),
        runtime_recent_action={
            "research_progress": {
                "missing_signals": [
                    "至少读取 2 个来源页面正文",
                    "至少覆盖 2 个不同站点来源",
                ]
            }
        },
    )
    assert payload is not None
    hints = str(payload.get("next_hint") or "")
    assert "当前缺口" in hints
    assert "至少读取 2 个来源页面正文" in hints


def test_execute_step_with_prompt_should_return_loop_break_when_no_tools_and_empty_model_output() -> None:
    llm = _FakeNoToolLLM({"content": '{"success": false, "summary": "", "delivery_text": ""}'})
    step = Step(description="整理当前目录结果")

    payload, tool_events = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=step,
            runtime_tools=[],
            task_mode="general",
        )
    )

    assert payload["success"] is False
    blockers = [str(item) for item in list(payload.get("blockers") or [])]
    assert any("当前步骤无可用工具" in item for item in blockers)
    assert tool_events == []
