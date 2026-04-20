#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构辅助模块单测。"""

import logging
import time
import asyncio
import json
from io import StringIO
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from app.domain.models import Step, ToolResult
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_context import (
    ExecutionContext,
    build_execution_context,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import (
    ExecutionState,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.finalizer import (
    finalize_max_iterations,
    finalize_no_tool_call,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.loop_breaks import (
    build_loop_break_result,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.research.research_diagnosis import (
    build_research_diagnosis,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.research.research_query_builder import (
    build_research_query,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_effects import (
    apply_tool_preinvoke_effects,
    apply_rewrite_effects,
    apply_tool_result_effects,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_handlers import (
    execute_tool_with_policy,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.judge import (
    ConvergenceJudge,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import (
    ConstraintDecision,
    ConstraintEngineResult,
    ConstraintInput,
    ConstraintPolicyTraceEntry,
    ConstraintToolResultPayload,
    build_default_external_signals_snapshot,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.engine import (
    ConstraintEngine,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.policy_engine.engine import (
    ToolPolicyEngine,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.human_wait_policy import (
    evaluate_human_wait_policy,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.research_route_policy import (
    build_research_route_rewrite_decision,
    evaluate_research_route_policy,
    normalize_research_fetch_dedupe_key,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.repeat_loop_policy import (
    evaluate_repeat_loop_policy,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.task_mode_policy import (
    evaluate_task_mode_policy,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import (
    REASON_ALLOW,
    REASON_CONSTRAINT_ENGINE_ERROR,
    REASON_RESEARCH_SEARCH_TO_FETCH_REWRITE,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tool_runtime.tool_events import (
    ToolEventDispatcher,
    bind_tool_name,
    build_called_event,
    build_calling_event,
    build_tool_call_lifecycle,
    build_tool_feedback_message,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.tools import (
    execute_step_with_prompt,
)
from app.domain.services.workspace_runtime.policies import (
    build_browser_high_level_failure_key,
    build_browser_route_state_key,
    build_search_fingerprint,
    build_tool_fingerprint,
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


def test_constraint_engine_default_snapshot_should_fill_required_keys() -> None:
    snapshot = build_default_external_signals_snapshot({})
    assert snapshot["blocked_fingerprints"] == []
    assert snapshot["avoid_url_keys"] == []
    assert snapshot["avoid_domains"] == []
    assert snapshot["avoid_query_patterns"] == []
    assert snapshot["recent_failure_reasons"] == []


def test_constraint_engine_should_generate_research_fetch_required_for_explicit_url() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="读取明确 URL 的网页正文")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取 https://example.com 页面正文"}],
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
        research_has_explicit_url=True,
        current_user_message_text="请读取 https://example.com 正文",
    )
    execution_state = ExecutionState()
    execution_state.research_fetch_completed = False
    execution_state.consecutive_fetch_failure_count = 0

    engine = ConstraintEngine(logger=logger)
    result = engine.evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "读取该页面内容"},
            matched_tool=_AnyTool(),  # type: ignore[arg-type]
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        )
    )
    decision = result.constraint_decision
    assert decision.action == "allow"
    assert decision.reason_code == REASON_ALLOW
    assert decision.loop_break_reason == ""
    assert decision.tool_result_payload is None
    assert result.rewrite_applied is True
    assert result.rewrite_reason == REASON_RESEARCH_SEARCH_TO_FETCH_REWRITE
    assert result.final_function_name == "fetch_page"
    assert result.final_normalized_function_name == "fetch_page"
    assert result.final_function_args == {"url": "https://example.com"}
    assert any(
        trace.policy_name == "research_route_policy" and trace.reason_code == REASON_RESEARCH_SEARCH_TO_FETCH_REWRITE
        for trace in result.policy_trace
    )
    assert result.winning_policy == "research_route_policy"


def test_constraint_engine_evaluate_guard_should_accept_runtime_logger() -> None:
    logger = logging.getLogger("runtime-guard-test")
    step = Step(description="普通步骤")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "hello"}],
        available_tools=[],
        available_function_names={"search_web"},
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
    engine = ConstraintEngine(logger=logging.getLogger("module-default"))
    result = engine.evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="general",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "自然语言查询"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        ),
        logger=logger,
    )
    assert result.constraint_decision.action == "allow"
    assert result.constraint_decision.reason_code == "allow"


def test_constraint_engine_should_fail_closed_when_policy_raises() -> None:
    logger = logging.getLogger("runtime-guard-fail-closed")
    engine = ConstraintEngine(logger=logger)
    step = Step(description="普通步骤")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "hello"}],
        available_tools=[],
        available_function_names={"search_web"},
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
    with patch(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.engine.evaluate_task_mode_policy",
        side_effect=RuntimeError("boom"),
    ):
        result = engine.evaluate_guard(
            constraint_input=ConstraintInput(
                step=step,
                task_mode="general",
                function_name="search_web",
                normalized_function_name="search_web",
                function_args={"query": "自然语言查询"},
                matched_tool=object(),
                iteration_blocked_function_names=set(),
                execution_context=execution_context,
                execution_state=execution_state,
            ),
            logger=logger,
        )
    decision = result.constraint_decision
    assert decision.action == "block"
    assert decision.reason_code == "constraint_engine_error"
    assert decision.block_mode == "hard_block_break"
    assert decision.loop_break_reason == "constraint_engine_error"
    assert result.winning_policy == "task_mode_policy"
    assert decision.tool_result_payload is not None
    assert "约束引擎异常" in str(decision.tool_result_payload.message or "")
    assert result.tool_call_fingerprint != ""
    assert any(
        trace.policy_name == "task_mode_policy" and trace.reason_code == "constraint_engine_error"
        for trace in result.policy_trace
    )


def test_constraint_engine_should_fail_closed_when_rewrite_builder_raises() -> None:
    logger = logging.getLogger("runtime-rewrite-fail-closed")
    engine = ConstraintEngine(logger=logger)
    step = Step(description="读取明确 URL 页面")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "检索并总结"}],
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
        current_user_message_text="请检索网页并总结",
    )
    execution_state = ExecutionState()
    execution_state.research_search_ready = False
    execution_state.consecutive_fetch_failure_count = 0
    with patch(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.engine.build_research_route_rewrite_decision",
        side_effect=RuntimeError("rewrite boom"),
    ):
        result = engine.evaluate_guard(
            constraint_input=ConstraintInput(
                step=step,
                task_mode="research",
                function_name="search_web",
                normalized_function_name="search_web",
                function_args={"query": "主流 AI 编程助手及其支持的 IDE"},
                matched_tool=object(),
                iteration_blocked_function_names=set(),
                execution_context=execution_context,
                execution_state=execution_state,
            ),
            logger=logger,
            allow_rewrite=True,
        )
    decision = result.constraint_decision
    assert decision.action == "block"
    assert decision.reason_code == REASON_CONSTRAINT_ENGINE_ERROR
    assert decision.block_mode == "hard_block_break"
    assert decision.loop_break_reason == REASON_CONSTRAINT_ENGINE_ERROR
    assert decision.tool_result_payload is not None
    assert "约束引擎异常" in str(decision.tool_result_payload.message or "")


def test_constraint_engine_should_rewrite_in_guard_main_path() -> None:
    logger = logging.getLogger("runtime-rewrite-main-path")
    engine = ConstraintEngine(logger=logger)
    step = Step(description="打开 https://example.com/news 并提取页面要点")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "打开并提取要点"}],
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
        research_has_explicit_url=True,
        current_user_message_text="请读取该链接正文",
    )
    execution_state = ExecutionState()
    result = engine.evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "读取该页面内容"},
            matched_tool=_AnyTool(),  # type: ignore[arg-type]
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        ),
        logger=logger,
        allow_rewrite=True,
    )
    decision = result.constraint_decision
    assert decision.action == "allow"
    assert decision.reason_code == REASON_ALLOW
    assert result.rewrite_applied is True
    assert result.rewrite_reason == REASON_RESEARCH_SEARCH_TO_FETCH_REWRITE
    assert result.final_function_name == "fetch_page"
    assert result.final_function_args == {"url": "https://example.com/news"}


def test_evaluate_human_wait_policy_should_block_non_ask_user_in_human_wait() -> None:
    step = Step(description="请等待用户确认后继续")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "等待用户确认"}],
        available_tools=[],
        available_function_names={"search_web", "message_ask_user"},
        browser_route_enabled=False,
        blocked_function_names={"search_web"},
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
        allow_ask_user=True,
        research_route_enabled=False,
        research_has_explicit_url=False,
    )
    decision = evaluate_human_wait_policy(
        ConstraintInput(
            step=step,
            task_mode="human_wait",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "课程推荐"},
            matched_tool=object(),
            iteration_blocked_function_names={"search_web"},
            execution_context=execution_context,
            execution_state=ExecutionState(),
        )
    )
    assert decision is not None
    assert decision.reason_code == "human_wait_non_interrupt_tool_blocked"
    assert decision.block_mode == "hard_block_break"


def test_evaluate_human_wait_policy_should_block_ask_user_when_step_not_allow_wait() -> None:
    step = Step(description="先读取文件并总结")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取并总结"}],
        available_tools=[],
        available_function_names={"message_ask_user"},
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
    decision = evaluate_human_wait_policy(
        ConstraintInput(
            step=step,
            task_mode="general",
            function_name="message_ask_user",
            normalized_function_name="message_ask_user",
            function_args={"question": "请确认是否继续"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=ExecutionState(),
        )
    )
    assert decision is not None
    assert decision.reason_code == "ask_user_not_allowed"
    assert decision.block_mode == "soft_block_continue"


def test_evaluate_task_mode_policy_should_block_web_reading_file_tool() -> None:
    step = Step(description="读取网页正文")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取网页正文"}],
        available_tools=[],
        available_function_names={"read_file"},
        browser_route_enabled=False,
        blocked_function_names={"read_file"},
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
    decision = evaluate_task_mode_policy(
        ConstraintInput(
            step=step,
            task_mode="web_reading",
            function_name="read_file",
            normalized_function_name="read_file",
            function_args={"path": "/tmp/a.txt"},
            matched_tool=object(),
            iteration_blocked_function_names={"read_file"},
            execution_context=execution_context,
            execution_state=ExecutionState(),
        )
    )
    assert decision is not None
    assert decision.reason_code == "web_reading_file_tool_blocked"
    assert decision.block_mode == "hard_block_break"


def test_policy_order_should_prefer_task_mode_before_human_wait() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="等待用户选择后继续")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "等待用户确认"}],
        available_tools=[],
        available_function_names={"search_web", "message_ask_user"},
        browser_route_enabled=False,
        blocked_function_names={"search_web"},
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
        allow_ask_user=True,
        research_route_enabled=False,
        research_has_explicit_url=False,
    )
    execution_state = ExecutionState()
    engine = ConstraintEngine(logger=logger)
    result = engine.evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="human_wait",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "课程推荐"},
            matched_tool=object(),
            iteration_blocked_function_names={"search_web"},
            execution_context=execution_context,
            execution_state=execution_state,
        ),
        logger=logger,
        allow_rewrite=False,
    )
    decision = result.constraint_decision
    assert decision.action == "block"
    assert decision.reason_code == "human_wait_non_interrupt_tool_blocked"
    assert decision.tool_result_payload is not None
    assert "等待用户确认/选择" in str(decision.tool_result_payload.message or "")


def test_evaluate_research_route_policy_should_block_keyword_stacked_query() -> None:
    step = Step(description="检索并总结")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web"},
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
    decision = evaluate_research_route_policy(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "AI编程助手IDE支持价格对比最新"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=ExecutionState(),
        )
    )
    assert decision is not None
    assert decision.reason_code == "research_query_style_blocked"
    assert decision.block_mode == "hard_block_break"


def test_evaluate_repeat_loop_policy_should_block_search_repeat() -> None:
    state = ExecutionState()
    search_args = {"query": "主流 AI 编程助手及其支持的 IDE"}
    state.search_repeat_counter[build_search_fingerprint(search_args)] = 2
    decision = evaluate_repeat_loop_policy(
        ConstraintInput(
            step=Step(description="检索并总结"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args=search_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=ExecutionContext(
                normalized_user_content=[],
                available_tools=[],
                available_function_names={"search_web"},
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
            ),
            execution_state=state,
        )
    )
    assert decision is not None
    assert decision.reason_code == "search_repeat"


def test_evaluate_repeat_loop_policy_should_use_normalized_search_query_fingerprint() -> None:
    state = ExecutionState()
    normalized_search_args = {"query": "上海 周末 亲子 游玩 有哪些 推荐 景点"}
    state.search_repeat_counter[build_search_fingerprint(normalized_search_args)] = 2
    decision = evaluate_repeat_loop_policy(
        ConstraintInput(
            step=Step(description="检索并总结"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "请帮我查一下 上海 周末 亲子 游玩 有哪些 推荐 景点"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=ExecutionContext(
                normalized_user_content=[],
                available_tools=[],
                available_function_names={"search_web"},
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
            ),
            execution_state=state,
        )
    )
    assert decision is not None
    assert decision.reason_code == "search_repeat"


def test_evaluate_repeat_loop_policy_should_block_fetch_repeat_on_current_round() -> None:
    state = ExecutionState()
    state.fetch_repeat_counter["https://example.com/page"] = 2
    decision = evaluate_repeat_loop_policy(
        ConstraintInput(
            step=Step(description="抓取并总结"),
            task_mode="research",
            function_name="fetch_page",
            normalized_function_name="fetch_page",
            function_args={"url": "https://example.com/page?from=bing"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=ExecutionContext(
                normalized_user_content=[],
                available_tools=[],
                available_function_names={"fetch_page"},
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
            ),
            execution_state=state,
        )
    )
    assert decision is not None
    assert decision.reason_code == "research_route_fingerprint_repeat"


def test_evaluate_repeat_loop_policy_should_block_repeat_tool_call_on_current_round() -> None:
    state = ExecutionState()
    search_args = {"query": "主流 AI 编程助手及其支持的 IDE"}
    state.last_tool_fingerprint = build_tool_fingerprint("search_web", search_args)
    state.same_tool_repeat_count = 3
    decision = evaluate_repeat_loop_policy(
        ConstraintInput(
            step=Step(description="检索并总结"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args=search_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=ExecutionContext(
                normalized_user_content=[],
                available_tools=[],
                available_function_names={"search_web"},
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
            ),
            execution_state=state,
        )
    )
    assert decision is not None
    assert decision.reason_code == "repeat_tool_call"


def test_evaluate_repeat_loop_policy_should_predict_same_tool_repeat_from_normalized_args() -> None:
    state = ExecutionState()
    normalized_search_args = {"query": "上海 周末 亲子 游玩 有哪些 推荐 景点"}
    state.last_tool_fingerprint = build_tool_fingerprint("search_web", normalized_search_args)
    state.same_tool_repeat_count = 3
    decision = evaluate_repeat_loop_policy(
        ConstraintInput(
            step=Step(description="检索并总结"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "请帮我查一下 上海 周末 亲子 游玩 有哪些 推荐 景点"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=ExecutionContext(
                normalized_user_content=[],
                available_tools=[],
                available_function_names={"search_web"},
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
            ),
            execution_state=state,
        )
    )
    assert decision is not None
    assert decision.reason_code == "repeat_tool_call"


def test_evaluate_repeat_loop_policy_should_use_snapshot_override_limit() -> None:
    state = ExecutionState()
    search_args = {"query": "主流 AI 编程助手及其支持的 IDE"}
    state.search_repeat_counter[build_search_fingerprint(search_args)] = 2
    decision = evaluate_repeat_loop_policy(
        ConstraintInput(
            step=Step(description="检索并总结"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args=search_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=ExecutionContext(
                normalized_user_content=[],
                available_tools=[],
                available_function_names={"search_web"},
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
            ),
            execution_state=state,
            external_signals_snapshot={"search_repeat_limit": 5},
        )
    )
    assert decision is None


def test_constraint_engine_should_block_explicit_url_search_call() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="请读取 https://example.com 网页正文")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取 https://example.com 正文"}],
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
        research_has_explicit_url=True,
        current_user_message_text="请读取 https://example.com 正文",
    )
    execution_state = ExecutionState()

    engine = ConstraintEngine(logger=logger)
    result = engine.evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "读取该页面内容"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        ),
        logger=logger,
        allow_rewrite=False,
    )
    decision = result.constraint_decision
    assert decision.action == "block"
    assert decision.reason_code == "research_route_fetch_required"
    assert decision.tool_result_payload is not None
    assert "fetch_page" in str(decision.tool_result_payload.message or "")


def test_constraint_engine_should_log_final_decision_with_required_fields() -> None:
    logger = logging.getLogger("constraint-engine-log-test")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    step = Step(description="测试约束日志")
    engine = ConstraintEngine(logger=logger)
    engine_result = ConstraintEngineResult(
        constraint_decision=ConstraintDecision(
            action="block",
            reason_code="reason-x",
            block_mode="hard_block_break",
            loop_break_reason="reason-x",
        ),
        final_function_name="fetch_page",
        final_normalized_function_name="fetch_page",
        final_function_args={"url": "https://example.com"},
        policy_trace=[
            ConstraintPolicyTraceEntry(
                policy_name="research_route_policy",
                action="block",
                reason_code="reason-x",
            )
        ],
        winning_policy="research_route_policy",
        tool_call_fingerprint="fingerprint-x",
        rewrite_applied=True,
        rewrite_reason="rewrite-x",
        rewrite_metadata={"rewrite_source": "explicit"},
    )
    constraint_input = ConstraintInput(
        step=step,
        task_mode="research",
        function_name="search_web",
        normalized_function_name="search_web",
        function_args={"query": "读取页面"},
        matched_tool=object(),
        iteration_blocked_function_names=set(),
        execution_context=ExecutionContext(
            normalized_user_content=[],
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
            research_has_explicit_url=True,
        ),
        execution_state=ExecutionState(),
    )

    engine._log_final_decision(logger, constraint_input, engine_result)

    handler.flush()
    output = stream.getvalue()
    logger.removeHandler(handler)

    assert "最终函数名=\"fetch_page\"" in output
    assert "最终动作=\"block\"" in output
    assert "命中策略=\"research_route_policy\"" in output
    assert "原因码=\"reason-x\"" in output
    assert "工具调用指纹=\"fingerprint-x\"" in output
    assert "阻断模式=\"hard_block_break\"" in output
    assert "loop_break_reason=\"reason-x\"" in output
    assert "是否改写=true" in output


def test_constraint_engine_should_soft_block_listing_click_target_mismatch() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="在列表页点击目标链接")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "点击目标链接"}],
        available_tools=[],
        available_function_names={"browser_click"},
        browser_route_enabled=True,
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
    execution_state = ExecutionState()
    execution_state.browser_page_type = "listing"
    execution_state.browser_link_match_ready = True
    execution_state.last_browser_route_index = 2
    execution_state.last_browser_route_url = "https://example.com/detail"
    decision = ConstraintEngine(logger=logger).evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="browser_interaction",
            function_name="browser_click",
            normalized_function_name="browser_click",
            function_args={"index": 1},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        ),
        logger=logger,
        allow_rewrite=False,
    ).constraint_decision
    assert decision.action == "block"
    assert decision.reason_code == "browser_click_target_blocked"
    assert decision.block_mode == "soft_block_continue"
    assert decision.loop_break_reason == ""


def test_constraint_engine_should_soft_block_failed_browser_high_level_retry() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="继续页面链接匹配")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "继续匹配链接"}],
        available_tools=[],
        available_function_names={"browser_find_link_by_text"},
        browser_route_enabled=True,
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
    execution_state = ExecutionState()
    execution_state.browser_page_type = "listing"
    execution_state.last_browser_route_url = "https://example.com/list"
    execution_state.last_browser_observation_fingerprint = "fingerprint-1"
    failure_key = build_browser_high_level_failure_key(
        function_name="browser_find_link_by_text",
        function_args={"text": "目标"},
        browser_route_state_key=build_browser_route_state_key(
            browser_page_type=execution_state.browser_page_type,
            browser_url=execution_state.last_browser_route_url,
            browser_observation_fingerprint=execution_state.last_browser_observation_fingerprint,
        ),
    )
    execution_state.failed_browser_high_level_keys.add(failure_key)
    decision = ConstraintEngine(logger=logger).evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="browser_interaction",
            function_name="browser_find_link_by_text",
            normalized_function_name="browser_find_link_by_text",
            function_args={"text": "目标"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        ),
        logger=logger,
        allow_rewrite=False,
    ).constraint_decision
    assert decision.action == "block"
    assert decision.reason_code == "browser_high_level_retry_blocked"
    assert decision.block_mode == "soft_block_continue"
    assert decision.loop_break_reason == ""


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


def test_finalize_max_iterations_should_include_query_rewrite_hint_for_low_recall() -> None:
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
                "is_low_recall": True,
                "missing_signals": ["候选链接过少，请改写主题描述提高召回"],
            }
        },
    )

    assert payload["success"] is False
    assert "单主题自然语言短句" in str(payload.get("next_hint") or "")


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
    assert len(execution_state.research_search_evidence_items) == 2
    assert execution_state.research_snippet_sufficient is False
    assert execution_state.last_search_evidence_quality["reason_code"] == "snippet_insufficient"


def test_apply_tool_result_effects_should_capture_search_snippet_evidence() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="调研 OpenAI 文档并提炼要点")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并总结"}],
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

    _ = apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name="search_web",
        normalized_function_name="search_web",
        function_args={"query": "OpenAI 文档"},
        tool_result=ToolResult(
            success=True,
            data={
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "A",
                        "content": "这是一段足够长的摘要内容，用于说明搜索结果已经包含较完整的信息，可以先基于摘要继续判断是否需要抓取正文。",
                    },
                    {
                        "url": "https://example.org/b",
                        "title": "B",
                        "content": "这也是一段足够长的摘要内容，用于补充第二个来源，避免当前研究只依赖单一站点。",
                    },
                ]
            },
        ),
        loop_break_reason="",
        browser_route_state_key="state-key",
        execution_context=execution_context,
        execution_state=execution_state,
    )

    assert len(execution_state.research_search_evidence_items) == 2
    assert execution_state.research_search_evidence_items[0]["snippet"] != ""
    assert execution_state.research_snippet_sufficient is True
    assert execution_state.last_search_evidence_quality["reason_code"] == "snippet_source_coverage_thin"


def test_apply_tool_result_effects_should_reset_fetch_failure_counter_on_search_success() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="检索网页并读取正文")
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
    execution_state.consecutive_fetch_failure_count = 3
    tool_result = ToolResult(
        success=True,
        data={"results": [{"url": "https://example.com/a"}]},
    )

    _ = apply_tool_result_effects(
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

    assert execution_state.consecutive_fetch_failure_count == 0


def test_apply_tool_result_effects_should_blacklist_failed_fetch_url_key() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="读取页面正文")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取页面"}],
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
        requested_max_tool_iterations=3,
        effective_max_tool_iterations=3,
        allow_ask_user=False,
        research_route_enabled=True,
        research_has_explicit_url=True,
    )
    execution_state = ExecutionState()

    _ = apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name="fetch_page",
        normalized_function_name="fetch_page",
        function_args={"url": "https://example.com/a?src=bing"},
        tool_result=ToolResult(success=False, message="timeout"),
        loop_break_reason="",
        browser_route_state_key="",
        execution_context=execution_context,
        execution_state=execution_state,
    )

    assert "https://example.com/a" in execution_state.research_failed_fetch_url_keys


def test_apply_rewrite_effects_should_track_explicit_rewrite_url_key() -> None:
    execution_state = ExecutionState()
    apply_rewrite_effects(
        rewrite_reason="research_search_to_fetch_rewrite",
        rewrite_metadata={
            "rewrite_source": "explicit",
            "rewrite_url": "https://example.com/article/1?from=search",
        },
        execution_state=execution_state,
    )
    assert "https://example.com/article/1" in execution_state.research_explicit_rewrite_url_keys


def test_apply_tool_result_effects_should_persist_cross_domain_repeat_blocks() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="调研并抓取多来源")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "抓取不同来源"}],
        available_tools=[],
        available_function_names={"fetch_page", "search_web"},
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
    execution_state.research_cross_domain_repeat_blocks = 0

    _ = apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name="fetch_page",
        normalized_function_name="fetch_page",
        function_args={"url": "https://same.example.com/page-a"},
        tool_result=ToolResult(
            success=False,
            message="当前研究步骤尚未覆盖足够来源，请优先抓取不同站点候选链接。",
            data={
                "next_cross_domain_url": "https://other.example.org/page-b",
                "cross_domain_repeat_block_count": 1,
            },
        ),
        # 真实软阻断路径：soft block 不会写 loop_break_reason，依赖 reason_code 信号。
        loop_break_reason="",
        guard_reason_code="research_route_cross_domain_fetch_limit",
        browser_route_state_key="",
        execution_context=execution_context,
        execution_state=execution_state,
        tool_executed=False,
    )
    assert execution_state.research_cross_domain_repeat_blocks == 1


def test_apply_tool_result_effects_should_not_mark_browser_high_level_failed_on_soft_block_retry_reason() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="继续执行高阶浏览器动作")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "继续点击"}],
        available_tools=[],
        available_function_names={"browser_find_link_by_text"},
        browser_route_enabled=True,
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
    execution_state = ExecutionState()
    browser_route_state_key = build_browser_route_state_key(
        browser_page_type="listing",
        browser_url="https://example.com/list",
        browser_observation_fingerprint="fingerprint-1",
    )
    tool_result = ToolResult(success=False, message="该高阶能力已失败，请改用低阶能力。")

    result = apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name="browser_find_link_by_text",
        normalized_function_name="browser_find_link_by_text",
        function_args={"text": "目标链接"},
        tool_result=tool_result,
        loop_break_reason="",
        guard_reason_code="browser_high_level_retry_blocked",
        browser_route_state_key=browser_route_state_key,
        execution_context=execution_context,
        execution_state=execution_state,
        tool_executed=False,
    )

    assert result.tool_result.data is None
    assert len(execution_state.failed_browser_high_level_keys) == 0


def test_build_research_route_rewrite_decision_should_not_fallback_to_candidate_when_explicit_url_blacklisted() -> None:
    step = Step(description="读取 https://same.example.com/a?from=ctx 并补充信息")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "请读取 https://same.example.com/a?from=user"}],
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
        research_has_explicit_url=True,
    )
    execution_state = ExecutionState()
    execution_state.research_search_ready = True
    execution_state.research_candidate_urls = [
        "https://candidate.example.org/1",
        "https://candidate2.example.org/2",
    ]
    execution_state.research_failed_fetch_url_keys = {"https://same.example.com/a"}

    lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "call-1",
            "function": {"name": "search_web", "arguments": {"query": "苏州 旅游"}}},
        parse_tool_call_args=lambda raw: raw if isinstance(raw, dict) else {},
    )
    assert lifecycle is not None
    decision = build_research_route_rewrite_decision(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name=lifecycle.function_name,
            normalized_function_name=lifecycle.normalized_function_name,
            function_args=lifecycle.function_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        )
    )
    assert decision is None


def test_build_research_route_rewrite_decision_should_not_rewrite_to_explicit_without_single_page_intent() -> None:
    step = Step(description="调研厦门周末亲子游攻略并总结可信来源")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "请参考 https://example.com/seed"}],
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
        research_has_explicit_url=True,
        current_user_message_text="请参考 https://example.com/seed",
    )
    execution_state = ExecutionState()
    lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "call-2",
            "function": {"name": "search_web", "arguments": {"query": "厦门 周末 亲子游"}}},
        parse_tool_call_args=lambda raw: raw if isinstance(raw, dict) else {},
    )
    assert lifecycle is not None
    decision = build_research_route_rewrite_decision(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name=lifecycle.function_name,
            normalized_function_name=lifecycle.normalized_function_name,
            function_args=lifecycle.function_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        )
    )
    assert decision is None


def test_build_research_route_rewrite_decision_should_not_rewrite_search_ready_without_explicit_url() -> None:
    step = Step(description="调研 OpenAI 文档并提炼要点")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "请调研并总结"}],
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
    execution_state.research_candidate_urls = ["https://example.com/a"]
    lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "call-search-ready",
            "function": {"name": "search_web", "arguments": {"query": "OpenAI 文档"}},
        },
        parse_tool_call_args=lambda raw: raw if isinstance(raw, dict) else {},
    )
    assert lifecycle is not None
    decision = build_research_route_rewrite_decision(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name=lifecycle.function_name,
            normalized_function_name=lifecycle.normalized_function_name,
            function_args=lifecycle.function_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        )
    )
    assert decision is None


def test_build_research_route_rewrite_decision_should_rewrite_explicit_only_once_for_single_page_intent() -> None:
    step = Step(description="请读取并抓取这个页面正文：https://example.com/a")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取 https://example.com/a"}],
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
        research_has_explicit_url=True,
        current_user_message_text="读取 https://example.com/a",
    )
    execution_state = ExecutionState()

    first_lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "call-3",
            "function": {"name": "search_web", "arguments": {"query": "读取页面"}}},
        parse_tool_call_args=lambda raw: raw if isinstance(raw, dict) else {},
    )
    assert first_lifecycle is not None
    first_decision = build_research_route_rewrite_decision(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name=first_lifecycle.function_name,
            normalized_function_name=first_lifecycle.normalized_function_name,
            function_args=first_lifecycle.function_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        )
    )
    assert first_decision is not None
    assert first_decision.reason_code == "research_search_to_fetch_rewrite"
    assert first_decision.rewrite_target is not None
    assert first_decision.rewrite_target.get("function_name") == "fetch_page"
    assert first_decision.metadata.get("rewrite_source") == "explicit"
    explicit_key = normalize_research_fetch_dedupe_key(
        (first_decision.rewrite_target.get("function_args") or {}).get("url")
    )
    apply_rewrite_effects(
        rewrite_reason=str(first_decision.reason_code or ""),
        rewrite_metadata=dict(first_decision.metadata or {}),
        execution_state=execution_state,
    )
    assert explicit_key in execution_state.research_explicit_rewrite_url_keys

    second_lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "call-4",
            "function": {"name": "search_web", "arguments": {"query": "读取页面"}}},
        parse_tool_call_args=lambda raw: raw if isinstance(raw, dict) else {},
    )
    assert second_lifecycle is not None
    second_decision = build_research_route_rewrite_decision(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name=second_lifecycle.function_name,
            normalized_function_name=second_lifecycle.normalized_function_name,
            function_args=second_lifecycle.function_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        )
    )
    assert second_decision is None


def test_build_research_route_rewrite_decision_should_rewrite_explicit_for_open_url_extract_summary_intent() -> None:
    step = Step(description="打开 https://example.com/news 并提取页面要点")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "打开并提取要点"}],
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
        research_has_explicit_url=True,
        current_user_message_text="打开并提取要点",
    )
    execution_state = ExecutionState()
    lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "call-4a",
            "function": {"name": "search_web", "arguments": {"query": "新闻要点"}}},
        parse_tool_call_args=lambda raw: raw if isinstance(raw, dict) else {},
    )
    assert lifecycle is not None
    decision = build_research_route_rewrite_decision(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name=lifecycle.function_name,
            normalized_function_name=lifecycle.normalized_function_name,
            function_args=lifecycle.function_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        )
    )
    assert decision is not None
    assert decision.reason_code == "research_search_to_fetch_rewrite"
    assert decision.rewrite_target is not None
    assert decision.rewrite_target.get("function_name") == "fetch_page"
    assert decision.metadata.get("rewrite_source") == "explicit"


def test_build_research_route_rewrite_decision_should_rewrite_when_explicit_url_only_in_current_user_input() -> None:
    step = Step(description="读取用户提供的页面并提炼核心信息")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "系统拼接内容"}],
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
        research_has_explicit_url=True,
        current_user_message_text="请读取这个页面：https://example.com/article/123",
    )
    execution_state = ExecutionState()
    lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "call-4c",
            "function": {"name": "search_web", "arguments": {"query": "提炼核心信息"}}},
        parse_tool_call_args=lambda raw: raw if isinstance(raw, dict) else {},
    )
    assert lifecycle is not None
    decision = build_research_route_rewrite_decision(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name=lifecycle.function_name,
            normalized_function_name=lifecycle.normalized_function_name,
            function_args=lifecycle.function_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        )
    )
    assert decision is not None
    assert decision.reason_code == "research_search_to_fetch_rewrite"
    assert decision.rewrite_target is not None
    assert decision.rewrite_target.get("function_name") == "fetch_page"
    assert decision.metadata.get("rewrite_source") == "explicit"


def test_build_research_route_rewrite_decision_should_not_rewrite_explicit_for_multi_source_research_intent() -> None:
    step = Step(description="对比 https://example.com/a 与其他来源并汇总结论")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "对比多个来源"}],
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
        research_has_explicit_url=True,
        current_user_message_text="对比多个来源",
    )
    execution_state = ExecutionState()
    lifecycle = build_tool_call_lifecycle(
        selected_tool_call={
            "id": "call-4b",
            "function": {"name": "search_web", "arguments": {"query": "对比结论"}}},
        parse_tool_call_args=lambda raw: raw if isinstance(raw, dict) else {},
    )
    assert lifecycle is not None
    decision = build_research_route_rewrite_decision(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name=lifecycle.function_name,
            normalized_function_name=lifecycle.normalized_function_name,
            function_args=lifecycle.function_args,
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
        )
    )
    assert decision is None


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


class _FakeSequentialToolCallLLM:
    def __init__(self, messages: List[Dict[str, Any]]) -> None:
        self._messages = list(messages)

    async def invoke(self, **kwargs: Any) -> Dict[str, Any]:
        if len(self._messages) == 0:
            return {"content": '{"success": true, "summary": "ok"}'}
        return dict(self._messages.pop(0))


class _FakeSearchFetchTool:
    name = "search"

    def __init__(self) -> None:
        self.invocations: List[tuple[str, Dict[str, Any]]] = []

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_page",
                    "description": "fetch",
                    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
                },
            },
        ]

    def has_tool(self, function_name: str) -> bool:
        return str(function_name or "").strip().lower() in {"search_web", "fetch_page"}

    async def invoke(self, function_name: str, **kwargs: Any) -> ToolResult:
        self.invocations.append((str(function_name or "").strip(), dict(kwargs)))
        if str(function_name or "").strip() == "search_web":
            return ToolResult(
                success=True,
                data={
                    "results": [
                        {"url": "https://alpha.example.com/a"},
                        {"url": "https://beta.example.org/b"},
                    ]
                },
            )
        return ToolResult(
            success=True,
            data={
                "url": str(kwargs.get("url") or ""),
                "final_url": str(kwargs.get("url") or ""),
                "content": "ok",
            },
        )


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


class _AnyTool:
    name = "any-tool"

    def has_tool(self, function_name: str) -> bool:
        return bool(str(function_name or "").strip())


class _SearchOnlyTool:
    name = "search-only"

    def has_tool(self, function_name: str) -> bool:
        return str(function_name or "").strip().lower() == "search_web"


class _FetchOnlyTool:
    name = "fetch-only"

    def has_tool(self, function_name: str) -> bool:
        return str(function_name or "").strip().lower() == "fetch_page"


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


def test_constraint_engine_should_resolve_rewritten_matched_tool_from_runtime_tools() -> None:
    logger = logging.getLogger("runtime-rewrite-tool-resolution")
    engine = ConstraintEngine(logger=logger)
    step = Step(description="打开 https://example.com/news 并提取页面要点")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "打开并提取要点"}],
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
        research_has_explicit_url=True,
        current_user_message_text="请读取该链接正文",
    )

    result = engine.evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "读取该页面内容"},
            matched_tool=_SearchOnlyTool(),  # type: ignore[arg-type]
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=ExecutionState(),
            runtime_tools=[_SearchOnlyTool(), _FetchOnlyTool()],  # type: ignore[list-item]
        ),
        logger=logger,
        allow_rewrite=True,
    )

    assert result.constraint_decision.action == "allow"
    assert result.rewrite_applied is True
    assert result.final_function_name == "fetch_page"
    assert result.winning_policy == "research_route_policy"


def test_constraint_engine_should_block_when_rewrite_target_tool_is_missing() -> None:
    logger = logging.getLogger("runtime-rewrite-target-missing")
    engine = ConstraintEngine(logger=logger)
    step = Step(description="打开 https://example.com/news 并提取页面要点")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "打开并提取要点"}],
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
        research_has_explicit_url=True,
        current_user_message_text="请读取该链接正文",
    )

    result = engine.evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "读取该页面内容"},
            matched_tool=_SearchOnlyTool(),  # type: ignore[arg-type]
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=ExecutionState(),
            runtime_tools=[_SearchOnlyTool()],  # type: ignore[list-item]
        ),
        logger=logger,
        allow_rewrite=True,
    )

    assert result.constraint_decision.action == "block"
    assert result.constraint_decision.reason_code == "invalid_tool"
    assert result.final_function_name == "fetch_page"


def test_constraint_engine_should_prefer_task_mode_policy_by_fixed_order() -> None:
    logger = logging.getLogger("runtime-fixed-policy-order")
    engine = ConstraintEngine(logger=logger)
    step = Step(description="普通步骤")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "普通步骤"}],
        available_tools=[],
        available_function_names={"browser_view"},
        browser_route_enabled=False,
        blocked_function_names={"browser_view"},
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

    result = engine.evaluate_guard(
        constraint_input=ConstraintInput(
            step=step,
            task_mode="general",
            function_name="browser_view",
            normalized_function_name="browser_view",
            function_args={},
            matched_tool=_FakeToolOnly(
                {
                    "type": "function",
                    "function": {
                        "name": "browser_view",
                        "description": "browser",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ),  # type: ignore[arg-type]
            iteration_blocked_function_names={"browser_view"},
            execution_context=execution_context,
            execution_state=ExecutionState(),
        ),
        logger=logger,
    )

    assert result.constraint_decision.action == "block"
    assert result.constraint_decision.reason_code == "task_mode_tool_blocked"
    assert result.winning_policy == "task_mode_policy"


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


def test_build_loop_break_result_should_append_query_rewrite_hint_when_low_recall() -> None:
    payload = build_loop_break_result(
        loop_break_reason="search_repeat",
        step=Step(description="检索并总结"),
        runtime_recent_action={
            "research_progress": {
                "is_low_recall": True,
                "missing_signals": ["候选链接过少，请改写主题描述提高召回"],
            }
        },
    )
    assert payload is not None
    hints = str(payload.get("next_hint") or "")
    assert "单主题自然语言短句" in hints


def test_build_loop_break_result_should_handle_research_query_style_blocked() -> None:
    payload = build_loop_break_result(
        loop_break_reason="research_query_style_blocked",
        step=Step(description="检索并总结"),
        runtime_recent_action={},
    )
    assert payload is not None
    assert payload.get("success") is False
    blockers = [str(item) for item in list(payload.get("blockers") or [])]
    assert any("关键词堆叠" in item for item in blockers)
    assert "单主题自然语言短句" in str(payload.get("next_hint") or "")


def test_execute_tool_with_policy_should_normalize_search_query_before_invoke() -> None:
    class _CaptureSearchTool:
        name = "search"

        def __init__(self) -> None:
            self.received_query = ""

        async def invoke(self, function_name, **kwargs):
            self.received_query = str(kwargs.get("query") or "")
            return ToolResult(success=True, data={"results": [{"url": "https://example.com"}]})

    tool = _CaptureSearchTool()
    step = Step(description="搜索并总结")

    _ = asyncio.run(
        execute_tool_with_policy(
            logger=logging.getLogger(__name__),
            step=step,
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "请帮我查一下 上海 周末 亲子 游玩 有哪些 推荐 景点"},
            matched_tool=tool,  # type: ignore[arg-type]
            tool_name="search",
            started_at=time.perf_counter(),
        )
    )

    assert tool.received_query != ""
    assert tool.received_query == "上海 周末 亲子 游玩 有哪些 推荐 景点"


def test_execute_tool_with_policy_should_return_normalized_search_query_as_executed_args() -> None:
    class _CaptureSearchTool:
        name = "search"

        async def invoke(self, function_name, **kwargs):
            return ToolResult(success=True, data={"query": kwargs.get("query"), "results": []})

    decision = asyncio.run(
        execute_tool_with_policy(
            logger=logging.getLogger(__name__),
            step=Step(description="搜索并总结"),
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "请帮我查一下 上海 周末 亲子 游玩 有哪些 推荐 景点"},
            matched_tool=_CaptureSearchTool(),  # type: ignore[arg-type]
            tool_name="search",
            started_at=time.perf_counter(),
        )
    )

    assert decision.executed_function_args["query"] == "上海 周末 亲子 游玩 有哪些 推荐 景点"


def test_policy_engine_should_not_count_blocked_search_call_as_real_invocation() -> None:
    logger = logging.getLogger(__name__)
    engine = ToolPolicyEngine(logger=logger)
    execution_state = ExecutionState()
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web"},
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

    result = asyncio.run(
        engine.evaluate_tool_call(
            step=Step(description="检索并总结"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "AI 编程助手 IDE 支持 价格 对比 最新"},
            matched_tool=None,
            runtime_tools=[],
            browser_route_state_key="",
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=execution_state,
            started_at=time.perf_counter(),
        )
    )

    assert result.tool_result.success is False
    assert execution_state.search_invocation_count == 0
    assert execution_state.search_repeat_counter == {}
    assert execution_state.same_tool_repeat_count == 0


def test_apply_tool_preinvoke_effects_should_track_repeat_counters() -> None:
    state = ExecutionState()
    search_args = {"query": "主流 AI 编程助手及其支持的 IDE"}

    apply_tool_preinvoke_effects(
        normalized_function_name="search_web",
        function_args=search_args,
        execution_state=state,
    )
    apply_tool_preinvoke_effects(
        normalized_function_name="search_web",
        function_args=search_args,
        execution_state=state,
    )
    apply_tool_preinvoke_effects(
        normalized_function_name="fetch_page",
        function_args={"url": "https://example.com/page?utm=test"},
        execution_state=state,
    )

    assert state.same_tool_repeat_count == 1
    assert state.last_tool_fingerprint != ""
    assert state.search_repeat_counter[build_search_fingerprint(search_args)] == 2
    assert state.fetch_repeat_counter["https://example.com/page"] == 1


def test_evaluate_research_route_policy_should_block_keyword_stacked_search_query() -> None:
    step = Step(description="检索并总结")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web"},
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
    decision = evaluate_research_route_policy(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "AI 编程助手 IDE 支持 价格 对比 最新"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=ExecutionState(),
        )
    )
    assert decision is not None
    assert decision.reason_code == "research_query_style_blocked"
    assert "单主题自然语言描述" in str(decision.message_for_model or "")


def test_evaluate_research_route_policy_should_allow_natural_language_search_query() -> None:
    step = Step(description="检索并总结")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web"},
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
    decision = evaluate_research_route_policy(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "主流 AI 编程助手及其支持的 IDE"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=ExecutionState(),
        )
    )
    assert decision is None


def test_evaluate_research_route_policy_should_allow_compact_natural_language_query_with_structure_word() -> None:
    step = Step(description="检索并总结")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web"},
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
    decision = evaluate_research_route_policy(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "主流AI编程助手支持哪些IDE"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=ExecutionState(),
        )
    )
    assert decision is None


def test_evaluate_research_route_policy_should_block_compact_keyword_stacked_search_query() -> None:
    step = Step(description="检索并总结")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "搜索并读取"}],
        available_tools=[],
        available_function_names={"search_web"},
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
    decision = evaluate_research_route_policy(
        ConstraintInput(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "AI编程助手IDE支持价格对比最新"},
            matched_tool=object(),
            iteration_blocked_function_names=set(),
            execution_context=execution_context,
            execution_state=ExecutionState(),
        )
    )
    assert decision is not None
    assert decision.reason_code == "research_query_style_blocked"


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


def test_execute_step_with_prompt_should_rewrite_search_web_to_fetch_page_when_explicit_url_present() -> None:
    llm = _FakeSequentialToolCallLLM(
        messages=[
            {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "上海周边周末自驾游"}, ensure_ascii=False),
                        },
                    }
                ]
            },
            {
                "content": json.dumps(
                    {
                        "success": True,
                        "summary": "已读取目标URL",
                        "delivery_text": "",
                        "attachments": [],
                        "blockers": [],
                        "facts_learned": [],
                        "open_questions": [],
                        "next_hint": "",
                    },
                    ensure_ascii=False,
                )
            },
        ]
    )
    tool = _FakeSearchFetchTool()
    step = Step(description="读取URL https://www.sohu.com/a/880046224_121956422 的页面内容")

    payload, tool_events = asyncio.run(
        execute_step_with_prompt(
            llm=llm,  # type: ignore[arg-type]
            step=step,
            runtime_tools=[tool],  # type: ignore[arg-type]
            task_mode="research",
            max_tool_iterations=2,
            user_content=[{"type": "text", "text": "请读取这个链接并总结"}],
        )
    )

    assert payload.get("success") is True
    assert len(tool.invocations) == 1
    assert tool.invocations[0][0] == "fetch_page"
    assert tool.invocations[0][1]["url"] == "https://www.sohu.com/a/880046224_121956422"
    called_events = [event for event in tool_events if str(event.status.value) == "called"]
    assert len(called_events) >= 1
    assert called_events[0].function_name == "fetch_page"


def test_policy_engine_should_count_only_final_rewritten_call_once() -> None:
    from app.infrastructure.runtime.langgraph.graphs.planner_react.policy_engine.engine import ToolPolicyEngine

    logger = logging.getLogger(__name__)
    engine = ToolPolicyEngine(logger=logger)
    execution_state = ExecutionState()
    step = Step(description="读取URL https://example.com/article 的页面内容")
    runtime_tool = _FakeSearchFetchTool()

    result = asyncio.run(
        engine.evaluate_tool_call(
            step=step,
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "读取文章正文"},
            matched_tool=runtime_tool,
            runtime_tools=[runtime_tool],
            browser_route_state_key="",
            iteration_blocked_function_names=set(),
            execution_context=ExecutionContext(
                normalized_user_content=[{"type": "text", "text": "读取 https://example.com/article 页面正文"}],
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
                research_has_explicit_url=True,
                current_user_message_text="读取 https://example.com/article 页面正文",
            ),
            execution_state=execution_state,
            started_at=time.perf_counter(),
        )
    )

    assert result.final_function_name == "fetch_page"
    assert execution_state.same_tool_repeat_count == 1
    assert execution_state.search_repeat_counter == {}
    assert execution_state.fetch_repeat_counter == {"https://example.com/article": 1}


def test_build_research_query_should_keep_single_topic_and_strip_trailing_constraints() -> None:
    assert build_research_query("请帮我查一下 厦门周边适合自驾游的自然风景目的地 车程2小时内 两天一夜") == (
        "厦门周边适合自驾游的自然风景目的地"
    )


def test_build_research_diagnosis_should_mark_search_low_recall() -> None:
    state = ExecutionState()
    state.search_invocation_count = 1
    diagnosis = build_research_diagnosis(state=state)
    assert diagnosis["code"] == "search_low_recall"


def test_build_research_diagnosis_should_mark_search_snippet_sufficient() -> None:
    state = ExecutionState()
    state.search_invocation_count = 1
    state.research_candidate_urls = ["https://example.com/a"]
    state.research_snippet_sufficient = True
    state.last_search_evidence_quality = {"reason_code": "snippet_sufficient"}
    diagnosis = build_research_diagnosis(state=state)
    assert diagnosis["code"] == "search_snippet_sufficient"


def test_build_research_diagnosis_should_mark_search_snippet_insufficient() -> None:
    state = ExecutionState()
    state.search_invocation_count = 1
    state.research_candidate_urls = ["https://example.com/a"]
    state.last_search_evidence_quality = {"need_fetch": True, "reason_code": "snippet_insufficient"}
    diagnosis = build_research_diagnosis(state=state)
    assert diagnosis["code"] == "search_snippet_insufficient"


def test_apply_tool_result_effects_should_convert_low_value_fetch_into_research_diagnosis_failure() -> None:
    logger = logging.getLogger(__name__)
    step = Step(description="读取页面正文")
    execution_context = ExecutionContext(
        normalized_user_content=[{"type": "text", "text": "读取页面"}],
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

    effects_result = apply_tool_result_effects(
        logger=logger,
        step=step,
        function_name="fetch_page",
        normalized_function_name="fetch_page",
        function_args={"url": "https://example.com/a"},
        tool_result=ToolResult(
            success=True,
            data={
                "url": "https://example.com/a",
                "final_url": "https://example.com/a",
                "title": "首页",
                "content": "太短了",
            },
        ),
        loop_break_reason="",
        browser_route_state_key="",
        execution_context=execution_context,
        execution_state=execution_state,
    )

    assert effects_result.tool_result.success is False
    assert execution_state.last_fetch_quality["is_useful"] is False
    diagnosis = dict(execution_state.runtime_recent_action.get("research_diagnosis") or {})
    assert diagnosis["code"] == "fetch_low_value"
    assert "有效信息" in str(effects_result.tool_result.message or "")


def test_runtime_logging_should_render_requested_and_executed_function_names() -> None:
    logger = logging.getLogger("runtime-log-field-test")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    from app.domain.services.runtime.contracts.runtime_logging import log_runtime

    log_runtime(
        logger,
        logging.INFO,
        "测试日志字段",
        requested_function_name="search_web",
        final_function_name="fetch_page",
        executed_function_name="fetch_page",
    )

    handler.flush()
    output = stream.getvalue()
    logger.removeHandler(handler)

    assert '原始函数名="search_web"' in output
    assert '最终函数名="fetch_page"' in output
    assert '实际执行函数名="fetch_page"' in output
