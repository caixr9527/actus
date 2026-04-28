#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime EntryCompiler 合同级规则测试。"""

import pytest

from app.domain.models import ExecutionStatus, Plan, Step, StepTaskModeHint
from app.domain.services.workspace_runtime.entry import (
    EntryCompiler,
    EntryContextProfile,
    EntryRiskLevel,
    EntryRoute,
    EntryToolBudget,
)
from app.domain.services.workspace_runtime.entry import reason_codes as rc


def _compile(
        user_message: str,
        *,
        has_input_parts: bool = False,
        has_active_plan: bool = False,
        contextual_followup_anchor: bool = False,
):
    return EntryCompiler().compile(
        user_message=user_message,
        has_input_parts=has_input_parts,
        has_active_plan=has_active_plan,
        contextual_followup_anchor=contextual_followup_anchor,
    )


@pytest.mark.parametrize(
    ("user_message", "reason_code"),
    [
        ("你好", rc.PHATIC_DIRECT_ANSWER),
        ("这段话是什么意思？", rc.SIMPLE_DIRECT_ANSWER),
    ],
)
def test_compile_should_route_simple_answer_without_context_or_tools(
        user_message: str,
        reason_code: str,
) -> None:
    contract = _compile(user_message)

    assert contract.route == EntryRoute.ANSWER
    assert contract.task_mode == StepTaskModeHint.GENERAL
    assert contract.context_profile == EntryContextProfile.NONE
    assert contract.tool_budget == EntryToolBudget.NONE
    assert contract.needs_summary is False
    assert contract.plan_only is False
    assert contract.risk_level == EntryRiskLevel.LOW
    assert reason_code in contract.reason_codes
    assert contract.upgrade_policy.allow_upgrade is False


def test_compile_should_use_minimal_history_for_contextual_followup() -> None:
    contract = _compile(
        "再详细一点",
        contextual_followup_anchor=True,
    )

    assert contract.route == EntryRoute.ANSWER
    assert contract.context_profile == EntryContextProfile.MINIMAL_HISTORY
    assert contract.tool_budget == EntryToolBudget.NONE
    assert contract.needs_summary is False
    assert rc.CONTEXTUAL_FOLLOWUP_DIRECT_ANSWER in contract.reason_codes
    assert contract.source.contextual_followup_anchor is True


@pytest.mark.parametrize(
    ("user_message", "task_mode", "reason_code", "tool_budget"),
    [
        ("搜索 OpenAI 官网", StepTaskModeHint.RESEARCH, rc.SINGLE_SEARCH_ATOMIC_ACTION, EntryToolBudget.SMALL_LOOP),
        (
            "打开 https://openai.com/docs 看一下当前页面",
            StepTaskModeHint.WEB_READING,
            rc.SINGLE_URL_ATOMIC_ACTION,
            EntryToolBudget.SMALL_LOOP,
        ),
        (
            "读取 /tmp/backend.log 并整理错误摘要",
            StepTaskModeHint.FILE_PROCESSING,
            rc.SINGLE_FILE_READ_ATOMIC_ACTION,
            EntryToolBudget.SMALL_LOOP,
        ),
    ],
)
def test_compile_should_route_single_tool_tasks_to_atomic_action(
        user_message: str,
        task_mode: StepTaskModeHint,
        reason_code: str,
        tool_budget: EntryToolBudget,
) -> None:
    contract = _compile(user_message)

    assert contract.route == EntryRoute.ATOMIC_ACTION
    assert contract.task_mode == task_mode
    assert contract.context_profile == EntryContextProfile.WORKSPACE
    assert contract.tool_budget == tool_budget
    assert contract.needs_summary is True
    assert contract.plan_only is False
    assert reason_code in contract.reason_codes
    assert contract.upgrade_policy.allow_upgrade is True
    assert contract.upgrade_policy.upgrade_on_second_tool_family is True
    assert contract.upgrade_policy.upgrade_on_open_questions is True


def test_compile_should_route_freshness_request_to_atomic_action() -> None:
    contract = _compile("现在 OpenAI 最新模型是什么？")

    assert contract.route == EntryRoute.ATOMIC_ACTION
    assert contract.task_mode == StepTaskModeHint.GENERAL
    assert contract.tool_budget == EntryToolBudget.SINGLE_CALL
    assert contract.freshness_score > 0
    assert rc.FRESHNESS_REQUIRES_TOOL in contract.reason_codes


@pytest.mark.parametrize(
    ("user_message", "reason_code"),
    [
        ("搜索并阅读 OpenAI Agents 文档，整理关键点", rc.SEARCH_AND_SYNTHESIS_REQUIRES_PLANNER),
        ("对比 LangGraph 和 AutoGen 的优缺点，推荐怎么选", rc.COMPARISON_REQUIRES_PLANNER),
        ("规划一个三阶段调研方案，先不要执行，只给步骤", rc.PLAN_ONLY_REQUIRES_PLANNER),
    ],
)
def test_compile_should_route_complex_requests_to_planner(
        user_message: str,
        reason_code: str,
) -> None:
    contract = _compile(user_message)

    assert contract.route == EntryRoute.PLANNED_TASK
    assert contract.context_profile == EntryContextProfile.FULL
    assert contract.tool_budget == EntryToolBudget.PLANNER_CONTROLLED
    assert contract.needs_summary is True
    assert reason_code in contract.reason_codes
    assert contract.upgrade_policy.allow_upgrade is False


def test_compile_should_keep_plan_only_on_planned_contract() -> None:
    contract = _compile("帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。")

    assert contract.route == EntryRoute.PLANNED_TASK
    assert contract.plan_only is True
    assert rc.PLAN_ONLY_REQUIRES_PLANNER in contract.reason_codes


def test_compile_should_route_preconfirm_request_to_wait() -> None:
    contract = _compile("在你开始搜索之前先让我确认")

    assert contract.route == EntryRoute.WAIT
    assert contract.context_profile == EntryContextProfile.WORKSPACE
    assert contract.tool_budget == EntryToolBudget.SMALL_LOOP
    assert contract.needs_summary is True
    assert contract.plan_only is False
    assert rc.WAIT_BEFORE_ACTION in contract.reason_codes


def test_compile_should_route_complex_wait_request_to_planner() -> None:
    contract = _compile("帮我制定周末出行方案；如果你需要我先确认预算和偏好，请先停下来问我。")

    assert contract.route == EntryRoute.PLANNED_TASK
    assert contract.context_profile == EntryContextProfile.FULL
    assert contract.tool_budget == EntryToolBudget.PLANNER_CONTROLLED
    assert rc.COMPLEX_WAIT_REQUIRES_PLANNER in contract.reason_codes


def test_compile_should_route_attachment_input_to_planner() -> None:
    contract = _compile("分析这个附件", has_input_parts=True)

    assert contract.route == EntryRoute.PLANNED_TASK
    assert contract.context_profile == EntryContextProfile.FULL
    assert contract.source.has_input_parts is True
    assert rc.INPUT_ATTACHMENT_REQUIRES_PLANNER in contract.reason_codes


def test_compile_should_route_empty_message_to_planner() -> None:
    contract = _compile("")

    assert contract.route == EntryRoute.PLANNED_TASK
    assert contract.context_profile == EntryContextProfile.FULL
    assert rc.EMPTY_MESSAGE_REQUIRES_PLANNER in contract.reason_codes


def test_compile_should_route_active_plan_to_resume_plan() -> None:
    contract = _compile("继续", has_active_plan=True)

    assert contract.route == EntryRoute.RESUME_PLAN
    assert contract.context_profile == EntryContextProfile.FULL
    assert contract.tool_budget == EntryToolBudget.PLANNER_CONTROLLED
    assert contract.plan_only is False
    assert contract.source.has_active_plan is True
    assert rc.ACTIVE_PLAN_RESUME in contract.reason_codes


def test_compile_state_should_detect_active_plan_from_domain_plan() -> None:
    contract = EntryCompiler().compile_state(
        {
            "user_message": "继续",
            "input_parts": [],
            "message_window": [],
            "recent_run_briefs": [],
            "conversation_summary": "",
            "plan": Plan(steps=[Step(title="未完成步骤")], status=ExecutionStatus.RUNNING),
        }
    )

    assert contract.route == EntryRoute.RESUME_PLAN
    assert rc.ACTIVE_PLAN_RESUME in contract.reason_codes


def test_compile_state_should_ignore_done_plan_when_routing() -> None:
    contract = EntryCompiler().compile_state(
        {
            "user_message": "继续",
            "input_parts": [],
            "message_window": [],
            "recent_run_briefs": [],
            "conversation_summary": "",
            "plan": Plan(steps=[Step(title="已完成步骤")], status=ExecutionStatus.COMPLETED),
        }
    )

    assert contract.route == EntryRoute.ANSWER
    assert rc.CONTEXTUAL_FOLLOWUP_DIRECT_ANSWER not in contract.reason_codes
