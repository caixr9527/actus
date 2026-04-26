#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""入口编译原因码。

原因码用于日志、测试和后续误判样本分析，表达 EntryCompiler 为什么选择某条入口通道。
"""

ACTIVE_PLAN_RESUME = "active_plan_resume"
INPUT_ATTACHMENT_REQUIRES_PLANNER = "input_attachment_requires_planner"
PHATIC_DIRECT_ANSWER = "phatic_direct_answer"
SIMPLE_DIRECT_ANSWER = "simple_direct_answer"
CONTEXTUAL_FOLLOWUP_DIRECT_ANSWER = "contextual_followup_direct_answer"
FRESHNESS_REQUIRES_TOOL = "freshness_requires_tool"
SINGLE_URL_ATOMIC_ACTION = "single_url_atomic_action"
SINGLE_FILE_READ_ATOMIC_ACTION = "single_file_read_atomic_action"
SINGLE_SEARCH_ATOMIC_ACTION = "single_search_atomic_action"
SINGLE_TOOL_ATOMIC_ACTION = "single_tool_atomic_action"
WAIT_BEFORE_ACTION = "wait_before_action"
COMPLEX_WAIT_REQUIRES_PLANNER = "complex_wait_requires_planner"
SEARCH_AND_SYNTHESIS_REQUIRES_PLANNER = "search_and_synthesis_requires_planner"
SEARCH_AND_READ_REQUIRES_PLANNER = "search_and_read_requires_planner"
COMPARISON_REQUIRES_PLANNER = "comparison_requires_planner"
PLANNING_REQUIRES_PLANNER = "planning_requires_planner"
PLAN_ONLY_REQUIRES_PLANNER = "plan_only_requires_planner"
HIGH_RISK_REQUIRES_PLANNER = "high_risk_requires_planner"
MULTI_STEP_REQUIRES_PLANNER = "multi_step_requires_planner"
LOW_CONFIDENCE_REQUIRES_PLANNER = "low_confidence_requires_planner"
EMPTY_MESSAGE_REQUIRES_PLANNER = "empty_message_requires_planner"
