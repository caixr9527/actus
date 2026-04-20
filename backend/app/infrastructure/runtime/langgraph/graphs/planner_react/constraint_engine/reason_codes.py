#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束层 reason_code 常量。"""

# 通用
REASON_ALLOW = "allow"
REASON_INVALID_TOOL = "invalid_tool"
REASON_CONSTRAINT_ENGINE_ERROR = "constraint_engine_error"
REASON_REWRITE_CHAIN_BLOCKED = "rewrite_chain_blocked"

# human_wait
REASON_HUMAN_WAIT_NON_INTERRUPT_TOOL_BLOCKED = "human_wait_non_interrupt_tool_blocked"
REASON_ASK_USER_NOT_ALLOWED = "ask_user_not_allowed"

# task_mode / artifact / final_delivery
REASON_RESEARCH_FILE_CONTEXT_REQUIRED = "research_file_context_required"
REASON_WEB_READING_FILE_TOOL_BLOCKED = "web_reading_file_tool_blocked"
REASON_GENERAL_INLINE_FILE_CONTEXT_REQUIRED = "general_inline_file_context_required"
REASON_FILE_PROCESSING_SHELL_EXPLICIT_REQUIRED = "file_processing_shell_explicit_required"
REASON_BROWSER_ROUTE_BLOCKED = "browser_route_blocked"
REASON_TASK_MODE_TOOL_BLOCKED = "task_mode_tool_blocked"
REASON_READ_ONLY_FILE_INTENT_WRITE_BLOCKED = "read_only_file_intent_write_blocked"
REASON_ARTIFACT_POLICY_FILE_OUTPUT_BLOCKED = "artifact_policy_file_output_blocked"
REASON_FINAL_DELIVERY_SEARCH_DRIFT_BLOCKED = "final_delivery_search_drift_blocked"
REASON_FINAL_INLINE_FILE_OUTPUT_BLOCKED = "final_inline_file_output_blocked"
REASON_FINAL_DELIVERY_SHELL_DRIFT_BLOCKED = "final_delivery_shell_drift_blocked"

# research route / rewrite
REASON_RESEARCH_QUERY_STYLE_BLOCKED = "research_query_style_blocked"
REASON_RESEARCH_ROUTE_SEARCH_REQUIRED = "research_route_search_required"
REASON_RESEARCH_ROUTE_FETCH_REQUIRED = "research_route_fetch_required"
REASON_RESEARCH_ROUTE_CROSS_DOMAIN_FETCH_LIMIT = "research_route_cross_domain_fetch_limit"
REASON_RESEARCH_SEARCH_TO_FETCH_REWRITE = "research_search_to_fetch_rewrite"

# repeat loop
REASON_SEARCH_REPEAT = "search_repeat"
REASON_RESEARCH_ROUTE_FINGERPRINT_REPEAT = "research_route_fingerprint_repeat"
REASON_REPEAT_TOOL_CALL = "repeat_tool_call"
REASON_REPEAT_TOOL_CALL_SUCCESS_FALLBACK = "repeat_tool_call_success_fallback"

# browser route 软阻断
REASON_BROWSER_CLICK_TARGET_BLOCKED = "browser_click_target_blocked"
REASON_BROWSER_HIGH_LEVEL_RETRY_BLOCKED = "browser_high_level_retry_blocked"
