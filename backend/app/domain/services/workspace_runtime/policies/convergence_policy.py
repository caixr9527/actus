#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具循环收敛策略。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Set

from app.domain.models import FetchedPage, SearchResults, Step, ToolResult
from app.domain.services.runtime.contracts.langgraph_settings import (
    ASK_USER_FUNCTION_NAME,
    NOTIFY_USER_FUNCTION_NAME,
)
from app.domain.services.runtime.normalizers import normalize_execution_response
from .common import compact_tool_value, hash_payload, truncate_tool_text

BLOCKED_TOOL_LOOP_BREAK_REASONS: Set[str] = {
    "human_wait_non_interrupt_tool_blocked",
    "read_only_file_intent_write_blocked",
    "research_file_context_required",
    "web_reading_file_tool_blocked",
    # P3-CASE3 修复：file_processing 在无显式命令意图时拦截 shell_execute。
    "file_processing_shell_explicit_required",
    "artifact_policy_file_output_blocked",
    "browser_route_blocked",
    "task_mode_tool_blocked",
    "research_route_search_required",
    "research_route_fetch_required",
    "research_query_style_blocked",
    # P3-2A/3A 收敛修复：检索链路重复/瞬时错误统一沉淀到 recent_action。
    "research_route_fingerprint_repeat",
    "research_route_transport_error",
    "browser_click_target_blocked",
    "browser_high_level_retry_blocked",
}


def build_tool_fingerprint(function_name: str, function_args: Dict[str, Any]) -> str:
    return hash_payload(
        {
            "function_name": function_name.strip().lower(),
            "args": compact_tool_value(function_args),
        }
    )


def build_search_fingerprint(function_args: Dict[str, Any]) -> str:
    return hash_payload(
        {
            "query": str(function_args.get("query") or "").strip().lower(),
            "engines": str(function_args.get("engines") or "").strip().lower(),
            "language": str(function_args.get("language") or "").strip().lower(),
            "time_range": str(function_args.get("time_range") or "").strip().lower(),
        }
    )


def build_loop_break_payload(
        *,
        step: Step,
        blocker: str,
        next_hint: str,
        runtime_recent_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "success": False,
        "summary": f"当前步骤暂时未能完成：{step.description}",
        "result": f"当前步骤暂时未能完成：{step.description}",
        "attachments": [],
        "blockers": [blocker],
        "next_hint": next_hint,
    }
    merged_recent_action = dict(runtime_recent_action or {})
    merged_recent_action["last_no_progress_reason"] = blocker
    if merged_recent_action:
        payload["runtime_recent_action"] = merged_recent_action
    return payload


def normalize_execution_payload(parsed: Dict[str, Any], *, default_summary: str) -> Dict[str, Any]:
    payload = normalize_execution_response(parsed)
    if not str(payload.get("summary") or "").strip():
        payload["summary"] = default_summary
        payload["result"] = default_summary
    runtime_recent_action = parsed.get("runtime_recent_action")
    if isinstance(runtime_recent_action, dict) and runtime_recent_action:
        payload["runtime_recent_action"] = runtime_recent_action
    return payload


def build_human_wait_missing_interrupt_payload(
        step: Step,
        *,
        runtime_recent_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_loop_break_payload(
        step=step,
        blocker="当前步骤需要等待用户确认/选择，但尚未成功发起等待请求。",
        next_hint="请调用 message_ask_user 发起确认/选择，不要继续搜索、读取或直接结束当前步骤。",
        runtime_recent_action=runtime_recent_action,
    )


def build_recent_failed_action(function_name: str, tool_result: ToolResult) -> Dict[str, Any]:
    return {
        "function_name": str(function_name or "").strip(),
        "message": truncate_tool_text(tool_result.message, max_chars=120),
    }


def build_recent_blocked_tool_call(
        *,
        function_name: str,
        tool_result: ToolResult,
        loop_break_reason: str,
) -> Dict[str, Any]:
    if loop_break_reason not in BLOCKED_TOOL_LOOP_BREAK_REASONS:
        return {}
    return {
        "function_name": str(function_name or "").strip(),
        "reason": loop_break_reason,
        "message": truncate_tool_text(tool_result.message, max_chars=160),
    }


def summarize_tool_result_data(function_name: str, tool_result: ToolResult) -> Dict[str, Any]:
    normalized_name = function_name.strip().lower()
    result_data = tool_result.data
    if normalized_name == NOTIFY_USER_FUNCTION_NAME:
        return {
            "notified": True,
            "message": "当前步骤进度通知已发送，请继续执行实际工具步骤，不要再次调用进度通知。",
        }
    if normalized_name == ASK_USER_FUNCTION_NAME:
        interrupt_payload = result_data.get("interrupt") if isinstance(result_data, dict) else None
        if isinstance(interrupt_payload, dict):
            return {
                "interrupt": {
                    "kind": str(interrupt_payload.get("kind") or "").strip(),
                    "prompt": truncate_tool_text(interrupt_payload.get("prompt"), max_chars=200),
                    "title": truncate_tool_text(interrupt_payload.get("title"), max_chars=120),
                }
            }
        return {"interrupt": True}
    if normalized_name == "write_file" and isinstance(result_data, dict):
        return {
            "filepath": str(
                result_data.get("filepath")
                or result_data.get("file_path")
                or result_data.get("path")
                or ""
            ).strip(),
            "message": tool_result.message,
        }
    if normalized_name == "read_file" and isinstance(result_data, dict):
        return {
            "filepath": str(
                result_data.get("filepath")
                or result_data.get("file_path")
                or result_data.get("path")
                or ""
            ).strip(),
            "content": result_data.get("content"),
        }
    if normalized_name in {"list_files", "find_files"} and isinstance(result_data, dict):
        files = result_data.get("files") or result_data.get("results") or []
        if not isinstance(files, list):
            files = []
        return {
            "dir_path": str(result_data.get("dir_path") or "").strip(),
            "files": files,
        }
    if normalized_name == "search_web" and isinstance(result_data, SearchResults):
        return {
            "query": str(result_data.query or "").strip(),
            "results": [
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                }
                for item in list(result_data.results or [])
            ],
        }
    if normalized_name == "fetch_page":
        # fetch_page 的正文已经由 sandbox 的 max_chars 控制；
        # 执行 LLM 这里应直接消费完整页面结果，不再做二次截断。
        if isinstance(result_data, FetchedPage):
            return {
                "url": str(result_data.url or "").strip(),
                "final_url": str(result_data.final_url or "").strip(),
                "status_code": int(result_data.status_code or 0),
                "content_type": str(result_data.content_type or "").strip(),
                "title": str(result_data.title or "").strip(),
                "content": str(result_data.content or ""),
                "excerpt": str(result_data.excerpt or ""),
                "content_length": int(result_data.content_length or 0),
                "truncated": bool(result_data.truncated),
                "max_chars": result_data.max_chars,
            }
        if isinstance(result_data, dict):
            return {
                "url": str(result_data.get("url") or "").strip(),
                "final_url": str(result_data.get("final_url") or "").strip(),
                "status_code": int(result_data.get("status_code") or 0),
                "content_type": str(result_data.get("content_type") or "").strip(),
                "title": str(result_data.get("title") or "").strip(),
                "content": str(result_data.get("content") or ""),
                "excerpt": str(result_data.get("excerpt") or ""),
                "content_length": int(result_data.get("content_length") or 0),
                "truncated": bool(result_data.get("truncated", False)),
                "max_chars": result_data.get("max_chars"),
            }
    return {"data": compact_tool_value(result_data)}


def build_tool_feedback_content(function_name: str, tool_result: ToolResult) -> str:
    payload = {
        "success": bool(tool_result.success),
        "message": tool_result.message,
        "data": summarize_tool_result_data(function_name, tool_result),
    }
    return json.dumps(payload, ensure_ascii=False)
