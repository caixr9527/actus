#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：循环收敛原因映射。"""

from typing import Any, Dict, Optional

from app.domain.models import Step
from app.domain.services.workspace_runtime.policies import (
    build_loop_break_payload as _build_loop_break_payload,
)
from app.infrastructure.runtime.langgraph.graphs.common.graph_parsers import safe_parse_json


def build_loop_break_result(
        *,
        loop_break_reason: str,
        step: Step,
        tool_result: Optional[Any] = None,
        runtime_recent_action: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if loop_break_reason == "repeat_tool_call_success_fallback":
        payload = _normalize_repeat_success_payload(tool_result)
        delivery_text = str(payload.get("delivery_text") or "").strip()
        summary_text = str(payload.get("summary") or "").strip() or delivery_text
        if not summary_text:
            summary_text = f"步骤执行完成：{step.description}"
        return {
            "success": True,
            "summary": summary_text,
            "result": summary_text,
            "delivery_text": delivery_text,
            "attachments": list(payload.get("attachments") or []),
            "blockers": [],
            "next_hint": "",
            "runtime_recent_action": runtime_recent_action or {},
        }
    if loop_break_reason == "repeat_tool_call":
        return _build_loop_break_payload(
            step=step,
            blocker="同一工具及参数被重复调用过多次，当前步骤已被强制收敛。",
            next_hint="请改用其他工具、调整参数，或将当前步骤拆小后再执行。",
            runtime_recent_action=runtime_recent_action,
        )
    if loop_break_reason == "search_repeat":
        next_hint = "请改写搜索主题描述、缩小范围，或改用 fetch_page / 文件读取继续。"
        return _build_loop_break_payload(
            step=step,
            blocker="同一搜索查询已重复触发多次，当前检索路径没有继续收获。",
            next_hint=_append_research_progress_hint(
                runtime_recent_action=runtime_recent_action,
                next_hint=next_hint,
            ),
            runtime_recent_action=runtime_recent_action,
        )
    if loop_break_reason == "research_query_style_blocked":
        return _build_loop_break_payload(
            step=step,
            blocker="当前检索查询属于关键词堆叠，未满足自然语言查询规范。",
            next_hint="请使用单主题自然语言短句表达查询目标，再继续调用 search_web。",
            runtime_recent_action=runtime_recent_action,
        )
    if loop_break_reason == "research_route_fingerprint_repeat":
        next_hint = "请切换其他候选 URL、改用其他工具，或结束当前步骤。"
        return _build_loop_break_payload(
            step=step,
            blocker="同一页面抓取请求已重复触发多次，当前检索路径没有新增信息。",
            next_hint=_append_research_progress_hint(
                runtime_recent_action=runtime_recent_action,
                next_hint=next_hint,
            ),
            runtime_recent_action=runtime_recent_action,
        )
    if loop_break_reason == "research_route_transport_error":
        return _build_loop_break_payload(
            step=step,
            blocker="检索/抓取链路出现瞬时网络错误，当前步骤已停止重试。",
            next_hint="请稍后重试，或先基于已有信息继续后续步骤。",
            runtime_recent_action=runtime_recent_action,
        )
    if loop_break_reason == "research_route_cross_domain_fetch_limit":
        return _build_loop_break_payload(
            step=step,
            blocker="当前研究步骤在同域页面重复抓取，尚未形成跨来源覆盖。",
            next_hint=_append_research_progress_hint(
                runtime_recent_action=runtime_recent_action,
                next_hint="请优先抓取不同域名的候选链接，再继续总结。",
            ),
            runtime_recent_action=runtime_recent_action,
        )
    if loop_break_reason == "browser_no_progress":
        return _build_loop_break_payload(
            step=step,
            blocker="浏览器连续观察未发现新的有效信息，当前页面路径已无进展。",
            next_hint="请更换页面、改用搜索/正文读取，或重新规划当前步骤。",
            runtime_recent_action=runtime_recent_action,
        )
    return None


def _append_research_progress_hint(*, runtime_recent_action: Optional[Dict[str, Any]], next_hint: str) -> str:
    research_progress = dict((runtime_recent_action or {}).get("research_progress") or {})
    missing_signals = [str(item).strip() for item in list(research_progress.get("missing_signals") or []) if
                       str(item).strip()]
    if bool(research_progress.get("is_low_recall")):
        next_hint = next_hint + " 建议先用单主题自然语言短句检索，必要时逐轮仅增加一个筛选条件。"
    if len(missing_signals) == 0:
        return next_hint
    return next_hint + " 当前缺口：" + "；".join(missing_signals[:2]) + "。"


def _normalize_repeat_success_payload(tool_result: Optional[Any]) -> Dict[str, Any]:
    if tool_result is None:
        return {"summary": "", "delivery_text": "", "attachments": []}
    message_text = str(getattr(tool_result, "message", "") or "").strip()
    data = getattr(tool_result, "data", None)
    parsed_data = data if isinstance(data, dict) else {}
    feedback_content = str(parsed_data.get("feedback_content") or "").strip()
    parsed_feedback = safe_parse_json(feedback_content) if feedback_content else {}
    summary_from_feedback = str(parsed_feedback.get("message") or "").strip()
    delivery_from_feedback = str(parsed_feedback.get("data", {}).get("content") or "").strip()
    attachments = []
    raw_attachments = parsed_feedback.get("attachments")
    if isinstance(raw_attachments, list):
        attachments = [str(item).strip() for item in raw_attachments if str(item).strip()]
    return {
        "summary": summary_from_feedback or message_text,
        "delivery_text": delivery_from_feedback or "",
        "attachments": attachments,
    }
