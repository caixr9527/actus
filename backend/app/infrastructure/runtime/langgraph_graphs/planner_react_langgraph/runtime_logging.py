#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""planner-react 运行链路统一日志工具。"""

import json
import logging
import time
import uuid
from contextvars import ContextVar, Token
from typing import Any, Mapping, Optional

from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState

LOG_PREFIX = "规划执行"
LOG_VALUE_MAX_CHARS = 240
FIELD_LABELS: dict[str, str] = {
    "trace_id": "追踪ID",
    "session_id": "会话ID",
    "run_id": "运行ID",
    "thread_id": "线程ID",
    "current_step_id": "当前步骤ID",
    "execution_count": "执行次数",
    "step_id": "步骤ID",
    "next_step_id": "下一步骤ID",
    "last_step_id": "最近步骤ID",
    "waiting_step_id": "等待步骤ID",
    "objective_key": "目标键",
    "plan_title": "计划标题",
    "step_title": "步骤标题",
    "step_description": "步骤描述",
    "step_count": "步骤数",
    "new_step_count": "新增步骤数",
    "total_step_count": "总步骤数",
    "current_step_count": "当前步骤数",
    "attachment_count": "附件数",
    "artifact_count": "产物数",
    "blocker_count": "阻塞数",
    "open_question_count": "问题数",
    "preference_count": "偏好数",
    "memory_candidate_count": "记忆候选数",
    "recalled_memory_count": "召回记忆数",
    "existing_memory_count": "已有记忆数",
    "pending_memory_write_count": "待写记忆数",
    "remaining_memory_write_count": "剩余待写记忆数",
    "persisted_memory_count": "写入记忆数",
    "kept_candidate_count": "保留候选数",
    "trimmed_message_count": "裁剪消息数",
    "message_window_size": "消息窗口大小",
    "compacted_message_window_size": "压缩后消息窗口大小",
    "runtime_tool_count": "运行时工具数",
    "available_tool_count": "可用工具数",
    "context_memory_count": "上下文记忆数",
    "context_recent_run_count": "近期成功运行数",
    "context_recent_attempt_count": "近期失败运行数",
    "max_tool_iterations": "最大工具轮次",
    "max_execution_steps": "最大执行步数",
    "iteration": "轮次",
    "iteration_count": "轮次数",
    "tool_call_id": "工具调用ID",
    "tool_name": "工具名",
    "function_name": "函数名",
    "candidate_count": "候选数",
    "selected_function": "选中函数",
    "arg_keys": "参数键",
    "response_keys": "返回键",
    "event_type": "事件类型",
    "language": "语言",
    "fact_count": "事实数",
    "raw_length": "原始长度",
    "graph_input_type": "输入类型",
    "wait_event_count": "等待事件数",
    "resume_value_type": "恢复值类型",
    "status": "状态",
    "success": "成功",
    "error": "错误",
    "reason": "原因",
    "interrupt_kind": "中断类型",
    "has_interrupt": "是否中断",
    "timeout_seconds": "超时秒数",
    "message_length": "消息长度",
    "content_length": "内容长度",
    "resumed_message_length": "恢复消息长度",
    "emitted_event_count": "已发事件数",
    "elapsed_ms": "耗时毫秒",
    "llm_elapsed_ms": "模型耗时毫秒",
    "tool_elapsed_ms": "工具耗时毫秒",
    "skill_elapsed_ms": "技能耗时毫秒",
    "write_elapsed_ms": "写入耗时毫秒",
    "recall_elapsed_ms": "召回耗时毫秒",
    "cleanup_elapsed_ms": "清理耗时毫秒",
    "has_repository": "是否有仓储",
    "has_checkpointer": "是否有检查点",
    "has_skill_runtime": "是否有技能运行时",
    "skill_id": "技能ID",
    "source_run_id": "来源运行ID",
    "source_step_id": "来源步骤ID",
}
STATE_CONTEXT_KEYS: tuple[str, ...] = (
    "session_id",
    "run_id",
    "thread_id",
    "current_step_id",
    "execution_count",
)
_TRACE_ID: ContextVar[Optional[str]] = ContextVar("planner_react_trace_id", default=None)


def _normalize_log_value(value: Any) -> str:
    if isinstance(value, str):
        normalized = " ".join(value.split())
    elif isinstance(value, (int, float, bool)) or value is None:
        normalized = value
    else:
        normalized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    rendered = json.dumps(normalized, ensure_ascii=False, default=str)
    if len(rendered) <= LOG_VALUE_MAX_CHARS:
        return rendered
    return json.dumps(f"{rendered[:LOG_VALUE_MAX_CHARS]}...", ensure_ascii=False)


def _merge_log_fields(
        *,
        state: Optional[PlannerReActLangGraphState] = None,
        fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged_fields: dict[str, Any] = {}
    trace_id = _TRACE_ID.get()
    if trace_id:
        merged_fields["trace_id"] = trace_id
    if isinstance(state, dict):
        for key in STATE_CONTEXT_KEYS:
            value = state.get(key)
            if value is None or value == "":
                continue
            merged_fields[key] = value

    for key, value in dict(fields or {}).items():
        if value is None or value == "":
            continue
        merged_fields[str(key)] = value
    return merged_fields


def format_runtime_log(
        event: str,
        *,
        state: Optional[PlannerReActLangGraphState] = None,
        **fields: Any,
) -> str:
    merged_fields = _merge_log_fields(state=state, fields=fields)
    ordered_parts = [f"{LOG_PREFIX} 事件={_normalize_log_value(event)}"]
    for key in sorted(merged_fields.keys()):
        rendered_key = FIELD_LABELS.get(key, key)
        ordered_parts.append(f"{rendered_key}={_normalize_log_value(merged_fields[key])}")
    return " ".join(ordered_parts)


def log_runtime(
        logger: logging.Logger,
        level: int,
        event: str,
        *,
        state: Optional[PlannerReActLangGraphState] = None,
        exc_info: bool = False,
        **fields: Any,
) -> None:
    logger.log(
        level,
        format_runtime_log(event, state=state, **fields),
        exc_info=exc_info,
    )


def now_perf() -> float:
    return time.perf_counter()


def elapsed_ms(start_time: float) -> int:
    return max(int((time.perf_counter() - start_time) * 1000), 0)


def build_trace_id(session_id: str, run_id: Optional[str] = None) -> str:
    trace_suffix = uuid.uuid4().hex[:8]
    scope = str(run_id or session_id or "runtime").strip()[:16] or "runtime"
    return f"{scope}-{trace_suffix}"


def bind_trace_id(trace_id: str) -> Token:
    return _TRACE_ID.set(str(trace_id).strip() or None)


def reset_trace_id(token: Token) -> None:
    _TRACE_ID.reset(token)
