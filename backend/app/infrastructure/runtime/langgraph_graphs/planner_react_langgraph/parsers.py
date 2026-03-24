#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 纯函数解析与归一化工具。"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from app.domain.models import (
    MessageEvent,
    Plan,
    Step,
    ToolEvent,
    ToolEventStatus,
    ExecutionStatus,
)
from app.domain.services.prompts import SUMMARIZE_PROMPT
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState

logger = logging.getLogger(__name__)

_SIMPLE_GREETING_NORMALIZED_SET = {
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "在吗",
    "hi",
    "hello",
    "hey",
}


def safe_parse_json(content: str | None) -> Dict[str, Any]:
    """宽松解析模型 JSON 输出，兼容 fenced code 与前后赘述。"""
    if not content:
        return {}

    candidates: List[str] = [content]

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    start_index = content.find("{")
    end_index = content.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        candidates.append(content[start_index:end_index + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    logger.warning("LangGraph 解析JSON失败，使用回退逻辑")
    return {}


def normalize_attachments(raw_attachments: Any) -> List[str]:
    if isinstance(raw_attachments, str):
        return [raw_attachments]
    if isinstance(raw_attachments, list):
        return [str(item) for item in raw_attachments if str(item).strip()]
    return []


def merge_attachment_paths(*path_groups: List[str]) -> List[str]:
    """按出现顺序去重合并附件路径。"""
    merged_paths: List[str] = []
    seen: set[str] = set()
    for path_group in path_groups:
        for raw_path in path_group:
            path = str(raw_path or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            merged_paths.append(path)
    return merged_paths


def extract_write_file_paths_from_tool_events(tool_events: List[ToolEvent]) -> List[str]:
    """从 write_file 工具事件提取产物路径，作为附件兜底来源。"""
    attachment_paths: List[str] = []
    for tool_event in tool_events:
        if tool_event.status != ToolEventStatus.CALLED:
            continue
        function_name = str(tool_event.function_name or "").strip().lower()
        if function_name != "write_file":
            continue

        arg_path = str(tool_event.function_args.get("filepath") or "").strip()
        if arg_path:
            attachment_paths.append(arg_path)
            continue

        function_result = tool_event.function_result
        result_data = function_result.data if function_result is not None and hasattr(function_result, "data") else None
        if isinstance(result_data, dict):
            result_path = str(
                result_data.get("filepath")
                or result_data.get("file_path")
                or result_data.get("path")
                or ""
            ).strip()
            if result_path:
                attachment_paths.append(result_path)

    return merge_attachment_paths(attachment_paths)


def collect_plan_attachment_paths(plan: Optional[Plan]) -> List[str]:
    """汇总计划中已完成步骤产生的附件路径。"""
    if plan is None:
        return []

    collected_paths: List[str] = []
    for step in plan.steps:
        if not step.done:
            continue
        collected_paths.extend(normalize_attachments(step.attachments))
    return merge_attachment_paths(collected_paths)


def normalize_user_text(text: str) -> str:
    """归一化用户输入，便于做轻量意图判断。"""
    return re.sub(r"[\s\W_]+", "", text.strip().lower(), flags=re.UNICODE)


def is_simple_greeting_message(user_message: str) -> bool:
    """识别纯问候输入，避免进入完整 planner/step 链路。"""
    normalized = normalize_user_text(user_message)
    return normalized in _SIMPLE_GREETING_NORMALIZED_SET


def build_simple_greeting_reply(user_message: str) -> str:
    """根据输入语种返回简洁问候回复。"""
    has_ascii_letters = re.search(r"[a-z]", user_message, flags=re.IGNORECASE) is not None
    has_cjk = re.search(r"[\u4e00-\u9fff]", user_message) is not None
    if has_ascii_letters and not has_cjk:
        return "Hello! I'm your assistant, happy to help."
    return "你好！我是助手，很高兴为您服务。"


def build_fallback_plan_title(user_message: str) -> str:
    """规划 JSON 缺失标题时，生成可读回退标题。"""
    normalized = user_message.strip()
    if not normalized:
        return "任务计划"

    title = normalized[:24]
    return title if len(normalized) <= 24 else f"{title}..."


def get_last_assistant_message_event(events: List[Any]) -> Optional[MessageEvent]:
    """获取事件列表中最后一条 assistant 消息事件。"""
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.role == "assistant":
            return event
    return None


def collect_message_attachment_paths(message_event: Optional[MessageEvent]) -> List[str]:
    """提取消息事件中的附件路径，并按顺序去重。"""
    if message_event is None:
        return []

    attachment_paths: List[str] = []
    for attachment in list(message_event.attachments or []):
        filepath = str(getattr(attachment, "filepath", "") or "").strip()
        if filepath:
            attachment_paths.append(filepath)
    return merge_attachment_paths(attachment_paths)


def build_summarize_prompt(state: PlannerReActLangGraphState) -> str:
    """构建带上下文的总结提示词，降低无关总结风险。"""
    plan = state.get("plan")
    plan_snapshot = plan.model_dump(mode="json") if plan is not None else {}
    final_message = str(state.get("final_message") or "")
    user_message = str(state.get("user_message") or "")
    execution_count = int(state.get("execution_count") or 0)

    return (
        f"{SUMMARIZE_PROMPT}\n\n"
        "请严格基于以下运行上下文输出总结，禁止引入上下文之外的场景或数据：\n"
        f"- 用户原始消息: {user_message}\n"
        f"- 执行轮次: {execution_count}\n"
        f"- 最近一步结果: {final_message}\n"
        f"- 计划快照(JSON): {json.dumps(plan_snapshot, ensure_ascii=False)}\n"
    )


def should_accept_summary_message(
        state: PlannerReActLangGraphState,
        candidate_message: str,
        fallback_message: str,
) -> bool:
    """判断是否接受 summarize 文本，避免单步任务被无关长文覆盖。"""
    candidate = candidate_message.strip()
    if not candidate:
        return False

    execution_count = int(state.get("execution_count") or 0)
    if execution_count <= 1 and fallback_message.strip():
        fallback = fallback_message.strip()
        if fallback not in candidate and candidate not in fallback:
            logger.warning("LangGraph summarize 与单步执行结果无关联，回退到步骤结果")
            return False

        max_allowed_length = len(fallback_message.strip()) * 3 + 120
        if len(candidate) > max_allowed_length:
            logger.warning("LangGraph summarize 产出异常长文本，回退到步骤结果以避免无关回复")
            return False

    return True


def build_step_from_payload(payload: Any, fallback_index: int) -> Step:
    """将模型返回的步骤片段规范化为领域 Step。"""
    if isinstance(payload, dict):
        step_id = str(payload.get("id") or str(uuid.uuid4()))
        description = str(payload.get("description") or f"步骤{fallback_index + 1}")
        return Step(
            id=step_id,
            description=description,
            status=ExecutionStatus.PENDING,
        )

    return Step(
        id=str(uuid.uuid4()),
        description=str(payload).strip() or f"步骤{fallback_index + 1}",
        status=ExecutionStatus.PENDING,
    )


def format_attachments_for_prompt(attachments: List[str]) -> str:
    if not attachments:
        return "无"
    return "\n".join(f"- {item}" for item in attachments)


def should_emit_planner_message(user_message: str, planner_message: str, steps: List[Step]) -> bool:
    """判断是否需要输出规划消息，避免问候/回显噪音。"""
    normalized_planner_message = planner_message.strip()
    if not normalized_planner_message:
        return False

    normalized_user_message = user_message.strip()
    if normalized_user_message and normalized_planner_message == normalized_user_message:
        return False

    if len(steps) == 1:
        first_step_description = str(steps[0].description or "").strip()
        if first_step_description and normalized_planner_message == first_step_description:
            return False

    return True
