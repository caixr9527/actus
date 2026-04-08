#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 纯函数解析与归一化工具。"""

import uuid
from typing import Any, List

from app.domain.models import (
    Step,
    ToolEvent,
    ToolEventStatus,
    ExecutionStatus,
    build_step_objective_key,
)
from app.domain.services.runtime.normalizers import normalize_text_list


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


def build_fallback_plan_title(user_message: str) -> str:
    """规划 JSON 缺失标题时，生成可读回退标题。"""
    normalized = user_message.strip()
    if not normalized:
        return "新会话"

    title = normalized[:24]
    return title if len(normalized) <= 24 else f"{title}..."


def build_step_from_payload(payload: Any, fallback_index: int) -> Step:
    """将模型返回的步骤片段规范化为领域 Step。"""
    if isinstance(payload, dict):
        step_id = str(payload.get("id") or str(uuid.uuid4()))
        raw_description = str(payload.get("description") or "").strip()
        raw_title = str(payload.get("title") or "").strip()
        description = raw_description or raw_title or f"步骤{fallback_index + 1}"
        title = raw_title or description
        success_criteria = normalize_text_list(payload.get("success_criteria"))
        return Step(
            id=step_id,
            title=title,
            description=description,
            objective_key=str(payload.get("objective_key") or build_step_objective_key(title, description)),
            success_criteria=success_criteria or [description],
            status=ExecutionStatus.PENDING,
        )

    description = str(payload).strip() or f"步骤{fallback_index + 1}"
    return Step(
        id=str(uuid.uuid4()),
        title=description,
        description=description,
        objective_key=build_step_objective_key(description, description),
        success_criteria=[description],
        status=ExecutionStatus.PENDING,
    )


def should_emit_planner_message(user_message: str, planner_message: str, steps: List[Step]) -> bool:
    """判断是否需要输出规划消息，避免问候/回显噪音。"""
    planner_message = planner_message.strip()
    if not planner_message:
        return False

    user_message = user_message.strip()
    if user_message and planner_message == user_message:
        return False

    if len(steps) == 1:
        first_step_description = str(steps[0].description or "").strip()
        if first_step_description and planner_message == first_step_description:
            return False

    return True
