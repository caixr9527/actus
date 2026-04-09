#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 纯函数解析与归一化工具。"""

import re
import uuid
from typing import Any, List

from app.domain.models import (
    Step,
    StepArtifactPolicy,
    StepOutputMode,
    StepTaskModeHint,
    ToolEvent,
    ToolEventStatus,
    ExecutionStatus,
    build_step_objective_key,
)
from app.domain.services.runtime.normalizers import normalize_controlled_value, normalize_text_list

_EXPLICIT_FILE_OUTPUT_REQUEST_PATTERN = re.compile(
    r"((保存|写入|导出|输出|生成|创建|落盘|整理).{0,10}(文件|文档|txt|md|markdown|json|csv))"
    r"|((文件|文档|txt|md|markdown|json|csv).{0,10}(保存|写入|导出|输出|生成|创建|落盘))",
    re.IGNORECASE,
)


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


def _user_explicitly_requests_file_output(user_message: str) -> bool:
    """仅识别用户自己明确提出的文件产出要求，用于结构化产物策略兜底。"""
    return bool(_EXPLICIT_FILE_OUTPUT_REQUEST_PATTERN.search(str(user_message or "").strip()))


def _resolve_step_artifact_strategy(
        *,
        task_mode_hint: Any,
        raw_output_mode: Any,
        raw_artifact_policy: Any,
        user_message: str,
) -> tuple[str, str]:
    """按任务模式与用户原始诉求收敛结构化产物策略。"""
    output_mode = normalize_controlled_value(raw_output_mode, StepOutputMode)
    artifact_policy = normalize_controlled_value(raw_artifact_policy, StepArtifactPolicy)
    user_requests_file_output = _user_explicitly_requests_file_output(user_message)

    # 等待步骤和默认检索步骤都不应该再依赖 write_file 产出临时文件。
    if task_mode_hint == "human_wait":
        return "none", "forbid_file_output"
    if task_mode_hint in {"research", "web_reading"} and not user_requests_file_output:
        return "none", "forbid_file_output"

    if artifact_policy == "require_file_output":
        return "file", "require_file_output"
    if output_mode == "file" and artifact_policy == "forbid_file_output":
        artifact_policy = "allow_file_output"
    if output_mode == "file" and not artifact_policy:
        artifact_policy = "allow_file_output"
    if artifact_policy and not output_mode:
        output_mode = "file" if artifact_policy in {"allow_file_output", "require_file_output"} else "none"

    return output_mode or "none", artifact_policy or "default"


def build_step_from_payload(payload: Any, fallback_index: int, *, user_message: str = "") -> Step:
    """将模型返回的步骤片段规范化为领域 Step。"""
    if isinstance(payload, dict):
        step_id = str(payload.get("id") or str(uuid.uuid4()))
        raw_description = str(payload.get("description") or "").strip()
        raw_title = str(payload.get("title") or "").strip()
        description = raw_description or raw_title or f"步骤{fallback_index + 1}"
        title = raw_title or description
        success_criteria = normalize_text_list(payload.get("success_criteria"))
        # 先消费 planner/replan 输出的结构化模式，再回退到执行器里的文本判定规则。
        raw_task_mode_hint = normalize_controlled_value(payload.get("task_mode_hint"), StepTaskModeHint)
        task_mode_hint = raw_task_mode_hint or None
        # 产物策略只做结构化标准化与默认回填，不再做文本删句修正。
        output_mode, artifact_policy = _resolve_step_artifact_strategy(
            task_mode_hint=task_mode_hint,
            raw_output_mode=payload.get("output_mode"),
            raw_artifact_policy=payload.get("artifact_policy"),
            user_message=user_message,
        )
        return Step(
            id=step_id,
            title=title,
            description=description,
            task_mode_hint=task_mode_hint,
            output_mode=output_mode,
            artifact_policy=artifact_policy,
            objective_key=str(payload.get("objective_key") or build_step_objective_key(title, description)),
            success_criteria=success_criteria or [description],
            status=ExecutionStatus.PENDING,
        )

    description = str(payload).strip() or f"步骤{fallback_index + 1}"
    return Step(
        id=str(uuid.uuid4()),
        title=description,
        description=description,
        output_mode="none",
        artifact_policy="default",
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
