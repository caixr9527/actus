#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 共享归一化纯函数。"""

from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import BaseModel


def normalize_string_value(
        value: Any,
        *,
        default: str = "",
        lower: bool = False,
) -> str:
    """通用字符串归一化，统一处理空值、裁剪空白与大小写。"""
    normalized_value = str(value or default).strip()
    return normalized_value.lower() if lower else normalized_value


def truncate_text(value: Any, *, max_chars: int) -> str:
    normalized_value = normalize_string_value(value)
    if len(normalized_value) <= max_chars:
        return normalized_value
    return normalized_value[:max_chars]


def normalize_step_result_text(value: Any, *, fallback: str = "") -> str:
    """统一规整步骤结果，避免把 None/空值写成无意义文本。"""
    if value is None:
        return normalize_string_value(fallback)

    normalized_value = normalize_string_value(value)
    if not normalized_value or normalized_value.lower() == "none":
        return normalize_string_value(fallback)
    return normalized_value


def normalize_controlled_value(value: Any, enum_class: type[Enum]) -> str:
    """标准化受控枚举值，仅接受指定枚举中的候选。"""
    if isinstance(value, Enum):
        normalized_value = normalize_string_value(value.value, lower=True)
    else:
        normalized_value = normalize_string_value(value, lower=True)
    allowed_values = {
        normalize_string_value(item.value, lower=True)
        for item in enum_class
    }
    return normalized_value if normalized_value in allowed_values else ""


def _extract_mapping_fields(raw: Any, field_names: Iterable[str]) -> Dict[str, Any]:
    """统一把 dict / Pydantic 模型 / 普通对象拉平成字段字典。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, BaseModel):
        dumped = raw.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped

    extracted: Dict[str, Any] = {}
    for field_name in field_names:
        if hasattr(raw, field_name):
            extracted[field_name] = getattr(raw, field_name)
    return extracted


def _normalize_items(
        items: Iterable[Any],
        *,
        dedupe: bool = True,
        max_items: Optional[int] = None,
        predicate: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    normalized_items: List[str] = []
    seen_items: set[str] = set()
    for item in items:
        normalized_item = normalize_string_value(item)
        if not normalized_item:
            continue
        if predicate is not None and not predicate(normalized_item):
            continue
        if dedupe:
            if normalized_item in seen_items:
                continue
            seen_items.add(normalized_item)
        normalized_items.append(normalized_item)
        if max_items is not None and len(normalized_items) >= max_items:
            break
    return normalized_items


def normalize_text_list(
        raw: Any,
        *,
        dedupe: bool = True,
        max_items: Optional[int] = None,
) -> List[str]:
    if not isinstance(raw, list):
        return []
    return _normalize_items(raw, dedupe=dedupe, max_items=max_items)


def normalize_ref_list(
        raw: Any,
        *,
        dedupe: bool = True,
        max_items: Optional[int] = None,
) -> List[str]:
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    return _normalize_items(items, dedupe=dedupe, max_items=max_items)


def is_attachment_filepath(ref: Any) -> bool:
    """判断引用是否为可直接交付的绝对文件路径。"""
    return normalize_string_value(ref).startswith("/")


def normalize_file_path_list(
        raw: Any,
        *,
        dedupe: bool = True,
        max_items: Optional[int] = None,
) -> List[str]:
    """标准化可直接交付的绝对文件路径列表。"""
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    return _normalize_items(
        items,
        dedupe=dedupe,
        max_items=max_items,
        predicate=is_attachment_filepath,
    )


def truncate_text_list(
        raw: Any,
        *,
        max_items: int,
        max_chars: int,
) -> List[str]:
    """统一规整并截断字符串列表。"""
    return [
        truncate_text(item, max_chars=max_chars)
        for item in normalize_text_list(raw)[:max_items]
    ]


def append_unique_text(items: List[Any], value: Any) -> List[str]:
    normalized_items = normalize_text_list(items)
    normalized_value = normalize_text_list([value], max_items=1)
    if len(normalized_value) == 0:
        return normalized_items
    candidate = normalized_value[0]
    if candidate not in normalized_items:
        normalized_items.append(candidate)
    return normalized_items


def merge_unique_strings(*groups: Any) -> List[str]:
    merged_items: List[str] = []
    for group in groups:
        merged_items.extend(normalize_ref_list(group, dedupe=False))
    return _normalize_items(merged_items)


def normalize_delivery_payload(raw: Any) -> Dict[str, Any]:
    """标准化最终交付载荷，统一收口正文、结构化分段与来源引用。"""
    payload: Dict[str, Any] = {
        "text": "",
        "sections": [],
        "source_refs": [],
    }
    if not isinstance(raw, dict):
        return payload

    payload["text"] = normalize_string_value(raw.get("text"))

    normalized_sections: List[Dict[str, str]] = []
    for item in list(raw.get("sections") or []):
        if not isinstance(item, dict):
            continue
        title = normalize_string_value(item.get("title"))
        content = normalize_string_value(item.get("content"))
        if not title and not content:
            continue
        normalized_sections.append(
            {
                "title": title,
                "content": content,
            }
        )
    payload["sections"] = normalized_sections
    # source_refs 会进入最终交付附件链路，因此这里只保留可直接交付的文件路径。
    payload["source_refs"] = normalize_file_path_list(raw.get("source_refs"))
    return payload


def normalize_step_outcome_payload(raw: Any) -> Optional[Dict[str, Any]]:
    """统一规整步骤结果载荷，收紧附件路径并清理无效文本。"""
    if raw is None:
        return None

    source = _extract_mapping_fields(
        raw,
        (
            "done",
            "summary",
            "delivery_text",
            "produced_artifacts",
            "blockers",
            "facts_learned",
            "open_questions",
            "next_hint",
            "reused_from_run_id",
            "reused_from_step_id",
        ),
    )

    next_hint = normalize_step_result_text(source.get("next_hint"))
    reused_from_run_id = normalize_string_value(source.get("reused_from_run_id"))
    reused_from_step_id = normalize_string_value(source.get("reused_from_step_id"))
    return {
        "done": bool(source.get("done", False)),
        "summary": normalize_step_result_text(source.get("summary")),
        "delivery_text": normalize_string_value(source.get("delivery_text")),
        "produced_artifacts": normalize_file_path_list(source.get("produced_artifacts")),
        "blockers": normalize_text_list(source.get("blockers")),
        "facts_learned": normalize_text_list(source.get("facts_learned")),
        "open_questions": normalize_text_list(source.get("open_questions")),
        "next_hint": next_hint or None,
        "reused_from_run_id": reused_from_run_id or None,
        "reused_from_step_id": reused_from_step_id or None,
    }


def normalize_step_payload(raw: Any) -> Optional[Dict[str, Any]]:
    """统一规整 Step 载荷，避免 plan/事件/恢复链路各自漂移。"""
    if raw is None:
        return None

    from app.domain.models.plan import (
        ExecutionStatus,
        StepArtifactPolicy,
        StepDeliveryContextState,
        StepDeliveryRole,
        StepOutputMode,
        StepTaskModeHint,
    )

    source = _extract_mapping_fields(
        raw,
        (
            "id",
            "title",
            "description",
            "task_mode_hint",
            "output_mode",
            "artifact_policy",
            "delivery_role",
            "delivery_context_state",
            "objective_key",
            "success_criteria",
            "status",
            "outcome",
            "error",
        ),
    )
    if not source:
        return None

    normalized_step: Dict[str, Any] = {
        "title": normalize_string_value(source.get("title")),
        "description": normalize_string_value(source.get("description")),
        "objective_key": normalize_string_value(source.get("objective_key")),
        "success_criteria": normalize_text_list(source.get("success_criteria")),
        "status": (
            normalize_controlled_value(source.get("status"), ExecutionStatus)
            or ExecutionStatus.PENDING.value
        ),
    }
    step_id = normalize_string_value(source.get("id"))
    if step_id:
        normalized_step["id"] = step_id

    error = normalize_string_value(source.get("error"))
    if error:
        normalized_step["error"] = error

    task_mode_hint = normalize_controlled_value(source.get("task_mode_hint"), StepTaskModeHint)
    if task_mode_hint:
        normalized_step["task_mode_hint"] = task_mode_hint
    output_mode = normalize_controlled_value(source.get("output_mode"), StepOutputMode)
    if output_mode:
        normalized_step["output_mode"] = output_mode
    artifact_policy = normalize_controlled_value(source.get("artifact_policy"), StepArtifactPolicy)
    if artifact_policy:
        normalized_step["artifact_policy"] = artifact_policy
    delivery_role = normalize_controlled_value(source.get("delivery_role"), StepDeliveryRole)
    if delivery_role:
        normalized_step["delivery_role"] = delivery_role
    delivery_context_state = normalize_controlled_value(
        source.get("delivery_context_state"),
        StepDeliveryContextState,
    )
    if delivery_context_state:
        normalized_step["delivery_context_state"] = delivery_context_state

    normalized_outcome = normalize_step_outcome_payload(source.get("outcome"))
    if normalized_outcome is not None:
        normalized_step["outcome"] = normalized_outcome

    return normalized_step


def normalize_plan_payload(raw: Any) -> Optional[Dict[str, Any]]:
    """统一规整 Plan 载荷，保证步骤列表与状态字段语义稳定。"""
    if raw is None:
        return None

    from app.domain.models.plan import ExecutionStatus

    source = _extract_mapping_fields(
        raw,
        ("id", "title", "goal", "language", "steps", "message", "status", "error"),
    )
    if not source:
        return None

    normalized_plan: Dict[str, Any] = {
        "title": normalize_string_value(source.get("title")),
        "goal": normalize_string_value(source.get("goal")),
        "language": normalize_string_value(source.get("language")),
        "message": normalize_string_value(source.get("message")),
        "status": (
            normalize_controlled_value(source.get("status"), ExecutionStatus)
            or ExecutionStatus.PENDING.value
        ),
        "steps": [
            normalized_step
            for item in list(source.get("steps") or [])
            if (normalized_step := normalize_step_payload(item)) is not None
        ],
    }
    plan_id = normalize_string_value(source.get("id"))
    if plan_id:
        normalized_plan["id"] = plan_id

    error = normalize_string_value(source.get("error"))
    if error:
        normalized_plan["error"] = error

    return normalized_plan


def normalize_event_payload(raw: Any) -> Any:
    """统一规整事件中的 Step/Plan，收口历史回放与 SSE 的旁路。"""
    from app.domain.models.event import PlanEvent, StepEvent
    from app.domain.models.plan import Plan, Step

    if isinstance(raw, StepEvent):
        normalized_step = normalize_step_payload(raw.step)
        if normalized_step is None:
            return raw
        return raw.model_copy(update={"step": Step.model_validate(normalized_step)})

    if isinstance(raw, PlanEvent):
        normalized_plan = normalize_plan_payload(raw.plan)
        if normalized_plan is None:
            return raw
        return raw.model_copy(update={"plan": Plan.model_validate(normalized_plan)})

    return raw


def normalize_message_window_entry(
        raw_entry: Any,
        *,
        default_role: str,
        max_message_chars: Optional[int] = None,
        max_attachment_paths: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """统一规整消息窗口条目，确保附件路径与基础字段语义稳定。"""
    if not isinstance(raw_entry, dict):
        return None

    normalized_role = normalize_string_value(raw_entry.get("role"), default=default_role) or default_role
    normalized_message = normalize_string_value(raw_entry.get("message"))
    if max_message_chars is not None:
        normalized_message = truncate_text(normalized_message, max_chars=max_message_chars)
    normalized_attachments = normalize_file_path_list(
        raw_entry.get("attachment_paths"),
        max_items=max_attachment_paths,
    )
    input_part_count_raw = raw_entry.get("input_part_count")
    try:
        input_part_count = max(int(input_part_count_raw or 0), 0)
    except Exception:
        input_part_count = 0

    if not normalized_message and len(normalized_attachments) == 0 and input_part_count == 0:
        return None

    normalized_entry: Dict[str, Any] = {
        "role": normalized_role,
        "message": normalized_message,
    }
    if len(normalized_attachments) > 0:
        normalized_entry["attachment_paths"] = normalized_attachments
    if input_part_count > 0:
        normalized_entry["input_part_count"] = input_part_count
    return normalized_entry


def normalize_execution_response(raw: Any) -> Dict[str, Any]:
    """统一规整执行阶段模型输出，拆分轻摘要、重正文与附件等结构化字段。"""
    if not isinstance(raw, dict):
        raw = {}

    summary = normalize_step_result_text(
        raw.get("summary"),
        fallback=normalize_step_result_text(raw.get("result")),
    )
    delivery_text = normalize_string_value(raw.get("delivery_text"))

    return {
        "success": bool(raw.get("success", True)),
        "summary": summary,
        "result": summary,
        "delivery_text": delivery_text,
        "attachments": normalize_file_path_list(raw.get("attachments")),
        "blockers": normalize_text_list(raw.get("blockers")),
        "facts_learned": normalize_text_list(raw.get("facts_learned")),
        "open_questions": normalize_text_list(raw.get("open_questions")),
        "next_hint": normalize_step_result_text(raw.get("next_hint")),
    }


def build_delivery_text(raw: Any, *, fallback: str = "") -> str:
    """从最终交付载荷构造用户可见正文；没有正文时回退到指定文本。"""
    payload = normalize_delivery_payload(raw)
    if payload["text"]:
        return payload["text"]

    section_texts: List[str] = []
    for item in list(payload.get("sections") or []):
        title = normalize_string_value(item.get("title"))
        content = normalize_string_value(item.get("content"))
        if title and content:
            section_texts.append(f"## {title}\n{content}")
            continue
        if content:
            section_texts.append(content)
            continue
        if title:
            section_texts.append(title)
    if section_texts:
        return "\n\n".join(section_texts)

    return normalize_string_value(fallback)
