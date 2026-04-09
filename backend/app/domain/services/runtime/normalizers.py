#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 共享归一化纯函数。"""

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


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


def _normalize_items(
        items: Iterable[Any],
        *,
        dedupe: bool = True,
        max_items: Optional[int] = None,
) -> List[str]:
    normalized_items: List[str] = []
    seen_items: set[str] = set()
    for item in items:
        normalized_item = normalize_string_value(item)
        if not normalized_item:
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


def normalize_attachment_path_list(
        raw: Any,
        *,
        dedupe: bool = True,
        max_items: Optional[int] = None,
) -> List[str]:
    """标准化附件路径/引用列表。"""
    return normalize_ref_list(raw, dedupe=dedupe, max_items=max_items)


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
    payload["source_refs"] = normalize_ref_list(raw.get("source_refs"))
    return payload


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


def is_attachment_filepath(ref: Any) -> bool:
    """判断引用是否为可直接交付的绝对文件路径。"""
    return normalize_string_value(ref).startswith("/")
