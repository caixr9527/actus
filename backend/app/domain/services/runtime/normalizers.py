#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 共享归一化纯函数。"""

from enum import Enum
from typing import Any, Iterable, List, Optional


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
