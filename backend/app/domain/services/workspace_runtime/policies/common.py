#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace Runtime 策略公共辅助函数。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from pydantic import BaseModel

from app.domain.services.runtime.normalizers import truncate_text

TOOL_RESULT_MAX_TEXT_CHARS = 2400
TOOL_RESULT_MAX_LIST_ITEMS = 12
TOOL_RESULT_MAX_DICT_ITEMS = 12


def truncate_tool_text(value: Any, *, max_chars: int = TOOL_RESULT_MAX_TEXT_CHARS) -> str:
    return truncate_text(value, max_chars=max_chars)


def build_log_text_preview(value: Any, *, max_chars: int = 120) -> str:
    """构建日志可读预览：压平空白并按长度截断。"""
    normalized = " ".join(str(value or "").split()).strip()
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars]}..."


def compact_tool_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, BaseModel):
        return compact_tool_value(value.model_dump(mode="json"), depth=depth)
    if isinstance(value, str):
        return truncate_tool_text(value)
    if depth >= 2:
        return truncate_tool_text(value, max_chars=400)
    if isinstance(value, list):
        return [
            compact_tool_value(item, depth=depth + 1)
            for item in value[:TOOL_RESULT_MAX_LIST_ITEMS]
        ]
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= TOOL_RESULT_MAX_DICT_ITEMS:
                break
            compacted[str(key)] = compact_tool_value(item, depth=depth + 1)
        return compacted
    return truncate_tool_text(value, max_chars=400)


def hash_payload(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()
