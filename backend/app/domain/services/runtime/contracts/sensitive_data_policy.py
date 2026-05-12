#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长期记忆写入前使用的敏感信息纯规则。"""

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class SensitiveDataDetectionResult(BaseModel):
    """敏感信息检测结果，不暴露原始敏感内容。"""

    has_secret: bool = False
    has_pii: bool = False
    categories: list[str] = Field(default_factory=list)
    redacted_text: str = ""


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("token", re.compile(r"(?i)['\"]?\b(?:access|refresh|id)?_?token\b['\"]?\s*[:=]\s*['\"]?[^'\"\s,}]{8,}")),
    ("api_key", re.compile(r"(?i)['\"]?\b(?:api[_-]?key|secret[_-]?key)\b['\"]?\s*[:=]\s*['\"]?[^'\"\s,}]{8,}")),
    ("password", re.compile(r"(?i)['\"]?\bpassword\b['\"]?\s*[:=]\s*['\"]?[^'\"\s,}]{4,}")),
    ("cookie", re.compile(r"(?i)['\"]?\bcookie\b['\"]?\s*[:=]\s*['\"]?[^'\"\n,}]{8,}")),
)

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("phone", re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)")),
    ("email", re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
)


def detect_sensitive_text(text: str) -> SensitiveDataDetectionResult:
    """检测最低要求的 secret 与 PII 类型，并返回脱敏文本。"""
    source_text = str(text or "")
    redacted_text = source_text
    categories: list[str] = []
    has_secret = False
    has_pii = False

    for category, pattern in _SECRET_PATTERNS:
        if pattern.search(source_text):
            has_secret = True
            categories.append(category)
            redacted_text = pattern.sub(f"[REDACTED_{category.upper()}]", redacted_text)

    for category, pattern in _PII_PATTERNS:
        if pattern.search(source_text):
            has_pii = True
            categories.append(category)
            redacted_text = pattern.sub(f"[REDACTED_{category.upper()}]", redacted_text)

    return SensitiveDataDetectionResult(
        has_secret=has_secret,
        has_pii=has_pii,
        categories=categories,
        redacted_text=redacted_text,
    )


def redact_sensitive_text(text: str) -> str:
    """返回脱敏后的文本。"""
    return detect_sensitive_text(text).redacted_text


def assert_memory_content_safe(text: str) -> str:
    """长期记忆禁止写入 secret；PII 必须脱敏后再写入。"""
    result = detect_sensitive_text(text)
    if result.has_secret:
        raise ValueError("长期记忆内容包含不允许保存的敏感凭证")
    return result.redacted_text


def assert_memory_payload_safe(payload: Any) -> Any:
    """递归治理长期记忆正文载荷，确保 secret 拒写、PII 脱敏。"""
    if isinstance(payload, (dict, list)):
        serialized_payload = str(payload)
        try:
            serialized_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            pass
        if detect_sensitive_text(serialized_payload).has_secret:
            raise ValueError("长期记忆内容包含不允许保存的敏感凭证")
    if isinstance(payload, str):
        return assert_memory_content_safe(payload)
    if isinstance(payload, dict):
        normalized_payload: dict[Any, Any] = {}
        for key, value in payload.items():
            safe_key = assert_memory_content_safe(str(key))
            normalized_payload[safe_key] = assert_memory_payload_safe(value)
        return normalized_payload
    if isinstance(payload, list):
        return [assert_memory_payload_safe(item) for item in payload]
    if isinstance(payload, (int, float)):
        normalized_value = str(payload)
        result = detect_sensitive_text(normalized_value)
        if result.has_secret:
            raise ValueError("长期记忆内容包含不允许保存的敏感凭证")
        if result.has_pii:
            return result.redacted_text
    return payload
