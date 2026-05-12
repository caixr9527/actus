#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox capability profile 错误消息脱敏工具。"""

from __future__ import annotations

import re


_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(password|passwd|token|cookie|api[_-]?key|secret)\s*[:=]\s*[^,\s;]+"
)
_URL_PATTERN = re.compile(r"(?i)\bhttps?://[^\s]+")


def sanitize_sandbox_profile_message(message: str) -> str:
    text = str(message or "").replace("\n", " ").strip()
    text = _URL_PATTERN.sub("[masked-url]", text)
    text = _SENSITIVE_ASSIGNMENT_PATTERN.sub(lambda match: f"{match.group(1)}=[masked]", text)
    if len(text) > 160:
        return f"{text[:157]}..."
    return text
