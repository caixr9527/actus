#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangGraph 图级共享解析与归一化纯函数。"""

import json
import logging
import re
from typing import Any, Dict, List

from app.domain.services.runtime.normalizers import normalize_ref_list

logger = logging.getLogger(__name__)


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

    logger.warning("模型结果JSON解析失败，content_length=%s", len(content or ""))
    return {}


def normalize_attachments(raw_attachments: Any) -> List[str]:
    return normalize_ref_list(raw_attachments)


def format_attachments_for_prompt(attachments: List[str]) -> str:
    if not attachments:
        return "无"
    return "\n".join(f"- {item}" for item in attachments)
