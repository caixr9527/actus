#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究链路 URL 提取共享函数。"""

import re

from app.domain.models import Step
from app.domain.services.runtime.normalizers import normalize_url_value
from app.domain.services.workspace_runtime.policies import (
    build_step_candidate_text as _build_step_candidate_text,
)
from .execution_context import ExecutionContext

_URL_PATTERN = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)


def extract_explicit_url_from_research_context(*, step: Step, ctx: ExecutionContext) -> str:
    """从步骤文本与用户输入上下文提取显式 URL（优先 step 文本）。"""
    step_text = _build_step_candidate_text(step)
    step_url_match = _URL_PATTERN.search(str(step_text or ""))
    if step_url_match:
        return normalize_url_value(step_url_match.group(0))

    content_text = "\n".join(
        [
            str((item or {}).get("text") or "").strip()
            for item in list(getattr(ctx, "normalized_user_content", []) or [])
            if isinstance(item, dict) and str((item or {}).get("type") or "").strip().lower() == "text"
        ]
    )
    content_url_match = _URL_PATTERN.search(content_text)
    if content_url_match:
        return normalize_url_value(content_url_match.group(0))
    return ""
