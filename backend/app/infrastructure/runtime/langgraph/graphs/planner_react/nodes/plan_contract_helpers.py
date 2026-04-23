#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层计划契约后处理 helper。

本模块只负责 planner / replan 节点在拿到步骤契约校验结果后的统一后处理，
不承载领域规则本身。

当前职责：
- 剔除 `summary_only_step_forbidden` 对应的纯整理步骤；
- 保留其余可执行步骤与剩余硬错误，避免单个违规步骤清空整批。
"""

from typing import Any, List, Tuple

from app.domain.models import Step


def filter_summary_only_step_issues(
        steps: List[Step],
        issues: List[Any],
) -> Tuple[List[Step], List[Any], int]:
    """剔除纯整理步骤对应的问题，保留其余可执行步骤。"""
    removable_step_ids = {
        str(item.step_id or "").strip()
        for item in list(issues or [])
        if str(getattr(item, "issue_code", "") or "").strip() == "summary_only_step_forbidden"
        and str(getattr(item, "step_id", "") or "").strip()
    }
    if len(removable_step_ids) == 0:
        return list(steps or []), list(issues or []), 0
    filtered_steps = [
        step for step in list(steps or [])
        if str(getattr(step, "id", "") or "").strip() not in removable_step_ids
    ]
    remaining_issues = [
        item for item in list(issues or [])
        if str(getattr(item, "issue_code", "") or "").strip() != "summary_only_step_forbidden"
    ]
    return filtered_steps, remaining_issues, len(removable_step_ids)
