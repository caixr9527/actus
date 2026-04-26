#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层确认事实 helper。

本模块只负责 human_wait 恢复后的确认事实规整、合并和执行文本拼接，
不决定节点流转，也不改写 step.description。
"""

import json
from typing import Any, Dict, Optional

from app.domain.models import Step


def _normalize_confirmed_fact_map(raw_facts: Any) -> Dict[str, Any]:
    """规整确认事实字典：只保留非空键。"""
    if not isinstance(raw_facts, dict):
        return {}
    normalized_facts: Dict[str, Any] = {}
    for raw_key, raw_value in raw_facts.items():
        normalized_key = str(raw_key or "").strip()
        if not normalized_key:
            continue
        normalized_facts[normalized_key] = raw_value
    return normalized_facts


def _fact_value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return len(value.strip()) == 0
    return False


def _fact_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value).strip()


def _merge_confirmed_facts(
        *,
        memory_facts: Dict[str, Any],
        new_facts: Dict[str, Any],
) -> Dict[str, Any]:
    """合并确认事实：最新确认信息优先，覆盖旧值。"""
    merged_facts = dict(memory_facts or {})
    merged_facts.update(new_facts or {})
    return _normalize_confirmed_fact_map(merged_facts)


def _build_step_execution_text(step: Step, *, working_memory: Dict[str, Any]) -> str:
    """
    description-only 主干：执行提示词只基于原始 description，
    human_wait 恢复得到的确认事实作为临时上下文追加，不回写 step.description。
    """
    step_description = str(step.description or "").strip()
    if not step_description:
        return ""
    confirmed_facts = _normalize_confirmed_fact_map(working_memory.get("confirmed_facts"))
    fact_lines = [
        f"- {key}: {_fact_value_to_text(value)}"
        for key, value in confirmed_facts.items()
        if not _fact_value_is_missing(value)
    ]
    if len(fact_lines) == 0:
        return step_description
    return (
            f"{step_description}\n\n"
            f"已确认用户输入（仅作当前步骤执行上下文，不改写原步骤描述）:\n"
            + "\n".join(fact_lines)
    ).strip()


def _extract_confirmed_facts_from_resume(
        *,
        waiting_step: Optional[Step],
        payload: Dict[str, Any],
        resume_value: Any,
        resumed_message: str,
) -> Dict[str, Any]:
    """
    从 human_wait 恢复输入中提取结构化确认事实。
    优先使用 payload 的显式字段，避免依赖步骤外部约定做隐式推断。
    """
    _ = waiting_step  # description-only 主干：等待恢复阶段不依赖步骤内部扩展字段。
    confirmed_facts: Dict[str, Any] = {}
    normalized_resumed_message = str(resumed_message or "").strip()
    candidate_value = resume_value if not _fact_value_is_missing(resume_value) else normalized_resumed_message
    if _fact_value_is_missing(candidate_value):
        return confirmed_facts

    payload_response_key = str(payload.get("response_key") or payload.get("slot_key") or "").strip()
    if payload_response_key:
        confirmed_facts[payload_response_key] = candidate_value

    options = payload.get("options")
    if isinstance(options, list):
        matched_option = None
        for option in options:
            if not isinstance(option, dict):
                continue
            if option.get("resume_value") == resume_value:
                matched_option = option
                break
        if isinstance(matched_option, dict):
            option_slot_key = str(
                matched_option.get("slot_key")
                or matched_option.get("response_key")
                or ""
            ).strip()
            if option_slot_key and option_slot_key not in confirmed_facts:
                confirmed_facts[option_slot_key] = candidate_value
            option_slot_updates = matched_option.get("slot_updates")
            if isinstance(option_slot_updates, dict):
                for raw_key, raw_value in option_slot_updates.items():
                    normalized_key = str(raw_key or "").strip()
                    if not normalized_key:
                        continue
                    confirmed_facts[normalized_key] = raw_value

    if not payload_response_key and len(confirmed_facts) == 0:
        confirmed_facts["latest_user_input"] = candidate_value
    return _normalize_confirmed_fact_map(confirmed_facts)
