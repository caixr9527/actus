#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 最终输出阶段所有权契约。

本模块只表达 P0-2 的领域边界：哪些阶段可以生成最终正文、
最终附件和 final 消息事件。它不依赖 LangGraph、数据库或接口层。
"""

from enum import Enum
from typing import Any, Mapping


class RuntimeOutputStage(str, Enum):
    """Runtime 输出阶段标识，用于校验最终输出字段写入权。"""

    PLANNER = "planner"
    PLANNER_DIRECT_FALLBACK = "planner_direct_fallback"
    REPLAN = "replan"
    EXECUTE = "execute"
    WAIT = "wait"
    REUSE = "reuse"
    SUMMARY = "summary"
    DIRECT_ANSWER = "direct_answer"
    FINALIZE = "finalize"
    MEMORY = "memory"


FINAL_TEXT_FIELDS = frozenset({"final_message", "final_answer_text"})
FINAL_ATTACHMENT_FIELDS = frozenset({"selected_artifacts"})
FINAL_EVENT_STAGES = frozenset({"final"})


_FINAL_TEXT_STAGES = frozenset(
    {
        RuntimeOutputStage.SUMMARY,
        RuntimeOutputStage.DIRECT_ANSWER,
        RuntimeOutputStage.PLANNER_DIRECT_FALLBACK,
    }
)
_FINAL_ATTACHMENT_STAGES = frozenset({RuntimeOutputStage.SUMMARY})
_FINAL_EVENT_STAGES = frozenset(
    {
        RuntimeOutputStage.SUMMARY,
        RuntimeOutputStage.DIRECT_ANSWER,
        RuntimeOutputStage.PLANNER_DIRECT_FALLBACK,
    }
)


def can_write_final_answer_text(
        stage: RuntimeOutputStage,
        *,
        plan_only: bool = False,
) -> bool:
    """判断阶段是否允许生成用户可见最终正文。"""
    if stage == RuntimeOutputStage.PLANNER and plan_only:
        return True
    return stage in _FINAL_TEXT_STAGES


def can_write_final_message(
        stage: RuntimeOutputStage,
        *,
        plan_only: bool = False,
) -> bool:
    """判断阶段是否允许生成当前 run 轻量最终总结。"""
    if stage == RuntimeOutputStage.PLANNER and plan_only:
        return True
    return stage in _FINAL_TEXT_STAGES


def can_write_selected_artifacts(stage: RuntimeOutputStage) -> bool:
    """判断阶段是否允许选择当前 run 的最终交付附件。"""
    return stage in _FINAL_ATTACHMENT_STAGES


def can_emit_final_message_event(
        stage: RuntimeOutputStage,
        *,
        plan_only: bool = False,
) -> bool:
    """判断阶段是否允许发出 MessageEvent(stage="final")。"""
    if stage == RuntimeOutputStage.PLANNER and plan_only:
        return True
    return stage in _FINAL_EVENT_STAGES


def _has_changed(
        *,
        before_state: Mapping[str, Any],
        updates: Mapping[str, Any],
        field_name: str,
) -> bool:
    """判断 update 是否对字段产生了新写入。

    execute/wait/reuse 会同值带回部分最终字段以保持状态键稳定，
    这种保留不是写入所有权；只有与 before_state 不同才视为生成新值。
    """
    if field_name not in updates:
        return False
    before_value = before_state.get(field_name)
    update_value = updates.get(field_name)
    if field_name in FINAL_TEXT_FIELDS:
        before_value = "" if before_value is None else before_value
        update_value = "" if update_value is None else update_value
    if field_name in FINAL_ATTACHMENT_FIELDS:
        before_value = [] if before_value is None else before_value
        update_value = [] if update_value is None else update_value
    return update_value != before_value


def assert_state_update_allowed(
        *,
        stage: RuntimeOutputStage,
        before_state: Mapping[str, Any],
        updates: Mapping[str, Any],
        plan_only: bool = False,
) -> None:
    """校验 state updates 是否遵守最终输出字段所有权。

    抛出 ValueError 表示当前阶段试图生成不属于自己的最终输出字段。
    """
    if _has_changed(
        before_state=before_state,
        updates=updates,
        field_name="final_message",
    ) and not can_write_final_message(stage, plan_only=plan_only):
        raise ValueError(f"{stage.value} cannot write final_message")

    if _has_changed(
        before_state=before_state,
        updates=updates,
        field_name="final_answer_text",
    ) and not can_write_final_answer_text(stage, plan_only=plan_only):
        raise ValueError(f"{stage.value} cannot write final_answer_text")

    if _has_changed(
        before_state=before_state,
        updates=updates,
        field_name="selected_artifacts",
    ) and not can_write_selected_artifacts(stage):
        raise ValueError(f"{stage.value} cannot write selected_artifacts")
