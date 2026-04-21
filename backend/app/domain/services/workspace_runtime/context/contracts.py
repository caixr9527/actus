#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 上下文工程的结构化模型。"""

from typing import Any, Dict, List, Literal, TypedDict

PromptStage = Literal["planner", "execute", "replan", "summary", "direct_answer"]


class PendingConfirmationPacket(TypedDict, total=False):
    """当前仍阻塞执行的待确认事项。"""

    kind: str
    title: str
    prompt: str
    details: str
    suggest_user_takeover: str
    attachments: List[str]
    options: List[Dict[str, Any]]
    confirm_label: str
    cancel_label: str
    submit_label: str
    response_key: str


class PromptContextPacket(TypedDict, total=False):
    """统一提供给 Prompt 的结构化上下文数据包。"""

    stage: str
    task_mode: str
    user_goal: str
    current_step: Dict[str, Any]
    open_questions: List[str]
    pending_confirmation: PendingConfirmationPacket
    environment_digest: Dict[str, Any]
    observation_digest: Dict[str, Any]
    recent_action_digest: Dict[str, Any]
    working_memory_digest: Dict[str, Any]
    retrieved_memory_digest: List[Dict[str, Any]]
    stable_background: Dict[str, Any]
    prompt_visible_fields: List[str]
    audit_refs: Dict[str, Any]
