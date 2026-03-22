#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/22 21:30
@Author : caixiaorong01@outlook.com
@File   : human_task.py
"""
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class HumanTaskStatus(str, Enum):
    """人机任务状态。"""

    WAITING = "waiting"
    RESUMED = "resumed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class HumanTaskResumePoint(BaseModel):
    """恢复点快照。"""

    session_id: str
    run_id: Optional[str] = None
    thread_id: Optional[str] = None
    checkpoint_namespace: Optional[str] = None
    checkpoint_id: Optional[str] = None
    current_step_id: Optional[str] = None


class HumanTaskTimeoutPolicy(BaseModel):
    """超时策略。"""

    timeout_seconds: Optional[int] = None
    timeout_at: Optional[datetime] = None
    on_timeout: Literal["mark_timeout", "none"] = "mark_timeout"


class HumanTaskResumeCommand(BaseModel):
    """前端可直接消费的恢复命令。"""

    type: Literal["chat_message"] = "chat_message"
    session_id: str
    resume_token: str
    hint: str = "请在当前会话继续输入，系统会自动恢复执行。"


class HumanTask(BaseModel):
    """标准化人机任务模型。"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: HumanTaskStatus = HumanTaskStatus.WAITING
    kind: Literal["wait_for_user_input"] = "wait_for_user_input"
    reason: str = "wait_event"
    question: str = ""
    attachments: List[str] = Field(default_factory=list)
    suggest_user_takeover: Literal["none", "browser"] = "none"
    resume_token: Optional[str] = None
    resume_command: Optional[HumanTaskResumeCommand] = None
    resume_point: Optional[HumanTaskResumePoint] = None
    timeout: HumanTaskTimeoutPolicy = Field(default_factory=HumanTaskTimeoutPolicy)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @staticmethod
    def _normalize_attachments(raw: Optional[Union[str, List[str]]]) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            value = raw.strip()
            return [value] if value else []
        normalized: List[str] = []
        for item in raw:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    @classmethod
    def build_wait_for_user_input(
            cls,
            session_id: str,
            question: str,
            reason: str = "ask_user",
            attachments: Optional[Union[str, List[str]]] = None,
            suggest_user_takeover: Optional[str] = None,
            timeout_seconds: Optional[int] = None,
            run_id: Optional[str] = None,
            thread_id: Optional[str] = None,
            checkpoint_namespace: Optional[str] = None,
            checkpoint_id: Optional[str] = None,
            current_step_id: Optional[str] = None,
            resume_token: Optional[str] = None,
    ) -> "HumanTask":
        takeover = suggest_user_takeover if suggest_user_takeover in {"none", "browser"} else "none"
        normalized_token = str(resume_token or uuid.uuid4())
        normalized_timeout_seconds = int(timeout_seconds) if timeout_seconds and timeout_seconds > 0 else None
        timeout_at = (
            datetime.now() + timedelta(seconds=normalized_timeout_seconds)
            if normalized_timeout_seconds is not None
            else None
        )

        return cls(
            reason=reason,
            question=question,
            attachments=cls._normalize_attachments(attachments),
            suggest_user_takeover=takeover,
            resume_token=normalized_token,
            resume_command=HumanTaskResumeCommand(
                session_id=session_id,
                resume_token=normalized_token,
            ),
            resume_point=HumanTaskResumePoint(
                session_id=session_id,
                run_id=run_id,
                thread_id=thread_id,
                checkpoint_namespace=checkpoint_namespace,
                checkpoint_id=checkpoint_id,
                current_step_id=current_step_id,
            ),
            timeout=HumanTaskTimeoutPolicy(
                timeout_seconds=normalized_timeout_seconds,
                timeout_at=timeout_at,
            ),
        )

    def with_resume_point(
            self,
            session_id: str,
            run_id: Optional[str] = None,
            thread_id: Optional[str] = None,
            checkpoint_namespace: Optional[str] = None,
            checkpoint_id: Optional[str] = None,
            current_step_id: Optional[str] = None,
    ) -> "HumanTask":
        updated = self.model_copy(deep=True)
        token = str(updated.resume_token or uuid.uuid4())
        updated.resume_token = token
        updated.resume_command = HumanTaskResumeCommand(
            session_id=session_id,
            resume_token=token,
        )
        updated.resume_point = HumanTaskResumePoint(
            session_id=session_id,
            run_id=run_id,
            thread_id=thread_id,
            checkpoint_namespace=checkpoint_namespace,
            checkpoint_id=checkpoint_id,
            current_step_id=current_step_id,
        )
        updated.updated_at = datetime.now()
        return updated

