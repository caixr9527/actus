#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/19 18:50
@Author : caixiaorong01@outlook.com
@File   : plan.py
"""
import uuid
import hashlib
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


def build_step_objective_source(title: str, description: str) -> str:
    """构建步骤 objective_key 的稳定语义源文本。"""
    normalized_parts: List[str] = []
    for raw_item in [title, description]:
        normalized_item = str(raw_item or "").strip()
        if normalized_item and normalized_item not in normalized_parts:
            normalized_parts.append(normalized_item)
    if not normalized_parts:
        return "empty-step"
    return " | ".join(normalized_parts)


def build_step_objective_key(title: str, description: str) -> str:
    """根据稳定语义源文本生成 objective_key。"""
    source_text = build_step_objective_source(title=title, description=description)
    return hashlib.md5(source_text.encode("utf-8")).hexdigest()[:16]


class ExecutionStatus(str, Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepOutcome(BaseModel):
    """步骤执行结果。"""

    done: bool = False
    summary: str = ""
    produced_artifacts: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    facts_learned: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    next_hint: Optional[str] = None
    reused_from_run_id: Optional[str] = None
    reused_from_step_id: Optional[str] = None


class Step(BaseModel):
    """步骤/子任务。"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    objective_key: str = ""
    success_criteria: List[str] = Field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PENDING
    outcome: Optional[StepOutcome] = None
    error: Optional[str] = None

    @property
    def done(self) -> bool:
        """判断任务是否完成。"""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        ]


class Plan(BaseModel):
    """计划模型，拆分出来的子任务/子步骤。"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    goal: str = ""
    language: str = ""
    steps: List[Step] = Field(default_factory=list)
    message: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    error: Optional[str] = None

    @property
    def done(self) -> bool:
        """判断任务是否完成。"""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        ]

    def get_next_step(self) -> Optional[Step]:
        """获取下一个未完成的步骤。"""
        return next((step for step in self.steps if not step.done), None)
