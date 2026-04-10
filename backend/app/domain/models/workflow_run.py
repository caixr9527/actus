#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21 10:20
@Author : caixiaorong01@outlook.com
@File   : workflow_run.py
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .event import Event
from .plan import (
    ExecutionStatus,
    StepArtifactPolicy,
    StepDeliveryContextState,
    StepDeliveryRole,
    StepOutcome,
    StepOutputMode,
    StepTaskModeHint,
)


class WorkflowRunStatus(str, Enum):
    """运行状态"""
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowRun(BaseModel):
    """一次运行主记录"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    status: WorkflowRunStatus = WorkflowRunStatus.PENDING
    checkpoint_namespace: Optional[str] = None
    checkpoint_id: Optional[str] = None
    current_step_id: Optional[str] = None
    runtime_metadata: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)


class WorkflowRunEventRecord(BaseModel):
    """运行事件记录"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    session_id: str
    event_id: str
    event_type: str
    event_payload: Event
    created_at: datetime = Field(default_factory=datetime.now)


class WorkflowRunStepRecord(BaseModel):
    """运行步骤快照"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    step_id: str
    step_index: int = 0
    title: str = ""
    description: str = ""
    objective_key: str = ""
    success_criteria: List[str] = Field(default_factory=list)
    # 步骤快照需要保留结构化语义，避免恢复/排障时丢失运行时决策上下文。
    task_mode_hint: Optional[StepTaskModeHint] = None
    output_mode: Optional[StepOutputMode] = None
    artifact_policy: Optional[StepArtifactPolicy] = None
    delivery_role: Optional[StepDeliveryRole] = None
    delivery_context_state: Optional[StepDeliveryContextState] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    outcome: Optional[StepOutcome] = None
    error: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)


class WorkflowRunSummary(BaseModel):
    """单次运行完成后的情节记忆摘要。"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    session_id: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    goal: str = ""
    title: str = ""
    final_answer_summary: str = ""
    final_answer_text: str = ""
    status: WorkflowRunStatus = WorkflowRunStatus.COMPLETED
    completed_steps: int = 0
    total_steps: int = 0
    step_ledger: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    facts_learned: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SessionContextSnapshot(BaseModel):
    """会话级上下文聚合快照。"""

    session_id: str
    user_id: Optional[str] = None
    last_run_id: Optional[str] = None
    summary_text: str = ""
    recent_run_briefs: List[Dict[str, Any]] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    artifact_refs: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
