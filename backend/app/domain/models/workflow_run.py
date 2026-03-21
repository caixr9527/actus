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
from .file import File
from .memory import Memory
from .plan import ExecutionStatus


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
    plan_snapshot: Dict[str, Any] = Field(default_factory=dict)
    files_snapshot: List[File] = Field(default_factory=list)
    memories_snapshot: Dict[str, Memory] = Field(default_factory=dict)
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
    description: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    success: bool = False
    attachments: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)

