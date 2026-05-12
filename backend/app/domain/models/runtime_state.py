#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 生命周期状态治理领域模型。"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .session import SessionStatus
from .workflow_run import WorkflowRunStatus


class CheckpointRef(BaseModel):
    """LangGraph checkpoint 引用，只作为恢复依据，不作为用户历史。"""

    namespace: str = ""
    checkpoint_id: str | None = None


class RuntimeStateSource(str, Enum):
    """Runtime 状态快照或状态变化的来源。"""

    REQUEST = "request"
    STREAM_EVENT = "stream_event"
    GRAPH_STATE = "graph_state"
    CHECKPOINT = "checkpoint"
    CANCEL = "cancel"
    RECONCILE = "reconcile"


class RuntimeCommand(str, Enum):
    """Runtime 状态机可处理的外部输入或运行事件。"""

    START = "start"
    USER_MESSAGE = "user_message"
    RESUME = "resume"
    CONTINUE_CANCELLED = "continue_cancelled"
    WAIT = "wait"
    COMPLETE = "complete"
    FAIL = "fail"
    CANCEL = "cancel"
    RECONCILE = "reconcile"


class RuntimeEventProjection(BaseModel):
    """事件写入时需要同步维护的会话展示投影。"""

    title: str | None = None
    latest_message: str | None = None
    latest_message_at: datetime | None = None
    increment_unread: bool = False
    stream_event_id: str | None = None


class RuntimeTransition(BaseModel):
    """一次 Runtime 状态转移决策。"""

    command: RuntimeCommand
    from_session_status: SessionStatus
    from_run_status: WorkflowRunStatus | None = None
    to_session_status: SessionStatus
    to_run_status: WorkflowRunStatus | None = None
    reason: str


class RuntimeEventPersistResult(BaseModel):
    """Runtime 事件持久化后的状态收敛结果。"""

    event_inserted: bool
    transition_applied: bool
    from_session_status: SessionStatus
    to_session_status: SessionStatus
    from_run_status: WorkflowRunStatus | None = None
    to_run_status: WorkflowRunStatus | None = None
    ignored_reason: str = ""


class RuntimeStateSnapshot(BaseModel):
    """Runtime 当前状态快照，所有状态校验都应基于该快照完成。"""

    session_id: str
    workspace_id: str | None = None
    run_id: str | None = None
    session_status: SessionStatus
    run_status: WorkflowRunStatus | None = None
    workspace_run_id: str | None = None
    session_run_id: str | None = None
    checkpoint_ref: CheckpointRef | None = None
    has_checkpoint: bool = False
    pending_interrupt: dict[str, Any] = Field(default_factory=dict)
    graph_projection_status: WorkflowRunStatus | None = None
    current_step_id: str | None = None
    last_event_id: str | None = None
    last_event_type: str | None = None
    last_event_at: datetime | None = None
    # 该字段由后续 coordinator 基于最新 cancelled plan 计算，状态机只消费显式条件。
    has_continuable_cancelled_plan: bool = False
    source: RuntimeStateSource = RuntimeStateSource.RECONCILE


class RuntimeReconcileResult(BaseModel):
    """Runtime 对账结果。"""

    snapshot_before: RuntimeStateSnapshot
    snapshot_after: RuntimeStateSnapshot
    actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ResumeGateInspection(BaseModel):
    """恢复入口检查结果。"""

    session_id: str
    run_id: str | None = None
    can_resume: bool
    reason: str = ""
    pending_interrupt: dict[str, Any] = Field(default_factory=dict)
    checkpoint_ref: CheckpointRef | None = None
    snapshot: RuntimeStateSnapshot
