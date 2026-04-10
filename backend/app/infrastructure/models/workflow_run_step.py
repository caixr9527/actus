#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21 10:20
@Author : caixiaorong01@outlook.com
@File   : workflow_run_step.py
"""
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    String,
    DateTime,
    Text,
    Integer,
    Index,
    text,
    PrimaryKeyConstraint,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import (
    WorkflowRunStepRecord,
    ExecutionStatus,
    StepArtifactPolicy,
    StepDeliveryContextState,
    StepDeliveryRole,
    StepOutcome,
    StepOutputMode,
    StepTaskModeHint,
)
from app.domain.services.runtime.normalizers import normalize_step_outcome_payload
from .base import Base


class WorkflowRunStepModel(Base):
    """运行步骤快照 ORM 模型"""
    __tablename__ = "workflow_run_steps"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_workflow_run_steps_id"),
        UniqueConstraint("run_id", "step_id", name="uq_workflow_run_steps_run_step_id"),
        Index("ix_workflow_run_steps_run_id", "run_id"),
        Index("ix_workflow_run_steps_step_index", "step_index"),
    )

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    run_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("workflow_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    step_id: Mapped[str] = mapped_column(String(255), nullable=False)
    step_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("''::text"),
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("''::text"),
    )
    objective_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        server_default=text("''::character varying"),
    )
    success_criteria: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text("'pending'::character varying"),
    )
    task_mode_hint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    output_mode: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    artifact_policy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    delivery_role: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    delivery_context_state: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    outcome: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, record: WorkflowRunStepRecord) -> "WorkflowRunStepModel":
        return cls(
            id=record.id,
            run_id=record.run_id,
            step_id=record.step_id,
            step_index=record.step_index,
            title=record.title,
            description=record.description,
            objective_key=record.objective_key,
            success_criteria=list(record.success_criteria or []),
            status=record.status.value,
            task_mode_hint=record.task_mode_hint.value if record.task_mode_hint is not None else None,
            output_mode=record.output_mode.value if record.output_mode is not None else None,
            artifact_policy=record.artifact_policy.value if record.artifact_policy is not None else None,
            delivery_role=record.delivery_role.value if record.delivery_role is not None else None,
            delivery_context_state=(
                record.delivery_context_state.value
                if record.delivery_context_state is not None
                else None
            ),
            # 步骤投影与 runtime 主链共用同一套 outcome 归一化，避免附件语义再次漂移。
            outcome=normalize_step_outcome_payload(record.outcome),
            error=record.error,
            updated_at=record.updated_at,
            created_at=record.created_at,
        )

    def to_domain(self) -> WorkflowRunStepRecord:
        normalized_outcome = normalize_step_outcome_payload(self.outcome)
        return WorkflowRunStepRecord(
            id=self.id,
            run_id=self.run_id,
            step_id=self.step_id,
            step_index=self.step_index,
            title=self.title,
            description=self.description,
            objective_key=self.objective_key,
            success_criteria=list(self.success_criteria or []),
            status=ExecutionStatus(self.status),
            task_mode_hint=StepTaskModeHint(self.task_mode_hint) if self.task_mode_hint else None,
            output_mode=StepOutputMode(self.output_mode) if self.output_mode else None,
            artifact_policy=StepArtifactPolicy(self.artifact_policy) if self.artifact_policy else None,
            delivery_role=StepDeliveryRole(self.delivery_role) if self.delivery_role else None,
            delivery_context_state=(
                StepDeliveryContextState(self.delivery_context_state)
                if self.delivery_context_state
                else None
            ),
            outcome=StepOutcome.model_validate(normalized_outcome) if normalized_outcome is not None else None,
            error=self.error,
            updated_at=self.updated_at,
            created_at=self.created_at,
        )
