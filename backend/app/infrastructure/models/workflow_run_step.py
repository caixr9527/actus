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

from app.domain.models import WorkflowRunStepRecord, ExecutionStatus, StepOutcome
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
            outcome=record.outcome.model_dump(mode="json") if record.outcome is not None else None,
            error=record.error,
            updated_at=record.updated_at,
            created_at=record.created_at,
        )

    def to_domain(self) -> WorkflowRunStepRecord:
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
            outcome=StepOutcome.model_validate(self.outcome) if isinstance(self.outcome, dict) else None,
            error=self.error,
            updated_at=self.updated_at,
            created_at=self.created_at,
        )
