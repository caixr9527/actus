#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21 10:20
@Author : caixiaorong01@outlook.com
@File   : workflow_run.py
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    String,
    DateTime,
    Text,
    Index,
    text,
    PrimaryKeyConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import WorkflowRun
from .base import Base


class WorkflowRunModel(Base):
    """运行主记录 ORM 模型"""
    __tablename__ = "workflow_runs"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_workflow_runs_id"),
        Index("ix_workflow_runs_session_id", "session_id"),
        Index("ix_workflow_runs_user_id", "user_id"),
        Index("ix_workflow_runs_status", "status"),
    )

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    thread_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text("'pending'::character varying"),
    )
    checkpoint_namespace: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    checkpoint_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    current_step_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    plan_snapshot: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    files_snapshot: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    runtime_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_event_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
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
    def from_domain(cls, run: WorkflowRun) -> "WorkflowRunModel":
        return cls(
            **run.model_dump(
                mode="python",
                exclude={
                    "plan_snapshot",
                    "files_snapshot",
                    "runtime_metadata",
                    "updated_at",
                    "created_at",
                },
            ),
            **run.model_dump(
                mode="json",
                include={"plan_snapshot", "files_snapshot", "runtime_metadata"},
            ),
        )

    def to_domain(self) -> WorkflowRun:
        return WorkflowRun.model_validate(self, from_attributes=True)

    def update_from_domain(self, run: WorkflowRun) -> None:
        base_data = run.model_dump(
            mode="python",
            exclude={
                "plan_snapshot",
                "files_snapshot",
                "runtime_metadata",
                "updated_at",
                "created_at",
            },
        )
        json_data = run.model_dump(
            mode="json",
            include={"plan_snapshot", "files_snapshot", "runtime_metadata"},
        )
        for field, value in {**base_data, **json_data}.items():
            setattr(self, field, value)
