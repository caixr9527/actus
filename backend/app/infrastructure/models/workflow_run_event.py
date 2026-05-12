#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21 10:20
@Author : caixiaorong01@outlook.com
@File   : workflow_run_event.py
"""
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import TypeAdapter
from sqlalchemy import (
    String,
    DateTime,
    Index,
    text,
    PrimaryKeyConstraint,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import WorkflowRunEventRecord, Event
from app.domain.services.runtime.normalizers import normalize_event_payload
from .base import Base


class WorkflowRunEventModel(Base):
    """运行事件流水 ORM 模型"""
    __tablename__ = "workflow_run_events"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_workflow_run_events_id"),
        UniqueConstraint("run_id", "event_id", name="uq_workflow_run_events_run_event_id"),
        Index("ix_workflow_run_events_run_id", "run_id"),
        Index("ix_workflow_run_events_session_id", "session_id"),
        Index("ix_workflow_run_events_user_session_event", "user_id", "session_id", "event_id"),
        Index("ix_workflow_run_events_created_at", "created_at"),
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
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text("''::character varying"),
    )
    event_payload: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, record: WorkflowRunEventRecord) -> "WorkflowRunEventModel":
        return cls(
            id=record.id,
            run_id=record.run_id,
            session_id=record.session_id,
            user_id=record.user_id,
            event_id=record.event_id,
            event_type=record.event_type,
            event_payload=record.event_payload.model_dump(mode="json"),
            created_at=record.created_at,
        )

    def to_domain(self) -> WorkflowRunEventRecord:
        # 历史事件读取边界统一规整 step/plan 载荷，避免旧脏值继续透传到回放与 SSE。
        event = normalize_event_payload(TypeAdapter(Event).validate_python(self.event_payload))
        return WorkflowRunEventRecord(
            id=self.id,
            run_id=self.run_id,
            session_id=self.session_id,
            user_id=self.user_id,
            event_id=self.event_id,
            event_type=self.event_type,
            event_payload=event,
            created_at=self.created_at,
        )
