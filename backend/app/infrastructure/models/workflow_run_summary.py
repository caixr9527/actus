#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行摘要 ORM 模型。"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, PrimaryKeyConstraint, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import WorkflowRunStatus, WorkflowRunSummary
from .base import Base


class WorkflowRunSummaryModel(Base):
    """单次运行摘要 ORM 模型。"""

    __tablename__ = "workflow_run_summaries"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_workflow_run_summaries_id"),
        UniqueConstraint("run_id", name="uq_workflow_run_summaries_run_id"),
        Index("ix_workflow_run_summaries_session_id", "session_id"),
        Index("ix_workflow_run_summaries_user_id", "user_id"),
        Index("ix_workflow_run_summaries_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String(255), ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    thread_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    goal: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''::text"))
    title: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''::text"))
    final_answer_summary: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''::text"))
    final_answer_text: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''::text"))
    status: Mapped[str] = mapped_column(String(64), nullable=False, server_default=text("'completed'::character varying"))
    completed_steps: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    total_steps: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    step_ledger: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    artifacts: Mapped[List[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    open_questions: Mapped[List[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    blockers: Mapped[List[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    facts_learned: Mapped[List[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP(0)"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, summary: WorkflowRunSummary) -> "WorkflowRunSummaryModel":
        return cls(
            **summary.model_dump(
                mode="python",
                exclude={"step_ledger", "artifacts", "open_questions", "blockers", "facts_learned"},
            ),
            **summary.model_dump(
                mode="json",
                include={"step_ledger", "artifacts", "open_questions", "blockers", "facts_learned"},
            ),
            status=summary.status.value,
        )

    def to_domain(self) -> WorkflowRunSummary:
        return WorkflowRunSummary(
            id=self.id,
            run_id=self.run_id,
            session_id=self.session_id,
            user_id=self.user_id,
            thread_id=self.thread_id,
            goal=self.goal,
            title=self.title,
            final_answer_summary=self.final_answer_summary,
            final_answer_text=self.final_answer_text,
            status=WorkflowRunStatus(self.status),
            completed_steps=self.completed_steps,
            total_steps=self.total_steps,
            step_ledger=list(self.step_ledger or []),
            artifacts=list(self.artifacts or []),
            open_questions=list(self.open_questions or []),
            blockers=list(self.blockers or []),
            facts_learned=list(self.facts_learned or []),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def update_from_domain(self, summary: WorkflowRunSummary) -> None:
        self.session_id = summary.session_id
        self.user_id = summary.user_id
        self.thread_id = summary.thread_id
        self.goal = summary.goal
        self.title = summary.title
        self.final_answer_summary = summary.final_answer_summary
        self.final_answer_text = summary.final_answer_text
        self.status = summary.status.value
        self.completed_steps = summary.completed_steps
        self.total_steps = summary.total_steps
        self.step_ledger = summary.model_dump(mode="json", include={"step_ledger"})["step_ledger"]
        self.artifacts = list(summary.artifacts or [])
        self.open_questions = list(summary.open_questions or [])
        self.blockers = list(summary.blockers or [])
        self.facts_learned = list(summary.facts_learned or [])
