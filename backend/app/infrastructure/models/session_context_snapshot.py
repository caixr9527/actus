#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""会话上下文快照 ORM 模型。"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Index, PrimaryKeyConstraint, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import SessionContextSnapshot
from .base import Base


class SessionContextSnapshotModel(Base):
    """会话上下文快照 ORM 模型。"""

    __tablename__ = "session_context_snapshots"
    __table_args__ = (
        PrimaryKeyConstraint("session_id", name="pk_session_context_snapshots_session_id"),
        Index("ix_session_context_snapshots_user_id", "user_id"),
        Index("ix_session_context_snapshots_updated_at", "updated_at"),
    )

    session_id: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''::text"))
    recent_run_briefs: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    open_questions: Mapped[List[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    artifact_refs: Mapped[List[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP(0)"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, snapshot: SessionContextSnapshot) -> "SessionContextSnapshotModel":
        return cls(
            **snapshot.model_dump(
                mode="python",
                exclude={"recent_run_briefs", "open_questions", "artifact_refs"},
            ),
            **snapshot.model_dump(
                mode="json",
                include={"recent_run_briefs", "open_questions", "artifact_refs"},
            ),
        )

    def to_domain(self) -> SessionContextSnapshot:
        return SessionContextSnapshot(
            session_id=self.session_id,
            user_id=self.user_id,
            last_run_id=self.last_run_id,
            summary_text=self.summary_text,
            recent_run_briefs=list(self.recent_run_briefs or []),
            open_questions=list(self.open_questions or []),
            artifact_refs=list(self.artifact_refs or []),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def update_from_domain(self, snapshot: SessionContextSnapshot) -> None:
        self.user_id = snapshot.user_id
        self.last_run_id = snapshot.last_run_id
        self.summary_text = snapshot.summary_text
        self.recent_run_briefs = snapshot.model_dump(mode="json", include={"recent_run_briefs"})["recent_run_briefs"]
        self.open_questions = list(snapshot.open_questions or [])
        self.artifact_refs = list(snapshot.artifact_refs or [])
