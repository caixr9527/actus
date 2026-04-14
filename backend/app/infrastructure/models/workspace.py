#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/12 17:31
@Author : caixiaorong01@outlook.com
@File   : workspace.py
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import Workspace
from .base import Base


class WorkspaceModel(Base):
    """工作区 ORM 模型。"""

    __tablename__ = "workspaces"
    __table_args__ = (
        Index("ix_workspaces_session_id", "session_id"),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    current_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sandbox_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    shell_session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    cwd: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        server_default=text("''::character varying"),
    )
    browser_snapshot: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    environment_summary: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text("'active'::character varying"),
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
        onupdate=datetime.now,
    )

    @classmethod
    def from_domain(cls, workspace: Workspace) -> "WorkspaceModel":
        return cls(**workspace.model_dump(mode="python"))

    def to_domain(self) -> Workspace:
        return Workspace.model_validate(self, from_attributes=True)

    def update_from_domain(self, workspace: Workspace) -> None:
        for field, value in workspace.model_dump(mode="python").items():
            setattr(self, field, value)
