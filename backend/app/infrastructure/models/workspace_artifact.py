#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/12 17:31
@Author : caixiaorong01@outlook.com
@File   : workspace_artifact.py
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Index, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import WorkspaceArtifact
from .base import Base


class WorkspaceArtifactModel(Base):
    """工作区产物索引 ORM 模型。"""

    __tablename__ = "workspace_artifacts"
    __table_args__ = (
        Index("ix_workspace_artifacts_workspace_id", "workspace_id"),
        UniqueConstraint(
            "workspace_id",
            "path",
            name="uq_workspace_artifacts_workspace_id_path",
        ),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(2048), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False)
    summary: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
        server_default=text("''::character varying"),
    )
    source_step_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_capability: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    delivery_state: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        server_default=text("''::character varying"),
    )
    artifact_metadata: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
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
    def from_domain(cls, artifact: WorkspaceArtifact) -> "WorkspaceArtifactModel":
        payload = artifact.model_dump(mode="json")
        payload["artifact_metadata"] = payload.pop("metadata", {})
        return cls(**payload)

    def to_domain(self) -> WorkspaceArtifact:
        payload = {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "path": self.path,
            "artifact_type": self.artifact_type,
            "summary": self.summary,
            "source_step_id": self.source_step_id,
            "source_capability": self.source_capability,
            "delivery_state": self.delivery_state,
            "metadata": dict(self.artifact_metadata or {}),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return WorkspaceArtifact.model_validate(payload)

    def update_from_domain(self, artifact: WorkspaceArtifact) -> None:
        payload = artifact.model_dump(mode="json")
        payload["artifact_metadata"] = payload.pop("metadata", {})
        for field, value in payload.items():
            setattr(self, field, value)
