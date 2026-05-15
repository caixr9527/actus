#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace artifact revision ORM 模型。"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Index, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import WorkspaceArtifactRevision
from .base import Base


class WorkspaceArtifactRevisionModel(Base):
    """Artifact revision 历史与版本锁 ORM 模型。"""

    __tablename__ = "workspace_artifact_revisions"
    __table_args__ = (
        UniqueConstraint("artifact_id", "revision_no", name="uq_workspace_artifact_revisions_artifact_revision_no"),
        Index("ix_workspace_artifact_revisions_user_workspace_revision", "user_id", "workspace_id", "revision_id"),
        Index("ix_workspace_artifact_revisions_user_workspace_artifact_hash", "user_id", "workspace_id", "artifact_id", "content_hash"),
        Index("ix_workspace_artifact_revisions_user_workspace_artifact", "user_id", "workspace_id", "artifact_id"),
        Index("ix_workspace_artifact_revisions_source_event", "user_id", "workspace_id", "source_event_id"),
        Index(
            "uq_war_tool_idem",
            "user_id",
            "workspace_id",
            "source_event_id",
            "tool_call_id",
            "source_kind",
            "content_hash",
            unique=True,
            postgresql_where=text(
                "source_kind IN ('tool_write_file', 'tool_replace_file') "
                "AND source_event_id IS NOT NULL "
                "AND tool_call_id IS NOT NULL "
                "AND content_hash IS NOT NULL"
            ),
        ),
        Index(
            "uq_war_final_snapshot_idem",
            "user_id",
            "session_id",
            "source_run_id",
            "source_message_event_id",
            "source_final_answer_hash",
            unique=True,
            postgresql_where=text(
                "source_kind = 'final_answer_snapshot' "
                "AND source_run_id IS NOT NULL "
                "AND source_message_event_id IS NOT NULL "
                "AND source_final_answer_hash IS NOT NULL"
            ),
        ),
        Index(
            "uq_war_derived_export_idem",
            "user_id",
            "workspace_id",
            "source_revision_id",
            "source_kind",
            "content_hash",
            unique=True,
            postgresql_where=text(
                "source_kind = 'derived_export' "
                "AND source_revision_id IS NOT NULL "
                "AND content_hash IS NOT NULL"
            ),
        ),
    )

    revision_id: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False)
    artifact_id: Mapped[str] = mapped_column(String(255), nullable=False)
    revision_no: Mapped[int] = mapped_column(nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(255), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    step_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    path: Mapped[str] = mapped_column(String(2048), nullable=False)
    storage_ref: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    hash_algorithm: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'sha256'::character varying"))
    size_bytes: Mapped[Optional[int]] = mapped_column(nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False)
    delivery_state: Mapped[str] = mapped_column(String(64), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_message_event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_revision_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_fact_ids: Mapped[List[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    source_evidence_ids: Mapped[List[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    source_final_answer_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    derived_content_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    function_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    profile_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    profile_status: Mapped[str] = mapped_column(String(64), nullable=False, server_default=text("'missing'::character varying"))
    origin: Mapped[str] = mapped_column(String(64), nullable=False)
    trust_level: Mapped[str] = mapped_column(String(64), nullable=False)
    privacy_level: Mapped[str] = mapped_column(String(64), nullable=False)
    retention_policy: Mapped[str] = mapped_column(String(64), nullable=False)
    revision_metadata: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP(0)"))

    @classmethod
    def from_domain(cls, revision: WorkspaceArtifactRevision) -> "WorkspaceArtifactRevisionModel":
        payload = revision.model_dump(mode="json")
        payload["revision_metadata"] = payload.pop("metadata", {})
        payload["created_at"] = revision.created_at
        return cls(**payload)

    def to_domain(self) -> WorkspaceArtifactRevision:
        payload = {
            "revision_id": self.revision_id,
            "artifact_id": self.artifact_id,
            "revision_no": self.revision_no,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "path": self.path,
            "storage_ref": dict(self.storage_ref or {}),
            "content_hash": self.content_hash,
            "storage_hash": self.storage_hash,
            "hash_algorithm": self.hash_algorithm,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
            "artifact_type": self.artifact_type,
            "delivery_state": self.delivery_state,
            "source_kind": self.source_kind,
            "source_event_id": self.source_event_id,
            "source_run_id": self.source_run_id,
            "source_message_event_id": self.source_message_event_id,
            "source_revision_id": self.source_revision_id,
            "source_fact_ids": list(self.source_fact_ids or []),
            "source_evidence_ids": list(self.source_evidence_ids or []),
            "source_final_answer_hash": self.source_final_answer_hash,
            "derived_content_hash": self.derived_content_hash,
            "tool_call_id": self.tool_call_id,
            "function_name": self.function_name,
            "profile_hash": self.profile_hash,
            "profile_status": self.profile_status,
            "origin": self.origin,
            "trust_level": self.trust_level,
            "privacy_level": self.privacy_level,
            "retention_policy": self.retention_policy,
            "metadata": dict(self.revision_metadata or {}),
            "created_at": self.created_at,
        }
        return WorkspaceArtifactRevision.model_validate(payload)
