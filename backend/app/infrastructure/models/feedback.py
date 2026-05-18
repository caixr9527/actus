#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback Ledger ORM 模型。"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.feedback import (
    FeedbackRecord,
    FeedbackRecordResult,
    build_feedback_record_from_result,
    build_feedback_record_result,
)
from app.domain.services.runtime.contracts.data_access_contract import DataOrigin
from .base import Base


class FeedbackRecordModel(Base):
    """feedback_records 表 ORM 映射。"""

    __tablename__ = "feedback_records"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "session_id",
            "feedback_scope_kind",
            "scope_id",
            "dedupe_key",
            name="uq_feedback_records_user_session_scope_dedupe",
        ),
        Index("ix_feedback_user_session_run_created", "user_id", "session_id", "run_id", "created_at"),
        Index("ix_feedback_user_session_run_step", "user_id", "session_id", "run_id", "step_id"),
        Index("ix_feedback_user_session_run_kind_status", "user_id", "session_id", "run_id", "kind", "status"),
        Index("ix_feedback_user_session_run_severity_status", "user_id", "session_id", "run_id", "severity", "status"),
        Index("ix_feedback_user_session_scope_status", "user_id", "session_id", "feedback_scope_kind", "scope_id", "status"),
        Index("ix_feedback_user_session_source_event", "user_id", "session_id", "source_event_id"),
        Index("ix_feedback_user_session_target", "user_id", "session_id", "target_type", "target_id"),
        Index(
            "ix_feedback_user_session_target_revision",
            "user_id",
            "session_id",
            "target_type",
            "target_id",
            "target_revision_id",
        ),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(255), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    feedback_scope_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    step_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(64), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    source_event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[str] = mapped_column(String(255), nullable=False)
    target_revision_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_content_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    feedback_key: Mapped[str] = mapped_column(String(255), nullable=False)
    dedupe_key: Mapped[str] = mapped_column(String(255), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(255), nullable=False)
    resolution_reason_code: Mapped[str | None] = mapped_column(String(255), nullable=True)
    decay_policy: Mapped[str] = mapped_column(String(255), nullable=False)
    ttl_scope: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    origin: Mapped[str] = mapped_column(String(64), nullable=False)
    trust_level: Mapped[str] = mapped_column(String(64), nullable=False)
    privacy_level: Mapped[str] = mapped_column(String(64), nullable=False)
    retention_policy: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_record_refs: Mapped[list[dict[str, str | None]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    source_ref: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    target_ref: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    feedback_summary: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    prompt_safe_summary: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    resolution: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    classification: Mapped[dict[str, Any]] = mapped_column(
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
    )

    @classmethod
    def from_domain(cls, record: FeedbackRecord) -> "FeedbackRecordModel":
        return cls(
            id=record.id,
            user_id=record.user_id,
            session_id=record.session_id,
            workspace_id=record.workspace_id,
            run_id=record.run_id,
            feedback_scope_kind=record.feedback_scope_kind.value,
            scope_id=record.scope_id,
            source_run_id=record.source_run_id,
            target_run_id=record.target_run_id,
            step_id=record.step_id,
            kind=record.kind.value,
            category=record.category.value,
            status=record.status.value,
            severity=record.severity.value,
            source_kind=record.source_kind.value,
            source_event_id=record.source_event_id,
            target_type=record.target_type.value,
            target_id=record.target_id,
            target_revision_id=record.target_revision_id,
            target_content_hash=record.target_content_hash,
            feedback_key=record.feedback_key,
            dedupe_key=record.dedupe_key,
            reason_code=record.reason_code.value,
            resolution_reason_code=(
                record.resolution_reason_code.value if record.resolution_reason_code is not None else None
            ),
            decay_policy=record.decay_policy,
            ttl_scope=record.ttl_scope,
            expires_at=record.expires_at,
            origin=record.origin.value,
            trust_level=record.trust_level.value,
            privacy_level=record.privacy_level.value,
            retention_policy=record.retention_policy.value,
            profile_hash=record.profile_hash,
            source_record_refs=list(record.source_record_refs),
            source_ref=record.source_ref.model_dump(mode="json"),
            target_ref=record.target_ref.model_dump(mode="json"),
            feedback_summary=record.feedback_summary.model_dump(mode="json"),
            prompt_safe_summary=record.prompt_safe_summary.model_dump(mode="json"),
            resolution=record.resolution.model_dump(mode="json"),
            classification=record.classification.model_dump(mode="json"),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def to_domain(self) -> FeedbackRecord:
        record_result = FeedbackRecordResult.model_validate(
            {
                "feedback_id": self.id,
                "scope": {
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "workspace_id": self.workspace_id,
                    "feedback_scope_kind": self.feedback_scope_kind,
                    "scope_id": self.scope_id,
                    "run_id": self.run_id,
                    "source_run_id": self.source_run_id,
                    "target_run_id": self.target_run_id,
                    "current_run_id_at_record_time": self.run_id if self.feedback_scope_kind == "run" else self.run_id,
                },
                "source_ref": dict(self.source_ref or {}),
                "target_ref": dict(self.target_ref or {}),
                "kind": self.kind,
                "category": self.category,
                "status": self.status,
                "severity": self.severity,
                "reason_code": self.reason_code,
                "feedback_summary": dict(self.feedback_summary or {}),
                "prompt_safe_summary": dict(self.prompt_safe_summary or {}),
                "classification": dict(self.classification or {}),
                "resolution": dict(self.resolution or {}),
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        )
        return build_feedback_record_from_result(
            record=record_result,
            user_id=self.user_id,
            session_id=self.session_id,
            workspace_id=self.workspace_id,
            run_id=self.run_id,
            source_run_id=self.source_run_id,
            target_run_id=self.target_run_id,
            step_id=self.step_id,
            source_record_refs=list(self.source_record_refs or []),
            dedupe_key=self.dedupe_key,
            feedback_key=self.feedback_key,
            decay_policy=self.decay_policy,
            ttl_scope=self.ttl_scope,
            expires_at=self.expires_at,
            profile_hash=self.profile_hash,
            origin=DataOrigin(self.origin),
        )

    def to_result(self) -> FeedbackRecordResult:
        return build_feedback_record_result(self.to_domain())
