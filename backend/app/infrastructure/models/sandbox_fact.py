#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Fact Ledger ORM 模型。"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.sandbox_fact import SandboxFactRecord
from .base import Base


class SandboxFactModel(Base):
    """sandbox_facts 表 ORM 映射。"""

    __tablename__ = "sandbox_facts"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_sandbox_facts_idempotency_key"),
        Index("ix_sandbox_facts_user_session_created", "user_id", "session_id", "created_at"),
        Index("ix_sandbox_facts_user_run_step", "user_id", "run_id", "step_id"),
        Index("ix_sandbox_facts_user_scope", "user_id", "session_id", "fact_scope"),
        Index("ix_sandbox_facts_user_workspace_kind", "user_id", "workspace_id", "fact_kind"),
        Index("ix_sandbox_facts_source_event", "user_id", "session_id", "source_event_id"),
        Index("ix_sandbox_facts_tool_call", "user_id", "session_id", "tool_call_id"),
        Index("ix_sandbox_facts_profile_hash", "user_id", "workspace_id", "profile_hash"),
        Index("ix_sandbox_facts_supersedes", "user_id", "session_id", "supersedes_fact_id"),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(255), nullable=False)
    fact_scope: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    step_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sandbox_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    fact_kind: Mapped[str] = mapped_column(String(128), nullable=False)
    source_type: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_event_status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text("'missing'::character varying"),
    )
    tool_event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    function_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    subject_type: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_key: Mapped[str] = mapped_column(String(512), nullable=False)
    profile_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    profile_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    source_ref: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    subject_ref: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    profile_ref: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    related_fact_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    supersedes_fact_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    payload_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''::text"))
    payload: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    visibility: Mapped[str] = mapped_column(String(64), nullable=False)
    origin: Mapped[str] = mapped_column(String(64), nullable=False)
    trust_level: Mapped[str] = mapped_column(String(64), nullable=False)
    privacy_level: Mapped[str] = mapped_column(String(64), nullable=False)
    retention_policy: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, fact: SandboxFactRecord) -> "SandboxFactModel":
        dumped = fact.model_dump(mode="json")
        return cls(
            id=fact.id,
            user_id=fact.user_id,
            session_id=fact.session_id,
            workspace_id=fact.workspace_id,
            run_id=fact.run_id,
            step_id=fact.step_id,
            sandbox_id=fact.sandbox_id,
            source_ref=fact.source_ref.model_dump(mode="json"),
            subject_ref=fact.subject_ref.model_dump(mode="json"),
            profile_ref=fact.profile_ref.model_dump(mode="json"),
            related_fact_ids=list(fact.related_fact_ids),
            supersedes_fact_id=fact.supersedes_fact_id,
            summary=fact.summary,
            payload=dict(dumped.get("payload") or {}),
            payload_hash=fact.payload_hash,
            idempotency_key=fact.idempotency_key,
            created_at=fact.created_at,
            source_type=fact.source_ref.source_type.value,
            source_event_id=fact.source_ref.source_event_id,
            source_event_status=fact.source_ref.source_event_status,
            tool_event_id=fact.source_ref.tool_event_id,
            tool_call_id=fact.source_ref.tool_call_id,
            function_name=fact.source_ref.function_name,
            subject_type=fact.subject_ref.subject_type,
            subject_key=fact.subject_ref.subject_key,
            profile_id=fact.profile_ref.profile_id,
            profile_hash=fact.profile_ref.profile_hash,
            fact_scope=fact.fact_scope.value,
            fact_kind=fact.fact_kind.value,
            visibility=fact.visibility.value,
            origin=fact.origin.value,
            trust_level=fact.trust_level.value,
            privacy_level=fact.privacy_level.value,
            retention_policy=fact.retention_policy.value,
        )

    def to_domain(self) -> SandboxFactRecord:
        return SandboxFactRecord.model_validate(
            {
                "id": self.id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "workspace_id": self.workspace_id,
                "fact_scope": self.fact_scope,
                "run_id": self.run_id,
                "step_id": self.step_id,
                "sandbox_id": self.sandbox_id,
                "fact_kind": self.fact_kind,
                "source_ref": dict(self.source_ref or {}),
                "subject_ref": dict(self.subject_ref or {}),
                "profile_ref": dict(self.profile_ref or {}),
                "related_fact_ids": list(self.related_fact_ids or []),
                "supersedes_fact_id": self.supersedes_fact_id,
                "summary": self.summary,
                "payload": dict(self.payload or {}),
                "payload_hash": self.payload_hash,
                "idempotency_key": self.idempotency_key,
                "visibility": self.visibility,
                "origin": self.origin,
                "trust_level": self.trust_level,
                "privacy_level": self.privacy_level,
                "retention_policy": self.retention_policy,
                "created_at": self.created_at,
            }
        )
