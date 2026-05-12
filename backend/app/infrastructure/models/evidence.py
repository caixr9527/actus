#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence Ledger ORM 模型。"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, DateTime, Float, Index, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.evidence import EvidenceRecord
from .base import Base


class EvidenceModel(Base):
    """evidence_records 表 ORM 映射。"""

    __tablename__ = "evidence_records"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_evidence_records_idempotency_key"),
        Index("ix_evidence_user_session_created", "user_id", "session_id", "created_at"),
        Index("ix_evidence_user_run_step", "user_id", "run_id", "step_id"),
        Index("ix_evidence_user_scope", "user_id", "session_id", "evidence_scope"),
        Index("ix_evidence_source_event", "user_id", "session_id", "source_event_id"),
        Index("ix_evidence_fact", "user_id", "session_id", "primary_fact_id"),
        Index("ix_evidence_artifact", "user_id", "session_id", "primary_artifact_id"),
        Index("ix_evidence_claim", "user_id", "session_id", "run_id", "claim_key"),
        Index("ix_evidence_action_subject", "user_id", "session_id", "run_id", "action_key", "subject_key"),
        Index("ix_evidence_reusable_by_run", "user_id", "session_id", "run_id", "reusable"),
        Index("ix_evidence_result_refs_hash", "user_id", "session_id", "result_refs_hash"),
        Index("ix_evidence_supersedes", "user_id", "session_id", "supersedes_evidence_id"),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(255), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    step_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    evidence_scope: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_kind: Mapped[str] = mapped_column(String(128), nullable=False)
    action_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    claim_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    claim_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    subject_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    source_step_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    support_level: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(128), nullable=False)
    source_event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    primary_fact_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    primary_artifact_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    profile_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    source_ref: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    subject_ref: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    payload: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    result_refs: Mapped[list[Dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    related_evidence_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    evidence_metadata: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    payload_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    result_refs_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    supersedes_evidence_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''::text"))
    confidence: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))
    reusable: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    reuse_policy: Mapped[str] = mapped_column(String(64), nullable=False)
    staleness_policy: Mapped[str] = mapped_column(String(64), nullable=False)
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
    def from_domain(cls, evidence: EvidenceRecord) -> "EvidenceModel":
        dumped = evidence.model_dump(mode="json")
        return cls(
            id=evidence.id,
            user_id=evidence.user_id,
            session_id=evidence.session_id,
            workspace_id=evidence.workspace_id,
            run_id=evidence.run_id,
            step_id=evidence.step_id,
            evidence_scope=evidence.evidence_scope.value,
            evidence_kind=evidence.evidence_kind.value,
            action_key=evidence.action_key,
            claim_key=evidence.claim_key,
            claim_text=evidence.claim_text,
            subject_key=evidence.subject_key,
            source_step_id=evidence.source_step_id,
            support_level=evidence.support_level.value,
            quality_status=evidence.quality_status.value,
            source_type=evidence.source_ref.source_type.value,
            source_event_id=evidence.source_event_id,
            tool_call_id=evidence.tool_call_id,
            primary_fact_id=evidence.primary_fact_id,
            primary_artifact_id=evidence.primary_artifact_id,
            profile_hash=evidence.profile_hash,
            source_ref=evidence.source_ref.model_dump(mode="json"),
            subject_ref=evidence.subject_ref.model_dump(mode="json"),
            payload=dict(dumped.get("payload") or {}),
            result_refs=list(dumped.get("result_refs") or []),
            related_evidence_ids=list(evidence.related_evidence_ids),
            evidence_metadata={},
            payload_hash=evidence.payload_hash,
            result_refs_hash=evidence.result_refs_hash,
            idempotency_key=evidence.idempotency_key,
            supersedes_evidence_id=evidence.supersedes_evidence_id,
            summary=evidence.summary,
            confidence=evidence.confidence,
            reusable=evidence.reusable,
            reuse_policy=evidence.reuse_policy.value,
            staleness_policy=evidence.staleness_policy.value,
            visibility=evidence.visibility.value,
            origin=evidence.origin.value,
            trust_level=evidence.trust_level.value,
            privacy_level=evidence.privacy_level.value,
            retention_policy=evidence.retention_policy.value,
            created_at=evidence.created_at,
        )

    def to_domain(self) -> EvidenceRecord:
        return EvidenceRecord.model_validate(
            {
                "id": self.id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "workspace_id": self.workspace_id,
                "run_id": self.run_id,
                "step_id": self.step_id,
                "evidence_scope": self.evidence_scope,
                "evidence_kind": self.evidence_kind,
                "action_key": self.action_key,
                "claim_key": self.claim_key,
                "claim_text": self.claim_text,
                "subject_key": self.subject_key,
                "source_step_id": self.source_step_id,
                "support_level": self.support_level,
                "quality_status": self.quality_status,
                "source_ref": dict(self.source_ref or {}),
                "subject_ref": dict(self.subject_ref or {}),
                "summary": self.summary,
                "payload": dict(self.payload or {}),
                "payload_hash": self.payload_hash,
                "idempotency_key": self.idempotency_key,
                "confidence": self.confidence,
                "reusable": self.reusable,
                "reuse_policy": self.reuse_policy,
                "staleness_policy": self.staleness_policy,
                "visibility": self.visibility,
                "origin": self.origin,
                "trust_level": self.trust_level,
                "privacy_level": self.privacy_level,
                "retention_policy": self.retention_policy,
                "result_refs": list(self.result_refs or []),
                "result_refs_hash": self.result_refs_hash,
                "related_evidence_ids": list(self.related_evidence_ids or []),
                "supersedes_evidence_id": self.supersedes_evidence_id,
                "source_event_id": self.source_event_id,
                "tool_call_id": self.tool_call_id,
                "primary_fact_id": self.primary_fact_id,
                "primary_artifact_id": self.primary_artifact_id,
                "profile_hash": self.profile_hash,
                "created_at": self.created_at,
            }
        )
