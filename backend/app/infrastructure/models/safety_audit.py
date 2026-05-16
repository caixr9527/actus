#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safety Audit Ledger ORM 模型。"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, DateTime, Index, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.safety_audit import SafetyAuditRecord
from .base import Base


class SafetyAuditRecordModel(Base):
    """safety_audit_records 表 ORM 映射。"""

    __tablename__ = "safety_audit_records"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "session_id",
            "run_id",
            "action_id",
            name="uq_safety_audit_records_user_session_run_action",
        ),
        Index("ix_safety_audit_user_session_run_created", "user_id", "session_id", "run_id", "created_at"),
        Index("ix_safety_audit_user_session_run_step", "user_id", "session_id", "run_id", "step_id"),
        Index("ix_safety_audit_user_session_decision_risk", "user_id", "session_id", "decision", "risk_level"),
        Index("ix_safety_audit_tool_event_source", "user_id", "session_id", "tool_event_source_event_id"),
        Index("ix_safety_audit_decision_event", "user_id", "session_id", "decision_event_id"),
        Index("ix_safety_audit_confirmation_event", "user_id", "session_id", "confirmation_event_id"),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(255), nullable=False)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    step_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    action_id: Mapped[str] = mapped_column(String(255), nullable=False)
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    function_name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_function_name: Mapped[str] = mapped_column(String(255), nullable=False)
    final_function_name: Mapped[str] = mapped_column(String(255), nullable=False)
    final_normalized_function_name: Mapped[str] = mapped_column(String(255), nullable=False)
    decision: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(255), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(64), nullable=False)
    winning_policy: Mapped[str] = mapped_column(String(255), nullable=False)
    tool_call_fingerprint: Mapped[str] = mapped_column(String(255), nullable=False)
    capability_id: Mapped[str] = mapped_column(String(255), nullable=False)
    tool_family: Mapped[str] = mapped_column(String(128), nullable=False)
    decision_event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tool_event_source_event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    confirmation_event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_event_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    source_linked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    rewrite_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    rewrite_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    confirmation_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    origin: Mapped[str] = mapped_column(String(64), nullable=False)
    trust_level: Mapped[str] = mapped_column(String(64), nullable=False)
    privacy_level: Mapped[str] = mapped_column(String(64), nullable=False)
    retention_policy: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    requested_args_digest: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    final_args_digest: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    policy_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    rewrite_metadata_digest: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    related_refs: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    classification: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    risk_classification_digest: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, record: SafetyAuditRecord) -> "SafetyAuditRecordModel":
        return cls(
            id=record.id,
            user_id=record.user_id,
            session_id=record.session_id,
            workspace_id=record.workspace_id,
            run_id=record.run_id,
            step_id=record.step_id,
            action_id=record.action_id,
            tool_call_id=record.tool_call_id,
            function_name=record.function_name,
            normalized_function_name=record.normalized_function_name,
            final_function_name=record.final_function_name,
            final_normalized_function_name=record.final_normalized_function_name,
            decision=record.decision.value,
            reason_code=record.reason_code,
            risk_level=record.risk_level.value,
            winning_policy=record.winning_policy,
            tool_call_fingerprint=record.tool_call_fingerprint,
            capability_id=record.capability_id,
            tool_family=record.tool_family,
            decision_event_id=record.decision_event_id,
            tool_event_source_event_id=record.tool_event_source_event_id,
            confirmation_event_id=record.confirmation_event_id,
            source_event_type=record.source_event_type,
            source_linked_at=record.source_linked_at,
            rewrite_applied=record.rewrite_applied,
            rewrite_reason=record.rewrite_reason,
            confirmation_id=record.confirmation_id,
            origin=record.origin.value,
            trust_level=record.trust_level.value,
            privacy_level=record.privacy_level.value,
            retention_policy=record.retention_policy.value,
            profile_hash=record.profile_hash,
            requested_args_digest=record.requested_args_digest.model_dump(mode="json"),
            final_args_digest=record.final_args_digest.model_dump(mode="json"),
            policy_trace=[entry.model_dump(mode="json") for entry in record.policy_trace],
            rewrite_metadata_digest=record.rewrite_metadata_digest.model_dump(mode="json"),
            related_refs={
                "fact_ids": list(record.related_fact_ids),
                "evidence_ids": list(record.related_evidence_ids),
                "artifact_revisions": [
                    ref.model_dump(mode="json") for ref in record.related_artifact_revisions
                ],
            },
            classification=record.classification.model_dump(mode="json") if record.classification else {},
            risk_classification_digest=(
                record.risk_classification_digest.model_dump(mode="json")
                if record.risk_classification_digest
                else {}
            ),
            created_at=record.created_at,
        )

    def to_domain(self) -> SafetyAuditRecord:
        related_refs = dict(self.related_refs or {})
        return SafetyAuditRecord.model_validate(
            {
                "id": self.id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "workspace_id": self.workspace_id,
                "run_id": self.run_id,
                "step_id": self.step_id,
                "action_id": self.action_id,
                "tool_call_id": self.tool_call_id,
                "capability_id": self.capability_id,
                "tool_family": self.tool_family,
                "function_name": self.function_name,
                "normalized_function_name": self.normalized_function_name,
                "requested_args_digest": dict(self.requested_args_digest or {}),
                "final_function_name": self.final_function_name,
                "final_normalized_function_name": self.final_normalized_function_name,
                "final_args_digest": dict(self.final_args_digest or {}),
                "decision": self.decision,
                "reason_code": self.reason_code,
                "risk_level": self.risk_level,
                "policy_trace": list(self.policy_trace or []),
                "winning_policy": self.winning_policy,
                "tool_call_fingerprint": self.tool_call_fingerprint,
                "rewrite_applied": self.rewrite_applied,
                "rewrite_reason": self.rewrite_reason,
                "rewrite_metadata_digest": dict(self.rewrite_metadata_digest or {}),
                "confirmation_id": self.confirmation_id,
                "decision_event_id": self.decision_event_id,
                "tool_event_source_event_id": self.tool_event_source_event_id,
                "confirmation_event_id": self.confirmation_event_id,
                "source_event_type": self.source_event_type,
                "source_linked_at": self.source_linked_at,
                "related_fact_ids": list(related_refs.get("fact_ids") or []),
                "related_evidence_ids": list(related_refs.get("evidence_ids") or []),
                "related_artifact_revisions": list(related_refs.get("artifact_revisions") or []),
                "profile_hash": self.profile_hash,
                "origin": self.origin,
                "trust_level": self.trust_level,
                "privacy_level": self.privacy_level,
                "retention_policy": self.retention_policy,
                "classification": dict(self.classification or {}) or None,
                "risk_classification_digest": dict(self.risk_classification_digest or {}) or None,
                "created_at": self.created_at,
            }
        )
