#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safety Audit Ledger DB 仓储实现。"""

from datetime import datetime

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.safety_audit import SafetyAuditDecision, SafetyAuditRecord, SafetyAuditRiskLevel
from app.domain.repositories import SafetyAuditRepository
from app.infrastructure.models import SafetyAuditRecordModel


class SafetyAuditLinkageConflictError(ValueError):
    """尝试重复或冲突补齐 audit linkage。"""


class DBSafetyAuditRepository(SafetyAuditRepository):
    """基于数据库的 Safety Audit 仓储。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(1, min(int(limit or 100), 200))

    @staticmethod
    def _require_user_session_scope(*, user_id: str, session_id: str) -> None:
        if not str(user_id or "").strip() or not str(session_id or "").strip():
            raise ValueError("Safety Audit 查询必须提供 user_id 和 session_id")

    @staticmethod
    def _require_text(value: str | None, field_name: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{field_name} 不能为空")
        return normalized

    @staticmethod
    def _to_insert_values(record: SafetyAuditRecord) -> dict:
        model = SafetyAuditRecordModel.from_domain(record)
        return {
            "id": model.id,
            "user_id": model.user_id,
            "session_id": model.session_id,
            "workspace_id": model.workspace_id,
            "run_id": model.run_id,
            "step_id": model.step_id,
            "action_id": model.action_id,
            "tool_call_id": model.tool_call_id,
            "function_name": model.function_name,
            "normalized_function_name": model.normalized_function_name,
            "final_function_name": model.final_function_name,
            "final_normalized_function_name": model.final_normalized_function_name,
            "decision": model.decision,
            "reason_code": model.reason_code,
            "risk_level": model.risk_level,
            "winning_policy": model.winning_policy,
            "tool_call_fingerprint": model.tool_call_fingerprint,
            "capability_id": model.capability_id,
            "tool_family": model.tool_family,
            "decision_event_id": model.decision_event_id,
            "tool_event_source_event_id": model.tool_event_source_event_id,
            "confirmation_event_id": model.confirmation_event_id,
            "source_event_type": model.source_event_type,
            "source_linked_at": model.source_linked_at,
            "rewrite_applied": model.rewrite_applied,
            "rewrite_reason": model.rewrite_reason,
            "confirmation_id": model.confirmation_id,
            "origin": model.origin,
            "trust_level": model.trust_level,
            "privacy_level": model.privacy_level,
            "retention_policy": model.retention_policy,
            "profile_hash": model.profile_hash,
            "requested_args_digest": model.requested_args_digest,
            "final_args_digest": model.final_args_digest,
            "policy_trace": model.policy_trace,
            "rewrite_metadata_digest": model.rewrite_metadata_digest,
            "related_refs": model.related_refs,
            "classification": model.classification,
            "risk_classification_digest": model.risk_classification_digest,
            "created_at": model.created_at,
        }

    async def _get_by_action_scope(self, record: SafetyAuditRecord) -> SafetyAuditRecord | None:
        stmt = select(SafetyAuditRecordModel).where(
            SafetyAuditRecordModel.user_id == record.user_id,
            SafetyAuditRecordModel.session_id == record.session_id,
            SafetyAuditRecordModel.run_id == record.run_id,
            SafetyAuditRecordModel.action_id == record.action_id,
        )
        result = await self.db_session.execute(stmt)
        model = result.scalar_one_or_none()
        return model.to_domain() if model is not None else None

    async def save_once(self, record: SafetyAuditRecord) -> SafetyAuditRecord:
        insert_values = self._to_insert_values(record)
        stmt = (
            insert(SafetyAuditRecordModel.__table__)
            .values(**insert_values)
            .on_conflict_do_nothing(constraint="uq_safety_audit_records_user_session_run_action")
            .returning(SafetyAuditRecordModel.__table__.c.id)
        )
        try:
            result = await self.db_session.execute(stmt)
        except IntegrityError:
            existing = await self._get_by_action_scope(record)
            if existing is not None:
                return existing
            raise

        inserted_id = result.scalar_one_or_none()
        if inserted_id is not None:
            return record

        existing = await self._get_by_action_scope(record)
        if existing is None:
            raise RuntimeError("Safety Audit 幂等写入冲突后未找到已有记录")
        return existing

    async def get_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            audit_id: str,
    ) -> SafetyAuditRecord | None:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        audit_id = self._require_text(audit_id, "audit_id")
        stmt = select(SafetyAuditRecordModel).where(
            SafetyAuditRecordModel.user_id == user_id,
            SafetyAuditRecordModel.session_id == session_id,
            SafetyAuditRecordModel.id == audit_id,
        )
        result = await self.db_session.execute(stmt)
        model = result.scalar_one_or_none()
        return model.to_domain() if model is not None else None

    async def list_by_run(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            limit: int = 100,
    ) -> list[SafetyAuditRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        run_id = self._require_text(run_id, "run_id")
        stmt = (
            select(SafetyAuditRecordModel)
            .where(
                SafetyAuditRecordModel.user_id == user_id,
                SafetyAuditRecordModel.session_id == session_id,
                SafetyAuditRecordModel.run_id == run_id,
            )
            .order_by(SafetyAuditRecordModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def list_by_step(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            step_id: str,
            limit: int = 100,
    ) -> list[SafetyAuditRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        run_id = self._require_text(run_id, "run_id")
        step_id = self._require_text(step_id, "step_id")
        stmt = (
            select(SafetyAuditRecordModel)
            .where(
                SafetyAuditRecordModel.user_id == user_id,
                SafetyAuditRecordModel.session_id == session_id,
                SafetyAuditRecordModel.run_id == run_id,
                SafetyAuditRecordModel.step_id == step_id,
            )
            .order_by(SafetyAuditRecordModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def list_by_decision_and_risk(
            self,
            *,
            user_id: str,
            session_id: str,
            decision: SafetyAuditDecision,
            risk_level: SafetyAuditRiskLevel,
            limit: int = 100,
    ) -> list[SafetyAuditRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        stmt = (
            select(SafetyAuditRecordModel)
            .where(
                SafetyAuditRecordModel.user_id == user_id,
                SafetyAuditRecordModel.session_id == session_id,
                SafetyAuditRecordModel.decision == decision.value,
                SafetyAuditRecordModel.risk_level == risk_level.value,
            )
            .order_by(SafetyAuditRecordModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def list_by_tool_event_source(
            self,
            *,
            user_id: str,
            session_id: str,
            tool_event_source_event_id: str,
    ) -> list[SafetyAuditRecord]:
        return await self._list_by_linkage(
            user_id=user_id,
            session_id=session_id,
            field_name="tool_event_source_event_id",
            value=tool_event_source_event_id,
        )

    async def list_by_decision_event(
            self,
            *,
            user_id: str,
            session_id: str,
            decision_event_id: str,
    ) -> list[SafetyAuditRecord]:
        return await self._list_by_linkage(
            user_id=user_id,
            session_id=session_id,
            field_name="decision_event_id",
            value=decision_event_id,
        )

    async def list_by_confirmation_event(
            self,
            *,
            user_id: str,
            session_id: str,
            confirmation_event_id: str,
    ) -> list[SafetyAuditRecord]:
        return await self._list_by_linkage(
            user_id=user_id,
            session_id=session_id,
            field_name="confirmation_event_id",
            value=confirmation_event_id,
        )

    async def _list_by_linkage(
            self,
            *,
            user_id: str,
            session_id: str,
            field_name: str,
            value: str,
    ) -> list[SafetyAuditRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        value = self._require_text(value, field_name)
        column = getattr(SafetyAuditRecordModel, field_name)
        stmt = (
            select(SafetyAuditRecordModel)
            .where(
                SafetyAuditRecordModel.user_id == user_id,
                SafetyAuditRecordModel.session_id == session_id,
                column == value,
            )
            .order_by(SafetyAuditRecordModel.created_at.desc())
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def attach_linkage(
            self,
            *,
            user_id: str,
            session_id: str,
            audit_id: str,
            decision_event_id: str | None = None,
            tool_event_source_event_id: str | None = None,
            confirmation_event_id: str | None = None,
            source_event_type: str | None = None,
            source_linked_at: datetime | None = None,
    ) -> SafetyAuditRecord:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        audit_id = self._require_text(audit_id, "audit_id")
        existing = await self.get_by_scope(user_id=user_id, session_id=session_id, audit_id=audit_id)
        if existing is None:
            raise ValueError("Safety Audit 记录不存在或不属于当前用户会话")

        updates: dict = {}
        self._prepare_linkage_update(
            updates,
            current_value=existing.decision_event_id,
            field_name="decision_event_id",
            new_value=decision_event_id,
        )
        self._prepare_linkage_update(
            updates,
            current_value=existing.tool_event_source_event_id,
            field_name="tool_event_source_event_id",
            new_value=tool_event_source_event_id,
        )
        self._prepare_linkage_update(
            updates,
            current_value=existing.confirmation_event_id,
            field_name="confirmation_event_id",
            new_value=confirmation_event_id,
        )
        self._prepare_linkage_update(
            updates,
            current_value=existing.source_event_type,
            field_name="source_event_type",
            new_value=source_event_type,
        )
        if source_linked_at is not None:
            if existing.source_linked_at is not None and existing.source_linked_at != source_linked_at:
                raise SafetyAuditLinkageConflictError("source_linked_at 已存在且与本次回填不一致")
            if existing.source_linked_at is None:
                updates["source_linked_at"] = source_linked_at

        if not updates:
            return existing

        stmt = (
            update(SafetyAuditRecordModel)
            .where(
                SafetyAuditRecordModel.user_id == user_id,
                SafetyAuditRecordModel.session_id == session_id,
                SafetyAuditRecordModel.id == audit_id,
                *self._build_linkage_conflict_guards(updates),
            )
            .values(**updates)
        )
        result = await self.db_session.execute(stmt)
        if getattr(result, "rowcount", None) != 1:
            raise SafetyAuditLinkageConflictError("Safety Audit linkage 已被并发回填为其他值")
        updated = await self.get_by_scope(user_id=user_id, session_id=session_id, audit_id=audit_id)
        if updated is None:
            raise RuntimeError("Safety Audit linkage 回填后未找到记录")
        return updated

    @staticmethod
    def _prepare_linkage_update(
            updates: dict,
            *,
            current_value: str | None,
            field_name: str,
            new_value: str | None,
    ) -> None:
        normalized_new_value = str(new_value or "").strip()
        if not normalized_new_value:
            return
        if current_value is not None and current_value != normalized_new_value:
            raise SafetyAuditLinkageConflictError(f"{field_name} 已存在且与本次回填不一致")
        if current_value is None:
            updates[field_name] = normalized_new_value

    @staticmethod
    def _build_linkage_conflict_guards(updates: dict) -> list:
        guards = []
        for field_name, new_value in updates.items():
            column = getattr(SafetyAuditRecordModel, field_name)
            guards.append(or_(column.is_(None), column == new_value))
        return guards
