#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence Ledger DB 仓储实现。"""

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.evidence import EvidenceRecord, EvidenceScope
from app.domain.repositories import EvidenceRepository
from app.infrastructure.models import EvidenceModel


class DBEvidenceSupersededTargetError(ValueError):
    """CORRECTION/SUPERSEDED evidence 指向了不可写入的原 evidence。"""


class DBEvidenceRepository(EvidenceRepository):
    """基于数据库的 Evidence Ledger 仓储。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(1, min(int(limit or 100), 200))

    @staticmethod
    def _require_user_session_scope(*, user_id: str, session_id: str) -> None:
        if not str(user_id or "").strip() or not str(session_id or "").strip():
            raise ValueError("evidence 查询必须提供 user_id 和 session_id")

    @staticmethod
    def _to_insert_values(evidence: EvidenceRecord) -> dict:
        model = EvidenceModel.from_domain(evidence)
        return {
            "id": model.id,
            "user_id": model.user_id,
            "session_id": model.session_id,
            "workspace_id": model.workspace_id,
            "run_id": model.run_id,
            "step_id": model.step_id,
            "evidence_scope": model.evidence_scope,
            "evidence_kind": model.evidence_kind,
            "action_key": model.action_key,
            "claim_key": model.claim_key,
            "claim_text": model.claim_text,
            "subject_key": model.subject_key,
            "source_step_id": model.source_step_id,
            "support_level": model.support_level,
            "quality_status": model.quality_status,
            "source_type": model.source_type,
            "source_event_id": model.source_event_id,
            "tool_call_id": model.tool_call_id,
            "primary_fact_id": model.primary_fact_id,
            "primary_artifact_id": model.primary_artifact_id,
            "profile_hash": model.profile_hash,
            "source_ref": model.source_ref,
            "subject_ref": model.subject_ref,
            "payload": model.payload,
            "result_refs": model.result_refs,
            "related_evidence_ids": model.related_evidence_ids,
            "metadata": model.evidence_metadata,
            "payload_hash": model.payload_hash,
            "result_refs_hash": model.result_refs_hash,
            "idempotency_key": model.idempotency_key,
            "supersedes_evidence_id": model.supersedes_evidence_id,
            "summary": model.summary,
            "confidence": model.confidence,
            "reusable": model.reusable,
            "reuse_policy": model.reuse_policy,
            "staleness_policy": model.staleness_policy,
            "visibility": model.visibility,
            "origin": model.origin,
            "trust_level": model.trust_level,
            "privacy_level": model.privacy_level,
            "retention_policy": model.retention_policy,
            "created_at": model.created_at,
        }

    async def _get_by_idempotency_key(
            self,
            *,
            user_id: str,
            session_id: str,
            idempotency_key: str,
    ) -> EvidenceRecord | None:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        stmt = select(EvidenceModel).where(
            EvidenceModel.user_id == user_id,
            EvidenceModel.session_id == session_id,
            EvidenceModel.idempotency_key == idempotency_key,
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def _assert_superseded_target(self, evidence: EvidenceRecord) -> None:
        if evidence.supersedes_evidence_id is None:
            return
        stmt = select(EvidenceModel.id).where(
            EvidenceModel.user_id == evidence.user_id,
            EvidenceModel.session_id == evidence.session_id,
            EvidenceModel.id == evidence.supersedes_evidence_id,
        )
        result = await self.db_session.execute(stmt)
        if result.scalar_one_or_none() is None:
            raise DBEvidenceSupersededTargetError("修正/废弃 evidence 指向的原 evidence 不存在或不属于同一用户会话")

    async def save_once(self, evidence: EvidenceRecord) -> EvidenceRecord:
        await self._assert_superseded_target(evidence)
        insert_values = self._to_insert_values(evidence)
        stmt = (
            insert(EvidenceModel.__table__)
            .values(**insert_values)
            .on_conflict_do_nothing(constraint="uq_evidence_records_idempotency_key")
            .returning(EvidenceModel.__table__.c.id)
        )
        try:
            result = await self.db_session.execute(stmt)
        except IntegrityError:
            existing = await self._get_by_idempotency_key(
                user_id=evidence.user_id,
                session_id=evidence.session_id,
                idempotency_key=evidence.idempotency_key,
            )
            if existing is not None:
                return existing
            raise

        inserted_id = result.scalar_one_or_none()
        if inserted_id is not None:
            return evidence

        existing = await self._get_by_idempotency_key(
            user_id=evidence.user_id,
            session_id=evidence.session_id,
            idempotency_key=evidence.idempotency_key,
        )
        if existing is None:
            raise RuntimeError("evidence 幂等写入冲突后未找到已有记录")
        return existing

    async def list_by_ids(
            self,
            *,
            user_id: str,
            session_id: str,
            evidence_ids: list[str],
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        normalized_ids = _normalize_text_list(evidence_ids)
        if not normalized_ids:
            return []
        stmt = (
            select(EvidenceModel)
            .where(
                EvidenceModel.user_id == user_id,
                EvidenceModel.session_id == session_id,
                EvidenceModel.id.in_(normalized_ids),
            )
            .order_by(EvidenceModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def list_by_step(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            step_id: str,
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        stmt = (
            select(EvidenceModel)
            .where(
                EvidenceModel.user_id == user_id,
                EvidenceModel.session_id == session_id,
                EvidenceModel.run_id == run_id,
                EvidenceModel.step_id == step_id,
            )
            .order_by(EvidenceModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def list_by_run(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            evidence_scope: EvidenceScope | None = None,
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        stmt = select(EvidenceModel).where(
            EvidenceModel.user_id == user_id,
            EvidenceModel.session_id == session_id,
            EvidenceModel.run_id == run_id,
        )
        if evidence_scope is not None:
            stmt = stmt.where(EvidenceModel.evidence_scope == evidence_scope.value)
        stmt = stmt.order_by(EvidenceModel.created_at.desc()).limit(self._normalize_limit(limit))
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def list_by_fact_ids(
            self,
            *,
            user_id: str,
            session_id: str,
            fact_ids: list[str],
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        normalized_ids = _normalize_text_list(fact_ids)
        if not normalized_ids:
            return []
        fact_conditions = [EvidenceModel.primary_fact_id.in_(normalized_ids)]
        fact_conditions.extend(
            EvidenceModel.source_ref["fact_ids"].contains([fact_id])
            for fact_id in normalized_ids
        )
        stmt = (
            select(EvidenceModel)
            .where(
                EvidenceModel.user_id == user_id,
                EvidenceModel.session_id == session_id,
                or_(*fact_conditions),
            )
            .order_by(EvidenceModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def list_reusable_by_run(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        stmt = (
            select(EvidenceModel)
            .where(
                EvidenceModel.user_id == user_id,
                EvidenceModel.session_id == session_id,
                EvidenceModel.run_id == run_id,
                EvidenceModel.reusable.is_(True),
            )
            .order_by(
                EvidenceModel.source_step_id.asc(),
                EvidenceModel.action_key.asc(),
                EvidenceModel.subject_key.asc(),
                EvidenceModel.created_at.desc(),
            )
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def list_by_action_subject(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            action_key: str,
            subject_key: str,
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        stmt = (
            select(EvidenceModel)
            .where(
                EvidenceModel.user_id == user_id,
                EvidenceModel.session_id == session_id,
                EvidenceModel.run_id == run_id,
                EvidenceModel.action_key == action_key,
                EvidenceModel.subject_key == subject_key,
            )
            .order_by(EvidenceModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]


def _normalize_text_list(values: list[str]) -> list[str]:
    return [
        str(value or "").strip()
        for value in list(values or [])
        if str(value or "").strip()
    ]
