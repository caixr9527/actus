#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback Ledger DB 仓储实现。"""

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.feedback import (
    FeedbackRecord,
    FeedbackResolutionResult,
    FeedbackScopeKind,
    FeedbackStatus,
    FeedbackTargetType,
)
from app.domain.repositories import FeedbackRepository
from app.infrastructure.models import FeedbackRecordModel


class DBFeedbackRepository(FeedbackRepository):
    """基于数据库的 Feedback Ledger 仓储。"""

    ACTIVE_STATUSES = (FeedbackStatus.OPEN.value,)

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(1, min(int(limit or 100), 200))

    @staticmethod
    def _require_user_session_scope(*, user_id: str, session_id: str) -> None:
        if not str(user_id or "").strip() or not str(session_id or "").strip():
            raise ValueError("feedback 查询必须提供 user_id 和 session_id")

    @staticmethod
    def _require_text(value: str | None, field_name: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{field_name} 不能为空")
        return normalized

    @staticmethod
    def _to_insert_values(record: FeedbackRecord) -> dict:
        model = FeedbackRecordModel.from_domain(record)
        return {
            "id": model.id,
            "user_id": model.user_id,
            "session_id": model.session_id,
            "workspace_id": model.workspace_id,
            "run_id": model.run_id,
            "feedback_scope_kind": model.feedback_scope_kind,
            "scope_id": model.scope_id,
            "source_run_id": model.source_run_id,
            "target_run_id": model.target_run_id,
            "step_id": model.step_id,
            "kind": model.kind,
            "category": model.category,
            "status": model.status,
            "severity": model.severity,
            "source_kind": model.source_kind,
            "source_event_id": model.source_event_id,
            "target_type": model.target_type,
            "target_id": model.target_id,
            "target_revision_id": model.target_revision_id,
            "target_content_hash": model.target_content_hash,
            "feedback_key": model.feedback_key,
            "dedupe_key": model.dedupe_key,
            "reason_code": model.reason_code,
            "resolution_reason_code": model.resolution_reason_code,
            "decay_policy": model.decay_policy,
            "ttl_scope": model.ttl_scope,
            "expires_at": model.expires_at,
            "origin": model.origin,
            "trust_level": model.trust_level,
            "privacy_level": model.privacy_level,
            "retention_policy": model.retention_policy,
            "profile_hash": model.profile_hash,
            "source_record_refs": model.source_record_refs,
            "source_ref": model.source_ref,
            "target_ref": model.target_ref,
            "feedback_summary": model.feedback_summary,
            "prompt_safe_summary": model.prompt_safe_summary,
            "resolution": model.resolution,
            "classification": model.classification,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }

    async def _get_by_dedupe_scope(self, record: FeedbackRecord) -> FeedbackRecord | None:
        stmt = select(FeedbackRecordModel).where(
            FeedbackRecordModel.user_id == record.user_id,
            FeedbackRecordModel.session_id == record.session_id,
            FeedbackRecordModel.feedback_scope_kind == record.feedback_scope_kind.value,
            FeedbackRecordModel.scope_id == record.scope_id,
            FeedbackRecordModel.dedupe_key == record.dedupe_key,
        )
        result = await self.db_session.execute(stmt)
        model = result.scalar_one_or_none()
        return model.to_domain() if model is not None else None

    async def save_once(self, record: FeedbackRecord) -> FeedbackRecord:
        insert_values = self._to_insert_values(record)
        stmt = (
            insert(FeedbackRecordModel.__table__)
            .values(**insert_values)
            .on_conflict_do_nothing(constraint="uq_feedback_records_user_session_scope_dedupe")
            .returning(FeedbackRecordModel.__table__.c.id)
        )
        try:
            result = await self.db_session.execute(stmt)
        except IntegrityError:
            existing = await self._get_by_dedupe_scope(record)
            if existing is not None:
                return existing
            raise

        inserted_id = result.scalar_one_or_none()
        if inserted_id is not None:
            return record

        existing = await self._get_by_dedupe_scope(record)
        if existing is None:
            raise RuntimeError("feedback 幂等写入冲突后未找到已有记录")
        return existing

    async def get_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            feedback_scope_kind: FeedbackScopeKind,
            scope_id: str,
            feedback_id: str,
    ) -> FeedbackRecord | None:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        scope_id = self._require_text(scope_id, "scope_id")
        feedback_id = self._require_text(feedback_id, "feedback_id")
        stmt = select(FeedbackRecordModel).where(
            FeedbackRecordModel.user_id == user_id,
            FeedbackRecordModel.session_id == session_id,
            FeedbackRecordModel.feedback_scope_kind == feedback_scope_kind.value,
            FeedbackRecordModel.scope_id == scope_id,
            FeedbackRecordModel.id == feedback_id,
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
    ) -> list[FeedbackRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        run_id = self._require_text(run_id, "run_id")
        stmt = (
            select(FeedbackRecordModel)
            .where(
                FeedbackRecordModel.user_id == user_id,
                FeedbackRecordModel.session_id == session_id,
                FeedbackRecordModel.run_id == run_id,
            )
            .order_by(FeedbackRecordModel.created_at.desc())
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
    ) -> list[FeedbackRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        run_id = self._require_text(run_id, "run_id")
        step_id = self._require_text(step_id, "step_id")
        stmt = (
            select(FeedbackRecordModel)
            .where(
                FeedbackRecordModel.user_id == user_id,
                FeedbackRecordModel.session_id == session_id,
                FeedbackRecordModel.run_id == run_id,
                FeedbackRecordModel.step_id == step_id,
            )
            .order_by(FeedbackRecordModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def list_active_by_run(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        return await self.list_active_by_scope(
            user_id=user_id,
            session_id=session_id,
            feedback_scope_kind=FeedbackScopeKind.RUN,
            scope_id=run_id,
            limit=limit,
        )

    async def list_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            feedback_scope_kind: FeedbackScopeKind,
            scope_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        scope_id = self._require_text(scope_id, "scope_id")
        stmt = (
            select(FeedbackRecordModel)
            .where(
                FeedbackRecordModel.user_id == user_id,
                FeedbackRecordModel.session_id == session_id,
                FeedbackRecordModel.feedback_scope_kind == feedback_scope_kind.value,
                FeedbackRecordModel.scope_id == scope_id,
            )
            .order_by(FeedbackRecordModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def list_active_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            feedback_scope_kind: FeedbackScopeKind,
            scope_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        scope_id = self._require_text(scope_id, "scope_id")
        stmt = (
            select(FeedbackRecordModel)
            .where(
                FeedbackRecordModel.user_id == user_id,
                FeedbackRecordModel.session_id == session_id,
                FeedbackRecordModel.feedback_scope_kind == feedback_scope_kind.value,
                FeedbackRecordModel.scope_id == scope_id,
                FeedbackRecordModel.status.in_(self.ACTIVE_STATUSES),
            )
            .order_by(FeedbackRecordModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def list_by_target(
            self,
            *,
            user_id: str,
            session_id: str,
            target_type: FeedbackTargetType,
            target_id: str,
            target_revision_id: str | None = None,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        target_id = self._require_text(target_id, "target_id")
        stmt = select(FeedbackRecordModel).where(
            FeedbackRecordModel.user_id == user_id,
            FeedbackRecordModel.session_id == session_id,
            FeedbackRecordModel.target_type == target_type.value,
            FeedbackRecordModel.target_id == target_id,
        )
        if target_revision_id is None:
            stmt = stmt.where(FeedbackRecordModel.target_revision_id.is_(None))
        else:
            stmt = stmt.where(
                FeedbackRecordModel.target_revision_id == self._require_text(target_revision_id, "target_revision_id")
            )
        stmt = stmt.order_by(FeedbackRecordModel.created_at.desc()).limit(self._normalize_limit(limit))
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def list_by_source_event(
            self,
            *,
            user_id: str,
            session_id: str,
            source_event_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        source_event_id = self._require_text(source_event_id, "source_event_id")
        stmt = (
            select(FeedbackRecordModel)
            .where(
                FeedbackRecordModel.user_id == user_id,
                FeedbackRecordModel.session_id == session_id,
                FeedbackRecordModel.source_event_id == source_event_id,
            )
            .order_by(FeedbackRecordModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def list_by_source_event_for_projection(
            self,
            *,
            user_id: str,
            session_id: str,
            source_run_id: str,
            source_event_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        source_run_id = self._require_text(source_run_id, "source_run_id")
        source_event_id = self._require_text(source_event_id, "source_event_id")
        stmt = (
            select(FeedbackRecordModel)
            .where(
                FeedbackRecordModel.user_id == user_id,
                FeedbackRecordModel.session_id == session_id,
                FeedbackRecordModel.source_run_id == source_run_id,
                FeedbackRecordModel.source_event_id == source_event_id,
            )
            .order_by(FeedbackRecordModel.created_at.asc(), FeedbackRecordModel.id.asc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def list_by_resolution_aggregation_key(
            self,
            *,
            user_id: str,
            session_id: str,
            source_run_id: str,
            resolution_aggregation_key: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        source_run_id = self._require_text(source_run_id, "source_run_id")
        resolution_aggregation_key = self._require_text(
            resolution_aggregation_key,
            "resolution_aggregation_key",
        )
        stmt = (
            select(FeedbackRecordModel)
            .where(
                FeedbackRecordModel.user_id == user_id,
                FeedbackRecordModel.session_id == session_id,
                FeedbackRecordModel.source_run_id == source_run_id,
                FeedbackRecordModel.status != FeedbackStatus.OPEN.value,
                FeedbackRecordModel.resolution[("resolved_by_ref", "resolution_aggregation_key")].astext
                == resolution_aggregation_key,
            )
            .order_by(FeedbackRecordModel.updated_at.asc(), FeedbackRecordModel.id.asc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [model.to_domain() for model in result.scalars().all()]

    async def update_resolution(
            self,
            *,
            user_id: str,
            session_id: str,
            feedback_scope_kind: FeedbackScopeKind,
            scope_id: str,
            feedback_id: str,
            resolution: FeedbackResolutionResult,
            updated_at: datetime,
    ) -> FeedbackRecord:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        scope_id = self._require_text(scope_id, "scope_id")
        feedback_id = self._require_text(feedback_id, "feedback_id")
        stmt = (
            update(FeedbackRecordModel)
            .where(
                FeedbackRecordModel.user_id == user_id,
                FeedbackRecordModel.session_id == session_id,
                FeedbackRecordModel.feedback_scope_kind == feedback_scope_kind.value,
                FeedbackRecordModel.scope_id == scope_id,
                FeedbackRecordModel.id == feedback_id,
            )
            .values(
                status=resolution.status.value,
                resolution_reason_code=(
                    resolution.resolution_reason_code.value
                    if resolution.resolution_reason_code is not None
                    else None
                ),
                resolution=resolution.model_dump(mode="json"),
                updated_at=updated_at,
            )
        )
        result = await self.db_session.execute(stmt)
        if result.rowcount != 1:
            raise ValueError("反馈生命周期更新失败：记录不存在或不在当前 scope 内")
        record = await self.get_by_scope(
            user_id=user_id,
            session_id=session_id,
            feedback_scope_kind=feedback_scope_kind,
            scope_id=scope_id,
            feedback_id=feedback_id,
        )
        if record is None:
            raise RuntimeError("反馈生命周期更新后无法回读记录")
        return record
