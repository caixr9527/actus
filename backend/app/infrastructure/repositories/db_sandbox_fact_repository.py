#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Fact Ledger DB 仓储实现。"""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.sandbox_fact import (
    SandboxFactKind,
    SandboxFactRecord,
    SandboxFactScope,
)
from app.domain.repositories import SandboxFactRepository
from app.infrastructure.models import SandboxFactModel


class DBFactSupersededTargetError(ValueError):
    """SUPERSEDED fact 指向了不可写入的原 fact。"""


class DBSandboxFactRepository(SandboxFactRepository):
    """基于数据库的 Sandbox Fact 仓储。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(1, min(int(limit or 100), 200))

    @staticmethod
    def _require_user_session_scope(*, user_id: str, session_id: str) -> None:
        if not str(user_id or "").strip() or not str(session_id or "").strip():
            raise ValueError("sandbox fact 查询必须提供 user_id 和 session_id")

    @staticmethod
    def _to_insert_values(fact: SandboxFactRecord) -> dict:
        model = SandboxFactModel.from_domain(fact)
        return {
            "id": model.id,
            "user_id": model.user_id,
            "session_id": model.session_id,
            "workspace_id": model.workspace_id,
            "fact_scope": model.fact_scope,
            "run_id": model.run_id,
            "step_id": model.step_id,
            "sandbox_id": model.sandbox_id,
            "fact_kind": model.fact_kind,
            "source_type": model.source_type,
            "source_event_id": model.source_event_id,
            "source_event_status": model.source_event_status,
            "tool_event_id": model.tool_event_id,
            "tool_call_id": model.tool_call_id,
            "function_name": model.function_name,
            "subject_type": model.subject_type,
            "subject_key": model.subject_key,
            "profile_id": model.profile_id,
            "profile_hash": model.profile_hash,
            "source_ref": model.source_ref,
            "subject_ref": model.subject_ref,
            "profile_ref": model.profile_ref,
            "related_fact_ids": model.related_fact_ids,
            "supersedes_fact_id": model.supersedes_fact_id,
            "payload_hash": model.payload_hash,
            "idempotency_key": model.idempotency_key,
            "summary": model.summary,
            "payload": model.payload,
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
    ) -> SandboxFactRecord | None:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        stmt = select(SandboxFactModel).where(
            SandboxFactModel.user_id == user_id,
            SandboxFactModel.session_id == session_id,
            SandboxFactModel.idempotency_key == idempotency_key,
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def _assert_superseded_target(self, fact: SandboxFactRecord) -> None:
        if fact.fact_kind != SandboxFactKind.SUPERSEDED:
            return
        stmt = select(SandboxFactModel.id).where(
            SandboxFactModel.user_id == fact.user_id,
            SandboxFactModel.session_id == fact.session_id,
            SandboxFactModel.id == fact.supersedes_fact_id,
        )
        result = await self.db_session.execute(stmt)
        if result.scalar_one_or_none() is None:
            raise DBFactSupersededTargetError("SUPERSEDED fact 指向的原 fact 不存在或不属于同一用户会话")

    async def save_once(self, fact: SandboxFactRecord) -> SandboxFactRecord:
        await self._assert_superseded_target(fact)
        insert_values = self._to_insert_values(fact)
        stmt = (
            insert(SandboxFactModel.__table__)
            .values(**insert_values)
            .on_conflict_do_nothing(constraint="uq_sandbox_facts_idempotency_key")
            .returning(SandboxFactModel.__table__.c.id)
        )
        try:
            result = await self.db_session.execute(stmt)
        except IntegrityError:
            existing = await self._get_by_idempotency_key(
                user_id=fact.user_id,
                session_id=fact.session_id,
                idempotency_key=fact.idempotency_key,
            )
            if existing is not None:
                return existing
            raise

        inserted_id = result.scalar_one_or_none()
        if inserted_id is not None:
            return fact

        existing = await self._get_by_idempotency_key(
            user_id=fact.user_id,
            session_id=fact.session_id,
            idempotency_key=fact.idempotency_key,
        )
        if existing is None:
            raise RuntimeError("sandbox fact 幂等写入冲突后未找到已有记录")
        return existing

    async def list_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            fact_scope: SandboxFactScope | None = None,
            run_id: str | None = None,
            step_id: str | None = None,
            fact_kinds: list[SandboxFactKind] | None = None,
            limit: int = 100,
    ) -> list[SandboxFactRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        stmt = select(SandboxFactModel).where(
            SandboxFactModel.user_id == user_id,
            SandboxFactModel.session_id == session_id,
        )
        if fact_scope is not None:
            stmt = stmt.where(SandboxFactModel.fact_scope == fact_scope.value)
        if run_id is not None:
            stmt = stmt.where(SandboxFactModel.run_id == run_id)
        if step_id is not None:
            stmt = stmt.where(SandboxFactModel.step_id == step_id)
        if fact_kinds:
            stmt = stmt.where(SandboxFactModel.fact_kind.in_([kind.value for kind in fact_kinds]))
        stmt = stmt.order_by(SandboxFactModel.created_at.desc()).limit(self._normalize_limit(limit))
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def list_by_source_event(
            self,
            *,
            user_id: str,
            session_id: str,
            source_event_id: str,
    ) -> list[SandboxFactRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        stmt = (
            select(SandboxFactModel)
            .where(
                SandboxFactModel.user_id == user_id,
                SandboxFactModel.session_id == session_id,
                SandboxFactModel.source_event_id == source_event_id,
            )
            .order_by(SandboxFactModel.created_at.desc())
        )
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def list_by_ids(
            self,
            *,
            user_id: str,
            session_id: str,
            fact_ids: list[str],
            limit: int = 100,
    ) -> list[SandboxFactRecord]:
        self._require_user_session_scope(user_id=user_id, session_id=session_id)
        normalized_ids = [
            str(fact_id or "").strip()
            for fact_id in list(fact_ids or [])
            if str(fact_id or "").strip()
        ]
        if not normalized_ids:
            return []
        stmt = (
            select(SandboxFactModel)
            .where(
                SandboxFactModel.user_id == user_id,
                SandboxFactModel.session_id == session_id,
                SandboxFactModel.id.in_(normalized_ids),
            )
            .order_by(SandboxFactModel.created_at.desc())
            .limit(self._normalize_limit(limit))
        )
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]
