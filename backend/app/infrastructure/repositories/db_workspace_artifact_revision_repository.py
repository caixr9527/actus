#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Artifact revision DB 仓储实现。"""

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import WorkspaceArtifactRevision
from app.domain.repositories import WorkspaceArtifactRevisionRepository
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactRevisionSourceKind,
)
from app.infrastructure.models import WorkspaceArtifactModel, WorkspaceArtifactRevisionModel


class WorkspaceArtifactRevisionConflictError(RuntimeError):
    """revision 幂等写入后无法读取既有记录。"""


class WorkspaceArtifactRevisionScopeError(ValueError):
    """revision 与 artifact 归属 scope 不一致。"""


class DBWorkspaceArtifactRevisionRepository(WorkspaceArtifactRevisionRepository):
    """基于数据库的 artifact revision 仓库。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def insert_or_get_existing(self, revision: WorkspaceArtifactRevision) -> WorkspaceArtifactRevision:
        await self._ensure_artifact_scope(revision)
        existing = await self._get_existing_by_idempotency_key(revision)
        if existing is not None:
            return existing

        insert_values = self._to_insert_values(revision)
        stmt = (
            insert(WorkspaceArtifactRevisionModel.__table__)
            .values(**insert_values)
            .on_conflict_do_nothing()
            .returning(WorkspaceArtifactRevisionModel.revision_id)
        )
        try:
            result = await self.db_session.execute(stmt)
            inserted_revision_id = result.scalar_one_or_none()
        except IntegrityError:
            inserted_revision_id = None

        if inserted_revision_id:
            await self._update_current_projection(revision)
            inserted = await self.get_by_user_workspace_revision_id(
                user_id=revision.user_id,
                workspace_id=revision.workspace_id,
                revision_id=inserted_revision_id,
            )
            if inserted is None:
                raise WorkspaceArtifactRevisionConflictError("artifact revision 写入后无法读取")
            return inserted

        existing_after_conflict = await self._get_existing_by_idempotency_key(revision)
        if existing_after_conflict is not None:
            return existing_after_conflict
        raise WorkspaceArtifactRevisionConflictError("artifact revision 幂等冲突后无法读取既有记录")

    async def get_by_user_workspace_revision_id(
            self,
            *,
            user_id: str,
            workspace_id: str,
            revision_id: str,
    ) -> WorkspaceArtifactRevision | None:
        stmt = select(WorkspaceArtifactRevisionModel).where(
            WorkspaceArtifactRevisionModel.user_id == user_id,
            WorkspaceArtifactRevisionModel.workspace_id == workspace_id,
            WorkspaceArtifactRevisionModel.revision_id == revision_id,
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        if isinstance(record, WorkspaceArtifactRevision):
            return record
        return record.to_domain()

    async def list_by_user_workspace_artifact_id(
            self,
            *,
            user_id: str,
            workspace_id: str,
            artifact_id: str,
    ) -> list[WorkspaceArtifactRevision]:
        stmt = (
            select(WorkspaceArtifactRevisionModel)
            .where(
                WorkspaceArtifactRevisionModel.user_id == user_id,
                WorkspaceArtifactRevisionModel.workspace_id == workspace_id,
                WorkspaceArtifactRevisionModel.artifact_id == artifact_id,
            )
            .order_by(WorkspaceArtifactRevisionModel.revision_no.asc())
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [
            record if isinstance(record, WorkspaceArtifactRevision) else record.to_domain()
            for record in records
        ]

    async def _ensure_artifact_scope(self, revision: WorkspaceArtifactRevision) -> None:
        stmt = select(WorkspaceArtifactModel.id).where(
            WorkspaceArtifactModel.user_id == revision.user_id,
            WorkspaceArtifactModel.workspace_id == revision.workspace_id,
            WorkspaceArtifactModel.id == revision.artifact_id,
        )
        result = await self.db_session.execute(stmt)
        if result.scalar_one_or_none() is None:
            raise WorkspaceArtifactRevisionScopeError("artifact revision 与 artifact scope 不一致")

    async def _update_current_projection(self, revision: WorkspaceArtifactRevision) -> None:
        stmt = (
            update(WorkspaceArtifactModel)
            .where(
                WorkspaceArtifactModel.user_id == revision.user_id,
                WorkspaceArtifactModel.workspace_id == revision.workspace_id,
                WorkspaceArtifactModel.id == revision.artifact_id,
            )
            .values(
                current_revision_id=revision.revision_id,
                latest_content_hash=revision.content_hash,
                latest_size=revision.size_bytes,
                latest_mime_type=revision.mime_type,
                artifact_type=revision.artifact_type.value,
                delivery_state=revision.delivery_state.value,
            )
        )
        await self.db_session.execute(stmt)

    async def _get_existing_by_idempotency_key(
            self,
            revision: WorkspaceArtifactRevision,
    ) -> WorkspaceArtifactRevision | None:
        stmt = self._idempotency_select(revision)
        if stmt is None:
            return None
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        if isinstance(record, WorkspaceArtifactRevision):
            return record
        return record.to_domain()

    @staticmethod
    def _idempotency_select(revision: WorkspaceArtifactRevision):
        base = select(WorkspaceArtifactRevisionModel).where(
            WorkspaceArtifactRevisionModel.user_id == revision.user_id,
            WorkspaceArtifactRevisionModel.workspace_id == revision.workspace_id,
        )
        if revision.source_kind in {
            ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
            ArtifactRevisionSourceKind.TOOL_REPLACE_FILE,
        }:
            if not revision.source_event_id or not revision.tool_call_id or not revision.content_hash:
                return None
            return base.where(
                WorkspaceArtifactRevisionModel.source_event_id == revision.source_event_id,
                WorkspaceArtifactRevisionModel.tool_call_id == revision.tool_call_id,
                WorkspaceArtifactRevisionModel.source_kind == revision.source_kind.value,
                WorkspaceArtifactRevisionModel.content_hash == revision.content_hash,
            )
        if revision.source_kind == ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT:
            if not revision.source_run_id or not revision.source_message_event_id or not revision.source_final_answer_hash:
                return None
            return base.where(
                WorkspaceArtifactRevisionModel.session_id == revision.session_id,
                WorkspaceArtifactRevisionModel.source_run_id == revision.source_run_id,
                WorkspaceArtifactRevisionModel.source_message_event_id == revision.source_message_event_id,
                WorkspaceArtifactRevisionModel.source_final_answer_hash == revision.source_final_answer_hash,
                WorkspaceArtifactRevisionModel.source_kind == revision.source_kind.value,
            )
        if revision.source_kind == ArtifactRevisionSourceKind.DERIVED_EXPORT:
            if not revision.source_revision_id or not revision.content_hash:
                return None
            return base.where(
                WorkspaceArtifactRevisionModel.source_revision_id == revision.source_revision_id,
                WorkspaceArtifactRevisionModel.source_kind == revision.source_kind.value,
                WorkspaceArtifactRevisionModel.content_hash == revision.content_hash,
            )
        return None

    @staticmethod
    def _to_insert_values(revision: WorkspaceArtifactRevision) -> dict:
        payload = revision.model_dump(mode="json")
        payload["metadata"] = payload.pop("metadata", {})
        return payload
