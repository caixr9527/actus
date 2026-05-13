#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工作区产物仓储 DB 实现。"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import WorkspaceArtifact
from app.domain.repositories import WorkspaceArtifactRepository
from app.infrastructure.models import WorkspaceArtifactModel


class DBWorkspaceArtifactRepository(WorkspaceArtifactRepository):
    """基于数据库的工作区产物仓库。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    @staticmethod
    def _to_insert_values(artifact: WorkspaceArtifact) -> dict:
        return {
            "id": artifact.id,
            "workspace_id": artifact.workspace_id,
            "user_id": artifact.user_id,
            "session_id": artifact.session_id,
            "run_id": artifact.run_id,
            "path": artifact.path,
            "artifact_type": artifact.artifact_type,
            "summary": artifact.summary,
            "source_step_id": artifact.source_step_id,
            "source_capability": artifact.source_capability,
            "delivery_state": artifact.delivery_state,
            "current_revision_id": artifact.current_revision_id,
            "latest_content_hash": artifact.latest_content_hash,
            "latest_size": artifact.latest_size,
            "latest_mime_type": artifact.latest_mime_type,
            "artifact_status": artifact.artifact_status.value,
            "origin": artifact.origin.value,
            "trust_level": artifact.trust_level.value,
            "privacy_level": artifact.privacy_level.value,
            "retention_policy": artifact.retention_policy.value,
            "metadata": dict(artifact.metadata or {}),
            "created_at": artifact.created_at,
            "updated_at": artifact.updated_at,
        }

    async def save(self, artifact: WorkspaceArtifact) -> None:
        insert_values = self._to_insert_values(artifact)
        update_values = {
            key: value
            for key, value in insert_values.items()
            if key not in {"id", "workspace_id", "path", "created_at"}
        }
        update_values["updated_at"] = datetime.now()
        stmt = (
            insert(WorkspaceArtifactModel.__table__)
            .values(**insert_values)
            .on_conflict_do_update(
                constraint="uq_workspace_artifacts_workspace_id_path",
                set_=update_values,
            )
        )
        await self.db_session.execute(stmt)

    async def insert_current_index_if_absent(self, artifact: WorkspaceArtifact) -> None:
        stmt = (
            insert(WorkspaceArtifactModel.__table__)
            .values(**self._to_insert_values(artifact))
            .on_conflict_do_nothing(
                constraint="uq_workspace_artifacts_workspace_id_path",
            )
        )
        await self.db_session.execute(stmt)

    async def list_by_user_workspace_id(self, user_id: str, workspace_id: str) -> List[WorkspaceArtifact]:
        stmt = (
            select(WorkspaceArtifactModel)
            .where(
                WorkspaceArtifactModel.user_id == user_id,
                WorkspaceArtifactModel.workspace_id == workspace_id,
            )
            .order_by(WorkspaceArtifactModel.updated_at.desc())
        )
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def list_by_user_workspace_id_and_paths(
            self,
            user_id: str,
            workspace_id: str,
            paths: List[str],
    ) -> List[WorkspaceArtifact]:
        normalized_paths = [
            str(path or "").strip()
            for path in list(paths or [])
            if str(path or "").strip()
        ]
        if len(normalized_paths) == 0:
            return []
        stmt = (
            select(WorkspaceArtifactModel)
            .where(
                WorkspaceArtifactModel.user_id == user_id,
                WorkspaceArtifactModel.workspace_id == workspace_id,
                WorkspaceArtifactModel.path.in_(normalized_paths),
            )
        )
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def get_by_user_workspace_id_and_path(
            self,
            user_id: str,
            workspace_id: str,
            path: str,
    ) -> Optional[WorkspaceArtifact]:
        stmt = select(WorkspaceArtifactModel).where(
            WorkspaceArtifactModel.user_id == user_id,
            WorkspaceArtifactModel.workspace_id == workspace_id,
            WorkspaceArtifactModel.path == path,
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def get_by_user_workspace_id_and_id(
            self,
            user_id: str,
            workspace_id: str,
            artifact_id: str,
    ) -> Optional[WorkspaceArtifact]:
        stmt = select(WorkspaceArtifactModel).where(
            WorkspaceArtifactModel.user_id == user_id,
            WorkspaceArtifactModel.workspace_id == workspace_id,
            WorkspaceArtifactModel.id == artifact_id,
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None
