#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Artifact revision 强身份解析器。"""

from __future__ import annotations

from typing import Callable

from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionResolveCommand,
    ArtifactRevisionResolveResult,
    ArtifactRevisionResolveStatus,
)


class ArtifactRevisionResolver:
    """按 artifact_id + revision_id + content_hash 解析版本锁定产物。"""

    def __init__(self, *, uow_factory: Callable[[], IUnitOfWork]) -> None:
        self._uow_factory = uow_factory

    async def resolve(
            self,
            command: ArtifactRevisionResolveCommand,
    ) -> ArtifactRevisionResolveResult:
        async with self._uow_factory() as uow:
            revision = await uow.workspace_artifact_revision.get_by_identity(
                user_id=command.user_id,
                workspace_id=command.workspace_id,
                session_id=command.session_id,
                artifact_id=command.artifact_id,
                revision_id=command.revision_id,
                content_hash=command.content_hash,
            )
        if revision is None:
            return ArtifactRevisionResolveResult(
                status=ArtifactRevisionResolveStatus.NOT_FOUND,
                reason_code="artifact_revision_not_found",
            )
        if command.run_id and revision.run_id != command.run_id:
            return ArtifactRevisionResolveResult(
                status=ArtifactRevisionResolveStatus.NOT_FOUND,
                reason_code="artifact_run_scope_mismatch",
            )
        if command.source_run_id and revision.source_run_id != command.source_run_id:
            return ArtifactRevisionResolveResult(
                status=ArtifactRevisionResolveStatus.NOT_FOUND,
                reason_code="artifact_source_run_scope_mismatch",
            )
        if revision.delivery_state == ArtifactDeliveryState.EXPIRED:
            return ArtifactRevisionResolveResult(
                status=ArtifactRevisionResolveStatus.EXPIRED,
                reason_code="artifact_revision_expired",
            )
        if revision.delivery_state == ArtifactDeliveryState.QUARANTINED:
            return ArtifactRevisionResolveResult(
                status=ArtifactRevisionResolveStatus.QUARANTINED,
                reason_code="artifact_revision_quarantined",
            )
        return ArtifactRevisionResolveResult(
            status=ArtifactRevisionResolveStatus.RESOLVED,
            revision=revision,
        )
