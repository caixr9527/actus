#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-4 Artifact Ledger 应用服务。"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from app.domain.models import ArtifactEvent, WorkspaceArtifact, WorkspaceArtifactRevision
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactEventArtifactRef,
    ArtifactEventPayload,
    ArtifactRevisionIdentity,
    ArtifactRevisionEventRef,
    ArtifactRevisionRegistrationCommand,
    ArtifactStatus,
    ResolvedArtifactRevisionResult,
)

logger = logging.getLogger(__name__)


class ArtifactLedgerError(RuntimeError):
    """Artifact Ledger 应用服务错误。"""


class ArtifactLedgerScopeError(ArtifactLedgerError):
    """Artifact scope 缺失或不一致。"""


class ArtifactLedgerService:
    """Artifact 写入、revision 追加和 revision-aware 查询的唯一应用入口。"""

    def __init__(self, *, uow_factory: Callable[[], IUnitOfWork]) -> None:
        self._uow_factory = uow_factory

    async def register_revision(
            self,
            *,
            command: ArtifactRevisionRegistrationCommand,
    ) -> ResolvedArtifactRevisionResult:
        self._validate_scope(command.scope)
        normalized_path = str(command.path or "").strip()
        async with self._uow_factory() as uow:
            artifact = await self._get_or_create_artifact(
                uow=uow,
                command=command,
                normalized_path=normalized_path,
            )
            revision = WorkspaceArtifactRevision(
                artifact_id=artifact.id,
                revision_no=1,
                user_id=command.scope.user_id,
                session_id=str(command.scope.session_id),
                workspace_id=str(command.scope.workspace_id),
                run_id=command.scope.run_id,
                step_id=command.scope.current_step_id,
                path=normalized_path,
                storage_ref=command.storage_ref,
                content_hash=command.content_hash,
                storage_hash=command.storage_hash,
                hash_algorithm=command.hash_algorithm,
                size_bytes=command.size_bytes,
                mime_type=command.mime_type,
                artifact_type=command.artifact_type,
                delivery_state=command.delivery_state,
                source_kind=command.source_kind,
                source_event_id=command.source_event_id,
                source_run_id=command.source_run_id,
                source_message_event_id=command.source_message_event_id,
                source_revision_id=command.source_revision_id,
                source_fact_ids=list(command.source_fact_ids),
                source_evidence_ids=list(command.source_evidence_ids),
                source_final_answer_hash=command.source_final_answer_hash,
                derived_content_hash=command.derived_content_hash,
                tool_call_id=command.tool_call_id,
                function_name=command.function_name,
                profile_hash=command.profile_hash,
                profile_status=command.profile_status or "missing",
                origin=command.origin,
                trust_level=command.trust_level,
                privacy_level=command.privacy_level,
                retention_policy=command.retention_policy,
                metadata=dict(command.metadata or {}),
            )
            saved = await uow.workspace_artifact_revision.append_revision_for_artifact(revision)
            await self._persist_artifact_event_if_possible(uow=uow, revision=saved)
        return self._to_resolved(saved)

    async def resolve_authoritative_artifact_revisions(
            self,
            *,
            scope: AccessScopeResult,
            paths: list[str],
    ) -> list[ResolvedArtifactRevisionResult]:
        self._validate_scope(scope)
        normalized_paths = _normalize_unique_paths(paths)
        if not normalized_paths:
            return []
        async with self._uow_factory() as uow:
            artifacts = await uow.workspace_artifact.list_by_user_workspace_id_and_paths(
                user_id=scope.user_id,
                workspace_id=str(scope.workspace_id),
                paths=normalized_paths,
            )
            revision_by_path: dict[str, WorkspaceArtifactRevision] = {}
            for artifact in artifacts:
                revision_id = str(artifact.current_revision_id or "").strip()
                if not revision_id:
                    continue
                revision = await uow.workspace_artifact_revision.get_by_user_workspace_revision_id(
                    user_id=scope.user_id,
                    workspace_id=str(scope.workspace_id),
                    revision_id=revision_id,
                )
                if revision is None or revision.session_id != scope.session_id:
                    continue
                revision_by_path[revision.path] = revision
        return [
            self._to_resolved(revision_by_path[path])
            for path in normalized_paths
            if path in revision_by_path
        ]

    async def mark_artifact_revisions_delivery_state(
            self,
            *,
            scope: AccessScopeResult,
            revisions: list[ArtifactRevisionIdentity],
            delivery_state: ArtifactDeliveryState,
    ) -> list[ResolvedArtifactRevisionResult]:
        self._validate_scope(scope)
        if delivery_state not in {ArtifactDeliveryState.CANDIDATE, ArtifactDeliveryState.SELECTED}:
            raise ArtifactLedgerError("PR2 只允许写 candidate/selected delivery_state")
        if not revisions:
            return []
        async with self._uow_factory() as uow:
            updated = await uow.workspace_artifact_revision.update_delivery_state_by_identities(
                user_id=scope.user_id,
                workspace_id=str(scope.workspace_id),
                session_id=str(scope.session_id),
                identities=list(revisions),
                delivery_state=delivery_state,
            )
        return [self._to_resolved(revision) for revision in updated]

    async def _get_or_create_artifact(
            self,
            *,
            uow: IUnitOfWork,
            command: ArtifactRevisionRegistrationCommand,
            normalized_path: str,
    ) -> WorkspaceArtifact:
        artifact = await uow.workspace_artifact.get_by_user_workspace_id_and_path(
            user_id=command.scope.user_id,
            workspace_id=str(command.scope.workspace_id),
            path=normalized_path,
        )
        if artifact is not None:
            return artifact
        artifact = WorkspaceArtifact(
            workspace_id=str(command.scope.workspace_id),
            user_id=command.scope.user_id,
            session_id=command.scope.session_id,
            run_id=command.scope.run_id,
            path=normalized_path,
            artifact_type=command.artifact_type.value,
            summary=str((command.metadata or {}).get("summary") or "").strip(),
            source_step_id=command.scope.current_step_id,
            source_capability=command.function_name,
            delivery_state=command.delivery_state.value,
            current_revision_id=None,
            latest_content_hash=None,
            latest_size=None,
            latest_mime_type=None,
            artifact_status=ArtifactStatus.ACTIVE,
            origin=command.origin,
            trust_level=command.trust_level,
            privacy_level=command.privacy_level,
            retention_policy=command.retention_policy,
            metadata=dict(command.metadata or {}),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await uow.workspace_artifact.insert_current_index_if_absent(artifact=artifact)
        persisted_artifact = await uow.workspace_artifact.get_by_user_workspace_id_and_path(
            user_id=command.scope.user_id,
            workspace_id=str(command.scope.workspace_id),
            path=normalized_path,
        )
        if persisted_artifact is None:
            raise ArtifactLedgerError("artifact current 行写入后无法读取")
        return persisted_artifact

    @staticmethod
    def _validate_scope(scope: AccessScopeResult) -> None:
        if not str(scope.user_id or "").strip():
            raise ArtifactLedgerScopeError("artifact scope 必须包含 user_id")
        if not str(scope.session_id or "").strip():
            raise ArtifactLedgerScopeError("artifact scope 必须包含 session_id")
        if not str(scope.workspace_id or "").strip():
            raise ArtifactLedgerScopeError("artifact scope 必须包含 workspace_id")

    @staticmethod
    def _to_resolved(revision: WorkspaceArtifactRevision) -> ResolvedArtifactRevisionResult:
        return ResolvedArtifactRevisionResult(
            artifact_id=revision.artifact_id,
            revision_id=revision.revision_id,
            content_hash=revision.content_hash,
            path=revision.path,
            artifact_type=revision.artifact_type,
            delivery_state=revision.delivery_state,
            session_id=revision.session_id,
            run_id=revision.run_id,
            source_run_id=revision.source_run_id,
            source_step_id=revision.step_id,
            source_event_id=revision.source_event_id,
            source_kind=revision.source_kind,
        )

    @staticmethod
    async def _persist_artifact_event_if_possible(
            *,
            uow: IUnitOfWork,
            revision: WorkspaceArtifactRevision,
    ) -> None:
        """revision 成功后补写 ArtifactEvent，失败只记录诊断。"""
        run_id = str(revision.run_id or revision.source_run_id or "").strip()
        if not run_id:
            logger.warning(
                "artifact_event_projection_failed",
                extra={
                    "user_id": revision.user_id,
                    "session_id": revision.session_id,
                    "workspace_id": revision.workspace_id,
                    "artifact_id": revision.artifact_id,
                    "revision_id": revision.revision_id,
                    "reason_code": "artifact_event_run_id_missing",
                },
            )
            return
        event = ArtifactEvent(
            id=f"artifact-revision:{revision.revision_id}",
            payload=ArtifactEventPayload(
                artifact_refs=[
                    ArtifactEventArtifactRef(
                        artifact_id=revision.artifact_id,
                        path=revision.path,
                        artifact_type=revision.artifact_type,
                        delivery_state=revision.delivery_state,
                        current_revision_id=revision.revision_id,
                        latest_content_hash=revision.content_hash,
                    )
                ],
                revision_refs=[
                    ArtifactRevisionEventRef(
                        artifact_id=revision.artifact_id,
                        revision_id=revision.revision_id,
                        content_hash=revision.content_hash,
                        path=revision.path,
                        artifact_type=revision.artifact_type,
                        delivery_state=revision.delivery_state,
                        source_event_id=revision.source_event_id,
                    )
                ],
                counts={"revision_count": 1},
                summary="artifact revision registered",
                source_event_ids=[revision.source_event_id] if revision.source_event_id else [],
                runtime_metadata={
                    "schema_version": "artifact_event.v1",
                    "source_kind": revision.source_kind.value,
                },
            )
        )
        try:
            workflow_run_repo = getattr(uow, "workflow_run", None)
            if workflow_run_repo is None or not hasattr(workflow_run_repo, "add_event_record_if_absent"):
                raise AttributeError("workflow_run repository missing")
            await workflow_run_repo.add_event_record_if_absent(
                session_id=revision.session_id,
                run_id=run_id,
                event=event,
            )
        except AttributeError:
            logger.warning(
                "artifact_event_projection_failed",
                extra={
                    "user_id": revision.user_id,
                    "session_id": revision.session_id,
                    "workspace_id": revision.workspace_id,
                    "artifact_id": revision.artifact_id,
                    "revision_id": revision.revision_id,
                    "reason_code": "artifact_event_repository_missing",
                },
            )
        except Exception:
            logger.exception(
                "artifact_event_projection_failed",
                extra={
                    "user_id": revision.user_id,
                    "session_id": revision.session_id,
                    "workspace_id": revision.workspace_id,
                    "artifact_id": revision.artifact_id,
                    "revision_id": revision.revision_id,
                    "reason_code": "artifact_event_projection_failed",
                },
            )


def _normalize_unique_paths(paths: list[str]) -> list[str]:
    normalized: list[str] = []
    for path in paths:
        value = str(path or "").strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized
