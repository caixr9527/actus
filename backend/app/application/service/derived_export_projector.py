#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 final_answer_snapshot 派生导出文件 revision。"""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Callable

from app.application.service.artifact_ledger_service import ArtifactLedgerService
from app.domain.external import FileStorage, FileUploadPayload
from app.domain.models import MessageEvent, WorkspaceArtifactRevision
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.artifact_file_hash import verify_file_storage_stream
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionRegistrationCommand,
    ArtifactRevisionSourceKind,
    ArtifactStorageBackend,
    ArtifactStorageRef,
    ArtifactType,
    ResolvedArtifactRevisionResult,
)
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)


class DerivedExportError(RuntimeError):
    """派生导出失败。"""


class DerivedExportProjector:
    """基于已登记 final answer snapshot 生成受控导出文件。"""

    def __init__(
            self,
            *,
            uow_factory: Callable[[], IUnitOfWork],
            ledger_service: ArtifactLedgerService,
            file_storage: FileStorage,
    ) -> None:
        self._uow_factory = uow_factory
        self._ledger_service = ledger_service
        self._file_storage = file_storage

    async def export_latest_final_answer_as_markdown(
            self,
            *,
            scope: AccessScopeResult,
            source_run_id: str,
            filename: str = "final-answer.md",
    ) -> ResolvedArtifactRevisionResult:
        """从指定来源 run 的 final_answer_snapshot 派生 Markdown revision。"""
        normalized_source_run_id = str(source_run_id or "").strip()
        if not normalized_source_run_id:
            raise DerivedExportError("final_answer_snapshot_missing")
        current_run_id = str(scope.run_id or "").strip()
        if normalized_source_run_id == current_run_id:
            raise DerivedExportError("final_answer_snapshot_missing")
        snapshot, final_event = await self._load_latest_snapshot_and_event(
            scope=scope,
            source_run_id=normalized_source_run_id,
        )
        text = str(final_event.message or "")
        content = self._render_markdown(text).encode("utf-8")
        content_hash = "sha256:" + hashlib.sha256(content).hexdigest()
        uploaded = await self._file_storage.upload_file(
            upload_file=FileUploadPayload(
                filename=filename,
                file=BytesIO(content),
                content_type="text/markdown",
                size=len(content),
            ),
            user_id=scope.user_id,
        )
        stream, storage_file = await self._file_storage.download_file(uploaded.id, scope.user_id)
        verify_file_storage_stream(
            stream=stream,
            file=storage_file,
            expected_content_hash=content_hash,
            trusted_storage_hash=None,
        )
        storage_hash = content_hash
        return await self._ledger_service.register_revision(
            command=ArtifactRevisionRegistrationCommand(
                scope=scope,
                path=f"/exports/{uploaded.id}/{filename}",
                storage_ref=ArtifactStorageRef(
                    storage_backend=ArtifactStorageBackend.FILE_STORAGE,
                    object_key=uploaded.key,
                    file_id=uploaded.id,
                    storage_hash=storage_hash,
                    size_bytes=len(content),
                    mime_type="text/markdown",
                ),
                content_hash=content_hash,
                storage_hash=storage_hash,
                size_bytes=len(content),
                mime_type="text/markdown",
                artifact_type=ArtifactType.REPORT,
                delivery_state=ArtifactDeliveryState.CANDIDATE,
                source_kind=ArtifactRevisionSourceKind.DERIVED_EXPORT,
                source_event_id=str(final_event.id),
                source_run_id=normalized_source_run_id,
                source_message_event_id=snapshot.source_message_event_id,
                source_revision_id=snapshot.revision_id,
                source_final_answer_hash=snapshot.source_final_answer_hash,
                derived_content_hash=content_hash,
                origin=DataOrigin.AGENT_GENERATED,
                trust_level=DataTrustLevel.AGENT_GENERATED,
                privacy_level=PrivacyLevel.PRIVATE,
                retention_policy=RetentionPolicyKind.SESSION_BOUND,
                metadata={
                    "export_format": "markdown",
                    "source_revision_id": snapshot.revision_id,
                    "source_message_event_id": snapshot.source_message_event_id,
                },
            )
        )

    async def _load_latest_snapshot_and_event(
            self,
            *,
            scope: AccessScopeResult,
            source_run_id: str,
    ) -> tuple[WorkspaceArtifactRevision, MessageEvent]:
        async with self._uow_factory() as uow:
            snapshot = await uow.workspace_artifact_revision.get_latest_final_answer_snapshot(
                user_id=scope.user_id,
                workspace_id=str(scope.workspace_id),
                session_id=str(scope.session_id),
                source_run_id=source_run_id,
            )
            if snapshot is None:
                raise DerivedExportError("final_answer_snapshot_missing")
            event_id = str(snapshot.source_message_event_id or "").strip()
            if not event_id:
                raise DerivedExportError("final_message_event_missing")
            event_record = await uow.workflow_run.get_event_record_by_event_id(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                run_id=source_run_id,
                event_id=event_id,
            )
            if event_record is None or not isinstance(event_record.event_payload, MessageEvent):
                raise DerivedExportError("final_message_event_missing")
            final_event = event_record.event_payload
            if final_event.stage != "final":
                raise DerivedExportError("final_message_event_missing")
        final_hash = "sha256:" + hashlib.sha256(str(final_event.message or "").encode("utf-8")).hexdigest()
        if final_hash != snapshot.source_final_answer_hash:
            raise DerivedExportError("derived_export_hash_mismatch")
        return snapshot, final_event

    @staticmethod
    def _render_markdown(text: str) -> str:
        normalized = str(text or "")
        return normalized if normalized.endswith("\n") else f"{normalized}\n"
