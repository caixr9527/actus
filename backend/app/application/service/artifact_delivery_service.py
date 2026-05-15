#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Artifact revision 下载/预览应用服务。"""

from __future__ import annotations

from typing import BinaryIO, Callable, Literal

from app.application.errors import NotFoundError, ValidationError
from app.application.errors.exceptions import AppException
from app.application.errors import error_keys
from app.domain.services.runtime.artifact_file_hash import (
    ArtifactFileHashMismatchError,
    is_valid_sha256_content_hash,
    verify_file_storage_stream,
)
from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.domain.external import FileStorage
from app.domain.models import File, WorkspaceArtifactRevision
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.artifact_governance_contract import ArtifactStorageBackend
from app.domain.services.runtime.contracts.data_access_contract import DataAccessAction


class ArtifactDeliveryConflictError(AppException):
    """Artifact revision 当前状态不允许交付。"""

    def __init__(self, *, msg: str, reason_code: str) -> None:
        super().__init__(
            code=409,
            status_code=409,
            msg=msg,
            error_key=f"error.artifact.{reason_code}",
            error_params={"reason_code": reason_code},
        )


class ArtifactDeliveryFileResult:
    """下载/预览返回的已校验文件流。"""

    def __init__(
            self,
            *,
            stream: BinaryIO,
            file: File,
            revision: WorkspaceArtifactRevision,
            disposition: Literal["inline", "attachment"],
    ) -> None:
        self.stream = stream
        self.file = file
        self.revision = revision
        self.disposition = disposition


class ArtifactDeliveryService:
    """按 revision 身份读取可交付 file storage 内容。"""

    def __init__(
            self,
            *,
            uow_factory: Callable[[], IUnitOfWork],
            file_storage: FileStorage,
            access_control_service: RuntimeAccessControlService | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._access_control_service = access_control_service or RuntimeAccessControlService(
            uow_factory=uow_factory,
        )

    async def download_revision(
            self,
            *,
            user_id: str,
            session_id: str,
            artifact_id: str,
            revision_id: str,
            content_hash: str,
            run_id: str | None = None,
            source_run_id: str | None = None,
    ) -> ArtifactDeliveryFileResult:
        return await self._open_revision(
            user_id=user_id,
            session_id=session_id,
            artifact_id=artifact_id,
            revision_id=revision_id,
            content_hash=content_hash,
            run_id=run_id,
            source_run_id=source_run_id,
            disposition="attachment",
        )

    async def preview_revision(
            self,
            *,
            user_id: str,
            session_id: str,
            artifact_id: str,
            revision_id: str,
            content_hash: str,
            run_id: str | None = None,
            source_run_id: str | None = None,
    ) -> ArtifactDeliveryFileResult:
        return await self._open_revision(
            user_id=user_id,
            session_id=session_id,
            artifact_id=artifact_id,
            revision_id=revision_id,
            content_hash=content_hash,
            run_id=run_id,
            source_run_id=source_run_id,
            disposition="inline",
        )

    async def _open_revision(
            self,
            *,
            user_id: str,
            session_id: str,
            artifact_id: str,
            revision_id: str,
            content_hash: str,
            run_id: str | None,
            source_run_id: str | None,
            disposition: Literal["inline", "attachment"],
    ) -> ArtifactDeliveryFileResult:
        normalized_hash = str(content_hash or "").strip().lower()
        if not is_valid_sha256_content_hash(normalized_hash):
            raise ValidationError(
                msg="artifact content_hash 缺失或格式非法",
                error_key="error.artifact.content_hash_invalid",
                error_params={"reason_code": "artifact_content_hash_invalid"},
            )
        scope = await self._access_control_service.assert_session_access(
            user_id=user_id,
            session_id=session_id,
            action=DataAccessAction.DOWNLOAD,
        )
        if not scope.workspace_id:
            raise self._not_found(session_id=session_id)
        async with self._uow_factory() as uow:
            revision = await uow.workspace_artifact_revision.get_by_identity(
                user_id=user_id,
                workspace_id=str(scope.workspace_id),
                session_id=session_id,
                artifact_id=artifact_id,
                revision_id=revision_id,
                content_hash=normalized_hash,
            )
        if revision is None:
            raise self._not_found(session_id=session_id)
        self._assert_run_scope(revision=revision, run_id=run_id, source_run_id=source_run_id)
        if revision.storage_ref.storage_backend != ArtifactStorageBackend.FILE_STORAGE:
            raise ArtifactDeliveryConflictError(
                msg="artifact revision 当前存储不可下载或预览",
                reason_code="artifact_storage_not_deliverable",
            )
        file_id = str(revision.storage_ref.file_id or "").strip()
        if not file_id:
            raise ArtifactDeliveryConflictError(
                msg="artifact revision 缺少 file storage 文件",
                reason_code="artifact_storage_not_deliverable",
            )
        stream, file = await self._file_storage.download_file(file_id=file_id, user_id=user_id)
        try:
            verified = verify_file_storage_stream(
                stream=stream,
                file=file,
                expected_content_hash=revision.content_hash,
                trusted_storage_hash=None,
            )
        except ArtifactFileHashMismatchError as exc:
            raise ArtifactDeliveryConflictError(
                msg="artifact revision 内容 hash 已变化",
                reason_code=str(exc),
            ) from exc
        self._fill_file_projection(file=verified.file, revision=revision)
        return ArtifactDeliveryFileResult(
            stream=verified.stream,
            file=verified.file,
            revision=revision,
            disposition=disposition,
        )

    @staticmethod
    def _assert_run_scope(
            *,
            revision: WorkspaceArtifactRevision,
            run_id: str | None,
            source_run_id: str | None,
    ) -> None:
        expected_run_id = str(run_id or "").strip()
        if expected_run_id and revision.run_id != expected_run_id:
            raise NotFoundError(
                msg="artifact revision 不存在",
                error_key=error_keys.SESSION_NOT_FOUND,
            )
        expected_source_run_id = str(source_run_id or "").strip()
        if expected_source_run_id and revision.source_run_id != expected_source_run_id:
            raise NotFoundError(
                msg="artifact revision 不存在",
                error_key=error_keys.SESSION_NOT_FOUND,
            )

    @staticmethod
    def _fill_file_projection(*, file: File, revision: WorkspaceArtifactRevision) -> None:
        if not file.filepath:
            file.filepath = revision.path
        if not file.mime_type and revision.mime_type:
            file.mime_type = revision.mime_type
        if not file.size and revision.size_bytes:
            file.size = revision.size_bytes

    @staticmethod
    def _not_found(*, session_id: str) -> NotFoundError:
        return NotFoundError(
            msg="artifact revision 不存在",
            error_key=error_keys.SESSION_NOT_FOUND,
            error_params={"session_id": session_id},
        )
