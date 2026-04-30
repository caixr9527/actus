#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档输入请求前校验策略。"""

import logging
from typing import Iterable

from app.application.errors import BadRequestError, error_keys
from app.domain.models import File
from app.domain.services.runtime.contracts.document_input_contract import (
    DOCUMENT_INPUT_MAX_ATTACHMENTS,
    DOCUMENT_INPUT_MAX_FILE_SIZE_BYTES,
    DOCUMENT_REASON_EMPTY,
    DOCUMENT_REASON_TOO_LARGE,
    DOCUMENT_REASON_TOO_MANY_ATTACHMENTS,
    DOCUMENT_REASON_UNSUPPORTED_BINARY,
    DOCUMENT_REASON_UNSUPPORTED_DOCUMENT_FORMAT,
    DOCUMENT_REASON_UNSUPPORTED_MEDIA_AUDIO,
    DOCUMENT_REASON_UNSUPPORTED_MEDIA_IMAGE,
    DOCUMENT_REASON_UNSUPPORTED_MEDIA_VIDEO,
    is_supported_document_metadata,
    normalize_document_extension,
    resolve_unsupported_media_reason,
)

logger = logging.getLogger(__name__)

_REASON_ERROR_KEYS = {
    DOCUMENT_REASON_UNSUPPORTED_MEDIA_IMAGE: error_keys.DOCUMENT_INPUT_UNSUPPORTED_MEDIA_IMAGE,
    DOCUMENT_REASON_UNSUPPORTED_MEDIA_AUDIO: error_keys.DOCUMENT_INPUT_UNSUPPORTED_MEDIA_AUDIO,
    DOCUMENT_REASON_UNSUPPORTED_MEDIA_VIDEO: error_keys.DOCUMENT_INPUT_UNSUPPORTED_MEDIA_VIDEO,
    DOCUMENT_REASON_UNSUPPORTED_BINARY: error_keys.DOCUMENT_INPUT_UNSUPPORTED_BINARY,
    DOCUMENT_REASON_UNSUPPORTED_DOCUMENT_FORMAT: error_keys.DOCUMENT_INPUT_UNSUPPORTED_DOCUMENT_FORMAT,
    DOCUMENT_REASON_TOO_LARGE: error_keys.DOCUMENT_INPUT_TOO_LARGE,
    DOCUMENT_REASON_TOO_MANY_ATTACHMENTS: error_keys.DOCUMENT_INPUT_TOO_MANY_ATTACHMENTS,
    DOCUMENT_REASON_EMPTY: error_keys.DOCUMENT_INPUT_EMPTY,
}


class DocumentInputPreflightPolicy:
    """请求前文档附件校验策略。"""

    def __init__(
            self,
            *,
            max_attachments: int = DOCUMENT_INPUT_MAX_ATTACHMENTS,
            max_file_size_bytes: int = DOCUMENT_INPUT_MAX_FILE_SIZE_BYTES,
    ) -> None:
        self._max_attachments = max_attachments
        self._max_file_size_bytes = max_file_size_bytes

    def validate(
            self,
            files: Iterable[File],
            *,
            user_id: str,
            session_id: str,
            request_id: str | None = None,
    ) -> list[File]:
        """非文档媒体在 _create_task() 前拒绝，避免污染 runtime history。"""
        accepted_files = list(files or [])
        logger.info(
            "document_input_preflight_started user_id=%s session_id=%s file_count=%s request_id=%s",
            user_id,
            session_id,
            len(accepted_files),
            request_id or "",
        )
        if len(accepted_files) > self._max_attachments:
            self._raise_rejected(
                reason_code=DOCUMENT_REASON_TOO_MANY_ATTACHMENTS,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                file=None,
            )

        for file in accepted_files:
            reason_code = self._reject_reason(file)
            if reason_code is not None:
                self._raise_rejected(
                    reason_code=reason_code,
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    file=file,
                )

        logger.info(
            "document_input_preflight_passed user_id=%s session_id=%s file_count=%s accepted_document_count=%s request_id=%s",
            user_id,
            session_id,
            len(accepted_files),
            len(accepted_files),
            request_id or "",
        )
        return accepted_files

    def _reject_reason(self, file: File) -> str | None:
        if int(file.size or 0) <= 0:
            return DOCUMENT_REASON_EMPTY
        if int(file.size or 0) > self._max_file_size_bytes:
            return DOCUMENT_REASON_TOO_LARGE
        media_reason = resolve_unsupported_media_reason(
            extension=file.extension,
            mime_type=file.mime_type,
            filename=file.filename,
        )
        if media_reason is not None:
            return media_reason
        if not is_supported_document_metadata(
                extension=file.extension,
                mime_type=file.mime_type,
                filename=file.filename,
        ):
            return DOCUMENT_REASON_UNSUPPORTED_DOCUMENT_FORMAT
        return None

    def _raise_rejected(
            self,
            *,
            reason_code: str,
            user_id: str,
            session_id: str,
            request_id: str | None,
            file: File | None,
    ) -> None:
        extension = normalize_document_extension(
            getattr(file, "extension", "") if file is not None else "",
            getattr(file, "filename", "") if file is not None else "",
        )
        logger.warning(
            "document_input_preflight_rejected user_id=%s session_id=%s file_id=%s filename_extension=%s mime_type=%s size=%s reason_code=%s request_id=%s",
            user_id,
            session_id,
            getattr(file, "id", "") if file is not None else "",
            extension,
            getattr(file, "mime_type", "") if file is not None else "",
            int(getattr(file, "size", 0) or 0) if file is not None else 0,
            reason_code,
            request_id or "",
        )
        raise BadRequestError(
            msg="不支持该类型作为任务输入",
            error_key=_REASON_ERROR_KEYS.get(reason_code, error_keys.DOCUMENT_INPUT_UNSUPPORTED_DOCUMENT_FORMAT),
            error_params={
                "reason_code": reason_code,
                "file_id": getattr(file, "id", "") if file is not None else "",
                "extension": extension,
                "mime_type": getattr(file, "mime_type", "") if file is not None else "",
            },
        )
