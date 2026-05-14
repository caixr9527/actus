#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档输入解析应用服务。"""

import csv
import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any, BinaryIO, Protocol

from pydantic import BaseModel

from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.external import FileStorage
from app.domain.models import File
from app.domain.services.runtime.contracts.document_input_contract import (
    DOCUMENT_INPUT_EXCERPT_CHAR_LIMIT,
    DOCUMENT_INPUT_CONTEXT_CHAR_LIMIT,
    DOCUMENT_INPUT_MAX_FILE_SIZE_BYTES,
    DOCUMENT_REASON_DECODE_FAILED,
    DOCUMENT_REASON_EMPTY,
    DOCUMENT_REASON_PARSE_FAILED,
    DOCUMENT_REASON_TOO_LARGE,
    DOCUMENT_REASON_UNSUPPORTED_DOCUMENT_FORMAT,
    DocumentInputKind,
    DocumentInputPart,
    DocumentInputSourceRef,
    DocumentParseStatus,
    normalize_document_extension,
    resolve_document_kind,
)

logger = logging.getLogger(__name__)


class DocumentInputScopeMismatchError(RuntimeError):
    """文档输入来源 scope 与当前 runtime scope 不一致。"""


class DocumentAttachmentReader(Protocol):
    async def open_stream(self) -> BinaryIO:
        ...

    async def compute_sha256(self) -> str:
        ...

    @property
    def is_truncated(self) -> bool:
        ...


class FileStorageDocumentAttachmentReader:
    """唯一受控读取入口，负责读取上限、hash 和正文不外泄。"""

    def __init__(
            self,
            *,
            file_storage: FileStorage,
            file: File,
            user_id: str,
            max_bytes: int = DOCUMENT_INPUT_MAX_FILE_SIZE_BYTES,
    ) -> None:
        self._file_storage = file_storage
        self._file = file
        self._user_id = user_id
        self._max_bytes = max_bytes
        self._cached_bytes: bytes | None = None
        self._is_truncated = False

    @property
    def is_truncated(self) -> bool:
        return self._is_truncated

    async def open_stream(self) -> BinaryIO:
        data = await self._read_bytes()
        return BytesIO(data)

    async def compute_sha256(self) -> str:
        data = await self._read_bytes()
        return hashlib.sha256(data).hexdigest()

    async def _read_bytes(self) -> bytes:
        if self._cached_bytes is not None:
            return self._cached_bytes
        stream, _ = await self._file_storage.download_file(
            file_id=self._file.id,
            user_id=self._user_id,
        )
        chunks: list[bytes] = []
        total_read = 0
        while total_read <= self._max_bytes:
            chunk = stream.read(min(64 * 1024, self._max_bytes + 1 - total_read))
            if not chunk:
                break
            chunk_bytes = bytes(chunk)
            chunks.append(chunk_bytes)
            total_read += len(chunk_bytes)
            if total_read > self._max_bytes:
                self._is_truncated = True
                break
        data = b"".join(chunks)
        if len(data) > self._max_bytes:
            data = data[:self._max_bytes]
        self._cached_bytes = data
        return self._cached_bytes


@dataclass(slots=True)
class DocumentAttachmentSource:
    scope: AccessScopeResult
    file: File
    sandbox_filepath: str
    reader: DocumentAttachmentReader
    sha256: str | None = None


class DocumentPromptContextItem(BaseModel):
    file_id: str
    filename: str
    sandbox_filepath: str | None = None
    kind: str
    parse_status: str
    reason_code: str | None = None
    is_truncated: bool = False
    summary: str = ""
    text_excerpt: str = ""


class DocumentPromptContextResult(BaseModel):
    documents: list[DocumentPromptContextItem]
    document_count: int
    context_char_count: int
    truncated_document_count: int


class DocumentInputService:
    """P0-5 文档输入处理入口。"""

    def __init__(self, *, excerpt_char_limit: int = DOCUMENT_INPUT_EXCERPT_CHAR_LIMIT) -> None:
        self._excerpt_char_limit = excerpt_char_limit

    async def build_input_parts(
            self,
            *,
            scope: AccessScopeResult,
            attachments: list[DocumentAttachmentSource],
            request_id: str | None = None,
    ) -> list[DocumentInputPart]:
        """只处理 preflight 通过的文档，不承担权限查询和最终输出职责。"""
        parts: list[DocumentInputPart] = []
        for attachment in list(attachments or []):
            self._ensure_scope_matches(scope=scope, attachment=attachment, request_id=request_id)
            parts.append(await self._build_part(attachment, request_id=request_id))
        logger.info(
            "document_input_context_built user_id=%s session_id=%s run_id=%s document_count=%s context_char_count=%s truncated_document_count=%s request_id=%s",
            scope.user_id,
            scope.session_id or "",
            scope.run_id or "",
            len(parts),
            sum(len(part.text_excerpt or "") for part in parts),
            sum(1 for part in parts if part.is_truncated),
            request_id or "",
        )
        return parts

    def build_prompt_context(
            self,
            *,
            parts: list[DocumentInputPart],
            max_chars: int = DOCUMENT_INPUT_CONTEXT_CHAR_LIMIT,
    ) -> DocumentPromptContextResult:
        """将 DocumentInputPart 纯转换为 prompt-safe 文档上下文。"""
        documents: list[DocumentPromptContextItem] = []
        remaining_chars = max(0, int(max_chars))
        truncated_document_count = 0
        context_char_count = 0
        for part in list(parts or []):
            if not isinstance(part, DocumentInputPart):
                part = DocumentInputPart.model_validate(part)
            summary = str(part.summary or "")
            if len(summary) > remaining_chars:
                summary = summary[:remaining_chars]
                text_excerpt = ""
                is_context_truncated = True
            else:
                text_excerpt = str(part.text_excerpt or "")
                budget = max(0, remaining_chars - len(summary))
                if len(text_excerpt) > budget:
                    text_excerpt = text_excerpt[:budget]
                    is_context_truncated = True
                else:
                    is_context_truncated = False
            item = DocumentPromptContextItem(
                file_id=part.source.file_id,
                filename=part.source.filename,
                sandbox_filepath=part.source.sandbox_filepath,
                kind=part.kind.value,
                parse_status=part.parse_status.value,
                reason_code=part.reason_code,
                is_truncated=bool(part.is_truncated or is_context_truncated),
                summary=summary,
                text_excerpt=text_excerpt,
            )
            documents.append(item)
            used_chars = len(summary) + len(text_excerpt)
            context_char_count += used_chars
            remaining_chars = max(0, remaining_chars - used_chars)
            if item.is_truncated:
                truncated_document_count += 1
        return DocumentPromptContextResult(
            documents=documents,
            document_count=len(documents),
            context_char_count=context_char_count,
            truncated_document_count=truncated_document_count,
        )

    def _ensure_scope_matches(
            self,
            *,
            scope: AccessScopeResult,
            attachment: DocumentAttachmentSource,
            request_id: str | None,
    ) -> None:
        compared_fields = ("user_id", "session_id", "workspace_id", "run_id")
        for field_name in compared_fields:
            expected = str(getattr(scope, field_name, "") or "")
            actual = str(getattr(attachment.scope, field_name, "") or "")
            if expected and actual and expected != actual:
                logger.error(
                    "document_input_scope_mismatch field=%s expected=%s actual=%s file_id=%s request_id=%s",
                    field_name,
                    expected,
                    actual,
                    attachment.file.id,
                    request_id or "",
                )
                raise DocumentInputScopeMismatchError(
                    f"document input scope mismatch: {field_name}"
                )

    async def _build_part(
            self,
            attachment: DocumentAttachmentSource,
            *,
            request_id: str | None,
    ) -> DocumentInputPart:
        file = attachment.file
        source_ref = await self._build_source_ref(attachment)
        document_kind = resolve_document_kind(
            extension=file.extension,
            mime_type=file.mime_type,
            filename=file.filename,
        )
        logger.info(
            "document_input_parse_started user_id=%s session_id=%s run_id=%s file_id=%s document_kind=%s size=%s sha256_prefix=%s request_id=%s",
            source_ref.user_id,
            source_ref.session_id,
            source_ref.run_id or "",
            source_ref.file_id,
            document_kind.value,
            source_ref.size,
            str(source_ref.sha256 or "")[:12],
            request_id or "",
        )
        part = await self._parse_part(
            attachment=attachment,
            source_ref=source_ref,
            document_kind=document_kind,
        )
        logger.info(
            "document_input_parse_finished user_id=%s session_id=%s run_id=%s file_id=%s document_kind=%s parse_status=%s is_truncated=%s excerpt_char_count=%s reason_code=%s request_id=%s",
            source_ref.user_id,
            source_ref.session_id,
            source_ref.run_id or "",
            source_ref.file_id,
            document_kind.value,
            part.parse_status.value,
            part.is_truncated,
            len(part.text_excerpt or ""),
            part.reason_code or "",
            request_id or "",
        )
        return part

    async def _build_source_ref(self, attachment: DocumentAttachmentSource) -> DocumentInputSourceRef:
        file = attachment.file
        sha256 = attachment.sha256 or await attachment.reader.compute_sha256()
        return DocumentInputSourceRef(
            file_id=file.id,
            user_id=attachment.scope.user_id,
            session_id=attachment.scope.session_id or "",
            workspace_id=attachment.scope.workspace_id,
            run_id=attachment.scope.run_id,
            sandbox_filepath=attachment.sandbox_filepath,
            filename=file.filename,
            object_key=file.key or None,
            mime_type=file.mime_type or "",
            extension=normalize_document_extension(file.extension, file.filename),
            size=int(file.size or 0),
            sha256=sha256,
        )

    async def _parse_part(
            self,
            *,
            attachment: DocumentAttachmentSource,
            source_ref: DocumentInputSourceRef,
            document_kind: DocumentInputKind,
    ) -> DocumentInputPart:
        if int(source_ref.size or 0) <= 0:
            return self._failed_part(
                kind=document_kind,
                source_ref=source_ref,
                status=DocumentParseStatus.EMPTY,
                reason_code=DOCUMENT_REASON_EMPTY,
            )
        if document_kind in {
            DocumentInputKind.PDF,
            DocumentInputKind.WORD,
            DocumentInputKind.SPREADSHEET,
            DocumentInputKind.ARCHIVE,
            DocumentInputKind.UNKNOWN_DOCUMENT,
        }:
            # P0-5 只启用文本类 parser，复杂格式默认 unsupported_document_format。
            return self._failed_part(
                kind=document_kind,
                source_ref=source_ref,
                status=DocumentParseStatus.UNSUPPORTED,
                reason_code=DOCUMENT_REASON_UNSUPPORTED_DOCUMENT_FORMAT,
            )

        try:
            stream = await attachment.reader.open_stream()
            content_bytes = stream.read()
            raw_text = self._decode_text(content_bytes)
            preview = self._structured_preview(document_kind, raw_text)
            text = raw_text
            if document_kind == DocumentInputKind.HTML:
                text = self._extract_visible_html_text(text)
            excerpt, is_truncated = self._truncate(text)
            is_reader_truncated = bool(getattr(attachment.reader, "is_truncated", False))
            return DocumentInputPart(
                kind=document_kind,
                source=source_ref,
                parse_status=DocumentParseStatus.TOO_LARGE if is_reader_truncated else (
                    DocumentParseStatus.PARTIAL if is_truncated else DocumentParseStatus.PARSED
                ),
                text_excerpt=excerpt,
                summary=self._summary(document_kind, text),
                structured_preview=preview,
                reason_code=DOCUMENT_REASON_TOO_LARGE if is_reader_truncated else None,
                is_truncated=is_truncated or is_reader_truncated,
            )
        except UnicodeDecodeError:
            return self._failed_part(
                kind=document_kind,
                source_ref=source_ref,
                status=DocumentParseStatus.FAILED,
                reason_code=DOCUMENT_REASON_DECODE_FAILED,
            )
        except Exception as exc:
            logger.error(
                "document_input_parser_exception file_id=%s document_kind=%s exception_type=%s",
                source_ref.file_id,
                document_kind.value,
                exc.__class__.__name__,
            )
            return self._failed_part(
                kind=document_kind,
                source_ref=source_ref,
                status=DocumentParseStatus.FAILED,
                reason_code=DOCUMENT_REASON_PARSE_FAILED,
            )

    def _decode_text(self, data: bytes) -> str:
        if not data:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, DOCUMENT_REASON_EMPTY)
        return data.decode("utf-8-sig")

    def _truncate(self, text: str) -> tuple[str, bool]:
        normalized = str(text or "")
        if len(normalized) <= self._excerpt_char_limit:
            return normalized, False
        return normalized[:self._excerpt_char_limit], True

    def _summary(self, document_kind: DocumentInputKind, text: str) -> str:
        line_count = len([line for line in str(text or "").splitlines() if line.strip()])
        return f"{document_kind.value} document, {line_count} non-empty lines"

    def _structured_preview(self, document_kind: DocumentInputKind, text: str) -> dict[str, Any]:
        if document_kind == DocumentInputKind.CSV:
            return self._csv_preview(text)
        if document_kind == DocumentInputKind.JSON:
            return self._json_preview(text)
        if document_kind == DocumentInputKind.HTML:
            title = self._html_title(text)
            return {"title": title} if title else {}
        return {}

    def _csv_preview(self, text: str) -> dict[str, Any]:
        reader = csv.reader(StringIO(text))
        rows = [row for _, row in zip(range(6), reader)]
        if not rows:
            return {}
        return {"headers": rows[0], "rows": rows[1:]}

    def _json_preview(self, text: str) -> dict[str, Any]:
        first_line = str(text or "").splitlines()[0] if str(text or "").splitlines() else ""
        parsed = json.loads(first_line if first_line.strip() else text)
        if isinstance(parsed, dict):
            return {"type": "object", "keys": list(parsed.keys())[:20]}
        if isinstance(parsed, list):
            return {"type": "array", "length": len(parsed)}
        return {"type": type(parsed).__name__}

    def _html_title(self, text: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()

    def _extract_visible_html_text(self, text: str) -> str:
        without_scripts = re.sub(
            r"<(script|style)[^>]*>.*?</\1>",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
        return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()

    def _failed_part(
            self,
            *,
            kind: DocumentInputKind,
            source_ref: DocumentInputSourceRef,
            status: DocumentParseStatus,
            reason_code: str,
    ) -> DocumentInputPart:
        return DocumentInputPart(
            kind=kind,
            source=source_ref,
            parse_status=status,
            reason_code=reason_code,
        )
