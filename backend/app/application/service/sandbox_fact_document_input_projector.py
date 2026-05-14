#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DocumentInputPart 到 Sandbox Fact 的投影器。"""

from __future__ import annotations

import hashlib
import logging

from app.application.service.sandbox_fact_ledger_service import (
    DocumentContextFactInput,
    SandboxFactLedgerService,
)
from app.domain.models.sandbox_fact import SandboxFactScope, SandboxFactSourceType
from app.domain.services.runtime.contracts.artifact_governance_ports import ArtifactRevisionProjectorPort
from app.domain.services.runtime.contracts.document_input_contract import DocumentInputPart
from app.domain.services.runtime.contracts.sandbox_fact_ports import (
    SandboxFactProjectionContext,
)

logger = logging.getLogger(__name__)


class SandboxFactDocumentInputProjector:
    """将已归一文档输入写入 DOCUMENT_CONTEXT fact。"""

    def __init__(
            self,
            *,
            ledger_service: SandboxFactLedgerService,
            artifact_revision_projector: ArtifactRevisionProjectorPort | None = None,
    ) -> None:
        self._ledger_service = ledger_service
        self._artifact_revision_projector = artifact_revision_projector

    async def record_document_context(
            self,
            *,
            context: SandboxFactProjectionContext,
            parts: list[DocumentInputPart],
    ) -> None:
        source_event_id = str(context.source_event_id or "").strip()
        if not source_event_id:
            logger.warning(
                "sandbox_fact_record_failed",
                extra={"reason_code": "document_source_event_id_missing"},
            )
            return
        saved_facts = []
        for part in list(parts or []):
            fact_input = self._build_fact_input(part)
            saved_facts.append(await self._ledger_service.record_fact(context=context, fact_input=fact_input))
        if self._artifact_revision_projector is not None and saved_facts:
            await self._artifact_revision_projector.project_from_document_facts(
                scope=context.scope,
                facts=saved_facts,
            )

    @staticmethod
    def _build_fact_input(part: DocumentInputPart) -> DocumentContextFactInput:
        source = part.source
        text_excerpt = str(part.text_excerpt or "")
        read_content_sha256 = (
            f"sha256:{hashlib.sha256(text_excerpt.encode('utf-8')).hexdigest()}"
            if text_excerpt
            else None
        )
        return DocumentContextFactInput(
            fact_scope=SandboxFactScope.RUN,
            run_id=source.run_id,
            summary=f"Document context: {source.filename}",
            source_type=SandboxFactSourceType.DOCUMENT_INPUT,
            file_id=source.file_id,
            document_source_id=source.file_id,
            object_key=str(getattr(source, "object_key", "") or "").strip() or None,
            filename_extension=source.extension,
            mime_type=source.mime_type,
            size_bytes=source.size,
            parse_status=part.parse_status.value,
            reason_code=part.reason_code,
            full_file_sha256=source.sha256,
            read_content_sha256=read_content_sha256,
            is_truncated=part.is_truncated,
            excerpt_char_count=len(text_excerpt),
        )
