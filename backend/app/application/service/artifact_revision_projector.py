#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ToolEvent/SandboxFact 到 ArtifactRevision 的受控投影器。"""

from __future__ import annotations

from typing import Any

from app.application.service.artifact_ledger_service import ArtifactLedgerService
from app.domain.models import ToolEvent
from app.domain.models.sandbox_fact import SandboxFactKind, SandboxFactRecord
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionRegistrationCommand,
    ArtifactRevisionSourceKind,
    ArtifactStorageBackend,
    ArtifactStorageRef,
    ArtifactType,
)
from app.domain.services.runtime.contracts.artifact_governance_ports import (
    ArtifactRevisionProjectionResult,
    ArtifactRevisionProjectorPort,
)
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)


class ArtifactRevisionProjector(ArtifactRevisionProjectorPort):
    """基于已持久化 source event 和 SandboxFact 登记 artifact revision。"""

    def __init__(self, *, ledger_service: ArtifactLedgerService) -> None:
        self._ledger_service = ledger_service

    async def project_from_tool_event_facts(
            self,
            *,
            scope: AccessScopeResult,
            event: ToolEvent,
            facts: list[SandboxFactRecord],
    ) -> ArtifactRevisionProjectionResult:
        revision_count = 0
        skipped: list[str] = []
        for fact in list(facts or []):
            command = self._command_from_fact(scope=scope, event=event, fact=fact)
            if command is None:
                skipped.append("unsupported_or_incomplete_fact")
                continue
            await self._ledger_service.register_revision(command=command)
            revision_count += 1
        return ArtifactRevisionProjectionResult(
            revision_count=revision_count,
            skipped_count=len(skipped),
            reason_codes=skipped,
        )

    async def project_from_document_facts(
            self,
            *,
            scope: AccessScopeResult,
            facts: list[SandboxFactRecord],
    ) -> ArtifactRevisionProjectionResult:
        revision_count = 0
        skipped: list[str] = []
        placeholder_event = ToolEvent(
            tool_call_id="",
            tool_name="document_input",
            function_name="document_input",
            function_args={},
        )
        for fact in list(facts or []):
            command = self._command_from_fact(scope=scope, event=placeholder_event, fact=fact)
            if command is None:
                skipped.append("unsupported_or_incomplete_fact")
                continue
            await self._ledger_service.register_revision(command=command)
            revision_count += 1
        return ArtifactRevisionProjectionResult(
            revision_count=revision_count,
            skipped_count=len(skipped),
            reason_codes=skipped,
        )

    def _command_from_fact(
            self,
            *,
            scope: AccessScopeResult,
            event: ToolEvent,
            fact: SandboxFactRecord,
    ) -> ArtifactRevisionRegistrationCommand | None:
        if fact.fact_kind == SandboxFactKind.FILE_WRITE:
            return self._file_write_command(scope=scope, event=event, fact=fact)
        if fact.fact_kind == SandboxFactKind.BROWSER_SNAPSHOT:
            return self._browser_screenshot_command(scope=scope, fact=fact)
        if fact.fact_kind == SandboxFactKind.DOCUMENT_CONTEXT:
            return self._document_input_command(scope=scope, fact=fact)
        return None

    @staticmethod
    def _file_write_command(
            *,
            scope: AccessScopeResult,
            event: ToolEvent,
            fact: SandboxFactRecord,
    ) -> ArtifactRevisionRegistrationCommand | None:
        payload = dict(fact.payload or {})
        function_name = str(event.function_name or fact.source_ref.function_name or "").strip()
        if function_name not in {"write_file", "replace_in_file"}:
            return None
        content_hash = _first_text(
            payload.get("content_hash"),
            payload.get("after_content_sha256"),
            payload.get("storage_hash"),
        )
        file_id = _first_text(payload.get("file_id"), payload.get("storage_file_id"))
        if not content_hash or not file_id:
            return None
        size_bytes = _positive_int(payload.get("size_bytes"), payload.get("size_after"), payload.get("size"))
        mime_type = _first_text(payload.get("mime_type"), payload.get("content_type"), default="text/plain")
        if size_bytes is None or not mime_type:
            return None
        path = _first_text(payload.get("path"), fact.subject_ref.path, default="")
        object_key = _first_text(payload.get("object_key"), payload.get("key"))
        storage_ref = ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.FILE_STORAGE,
            object_key=object_key,
            file_id=file_id,
            storage_hash=_first_text(payload.get("storage_hash"), content_hash),
            size_bytes=size_bytes,
            mime_type=mime_type,
        )
        return ArtifactRevisionRegistrationCommand(
            scope=scope,
            path=path,
            storage_ref=storage_ref,
            content_hash=content_hash,
            storage_hash=storage_ref.storage_hash,
            size_bytes=size_bytes,
            mime_type=mime_type,
            artifact_type=ArtifactType.FILE,
            delivery_state=ArtifactDeliveryState.CANDIDATE,
            source_kind=(
                ArtifactRevisionSourceKind.TOOL_WRITE_FILE
                if function_name == "write_file"
                else ArtifactRevisionSourceKind.TOOL_REPLACE_FILE
            ),
            source_event_id=str(fact.source_ref.source_event_id or ""),
            source_run_id=fact.run_id,
            source_fact_ids=[fact.id],
            tool_call_id=str(event.tool_call_id or fact.source_ref.tool_call_id or ""),
            function_name=function_name,
            profile_hash=fact.profile_ref.profile_hash,
            profile_status=fact.profile_ref.status,
            origin=DataOrigin.AGENT_GENERATED,
            trust_level=DataTrustLevel.AGENT_GENERATED,
            privacy_level=PrivacyLevel.PRIVATE,
            retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
            metadata={
                "source_fact_kind": fact.fact_kind.value,
                "mutation_intent_hash": payload.get("mutation_intent_hash"),
            },
        )

    @staticmethod
    def _browser_screenshot_command(
            *,
            scope: AccessScopeResult,
            fact: SandboxFactRecord,
    ) -> ArtifactRevisionRegistrationCommand | None:
        payload = dict(fact.payload or {})
        content_hash = _first_text(payload.get("screenshot_content_hash"), payload.get("screenshot_storage_hash"))
        file_id = _first_text(payload.get("screenshot_file_id"))
        if not content_hash or not file_id:
            return None
        size_bytes = _positive_int(payload.get("screenshot_size"))
        mime_type = _first_text(payload.get("screenshot_mime_type"), default="image/png")
        if size_bytes is None or not mime_type:
            return None
        path = _first_text(payload.get("screenshot_filepath"), payload.get("screenshot_filename"), default="")
        storage_ref = ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.FILE_STORAGE,
            object_key=_first_text(payload.get("screenshot_key")),
            file_id=file_id,
            storage_hash=_first_text(payload.get("screenshot_storage_hash"), content_hash),
            size_bytes=size_bytes,
            mime_type=mime_type,
        )
        return ArtifactRevisionRegistrationCommand(
            scope=scope,
            path=path,
            storage_ref=storage_ref,
            content_hash=content_hash,
            storage_hash=storage_ref.storage_hash,
            size_bytes=size_bytes,
            mime_type=mime_type,
            artifact_type=ArtifactType.SCREENSHOT,
            delivery_state=ArtifactDeliveryState.CANDIDATE,
            source_kind=ArtifactRevisionSourceKind.BROWSER_SCREENSHOT,
            source_event_id=str(fact.source_ref.source_event_id or ""),
            source_run_id=fact.run_id,
            source_fact_ids=[fact.id],
            profile_hash=fact.profile_ref.profile_hash,
            profile_status=fact.profile_ref.status,
            origin=DataOrigin.AGENT_GENERATED,
            trust_level=DataTrustLevel.AGENT_GENERATED,
            privacy_level=PrivacyLevel.PRIVATE,
            retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
            metadata={
                "source_fact_kind": fact.fact_kind.value,
                "screenshot_filename": payload.get("screenshot_filename"),
                "url_origin": payload.get("url_origin"),
            },
        )

    @staticmethod
    def _document_input_command(
            *,
            scope: AccessScopeResult,
            fact: SandboxFactRecord,
    ) -> ArtifactRevisionRegistrationCommand | None:
        payload = dict(fact.payload or {})
        parse_status = _first_text(payload.get("parse_status"))
        is_truncated = bool(payload.get("is_truncated"))
        content_hash = _first_text(payload.get("full_file_sha256"))
        file_id = _first_text(payload.get("file_id"))
        size_bytes = _positive_int(payload.get("size_bytes"))
        mime_type = _first_text(payload.get("mime_type"))
        object_key = _first_text(payload.get("object_key"))
        if parse_status != "parsed" or is_truncated:
            return None
        if not content_hash or not file_id or size_bytes is None or not mime_type:
            return None
        storage_ref = ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.FILE_STORAGE,
            object_key=object_key,
            file_id=file_id,
            storage_hash=content_hash,
            size_bytes=size_bytes,
            mime_type=mime_type,
        )
        return ArtifactRevisionRegistrationCommand(
            scope=scope,
            path=_document_artifact_path(payload=payload, file_id=file_id),
            storage_ref=storage_ref,
            content_hash=content_hash,
            storage_hash=content_hash,
            size_bytes=size_bytes,
            mime_type=mime_type,
            artifact_type=ArtifactType.FILE,
            delivery_state=ArtifactDeliveryState.CANDIDATE,
            source_kind=ArtifactRevisionSourceKind.DOCUMENT_INPUT,
            source_event_id=str(fact.source_ref.source_event_id or ""),
            source_run_id=fact.run_id,
            source_fact_ids=[fact.id],
            profile_hash=fact.profile_ref.profile_hash,
            profile_status=fact.profile_ref.status,
            origin=DataOrigin.USER_UPLOAD,
            trust_level=DataTrustLevel.USER_PROVIDED,
            privacy_level=PrivacyLevel.PRIVATE,
            retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
            metadata={
                "source_fact_kind": fact.fact_kind.value,
                "file_id": file_id,
                "full_file_sha256": content_hash,
                "read_content_sha256": payload.get("read_content_sha256"),
                "parse_status": parse_status,
                "reason_code": payload.get("reason_code"),
                "is_truncated": is_truncated,
                "document_fact_id": fact.id,
            },
        )


def _first_text(*values: Any, default: str | None = None) -> str:
    for value in values:
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return str(default or "").strip()


def _positive_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _document_artifact_path(*, payload: dict[str, Any], file_id: str) -> str:
    extension = _first_text(payload.get("filename_extension"))
    suffix = extension if extension.startswith(".") else f".{extension}" if extension else ""
    return f"document-input/{file_id}{suffix}"
