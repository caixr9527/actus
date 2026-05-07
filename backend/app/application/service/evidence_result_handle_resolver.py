#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence ResultHandle 受控解析器。"""

from __future__ import annotations

import logging
from typing import Any

from app.domain.models.evidence import (
    DocumentEvidencePayload,
    EvidenceKind,
    EvidenceReadStrategy,
    EvidenceResolvedResult,
    EvidenceResolvedStatus,
    EvidenceResultHandle,
)
from app.domain.models.sandbox_fact import SandboxFactKind
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult

logger = logging.getLogger(__name__)


class EvidenceResultHandleResolver:
    """按 ResultHandle 指向的受控来源读取裁剪结果。"""

    def __init__(self, *, uow_factory) -> None:
        self._uow_factory = uow_factory

    async def resolve(
            self,
            *,
            scope: AccessScopeResult,
            handle: EvidenceResultHandle,
    ) -> EvidenceResolvedResult:
        result: EvidenceResolvedResult
        if not _has_required_scope(scope):
            result = _unresolved(handle, EvidenceResolvedStatus.SCOPE_MISMATCH, "evidence_scope_mismatch")
            _log_resolve_result(scope=scope, handle=handle, result=result)
            return result
        if handle.read_strategy == EvidenceReadStrategy.NOT_READABLE:
            result = _unresolved(handle, EvidenceResolvedStatus.NOT_READABLE, str(handle.reason_code or "not_readable"))
            _log_resolve_result(scope=scope, handle=handle, result=result)
            return result
        if handle.read_strategy == EvidenceReadStrategy.VERIFY_BEFORE_USE:
            result = _unresolved(
                handle,
                EvidenceResolvedStatus.REQUIRES_VERIFICATION,
                str(handle.reason_code or "verification_required"),
            )
            _log_resolve_result(scope=scope, handle=handle, result=result)
            return result
        if handle.read_strategy == EvidenceReadStrategy.USE_DIGEST_SUMMARY:
            result = self._resolve_digest_summary(handle)
            _log_resolve_result(scope=scope, handle=handle, result=result)
            return result
        async with self._uow_factory() as uow:
            if handle.read_strategy == EvidenceReadStrategy.READ_FACT_PAYLOAD:
                result = await self._resolve_fact_payload(uow=uow, scope=scope, handle=handle)
            elif handle.read_strategy == EvidenceReadStrategy.READ_ARTIFACT:
                result = await self._resolve_artifact(uow=uow, scope=scope, handle=handle)
            elif handle.read_strategy == EvidenceReadStrategy.READ_DOCUMENT_SOURCE:
                result = await self._resolve_document_source(uow=uow, scope=scope, handle=handle)
            else:
                result = _unresolved(handle, EvidenceResolvedStatus.NOT_READABLE, "read_strategy_not_supported")
        _log_resolve_result(scope=scope, handle=handle, result=result)
        return result

    @staticmethod
    def _resolve_digest_summary(handle: EvidenceResultHandle) -> EvidenceResolvedResult:
        if not str(handle.payload_hash or handle.content_hash or "").strip():
            return _unresolved(handle, EvidenceResolvedStatus.NOT_READABLE, "result_digest_hash_missing")
        return _resolved(
            handle,
            payload={
                "summary": handle.summary,
                "ref_id": handle.ref_id,
                "result_ref_type": handle.result_ref_type.value,
            },
            payload_hash=handle.payload_hash,
            content_hash=handle.content_hash,
        )

    async def _resolve_fact_payload(
            self,
            *,
            uow,
            scope: AccessScopeResult,
            handle: EvidenceResultHandle,
    ) -> EvidenceResolvedResult:
        if not handle.source_fact_id:
            return _unresolved(handle, EvidenceResolvedStatus.MISSING, "source_fact_id_missing")
        if not str(handle.payload_hash or "").strip():
            return _unresolved(handle, EvidenceResolvedStatus.NOT_READABLE, "fact_payload_hash_missing")
        facts = await uow.sandbox_fact.list_by_ids(
            user_id=scope.user_id,
            session_id=str(scope.session_id),
            fact_ids=[handle.source_fact_id],
            limit=1,
        )
        if not facts:
            return _unresolved(handle, EvidenceResolvedStatus.MISSING, "source_fact_missing")
        fact = facts[0]
        scope_status = _validate_scoped_record(
            scope=scope,
            session_id=fact.session_id,
            workspace_id=fact.workspace_id,
            run_id=fact.run_id,
        )
        if scope_status is not None:
            return _unresolved(handle, EvidenceResolvedStatus.SCOPE_MISMATCH, scope_status)
        if fact.payload_hash != handle.payload_hash:
            return _unresolved(handle, EvidenceResolvedStatus.STALE, "fact_payload_hash_mismatch")
        return _resolved(
            handle,
            payload={
                "fact_id": fact.id,
                "fact_kind": fact.fact_kind.value,
                "summary": fact.summary,
                "payload": _safe_payload(fact.payload),
            },
            payload_hash=fact.payload_hash,
            content_hash=handle.content_hash,
        )

    async def _resolve_artifact(
            self,
            *,
            uow,
            scope: AccessScopeResult,
            handle: EvidenceResultHandle,
    ) -> EvidenceResolvedResult:
        if not handle.artifact_id:
            return _unresolved(handle, EvidenceResolvedStatus.MISSING, "artifact_id_missing")
        artifact = await uow.workspace_artifact.get_by_user_workspace_id_and_id(
            user_id=scope.user_id,
            workspace_id=str(scope.workspace_id),
            artifact_id=handle.artifact_id,
        )
        if artifact is None:
            return _unresolved(handle, EvidenceResolvedStatus.MISSING, "artifact_missing")
        scope_status = _validate_scoped_record(
            scope=scope,
            session_id=artifact.session_id,
            workspace_id=artifact.workspace_id,
            run_id=artifact.run_id,
        )
        if scope_status is not None:
            return _unresolved(handle, EvidenceResolvedStatus.SCOPE_MISMATCH, scope_status)
        if handle.source_step_id and artifact.source_step_id != handle.source_step_id:
            return _unresolved(handle, EvidenceResolvedStatus.SCOPE_MISMATCH, "artifact_source_step_mismatch")
        current_hash = _artifact_current_hash(artifact.metadata)
        if not current_hash:
            return _unresolved(handle, EvidenceResolvedStatus.NOT_READABLE, "artifact_hash_missing")
        expected_hash = handle.content_hash or handle.payload_hash
        if expected_hash and current_hash != expected_hash:
            return _unresolved(handle, EvidenceResolvedStatus.STALE, "artifact_hash_changed")
        return _resolved(
            handle,
            payload={
                "artifact_id": artifact.id,
                "path": artifact.path,
                "artifact_type": artifact.artifact_type,
                "summary": artifact.summary,
                "source_step_id": artifact.source_step_id,
                "delivery_state": artifact.delivery_state,
                "metadata": _safe_artifact_metadata(artifact.metadata),
            },
            payload_hash=handle.payload_hash,
            content_hash=current_hash,
        )

    async def _resolve_document_source(
            self,
            *,
            uow,
            scope: AccessScopeResult,
            handle: EvidenceResultHandle,
    ) -> EvidenceResolvedResult:
        evidence = await self._load_source_evidence(uow=uow, scope=scope, handle=handle)
        if evidence is not None:
            if evidence.evidence_kind != EvidenceKind.DOCUMENT_EVIDENCE:
                return _unresolved(handle, EvidenceResolvedStatus.NOT_READABLE, "document_evidence_missing")
            payload = DocumentEvidencePayload.model_validate(evidence.payload).model_dump(mode="json")
            return _resolve_document_payload(handle=handle, payload=payload)
        if not handle.source_fact_id:
            return _unresolved(handle, EvidenceResolvedStatus.MISSING, "document_source_missing")
        facts = await uow.sandbox_fact.list_by_ids(
            user_id=scope.user_id,
            session_id=str(scope.session_id),
            fact_ids=[handle.source_fact_id],
            limit=1,
        )
        if not facts:
            return _unresolved(handle, EvidenceResolvedStatus.MISSING, "document_source_missing")
        fact = facts[0]
        if fact.fact_kind != SandboxFactKind.DOCUMENT_CONTEXT:
            return _unresolved(handle, EvidenceResolvedStatus.NOT_READABLE, "document_context_fact_missing")
        scope_status = _validate_scoped_record(
            scope=scope,
            session_id=fact.session_id,
            workspace_id=fact.workspace_id,
            run_id=fact.run_id,
        )
        if scope_status is not None:
            return _unresolved(handle, EvidenceResolvedStatus.SCOPE_MISMATCH, scope_status)
        return _resolve_document_payload(handle=handle, payload=_safe_payload(fact.payload), payload_hash=fact.payload_hash)

    async def _load_source_evidence(
            self,
            *,
            uow,
            scope: AccessScopeResult,
            handle: EvidenceResultHandle,
    ):
        if not handle.source_evidence_id:
            return None
        records = await uow.evidence.list_by_ids(
            user_id=scope.user_id,
            session_id=str(scope.session_id),
            evidence_ids=[handle.source_evidence_id],
            limit=1,
        )
        if not records:
            return None
        evidence = records[0]
        scope_status = _validate_scoped_record(
            scope=scope,
            session_id=evidence.session_id,
            workspace_id=evidence.workspace_id,
            run_id=evidence.run_id,
        )
        if scope_status is not None:
            return None
        return evidence


def _resolve_document_payload(
        *,
        handle: EvidenceResultHandle,
        payload: dict[str, Any],
        payload_hash: str | None = None,
) -> EvidenceResolvedResult:
    if handle.document_file_id and str(payload.get("file_id") or "") != handle.document_file_id:
        return _unresolved(handle, EvidenceResolvedStatus.SCOPE_MISMATCH, "document_file_id_mismatch")
    actual_hash = str(payload.get("read_content_sha256") or payload.get("full_file_sha256") or "").strip()
    expected_hash = str(handle.content_hash or handle.payload_hash or "").strip()
    if not expected_hash:
        return _unresolved(handle, EvidenceResolvedStatus.NOT_READABLE, "document_source_hash_missing")
    if actual_hash and expected_hash != actual_hash and expected_hash != payload_hash:
        return _unresolved(handle, EvidenceResolvedStatus.STALE, "document_source_hash_mismatch")
    resolved_payload = {
        "file_id": payload.get("file_id"),
        "parse_status": payload.get("parse_status"),
        "reason_code": payload.get("reason_code"),
        "full_file_sha256": payload.get("full_file_sha256"),
        "read_content_sha256": payload.get("read_content_sha256"),
        "is_truncated": payload.get("is_truncated"),
        "excerpt_char_count": payload.get("excerpt_char_count"),
    }
    return _resolved(
        handle,
        payload=resolved_payload,
        payload_hash=payload_hash or handle.payload_hash,
        content_hash=actual_hash or handle.content_hash,
    )


def _resolved(
        handle: EvidenceResultHandle,
        *,
        payload: dict[str, Any],
        payload_hash: str | None,
        content_hash: str | None,
) -> EvidenceResolvedResult:
    return EvidenceResolvedResult(
        status=EvidenceResolvedStatus.RESOLVED,
        result_ref_type=handle.result_ref_type,
        source_evidence_id=handle.source_evidence_id,
        source_fact_id=handle.source_fact_id,
        source_event_id=handle.source_event_id,
        artifact_id=handle.artifact_id,
        document_file_id=handle.document_file_id,
        subject_key=handle.subject_key,
        read_strategy=handle.read_strategy,
        summary=handle.summary,
        resolved_payload=payload,
        payload_hash=payload_hash,
        content_hash=content_hash,
    )


def _unresolved(
        handle: EvidenceResultHandle,
        status: EvidenceResolvedStatus,
        reason_code: str,
) -> EvidenceResolvedResult:
    return EvidenceResolvedResult(
        status=status,
        result_ref_type=handle.result_ref_type,
        source_evidence_id=handle.source_evidence_id,
        source_fact_id=handle.source_fact_id,
        source_event_id=handle.source_event_id,
        artifact_id=handle.artifact_id,
        document_file_id=handle.document_file_id,
        subject_key=handle.subject_key,
        read_strategy=handle.read_strategy,
        summary=handle.summary,
        reason_code=reason_code,
        allowed_verification_actions=handle.allowed_verification_actions,
    )


def _log_resolve_result(
        *,
        scope: AccessScopeResult,
        handle: EvidenceResultHandle,
        result: EvidenceResolvedResult,
) -> None:
    event_name = "evidence_result_handle_resolve_failed"
    if result.status == EvidenceResolvedStatus.RESOLVED:
        event_name = "evidence_result_handle_resolved"
    elif result.status == EvidenceResolvedStatus.STALE:
        event_name = "evidence_result_handle_stale"
    logger.info(
        "%s user_id=%s session_id=%s run_id=%s result_handle_id=%s read_strategy=%s status=%s reason_code=%s",
        event_name,
        scope.user_id,
        scope.session_id,
        scope.run_id,
        handle.result_handle_id,
        handle.read_strategy.value,
        result.status.value,
        result.reason_code,
    )


def _has_required_scope(scope: AccessScopeResult) -> bool:
    return bool(
        str(scope.user_id or "").strip()
        and str(scope.session_id or "").strip()
        and str(scope.workspace_id or "").strip()
    )


def _validate_scoped_record(
        *,
        scope: AccessScopeResult,
        session_id: str | None,
        workspace_id: str | None,
        run_id: str | None,
) -> str | None:
    if session_id != scope.session_id:
        return "session_scope_mismatch"
    if workspace_id and workspace_id != scope.workspace_id:
        return "workspace_scope_mismatch"
    if scope.run_id and run_id != scope.run_id:
        return "run_scope_mismatch"
    return None


def _artifact_current_hash(metadata: dict[str, Any]) -> str | None:
    for key in ("content_hash", "sha256", "current_hash"):
        value = str((metadata or {}).get(key) or "").strip()
        if value:
            return value
    return None


def _safe_artifact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(metadata or {}).items()
        if str(key) in {"content_hash", "sha256", "current_hash", "size", "mime_type"}
    }


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    blocked_keys = {"raw_stdout", "stdout", "full_text", "file_content", "page_content", "document_text", "html"}
    if isinstance(payload, dict):
        return {
            str(key): _safe_payload(value) if isinstance(value, dict) else value
            for key, value in payload.items()
            if str(key) not in blocked_keys
        }
    return {}
