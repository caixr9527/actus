import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from app.domain.models.evidence import (
    ActionEvidencePayload,
    DocumentEvidencePayload,
    EvidenceDigestResult,
    EvidenceDoNotRepeatResult,
    EvidenceDuplicateDecision,
    EvidenceGapPayload,
    EvidenceKind,
    EvidenceQualityStatus,
    EvidenceReadStrategy,
    EvidenceRecord,
    EvidenceResultHandle,
    EvidenceResultRef,
    EvidenceResultRefType,
    EvidenceReusePolicy,
    EvidenceReuseSnapshot,
    EvidenceResolvedResult,
    EvidenceResolvedStatus,
    EvidenceScope,
    EvidenceSourceRef,
    EvidenceSourceType,
    EvidenceStalenessPolicy,
    EvidenceSubjectRef,
    EvidenceSupportLevel,
    RuntimeEvidenceContextResult,
    build_evidence_idempotency_key,
    build_evidence_payload_hash,
    build_evidence_result_handle,
    build_evidence_result_handle_id_from_parts,
    build_evidence_result_refs_hash,
    classify_evidence_data,
    validate_evidence_payload,
)
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.domain.services.runtime.contracts.evidence_key_normalizer import (
    build_evidence_action_subject_key_from_tool_call,
    hash_query,
    hash_url,
    normalize_path,
)
from app.infrastructure.models.evidence import EvidenceModel
from app.infrastructure.repositories.db_evidence_repository import (
    DBEvidenceRepository,
    DBEvidenceSupersededTargetError,
)


def _result_ref(
        *,
        ref_id: str = "fact-1",
        result_ref_type: EvidenceResultRefType = EvidenceResultRefType.FACT_REF,
        source_evidence_id: str = "evidence-1",
        source_fact_id: str = "fact-1",
        source_event_id: str = "event-1",
        artifact_id: str | None = None,
        artifact_path: str | None = None,
        document_file_id: str | None = None,
        subject_key: str = "query:abc",
        payload_hash: str = "sha256:payload",
        content_hash: str | None = None,
        read_strategy: EvidenceReadStrategy = EvidenceReadStrategy.READ_FACT_PAYLOAD,
        reason_code: str | None = None,
        allowed_verification_actions: list[str] | None = None,
        summary: str = "安全摘要",
) -> EvidenceResultRef:
    return EvidenceResultRef(
        result_ref_type=result_ref_type,
        ref_id=ref_id,
        source_step_id="step-1",
        source_evidence_id=source_evidence_id,
        source_fact_id=source_fact_id,
        source_event_id=source_event_id,
        artifact_id=artifact_id,
        artifact_path=artifact_path,
        document_file_id=document_file_id,
        subject_key=subject_key,
        payload_hash=payload_hash,
        content_hash=content_hash,
        quality_status=EvidenceQualityStatus.VALID,
        support_level=EvidenceSupportLevel.STRONG,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        read_strategy=read_strategy,
        reason_code=reason_code,
        allowed_verification_actions=allowed_verification_actions or [],
        summary=summary,
    )


def _evidence(
        *,
        evidence_id: str = "evidence-1",
        user_id: str = "user-1",
        session_id: str = "session-1",
        workspace_id: str = "workspace-1",
        run_id: str | None = "run-1",
        step_id: str | None = "step-1",
        evidence_scope: EvidenceScope = EvidenceScope.STEP,
        evidence_kind: EvidenceKind = EvidenceKind.ACTION_EVIDENCE,
        payload: dict | None = None,
        result_refs: list[EvidenceResultRef] | None = None,
        source_event_id: str | None = "event-1",
        source_ref: EvidenceSourceRef | None = None,
        subject_ref: EvidenceSubjectRef | None = None,
        reusable: bool = True,
        supersedes_evidence_id: str | None = None,
) -> EvidenceRecord:
    actual_payload = validate_evidence_payload(
        evidence_kind=evidence_kind,
        payload=payload or _payload_for_kind(evidence_kind, source_event_id=source_event_id, supersedes_evidence_id=supersedes_evidence_id),
    ).model_dump(mode="json")
    payload_hash = build_evidence_payload_hash(actual_payload)
    actual_result_refs = list(result_refs if result_refs is not None else [_result_ref(source_evidence_id=evidence_id)])
    result_refs_hash = build_evidence_result_refs_hash(actual_result_refs)
    actual_source_ref = source_ref or EvidenceSourceRef(
        source_type=EvidenceSourceType.SANDBOX_FACT,
        source_event_id=source_event_id,
        fact_ids=["fact-1"],
        tool_call_id="tool-call-1",
        profile_hash="sha256:" + "a" * 64,
    )
    actual_subject_ref = subject_ref or EvidenceSubjectRef(
        subject_type="search",
        subject_key="query:abc",
    )
    idempotency_key = build_evidence_idempotency_key(
        user_id=user_id,
        session_id=session_id,
        run_id=run_id,
        step_id=step_id,
        evidence_scope=evidence_scope,
        evidence_kind=evidence_kind,
        source_event_id=actual_source_ref.source_event_id,
        primary_fact_id=(actual_source_ref.fact_ids or [None])[0],
        primary_artifact_id=(actual_source_ref.artifact_ids or [None])[0],
        action_key="search:abc",
        claim_key=None,
        payload_hash=payload_hash,
        result_refs_hash=result_refs_hash,
    )
    origin, trust_level, privacy_level, retention_policy = classify_evidence_data(
        evidence_kind=evidence_kind,
        source_type=actual_source_ref.source_type,
    )
    return EvidenceRecord(
        id=evidence_id,
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        run_id=run_id,
        step_id=step_id,
        evidence_scope=evidence_scope,
        evidence_kind=evidence_kind,
        action_key="search:abc",
        subject_key="query:abc",
        source_step_id=step_id if evidence_scope == EvidenceScope.STEP else None,
        support_level=EvidenceSupportLevel.STRONG if evidence_kind != EvidenceKind.EVIDENCE_GAP else EvidenceSupportLevel.GAP,
        quality_status=EvidenceQualityStatus.VALID if evidence_kind != EvidenceKind.EVIDENCE_GAP else EvidenceQualityStatus.MISSING_SOURCE,
        source_ref=actual_source_ref,
        subject_ref=actual_subject_ref,
        summary="记录了搜索动作",
        payload=actual_payload,
        payload_hash=payload_hash,
        idempotency_key=idempotency_key,
        confidence=0.9,
        reusable=reusable,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED if reusable else EvidenceReusePolicy.DO_NOT_REUSE,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        origin=origin,
        trust_level=trust_level,
        privacy_level=privacy_level,
        retention_policy=retention_policy,
        result_refs=actual_result_refs,
        result_refs_hash=result_refs_hash,
        supersedes_evidence_id=supersedes_evidence_id,
        created_at=datetime(2026, 5, 7, 10, 0, 0),
    )


def _payload_for_kind(
        evidence_kind: EvidenceKind,
        *,
        source_event_id: str | None = "event-1",
        supersedes_evidence_id: str | None = None,
) -> dict:
    payloads: dict[EvidenceKind, dict] = {
        EvidenceKind.ACTION_EVIDENCE: {
            "action_type": "search",
            "function_name": "search",
            "source_fact_ids": ["fact-1"],
            "source_event_id": source_event_id or "event-1",
            "result_status": "success",
            "reason_code": None,
            "excerpt": "结果摘要",
            "is_truncated": False,
        },
        EvidenceKind.CLAIM_SUPPORT: {
            "claim_text": "结论 A",
            "supporting_fact_ids": ["fact-1"],
            "supporting_artifact_ids": [],
            "support_level": "strong",
            "quality_status": "valid",
            "source_excerpt": "摘录",
            "limitations": [],
        },
        EvidenceKind.ARTIFACT_EVIDENCE: {
            "artifact_id": "artifact-1",
            "artifact_path": "/artifacts/a.txt",
            "artifact_type": "text",
            "source_fact_ids": ["fact-1"],
            "current_hash": "sha256:content",
            "hash_kind": "content_sha256",
            "delivery_candidate": True,
            "version_locked": False,
            "reason_code": "artifact_revision_not_available",
        },
        EvidenceKind.DOCUMENT_EVIDENCE: {
            "file_id": "file-1",
            "parse_status": "parsed",
            "reason_code": None,
            "full_file_sha256": "sha256:full",
            "read_content_sha256": "sha256:read",
            "is_truncated": False,
            "excerpt_char_count": 12,
            "source_fact_id": "fact-1",
        },
        EvidenceKind.SEARCH_EVIDENCE: {
            "query_excerpt": "query",
            "result_count": 1,
            "top_result_origins": ["https://example.com"],
            "source_fact_id": "fact-1",
            "snippet_quality": "usable",
            "needs_fetch": True,
        },
        EvidenceKind.PAGE_EVIDENCE: {
            "origin": "https://example.com",
            "title": "Example",
            "status_code": 200,
            "source_fact_id": "fact-1",
            "excerpt": "page",
            "is_truncated": False,
        },
        EvidenceKind.FILE_EVIDENCE: {
            "path": "/workspace/a.txt",
            "operation": "read",
            "mutation_intent_hash": None,
            "exists": True,
            "content_sha256": "sha256:file",
            "content_sha256_kind": "read_content_sha256",
            "source_fact_id": "fact-1",
            "excerpt": "file",
            "is_truncated": False,
        },
        EvidenceKind.BROWSER_EVIDENCE: {
            "action": "snapshot",
            "origin": "https://example.com",
            "title": "Example",
            "screenshot_artifact_id": "artifact-1",
            "source_fact_id": "fact-1",
            "excerpt": "browser",
            "is_truncated": False,
            "reason_code": None,
        },
        EvidenceKind.TOOL_FAILURE_EVIDENCE: {
            "function_name": "search",
            "reason_code": "tool_failed",
            "source_fact_id": "fact-1",
            "message_excerpt": "failed",
            "retry_count": 1,
            "timeout": False,
        },
        EvidenceKind.HUMAN_CONFIRMATION_EVIDENCE: {
            "interaction_id": "event-1",
            "interaction_type": "confirmation",
            "message_event_id": "event-1",
            "confirmed": True,
            "summary": "用户确认",
            "reason_code": None,
        },
        EvidenceKind.EVIDENCE_GAP: {
            "gap_type": "missing_source",
            "missing_source_types": ["sandbox_fact"],
            "claim_text": "缺少证据",
            "reason_code": "source_missing",
            "required_for": "step_completion",
        },
        EvidenceKind.CORRECTION: {
            "corrected_evidence_ids": [supersedes_evidence_id or "evidence-1"],
            "reason_code": "payload_corrected",
            "message_excerpt": "修正",
            "supersedes_evidence_id": supersedes_evidence_id or "evidence-1",
        },
        EvidenceKind.SUPERSEDED: {
            "supersedes_evidence_id": supersedes_evidence_id or "evidence-1",
            "reason_code": "replaced",
            "message_excerpt": "废弃",
        },
    }
    return payloads[evidence_kind]


@pytest.mark.parametrize("evidence_kind", list(EvidenceKind))
def test_evidence_payload_schema_should_be_strict_for_all_kinds(evidence_kind: EvidenceKind) -> None:
    valid_payload = _payload_for_kind(evidence_kind)
    parsed = validate_evidence_payload(evidence_kind=evidence_kind, payload=valid_payload)

    if evidence_kind == EvidenceKind.ACTION_EVIDENCE:
        assert isinstance(parsed, ActionEvidencePayload)
    with pytest.raises(ValidationError):
        validate_evidence_payload(
            evidence_kind=evidence_kind,
            payload={**valid_payload, "extra": "no"},
        )


def test_document_evidence_parse_status_should_use_p0_5_enum() -> None:
    valid = DocumentEvidencePayload.model_validate(
        {
            "file_id": "file-1",
            "parse_status": "parsed",
            "reason_code": None,
            "full_file_sha256": "sha256:full",
            "read_content_sha256": "sha256:read",
            "is_truncated": False,
            "excerpt_char_count": 12,
            "source_fact_id": "fact-1",
        }
    )

    assert valid.parse_status.value == "parsed"
    with pytest.raises(ValidationError):
        DocumentEvidencePayload.model_validate(
            {
                "file_id": "file-1",
                "parse_status": "success",
                "reason_code": None,
                "full_file_sha256": "sha256:full",
                "read_content_sha256": "sha256:read",
                "is_truncated": False,
                "excerpt_char_count": 12,
                "source_fact_id": "fact-1",
            }
        )


def test_evidence_record_should_validate_scope_and_source_invariants() -> None:
    assert _evidence().evidence_scope == EvidenceScope.STEP

    with pytest.raises(ValidationError):
        _evidence(step_id=None)

    with pytest.raises(ValidationError):
        _evidence(
            source_event_id=None,
            source_ref=EvidenceSourceRef(source_type=EvidenceSourceType.SANDBOX_FACT, fact_ids=["fact-1"]),
        )

    gap = _evidence(
        evidence_kind=EvidenceKind.EVIDENCE_GAP,
        source_event_id=None,
        source_ref=EvidenceSourceRef(source_type=EvidenceSourceType.SYSTEM_PROJECTION),
        result_refs=[],
        reusable=False,
    )
    assert gap.payload["reason_code"] == "source_missing"


def test_workspace_scope_should_not_bind_run_or_step() -> None:
    workspace_evidence = _evidence(
        evidence_scope=EvidenceScope.WORKSPACE,
        run_id=None,
        step_id=None,
        source_ref=EvidenceSourceRef(source_type=EvidenceSourceType.WORKFLOW_EVENT, source_event_id="event-1"),
        reusable=False,
        result_refs=[],
    )
    assert workspace_evidence.evidence_scope == EvidenceScope.WORKSPACE

    with pytest.raises(ValidationError):
        _evidence(evidence_scope=EvidenceScope.WORKSPACE, run_id="run-1", step_id=None, reusable=False, result_refs=[])


def test_evidence_data_classification_should_use_existing_enums() -> None:
    human = classify_evidence_data(
        evidence_kind=EvidenceKind.HUMAN_CONFIRMATION_EVIDENCE,
        source_type=EvidenceSourceType.USER_CONFIRMATION,
    )
    gap = classify_evidence_data(
        evidence_kind=EvidenceKind.EVIDENCE_GAP,
        source_type=EvidenceSourceType.SYSTEM_PROJECTION,
    )

    assert human[0] == DataOrigin.USER_MESSAGE
    assert human[1] == DataTrustLevel.USER_PROVIDED
    assert human[3] == RetentionPolicyKind.SESSION_BOUND
    assert gap == (
        DataOrigin.SYSTEM_OPERATIONAL,
        DataTrustLevel.SYSTEM_GENERATED,
        PrivacyLevel.INTERNAL,
        RetentionPolicyKind.SESSION_BOUND,
    )


def test_result_ref_hash_should_ignore_order_and_summary_but_include_required_fields() -> None:
    first = _result_ref(ref_id="fact-1", summary="摘要 A")
    second = _result_ref(ref_id="fact-2", source_fact_id="fact-2", summary="摘要 B")
    same_without_summary = _result_ref(ref_id="fact-1", summary="不同摘要")
    changed_strategy = _result_ref(ref_id="fact-1", read_strategy=EvidenceReadStrategy.USE_DIGEST_SUMMARY)

    assert build_evidence_result_refs_hash([first, second]) == build_evidence_result_refs_hash([second, first])
    assert build_evidence_result_refs_hash([first]) == build_evidence_result_refs_hash([same_without_summary])
    assert build_evidence_result_refs_hash([first]) != build_evidence_result_refs_hash([changed_strategy])


def test_idempotency_should_include_result_refs_hash() -> None:
    evidence = _evidence()
    changed_ref = _result_ref(ref_id="fact-2", source_fact_id="fact-2")
    changed_result_refs_hash = build_evidence_result_refs_hash([changed_ref])
    changed_key = build_evidence_idempotency_key(
        user_id=evidence.user_id,
        session_id=evidence.session_id,
        run_id=evidence.run_id,
        step_id=evidence.step_id,
        evidence_scope=evidence.evidence_scope,
        evidence_kind=evidence.evidence_kind,
        source_event_id=evidence.source_event_id,
        primary_fact_id=evidence.primary_fact_id,
        primary_artifact_id=evidence.primary_artifact_id,
        action_key=evidence.action_key,
        claim_key=evidence.claim_key,
        payload_hash=evidence.payload_hash,
        result_refs_hash=changed_result_refs_hash,
    )

    assert changed_key != evidence.idempotency_key


def test_hash_helpers_should_use_json_array_boundaries_for_newline_values() -> None:
    base_kwargs = {
        "user_id": "user\n1",
        "session_id": "session",
        "run_id": "run",
        "step_id": "step",
        "evidence_scope": EvidenceScope.STEP,
        "evidence_kind": EvidenceKind.ACTION_EVIDENCE,
        "source_event_id": "event",
        "primary_fact_id": "fact",
        "primary_artifact_id": "",
        "action_key": "a",
        "claim_key": "b\nc",
        "payload_hash": "sha256:payload",
        "result_refs_hash": "sha256:refs",
    }
    shifted_kwargs = {**base_kwargs, "action_key": "a\nb", "claim_key": "c"}
    base_handle_kwargs = {
        "result_ref_type": EvidenceResultRefType.FACT_REF,
        "ref_id": "ref\n1",
        "source_evidence_id": "evidence",
        "source_fact_id": "fact",
        "source_event_id": "event",
        "artifact_id": "",
        "document_file_id": "",
        "subject_key": "subject",
        "payload_hash": "sha256:p",
        "content_hash": "sha256:c",
        "read_strategy": EvidenceReadStrategy.READ_FACT_PAYLOAD,
    }
    shifted_handle_kwargs = {**base_handle_kwargs, "source_evidence_id": "evidence\nfact", "source_fact_id": ""}

    assert build_evidence_idempotency_key(**base_kwargs) != build_evidence_idempotency_key(**shifted_kwargs)
    assert build_evidence_result_handle_id_from_parts(**base_handle_kwargs) != build_evidence_result_handle_id_from_parts(**shifted_handle_kwargs)


def test_result_handle_id_should_be_stable_and_use_result_handle_id_as_index_key() -> None:
    ref = _result_ref(content_hash="sha256:content")
    handle = build_evidence_result_handle(ref)
    same_handle = build_evidence_result_handle(ref)
    changed_handle_id = build_evidence_result_handle_id_from_parts(
        result_ref_type=ref.result_ref_type,
        ref_id=ref.ref_id,
        source_evidence_id="different-evidence",
        source_fact_id=ref.source_fact_id,
        source_event_id=ref.source_event_id,
        artifact_id=ref.artifact_id,
        document_file_id=ref.document_file_id,
        subject_key=ref.subject_key,
        payload_hash=ref.payload_hash,
        content_hash=ref.content_hash,
        read_strategy=ref.read_strategy,
    )

    assert handle.result_handle_id == same_handle.result_handle_id
    assert changed_handle_id != handle.result_handle_id
    with pytest.raises(ValidationError):
        EvidenceResultHandle.model_validate({**handle.model_dump(mode="json"), "result_handle_id": "ref-id"})


@pytest.mark.parametrize(
    ("result_ref_type", "valid_kwargs", "invalid_kwargs"),
    [
        (
            EvidenceResultRefType.ARTIFACT_REF,
            {"ref_id": "artifact-1", "artifact_id": "artifact-1", "content_hash": "sha256:content", "read_strategy": EvidenceReadStrategy.READ_ARTIFACT},
            {"ref_id": "artifact-1", "artifact_id": None, "content_hash": None, "read_strategy": EvidenceReadStrategy.READ_ARTIFACT},
        ),
        (
            EvidenceResultRefType.FACT_REF,
            {"ref_id": "fact-1", "source_fact_id": "fact-1", "read_strategy": EvidenceReadStrategy.READ_FACT_PAYLOAD},
            {"ref_id": "fact-1", "source_fact_id": None, "read_strategy": EvidenceReadStrategy.READ_FACT_PAYLOAD},
        ),
        (
            EvidenceResultRefType.SOURCE_EVENT_REF,
            {"ref_id": "event-1", "source_event_id": "event-1", "read_strategy": EvidenceReadStrategy.USE_DIGEST_SUMMARY},
            {"ref_id": "event-1", "source_event_id": None, "read_strategy": EvidenceReadStrategy.USE_DIGEST_SUMMARY},
        ),
        (
            EvidenceResultRefType.DOCUMENT_SOURCE_REF,
            {"ref_id": "file-1", "document_file_id": "file-1", "read_strategy": EvidenceReadStrategy.READ_DOCUMENT_SOURCE},
            {"ref_id": "file-1", "document_file_id": None, "read_strategy": EvidenceReadStrategy.READ_DOCUMENT_SOURCE},
        ),
        (
            EvidenceResultRefType.USER_CONFIRMATION_REF,
            {"ref_id": "event-1", "source_event_id": "event-1", "read_strategy": EvidenceReadStrategy.USE_DIGEST_SUMMARY},
            {"ref_id": "event-1", "source_event_id": None, "read_strategy": EvidenceReadStrategy.USE_DIGEST_SUMMARY},
        ),
        (
            EvidenceResultRefType.VERIFICATION_REF,
            {"ref_id": "verify-1", "reason_code": "query_external_may_change", "read_strategy": EvidenceReadStrategy.VERIFY_BEFORE_USE, "allowed_verification_actions": ["verification_search"]},
            {"ref_id": "verify-1", "reason_code": None, "read_strategy": EvidenceReadStrategy.VERIFY_BEFORE_USE, "allowed_verification_actions": []},
        ),
    ],
)
def test_result_ref_type_should_require_recoverable_fields(
        result_ref_type: EvidenceResultRefType,
        valid_kwargs: dict,
        invalid_kwargs: dict,
) -> None:
    valid = _result_ref(result_ref_type=result_ref_type, payload_hash=None, **valid_kwargs)

    assert valid.result_ref_type == result_ref_type
    with pytest.raises(ValidationError):
        _result_ref(result_ref_type=result_ref_type, payload_hash=None, **invalid_kwargs)


def test_not_readable_structures_should_require_reason_code() -> None:
    with pytest.raises(ValidationError):
        _result_ref(read_strategy=EvidenceReadStrategy.NOT_READABLE)

    readable_ref = _result_ref(read_strategy=EvidenceReadStrategy.NOT_READABLE, reason_code="source_missing")
    handle = build_evidence_result_handle(readable_ref)
    assert handle.reason_code == "source_missing"

    with pytest.raises(ValidationError):
        EvidenceResultHandle.model_validate({**handle.model_dump(mode="json"), "reason_code": None})
    with pytest.raises(ValidationError):
        EvidenceResolvedResult(
            status=EvidenceResolvedStatus.NOT_READABLE,
            result_ref_type=EvidenceResultRefType.FACT_REF,
            source_fact_id="fact-1",
            read_strategy=EvidenceReadStrategy.NOT_READABLE,
        )


def test_resolved_result_should_require_auditable_payload_hash() -> None:
    valid = EvidenceResolvedResult(
        status=EvidenceResolvedStatus.RESOLVED,
        result_ref_type=EvidenceResultRefType.FACT_REF,
        source_fact_id="fact-1",
        read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
        resolved_payload={"summary": "可审计摘要"},
        payload_hash="sha256:payload",
    )

    assert valid.payload_hash == "sha256:payload"
    with pytest.raises(ValidationError):
        EvidenceResolvedResult(
            status=EvidenceResolvedStatus.RESOLVED,
            result_ref_type=EvidenceResultRefType.FACT_REF,
            source_fact_id="fact-1",
            read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
            resolved_payload={"summary": "缺少 hash"},
        )


def test_unresolved_result_should_require_reason_code() -> None:
    with pytest.raises(ValidationError):
        EvidenceResolvedResult(
            status=EvidenceResolvedStatus.REQUIRES_VERIFICATION,
            result_ref_type=EvidenceResultRefType.FACT_REF,
            source_fact_id="fact-1",
            read_strategy=EvidenceReadStrategy.VERIFY_BEFORE_USE,
        )

    result = EvidenceResolvedResult(
        status=EvidenceResolvedStatus.REQUIRES_VERIFICATION,
        result_ref_type=EvidenceResultRefType.FACT_REF,
        source_fact_id="fact-1",
        read_strategy=EvidenceReadStrategy.VERIFY_BEFORE_USE,
        reason_code="query_external_may_change",
    )
    assert result.reason_code == "query_external_may_change"


@pytest.mark.parametrize(
    "resolved_payload",
    [
        {"raw_stdout": "secret"},
        {"nested": {"html": "<html>raw</html>"}},
        {"items": [{"file_content": "raw"}]},
    ],
)
def test_resolved_result_should_reject_raw_payload_keys(resolved_payload: dict) -> None:
    with pytest.raises(ValidationError):
        EvidenceResolvedResult(
            status=EvidenceResolvedStatus.RESOLVED,
            result_ref_type=EvidenceResultRefType.FACT_REF,
            source_fact_id="fact-1",
            read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
            resolved_payload=resolved_payload,
            payload_hash="sha256:payload",
        )


def test_runtime_evidence_context_should_reject_missing_snapshot_and_bad_index_key() -> None:
    ref = _result_ref()
    handle = build_evidence_result_handle(ref)
    do_not_repeat = EvidenceDoNotRepeatResult(
        action_key="search:abc",
        subject_key="query:abc",
        reason_code="evidence_reuse_pending_resolution",
        source_step_id="step-1",
        evidence_ids=["evidence-1"],
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        support_level=EvidenceSupportLevel.STRONG,
        quality_status=EvidenceQualityStatus.VALID,
        result_status="success",
        duplicate_decision=EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION,
        reuse_result_ref=ref,
        result_handle_id=handle.result_handle_id,
        reuse_summary="复用摘要",
        message_for_model="已有结果可复用",
    )
    snapshot = EvidenceReuseSnapshot(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        cursor="cursor-1",
        do_not_repeat=[do_not_repeat],
        result_handles=[handle],
    )

    context = RuntimeEvidenceContextResult(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        has_previous_completed_steps=True,
        prompt_digest="摘要",
        evidence_reuse_snapshot=snapshot,
        result_handles=[handle],
        result_handle_index={handle.result_handle_id: handle},
        cursor="cursor-1",
    )
    digest = EvidenceDigestResult(run_id="run-1", current_step_id="step-2", summary_for_prompt="摘要", cursor="cursor-1")

    assert context.result_handle_index[handle.result_handle_id] == handle
    assert digest.summary_for_prompt == "摘要"
    with pytest.raises(ValidationError):
        RuntimeEvidenceContextResult(
            run_id="run-1",
            current_step_id="step-2",
            has_previous_completed_steps=True,
            cursor="cursor-1",
        )
    with pytest.raises(ValidationError):
        RuntimeEvidenceContextResult(
            run_id="run-1",
            current_step_id="step-2",
            has_previous_completed_steps=True,
            evidence_reuse_snapshot=snapshot,
            result_handles=[handle],
            result_handle_index={handle.ref_id: handle},
            cursor="cursor-1",
        )


def test_runtime_evidence_context_should_match_snapshot_handle_set() -> None:
    handle = build_evidence_result_handle(_result_ref())
    extra_handle = build_evidence_result_handle(
        _result_ref(
            ref_id="fact-2",
            source_evidence_id="evidence-2",
            source_fact_id="fact-2",
            payload_hash="sha256:payload-2",
        )
    )
    snapshot = EvidenceReuseSnapshot(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        cursor="cursor-1",
        result_handles=[handle],
    )

    with pytest.raises(ValidationError):
        RuntimeEvidenceContextResult(
            run_id="run-1",
            current_step_id="step-2",
            source_step_ids=["step-1"],
            has_previous_completed_steps=True,
            evidence_reuse_snapshot=snapshot,
            result_handles=[],
            result_handle_index={},
            cursor="cursor-1",
        )
    with pytest.raises(ValidationError):
        RuntimeEvidenceContextResult(
            run_id="run-1",
            current_step_id="step-2",
            source_step_ids=["step-1"],
            has_previous_completed_steps=True,
            evidence_reuse_snapshot=snapshot,
            result_handles=[handle, extra_handle],
            result_handle_index={
                handle.result_handle_id: handle,
                extra_handle.result_handle_id: extra_handle,
            },
            cursor="cursor-1",
        )


def test_evidence_reuse_snapshot_should_reject_dangling_or_incomplete_handle_refs() -> None:
    ref = _result_ref()
    handle = build_evidence_result_handle(ref)
    valid_item = EvidenceDoNotRepeatResult(
        action_key="search:abc",
        subject_key="query:abc",
        reason_code="evidence_reuse_pending_resolution",
        source_step_id="step-1",
        evidence_ids=["evidence-1"],
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        support_level=EvidenceSupportLevel.STRONG,
        quality_status=EvidenceQualityStatus.VALID,
        result_status="success",
        duplicate_decision=EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION,
        reuse_result_ref=ref,
        result_handle_id=handle.result_handle_id,
    )

    snapshot = EvidenceReuseSnapshot(run_id="run-1", cursor="cursor-1", do_not_repeat=[valid_item], result_handles=[handle])
    assert snapshot.do_not_repeat[0].result_handle_id == handle.result_handle_id

    with pytest.raises(ValidationError):
        EvidenceReuseSnapshot(
            run_id="run-1",
            cursor="cursor-1",
            do_not_repeat=[valid_item.model_copy(update={"result_handle_id": "missing-handle"})],
            result_handles=[handle],
        )
    with pytest.raises(ValidationError):
        EvidenceReuseSnapshot(
            run_id="run-1",
            cursor="cursor-1",
            do_not_repeat=[valid_item.model_copy(update={"result_handle_id": None})],
            result_handles=[handle],
        )
    with pytest.raises(ValidationError):
        EvidenceReuseSnapshot(
            run_id="run-1",
            cursor="cursor-1",
            do_not_repeat=[valid_item.model_copy(update={"reuse_result_ref": None})],
            result_handles=[handle],
        )


def test_key_normalizer_should_stabilize_query_url_and_path() -> None:
    assert hash_query(" Foo   Bar ") == hash_query("foo bar")
    assert hash_url("https://Example.com/a?b=2&a=1#frag") == hash_url("https://example.com/a?a=1&b=2")
    assert normalize_path("/workspace/./a//b/../c.txt") == "/workspace/a/c.txt"

    search_key = build_evidence_action_subject_key_from_tool_call("search", {"query": " Foo   Bar "})
    read_key = build_evidence_action_subject_key_from_tool_call("read_file", {"path": "/workspace/./a//b/../c.txt"})

    assert search_key.action_key == f"search:{hash_query('foo bar')}"
    assert read_key.subject_key == "file:/workspace/a/c.txt"


def test_evidence_model_should_round_trip_result_refs_and_enum_values() -> None:
    evidence = _evidence()
    model = EvidenceModel.from_domain(evidence)
    restored = model.to_domain()

    assert model.evidence_kind == "action_evidence"
    assert model.reuse_policy == "reuse_allowed"
    assert restored.result_refs == evidence.result_refs
    assert restored.result_refs_hash == evidence.result_refs_hash


def test_evidence_repository_should_save_once_by_idempotency_key() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: "evidence-1"))
    )
    repository = DBEvidenceRepository(db_session=db_session)

    saved = asyncio.run(repository.save_once(_evidence()))

    assert saved.id == "evidence-1"
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_evidence_records_idempotency_key DO NOTHING" in compiled_sql
    assert "evidence_records" in compiled_sql


def test_evidence_repository_duplicate_should_return_existing_record() -> None:
    existing = _evidence(evidence_id="existing-evidence")
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    existing_result = SimpleNamespace(scalar_one_or_none=lambda: EvidenceModel.from_domain(existing))
    db_session = SimpleNamespace(execute=AsyncMock(side_effect=[execute_result, existing_result]))
    repository = DBEvidenceRepository(db_session=db_session)

    saved = asyncio.run(repository.save_once(_evidence()))

    assert saved.id == "existing-evidence"


def test_evidence_repository_should_validate_superseded_target_scope() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    db_session = SimpleNamespace(execute=AsyncMock(return_value=execute_result))
    repository = DBEvidenceRepository(db_session=db_session)
    evidence = _evidence(
        evidence_id="evidence-super",
        evidence_kind=EvidenceKind.SUPERSEDED,
        payload=_payload_for_kind(EvidenceKind.SUPERSEDED, supersedes_evidence_id="missing"),
        source_ref=EvidenceSourceRef(source_type=EvidenceSourceType.SYSTEM_PROJECTION),
        result_refs=[],
        reusable=False,
        supersedes_evidence_id="missing",
    )

    with pytest.raises(DBEvidenceSupersededTargetError):
        asyncio.run(repository.save_once(evidence))


def test_evidence_queries_should_require_user_session_scope_and_strong_filters() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])))
    )
    repository = DBEvidenceRepository(db_session=db_session)

    result = asyncio.run(
        repository.list_by_step(
            user_id="user-1",
            session_id="session-1",
            run_id="run-1",
            step_id="step-1",
        )
    )

    assert result == []
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "evidence_records.user_id" in compiled_sql
    assert "evidence_records.session_id" in compiled_sql
    assert "evidence_records.run_id" in compiled_sql
    assert "evidence_records.step_id" in compiled_sql


def test_evidence_repository_should_reject_empty_user_session_scope() -> None:
    db_session = SimpleNamespace(execute=AsyncMock())
    repository = DBEvidenceRepository(db_session=db_session)

    with pytest.raises(ValueError):
        asyncio.run(repository.list_by_ids(user_id="", session_id="session-1", evidence_ids=["evidence-1"]))

    with pytest.raises(ValueError):
        asyncio.run(repository.list_by_run(user_id="user-1", session_id="", run_id="run-1"))

    db_session.execute.assert_not_awaited()


def test_evidence_repository_list_by_ids_should_restore_result_refs_from_model() -> None:
    evidence = _evidence()
    model = EvidenceModel.from_domain(evidence)
    db_session = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: [model])
            )
        )
    )
    repository = DBEvidenceRepository(db_session=db_session)

    result = asyncio.run(
        repository.list_by_ids(
            user_id=evidence.user_id,
            session_id=evidence.session_id,
            evidence_ids=[evidence.id],
        )
    )

    assert len(result) == 1
    restored = result[0]
    assert isinstance(restored, EvidenceRecord)
    assert restored.result_refs == evidence.result_refs
    assert restored.source_ref == evidence.source_ref
    assert restored.payload == evidence.payload


def test_evidence_repository_list_reusable_and_action_subject_should_filter_scope() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])))
    )
    repository = DBEvidenceRepository(db_session=db_session)

    asyncio.run(repository.list_reusable_by_run(user_id="user-1", session_id="session-1", run_id="run-1"))
    reusable_sql = str(db_session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert "evidence_records.reusable IS true" in reusable_sql

    asyncio.run(
        repository.list_by_action_subject(
            user_id="user-1",
            session_id="session-1",
            run_id="run-1",
            action_key="search:abc",
            subject_key="query:abc",
        )
    )
    action_sql = str(db_session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert "evidence_records.action_key" in action_sql
    assert "evidence_records.subject_key" in action_sql


def test_evidence_migration_should_define_table_indexes_and_unique_key() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "4f6a7b8c9d10_create_evidence_records.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert '"evidence_records"' in content
    assert "uq_evidence_records_idempotency_key" in content
    assert "ix_evidence_user_session_created" in content
    assert "ix_evidence_user_run_step" in content
    assert "ix_evidence_action_subject" in content
    assert "ix_evidence_result_refs_hash" in content


def test_result_contract_should_not_have_second_contract_entrypoint() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    forbidden_module = "_".join(["evidence", "result", "contract"])
    forbidden_path = repo_root / "app" / "domain" / "services" / "runtime" / "contracts" / f"{forbidden_module}.py"
    scanned_files = [
        path
        for root in (repo_root / "app", repo_root / "tests")
        for path in root.rglob("*.py")
        if path != Path(__file__)
    ]

    assert not forbidden_path.exists()
    assert all(forbidden_module not in path.read_text(encoding="utf-8") for path in scanned_files)
