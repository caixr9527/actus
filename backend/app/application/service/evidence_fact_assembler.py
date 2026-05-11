#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox fact 到 Evidence 写入命令的组织器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.application.service.evidence_ledger_inputs import EvidenceRecordInput
from app.domain.models import Step
from app.domain.models.evidence import (
    EvidenceKind,
    EvidenceQualityStatus,
    EvidenceReadStrategy,
    EvidenceResultRef,
    EvidenceResultRefType,
    EvidenceReusePolicy,
    EvidenceScope,
    EvidenceSourceRef,
    EvidenceSourceType,
    EvidenceStalenessPolicy,
    EvidenceSubjectRef,
    EvidenceSupportLevel,
)
from app.domain.models.sandbox_fact import SandboxFactKind, SandboxFactRecord
from app.domain.services.runtime.contracts.document_input_contract import DocumentParseStatus
from app.domain.services.runtime.contracts.evidence_key_normalizer import (
    EvidenceActionSubjectKeyResult,
    build_evidence_action_subject_key_from_fact,
)


@dataclass(frozen=True)
class EvidenceAssemblyResult:
    """单个 step 的 evidence 对账结果。"""

    evidence_inputs: list[EvidenceRecordInput]
    gap_inputs: list[EvidenceRecordInput]


class EvidenceStrategy(Protocol):
    """Fact 到 EvidenceRecordInput 的纯转换策略。"""

    fact_kinds: frozenset[SandboxFactKind]

    def assemble(
            self,
            *,
            fact: SandboxFactRecord,
            key_result: EvidenceActionSubjectKeyResult,
    ) -> list[EvidenceRecordInput]:
        ...


class CommandEvidenceStrategy(EvidenceStrategy):
    fact_kinds = frozenset({SandboxFactKind.COMMAND_EXECUTION, SandboxFactKind.SHELL_OUTPUT})

    def assemble(
            self,
            *,
            fact: SandboxFactRecord,
            key_result: EvidenceActionSubjectKeyResult,
    ) -> list[EvidenceRecordInput]:
        return [_command_evidence(fact=fact, key_result=key_result)]


class FileEvidenceStrategy(EvidenceStrategy):
    fact_kinds = frozenset({
        SandboxFactKind.FILE_READ,
        SandboxFactKind.FILE_WRITE,
        SandboxFactKind.FILE_DELETE,
        SandboxFactKind.FILE_LIST,
        SandboxFactKind.FILE_SEARCH,
        SandboxFactKind.FILE_SNAPSHOT,
    })

    def assemble(
            self,
            *,
            fact: SandboxFactRecord,
            key_result: EvidenceActionSubjectKeyResult,
    ) -> list[EvidenceRecordInput]:
        items = [_file_evidence(fact=fact, key_result=key_result)]
        artifact_evidence = _artifact_evidence_from_file_fact(fact=fact, key_result=key_result)
        if artifact_evidence is not None:
            items.append(artifact_evidence)
        return items


class SearchEvidenceStrategy(EvidenceStrategy):
    fact_kinds = frozenset({SandboxFactKind.SEARCH_RESULT})

    def assemble(
            self,
            *,
            fact: SandboxFactRecord,
            key_result: EvidenceActionSubjectKeyResult,
    ) -> list[EvidenceRecordInput]:
        return [_search_evidence(fact=fact, key_result=key_result)]


class PageEvidenceStrategy(EvidenceStrategy):
    fact_kinds = frozenset({SandboxFactKind.FETCHED_PAGE})

    def assemble(
            self,
            *,
            fact: SandboxFactRecord,
            key_result: EvidenceActionSubjectKeyResult,
    ) -> list[EvidenceRecordInput]:
        return [_page_evidence(fact=fact, key_result=key_result)]


class BrowserEvidenceStrategy(EvidenceStrategy):
    fact_kinds = frozenset({SandboxFactKind.BROWSER_SNAPSHOT, SandboxFactKind.BROWSER_ACTION})

    def assemble(
            self,
            *,
            fact: SandboxFactRecord,
            key_result: EvidenceActionSubjectKeyResult,
    ) -> list[EvidenceRecordInput]:
        return [_browser_evidence(fact=fact, key_result=key_result)]


class DocumentEvidenceStrategy(EvidenceStrategy):
    fact_kinds = frozenset({SandboxFactKind.DOCUMENT_CONTEXT})

    def assemble(
            self,
            *,
            fact: SandboxFactRecord,
            key_result: EvidenceActionSubjectKeyResult,
    ) -> list[EvidenceRecordInput]:
        return [_document_evidence(fact=fact, key_result=key_result)]


class ToolFailureEvidenceStrategy(EvidenceStrategy):
    fact_kinds = frozenset({SandboxFactKind.TOOL_FAILURE})

    def assemble(
            self,
            *,
            fact: SandboxFactRecord,
            key_result: EvidenceActionSubjectKeyResult,
    ) -> list[EvidenceRecordInput]:
        return [_tool_failure_evidence(fact=fact, key_result=key_result)]


class HumanConfirmationEvidenceStrategy(EvidenceStrategy):
    fact_kinds = frozenset({SandboxFactKind.HUMAN_INTERACTION})

    def assemble(
            self,
            *,
            fact: SandboxFactRecord,
            key_result: EvidenceActionSubjectKeyResult,
    ) -> list[EvidenceRecordInput]:
        return [_human_confirmation_evidence(fact=fact, key_result=key_result)]


DEFAULT_EVIDENCE_STRATEGIES: tuple[EvidenceStrategy, ...] = (
    CommandEvidenceStrategy(),
    FileEvidenceStrategy(),
    SearchEvidenceStrategy(),
    PageEvidenceStrategy(),
    BrowserEvidenceStrategy(),
    DocumentEvidenceStrategy(),
    ToolFailureEvidenceStrategy(),
    HumanConfirmationEvidenceStrategy(),
)


class EvidenceFactAssembler:
    """把已入库 sandbox fact 组织为 EvidenceRecordInput。

    本类只做纯转换，不查权限、不读 repository、不调用工具。
    """

    def __init__(self, strategies: tuple[EvidenceStrategy, ...] = DEFAULT_EVIDENCE_STRATEGIES) -> None:
        self._strategies_by_kind = _build_strategy_registry(strategies)

    def assemble_step(
            self,
            *,
            step: Step,
            facts: list[SandboxFactRecord],
    ) -> EvidenceAssemblyResult:
        evidence_inputs: list[EvidenceRecordInput] = []
        gap_inputs: list[EvidenceRecordInput] = []
        for fact in list(facts or []):
            assembled_items = self._assemble_fact(fact=fact)
            if assembled_items is None:
                gap_inputs.append(_gap_input_for_fact(fact=fact, reason_code="unsupported_fact_kind"))
                continue
            for assembled in assembled_items:
                if assembled.support_level == EvidenceSupportLevel.GAP:
                    gap_inputs.append(assembled)
                else:
                    evidence_inputs.append(assembled)
        if not facts:
            gap_inputs.append(_gap_input_for_step(step=step, reason_code="step_fact_missing"))
        return EvidenceAssemblyResult(evidence_inputs=evidence_inputs, gap_inputs=gap_inputs)

    def _assemble_fact(self, *, fact: SandboxFactRecord) -> list[EvidenceRecordInput] | None:
        key_result = build_evidence_action_subject_key_from_fact(fact)
        if key_result.normalization_status != "normalized":
            return [_gap_input_for_fact(fact=fact, reason_code=key_result.reason_code or "evidence_key_normalize_skipped")]
        if not fact.source_ref.source_event_id:
            return [_gap_input_for_fact(fact=fact, reason_code="evidence_source_event_missing")]

        strategy = self._strategies_by_kind.get(fact.fact_kind)
        if strategy is None:
            return None
        return strategy.assemble(fact=fact, key_result=key_result)


def _build_strategy_registry(strategies: tuple[EvidenceStrategy, ...]) -> dict[SandboxFactKind, EvidenceStrategy]:
    registry: dict[SandboxFactKind, EvidenceStrategy] = {}
    for strategy in strategies:
        for fact_kind in strategy.fact_kinds:
            if fact_kind in registry:
                raise ValueError(f"重复注册 EvidenceStrategy: {fact_kind.value}")
            registry[fact_kind] = strategy
    return registry


def _source_ref(fact: SandboxFactRecord) -> EvidenceSourceRef:
    return EvidenceSourceRef(
        source_type=(
            EvidenceSourceType.USER_CONFIRMATION
            if fact.fact_kind == SandboxFactKind.HUMAN_INTERACTION
            else EvidenceSourceType.SANDBOX_FACT
        ),
        source_event_id=fact.source_ref.source_event_id,
        fact_ids=[fact.id],
        tool_call_id=fact.source_ref.tool_call_id,
        message_event_id=(
            fact.source_ref.source_event_id
            if fact.fact_kind == SandboxFactKind.HUMAN_INTERACTION
            else None
        ),
    )


def _subject_ref(fact: SandboxFactRecord, *, subject_key: str | None = None) -> EvidenceSubjectRef:
    return EvidenceSubjectRef(
        subject_type=fact.subject_ref.subject_type,
        subject_key=str(subject_key or fact.subject_ref.subject_key or ""),
        path=fact.subject_ref.path,
        url_hash=fact.subject_ref.url_hash,
        artifact_path=fact.subject_ref.artifact_path,
    )


def _base_input(
        *,
        fact: SandboxFactRecord,
        key_result: EvidenceActionSubjectKeyResult,
        evidence_kind: EvidenceKind,
        payload: dict[str, Any],
        support_level: EvidenceSupportLevel = EvidenceSupportLevel.STRONG,
        quality_status: EvidenceQualityStatus = EvidenceQualityStatus.VALID,
        reusable: bool = True,
        reuse_policy: EvidenceReusePolicy = EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy: EvidenceStalenessPolicy = EvidenceStalenessPolicy.RUN_SCOPED,
        result_refs: list[EvidenceResultRef] | None = None,
) -> EvidenceRecordInput:
    has_result_refs = bool(result_refs)
    resolved_reuse_policy = (
        reuse_policy
        if reusable or (has_result_refs and reuse_policy != EvidenceReusePolicy.REUSE_ALLOWED)
        else EvidenceReusePolicy.DO_NOT_REUSE
    )
    return EvidenceRecordInput(
        evidence_scope=EvidenceScope.STEP,
        evidence_kind=evidence_kind,
        run_id=fact.run_id,
        step_id=fact.step_id,
        source_step_id=fact.step_id,
        action_key=key_result.action_key,
        source_ref=_source_ref(fact),
        subject_ref=_subject_ref(fact, subject_key=key_result.subject_key),
        support_level=support_level,
        quality_status=quality_status,
        summary=fact.summary,
        payload=payload,
        confidence=0.8 if quality_status == EvidenceQualityStatus.VALID else 0.4,
        reusable=reusable and has_result_refs,
        reuse_policy=resolved_reuse_policy if has_result_refs else EvidenceReusePolicy.DO_NOT_REUSE,
        staleness_policy=staleness_policy,
        result_refs=list(result_refs or []),
    )


def _fact_ref(
        *,
        fact: SandboxFactRecord,
        key_result: EvidenceActionSubjectKeyResult,
        read_strategy: EvidenceReadStrategy = EvidenceReadStrategy.READ_FACT_PAYLOAD,
        content_hash: str | None = None,
        reason_code: str | None = None,
) -> EvidenceResultRef:
    return EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.FACT_REF,
        ref_id=fact.id,
        source_step_id=fact.step_id,
        source_fact_id=fact.id,
        source_event_id=fact.source_ref.source_event_id,
        subject_key=key_result.subject_key,
        payload_hash=fact.payload_hash,
        content_hash=content_hash,
        quality_status=EvidenceQualityStatus.VALID if reason_code is None else EvidenceQualityStatus.PARTIAL,
        support_level=EvidenceSupportLevel.STRONG if reason_code is None else EvidenceSupportLevel.PARTIAL,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED if reason_code is None else EvidenceReusePolicy.VERIFY_BEFORE_REUSE,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        read_strategy=read_strategy,
        reason_code=reason_code,
        summary=fact.summary,
    )


def _document_evidence(*, fact: SandboxFactRecord, key_result: EvidenceActionSubjectKeyResult) -> EvidenceRecordInput:
    payload = dict(fact.payload or {})
    parse_status = str(payload.get("parse_status") or "failed")
    is_valid = parse_status == DocumentParseStatus.PARSED.value and not bool(payload.get("is_truncated"))
    quality_status = EvidenceQualityStatus.VALID if is_valid else EvidenceQualityStatus.PARTIAL
    support_level = EvidenceSupportLevel.STRONG if is_valid else EvidenceSupportLevel.PARTIAL
    result_ref = EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.DOCUMENT_SOURCE_REF,
        ref_id=str(payload.get("file_id") or fact.id),
        source_step_id=fact.step_id,
        source_fact_id=fact.id,
        source_event_id=fact.source_ref.source_event_id,
        document_file_id=str(payload.get("file_id") or ""),
        subject_key=key_result.subject_key,
        payload_hash=fact.payload_hash,
        content_hash=str(payload.get("read_content_sha256") or payload.get("full_file_sha256") or "") or None,
        quality_status=quality_status,
        support_level=support_level,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED if is_valid else EvidenceReusePolicy.VERIFY_BEFORE_REUSE,
        staleness_policy=EvidenceStalenessPolicy.STABLE,
        read_strategy=EvidenceReadStrategy.READ_DOCUMENT_SOURCE if is_valid else EvidenceReadStrategy.VERIFY_BEFORE_USE,
        reason_code=None if is_valid else str(payload.get("reason_code") or "document_not_fully_readable"),
        allowed_verification_actions=[] if is_valid else ["read_document_source"],
        summary=fact.summary,
    )
    return _base_input(
        fact=fact,
        key_result=key_result,
        evidence_kind=EvidenceKind.DOCUMENT_EVIDENCE,
        payload={
            "file_id": str(payload.get("file_id") or ""),
            "parse_status": parse_status,
            "reason_code": payload.get("reason_code"),
            "full_file_sha256": payload.get("full_file_sha256"),
            "read_content_sha256": payload.get("read_content_sha256"),
            "is_truncated": bool(payload.get("is_truncated")),
            "excerpt_char_count": int(payload.get("excerpt_char_count") or 0),
            "source_fact_id": fact.id,
        },
        support_level=support_level,
        quality_status=quality_status,
        reusable=is_valid,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED if is_valid else EvidenceReusePolicy.VERIFY_BEFORE_REUSE,
        staleness_policy=EvidenceStalenessPolicy.STABLE,
        result_refs=[result_ref],
    )


def _command_evidence(*, fact: SandboxFactRecord, key_result: EvidenceActionSubjectKeyResult) -> EvidenceRecordInput:
    payload = dict(fact.payload or {})
    if fact.fact_kind == SandboxFactKind.SHELL_OUTPUT:
        result_status = str(payload.get("process_status") or "unknown")
        reason_code = payload.get("reason_code")
        excerpt = str(payload.get("output_excerpt") or "")
        is_truncated = bool(payload.get("output_truncated"))
    else:
        exit_code = payload.get("exit_code")
        timeout = bool(payload.get("timeout"))
        result_status = "timeout" if timeout else "success" if exit_code == 0 else "failed"
        reason_code = payload.get("reason_code") or ("command_failed" if result_status == "failed" else None)
        excerpt = str(payload.get("stdout_excerpt") or payload.get("stderr_excerpt") or "")
        is_truncated = bool(payload.get("stdout_truncated") or payload.get("stderr_truncated"))
    is_valid = not bool(reason_code) and result_status in {"success", "completed", "running", "unknown"}
    result_ref = _fact_ref(
        fact=fact,
        key_result=key_result,
        read_strategy=EvidenceReadStrategy.USE_DIGEST_SUMMARY,
        reason_code=None if is_valid else str(reason_code or result_status or "command_not_reusable"),
    )
    return _base_input(
        fact=fact,
        key_result=key_result,
        evidence_kind=EvidenceKind.ACTION_EVIDENCE,
        payload={
            "action_type": key_result.action_type or fact.fact_kind.value,
            "function_name": str(fact.source_ref.function_name or ""),
            "source_fact_ids": [fact.id],
            "source_event_id": str(fact.source_ref.source_event_id or ""),
            "result_status": result_status,
            "reason_code": reason_code,
            "excerpt": excerpt,
            "is_truncated": is_truncated,
        },
        support_level=EvidenceSupportLevel.STRONG if is_valid else EvidenceSupportLevel.PARTIAL,
        quality_status=EvidenceQualityStatus.VALID if is_valid else EvidenceQualityStatus.PARTIAL,
        reusable=is_valid,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED if is_valid else EvidenceReusePolicy.VERIFY_BEFORE_REUSE,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        result_refs=[result_ref],
    )


def _search_evidence(*, fact: SandboxFactRecord, key_result: EvidenceActionSubjectKeyResult) -> EvidenceRecordInput:
    payload = dict(fact.payload or {})
    is_valid = not bool(payload.get("is_truncated")) and int(payload.get("result_count") or 0) > 0
    verification_reason_code = "external_evidence_may_change"
    result_ref = _fact_ref(
        fact=fact,
        key_result=key_result,
        reason_code=None if is_valid else str(payload.get("reason_code") or "search_result_partial"),
    )
    if not is_valid:
        result_ref.allowed_verification_actions = ["search_web"]
        result_ref.reason_code = result_ref.reason_code or verification_reason_code
    return _base_input(
        fact=fact,
        key_result=key_result,
        evidence_kind=EvidenceKind.SEARCH_EVIDENCE,
        payload={
            "query_excerpt": str(payload.get("query_excerpt") or ""),
            "query_hash": str(payload.get("query_hash") or ""),
            "verification_reason_code": verification_reason_code,
            "result_count": int(payload.get("result_count") or 0),
            "top_result_origins": [
                str(item.get("origin") or "")
                for item in list(payload.get("top_results") or [])
                if isinstance(item, dict) and str(item.get("origin") or "").strip()
            ][:5],
            "source_fact_id": fact.id,
            "snippet_quality": "partial" if payload.get("is_truncated") else "available",
            "needs_fetch": bool(payload.get("is_truncated")) or int(payload.get("result_count") or 0) == 0,
        },
        support_level=EvidenceSupportLevel.STRONG if is_valid else EvidenceSupportLevel.PARTIAL,
        quality_status=EvidenceQualityStatus.VALID if is_valid else EvidenceQualityStatus.PARTIAL,
        reusable=is_valid,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED if is_valid else EvidenceReusePolicy.VERIFY_BEFORE_REUSE,
        staleness_policy=(
            EvidenceStalenessPolicy.RUN_SCOPED
            if is_valid
            else EvidenceStalenessPolicy.EXTERNAL_MAY_CHANGE
        ),
        result_refs=[result_ref],
    )


def _page_evidence(*, fact: SandboxFactRecord, key_result: EvidenceActionSubjectKeyResult) -> EvidenceRecordInput:
    payload = dict(fact.payload or {})
    is_valid = not bool(payload.get("is_truncated")) and bool(str(payload.get("excerpt") or "").strip())
    verification_reason_code = "external_evidence_may_change"
    result_ref = _fact_ref(
        fact=fact,
        key_result=key_result,
        reason_code=None if is_valid else str(payload.get("reason_code") or "page_content_partial"),
    )
    if not is_valid:
        result_ref.allowed_verification_actions = ["fetch_page"]
        result_ref.reason_code = result_ref.reason_code or verification_reason_code
    return _base_input(
        fact=fact,
        key_result=key_result,
        evidence_kind=EvidenceKind.PAGE_EVIDENCE,
        payload={
            "origin": str(payload.get("final_url_origin") or ""),
            "url_hash": str(payload.get("fetched_url_hash") or ""),
            "verification_reason_code": verification_reason_code,
            "title": str(payload.get("title") or ""),
            "status_code": payload.get("status_code"),
            "source_fact_id": fact.id,
            "excerpt": str(payload.get("excerpt") or ""),
            "is_truncated": bool(payload.get("is_truncated")),
        },
        support_level=EvidenceSupportLevel.STRONG if is_valid else EvidenceSupportLevel.PARTIAL,
        quality_status=EvidenceQualityStatus.VALID if is_valid else EvidenceQualityStatus.PARTIAL,
        reusable=is_valid,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED if is_valid else EvidenceReusePolicy.VERIFY_BEFORE_REUSE,
        staleness_policy=(
            EvidenceStalenessPolicy.RUN_SCOPED
            if is_valid
            else EvidenceStalenessPolicy.EXTERNAL_MAY_CHANGE
        ),
        result_refs=[result_ref],
    )


def _file_evidence(*, fact: SandboxFactRecord, key_result: EvidenceActionSubjectKeyResult) -> EvidenceRecordInput:
    payload = dict(fact.payload or {})
    operation = str(payload.get("operation") or fact.fact_kind.value.replace("file_", ""))
    mutation_intent_hash = str(payload.get("mutation_intent_hash") or "").strip()
    content_hash = str(
        payload.get("content_sha256")
        or payload.get("read_content_sha256")
        or payload.get("after_content_sha256")
        or ""
    ).strip() or None
    is_valid = not bool(payload.get("is_truncated")) and not bool(payload.get("reason_code")) and bool(content_hash)
    result_ref = _fact_ref(
        fact=fact,
        key_result=key_result,
        content_hash=content_hash,
        reason_code=None if is_valid else str(payload.get("reason_code") or "file_evidence_partial"),
    )
    return _base_input(
        fact=fact,
        key_result=key_result,
        evidence_kind=EvidenceKind.FILE_EVIDENCE,
        payload={
            "path": str(payload.get("path") or payload.get("dir_path") or ""),
            "operation": operation,
            "mutation_intent_hash": mutation_intent_hash or None,
            "exists": bool(payload.get("exists", True)),
            "content_sha256": content_hash,
            "content_sha256_kind": str(payload.get("content_sha256_kind") or "unknown"),
            "source_fact_id": fact.id,
            "excerpt": str(payload.get("excerpt") or ""),
            "is_truncated": bool(payload.get("is_truncated")),
        },
        support_level=EvidenceSupportLevel.STRONG if is_valid else EvidenceSupportLevel.PARTIAL,
        quality_status=EvidenceQualityStatus.VALID if is_valid else EvidenceQualityStatus.PARTIAL,
        reusable=is_valid and fact.fact_kind != SandboxFactKind.FILE_DELETE,
        reuse_policy=(
            EvidenceReusePolicy.DO_NOT_REUSE
            if fact.fact_kind == SandboxFactKind.FILE_DELETE
            else EvidenceReusePolicy.REUSE_ALLOWED
        ),
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        result_refs=[result_ref],
    )


def _artifact_evidence_from_file_fact(
        *,
        fact: SandboxFactRecord,
        key_result: EvidenceActionSubjectKeyResult,
) -> EvidenceRecordInput | None:
    artifact_id = str(fact.source_ref.artifact_id or "").strip()
    if not artifact_id:
        return None
    payload = dict(fact.payload or {})
    content_hash = str(payload.get("after_content_sha256") or payload.get("content_sha256") or "").strip()
    if not content_hash:
        return None
    artifact_path = str(payload.get("path") or "")
    result_ref = EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.ARTIFACT_REF,
        ref_id=artifact_id,
        source_step_id=fact.step_id,
        source_fact_id=fact.id,
        source_event_id=fact.source_ref.source_event_id,
        artifact_id=artifact_id,
        artifact_path=artifact_path,
        artifact_hash_kind=str(payload.get("content_sha256_kind") or "content_hash"),
        subject_key=key_result.subject_key,
        content_hash=content_hash,
        quality_status=EvidenceQualityStatus.VALID,
        support_level=EvidenceSupportLevel.STRONG,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        read_strategy=EvidenceReadStrategy.READ_ARTIFACT,
        summary=fact.summary,
    )
    return EvidenceRecordInput(
        evidence_scope=EvidenceScope.STEP,
        evidence_kind=EvidenceKind.ARTIFACT_EVIDENCE,
        run_id=fact.run_id,
        step_id=fact.step_id,
        source_step_id=fact.step_id,
        action_key=key_result.action_key,
        source_ref=EvidenceSourceRef(
            source_type=EvidenceSourceType.ARTIFACT,
            source_event_id=fact.source_ref.source_event_id,
            fact_ids=[fact.id],
            artifact_ids=[artifact_id],
            tool_call_id=fact.source_ref.tool_call_id,
        ),
        subject_ref=EvidenceSubjectRef(
            subject_type="artifact",
            subject_key=key_result.subject_key or f"artifact:{artifact_id}",
            artifact_path=artifact_path,
        ),
        support_level=EvidenceSupportLevel.STRONG,
        quality_status=EvidenceQualityStatus.VALID,
        summary=fact.summary,
        payload={
            "artifact_id": artifact_id,
            "artifact_path": artifact_path,
            "artifact_type": "file",
            "source_fact_ids": [fact.id],
            "current_hash": content_hash,
            "hash_kind": str(payload.get("content_sha256_kind") or "content_hash"),
            "delivery_candidate": True,
        },
        confidence=0.8,
        reusable=True,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        result_refs=[result_ref],
    )


def _browser_evidence(*, fact: SandboxFactRecord, key_result: EvidenceActionSubjectKeyResult) -> EvidenceRecordInput:
    payload = dict(fact.payload or {})
    is_valid = not bool(payload.get("reason_code")) and not bool(payload.get("degrade_reason"))
    result_ref = _fact_ref(
        fact=fact,
        key_result=key_result,
        reason_code=None if is_valid else str(payload.get("reason_code") or payload.get("degrade_reason") or "browser_evidence_partial"),
    )
    return _base_input(
        fact=fact,
        key_result=key_result,
        evidence_kind=EvidenceKind.BROWSER_EVIDENCE,
        payload={
            "action": payload.get("action"),
            "origin": payload.get("url_origin"),
            "title": str(payload.get("title") or ""),
            "screenshot_artifact_id": payload.get("screenshot_artifact_id"),
            "source_fact_id": fact.id,
            "excerpt": str(payload.get("structured_summary") or payload.get("target_summary") or ""),
            "is_truncated": False,
            "reason_code": payload.get("reason_code") or payload.get("degrade_reason"),
        },
        support_level=EvidenceSupportLevel.STRONG if is_valid else EvidenceSupportLevel.PARTIAL,
        quality_status=EvidenceQualityStatus.VALID if is_valid else EvidenceQualityStatus.PARTIAL,
        reusable=is_valid,
        result_refs=[result_ref],
    )


def _tool_failure_evidence(*, fact: SandboxFactRecord, key_result: EvidenceActionSubjectKeyResult) -> EvidenceRecordInput:
    payload = dict(fact.payload or {})
    return _base_input(
        fact=fact,
        key_result=key_result,
        evidence_kind=EvidenceKind.TOOL_FAILURE_EVIDENCE,
        payload={
            "function_name": str(payload.get("function_name") or fact.source_ref.function_name or ""),
            "reason_code": str(payload.get("reason_code") or "tool_failed"),
            "source_fact_id": fact.id,
            "message_excerpt": str(payload.get("message_excerpt") or ""),
            "retry_count": int(payload.get("retry_count") or 0),
            "timeout": bool(payload.get("timeout")),
        },
        support_level=EvidenceSupportLevel.GAP,
        quality_status=EvidenceQualityStatus.FAILED,
        reusable=False,
        reuse_policy=EvidenceReusePolicy.DO_NOT_REUSE,
        result_refs=[],
    )


def _human_confirmation_evidence(*, fact: SandboxFactRecord, key_result: EvidenceActionSubjectKeyResult) -> EvidenceRecordInput:
    payload = dict(fact.payload or {})
    result_ref = EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.USER_CONFIRMATION_REF,
        ref_id=fact.source_ref.source_event_id or fact.id,
        source_step_id=fact.step_id,
        source_fact_id=fact.id,
        source_event_id=fact.source_ref.source_event_id,
        subject_key=key_result.subject_key,
        payload_hash=fact.payload_hash,
        quality_status=EvidenceQualityStatus.VALID,
        support_level=EvidenceSupportLevel.STRONG,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
        summary=fact.summary,
    )
    return _base_input(
        fact=fact,
        key_result=key_result,
        evidence_kind=EvidenceKind.HUMAN_CONFIRMATION_EVIDENCE,
        payload={
            "interaction_id": fact.source_ref.source_event_id or fact.id,
            "interaction_type": str(payload.get("interaction_type") or "confirmation"),
            "message_event_id": fact.source_ref.source_event_id or fact.id,
            "confirmed": payload.get("confirmed"),
            "summary": str(payload.get("message_excerpt") or fact.summary),
            "reason_code": payload.get("reason_code"),
        },
        reusable=True,
        result_refs=[result_ref],
    )


def _gap_input_for_step(*, step: Step, reason_code: str) -> EvidenceRecordInput:
    return EvidenceRecordInput(
        evidence_scope=EvidenceScope.STEP,
        evidence_kind=EvidenceKind.EVIDENCE_GAP,
        step_id=step.id,
        subject_ref=EvidenceSubjectRef(subject_type="step", subject_key=f"step:{step.id}"),
        source_ref=EvidenceSourceRef(source_type=EvidenceSourceType.SYSTEM_PROJECTION),
        support_level=EvidenceSupportLevel.GAP,
        quality_status=EvidenceQualityStatus.MISSING_SOURCE,
        summary=f"step evidence gap: {reason_code}",
        payload={
            "gap_type": "step_evidence",
            "missing_source_types": [EvidenceSourceType.SANDBOX_FACT.value],
            "claim_text": str(step.description or step.title or ""),
            "reason_code": reason_code,
            "required_for": "step_completion",
        },
    )


def _gap_input_for_fact(*, fact: SandboxFactRecord, reason_code: str) -> EvidenceRecordInput:
    return EvidenceRecordInput(
        evidence_scope=EvidenceScope.STEP,
        evidence_kind=EvidenceKind.EVIDENCE_GAP,
        run_id=fact.run_id,
        step_id=fact.step_id,
        source_step_id=fact.step_id,
        source_ref=EvidenceSourceRef(
            source_type=EvidenceSourceType.SANDBOX_FACT,
            source_event_id=fact.source_ref.source_event_id,
            fact_ids=[fact.id],
            tool_call_id=fact.source_ref.tool_call_id,
        ),
        subject_ref=_subject_ref(fact),
        support_level=EvidenceSupportLevel.GAP,
        quality_status=EvidenceQualityStatus.MISSING_SOURCE,
        summary=f"fact evidence gap: {reason_code}",
        payload={
            "gap_type": "fact_evidence",
            "missing_source_types": [EvidenceSourceType.SANDBOX_FACT.value],
            "claim_text": fact.summary,
            "reason_code": reason_code,
            "required_for": "evidence_reconcile",
        },
    )
