#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence Ledger PR2 应用服务。

本服务只处理显式 evidence 写入、治理校验和强过滤查询；digest/reconcile
属于 PR3，不在这里实现。
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Protocol

from app.application.service.evidence_ledger_inputs import EvidenceQueryInput
from app.application.service.evidence_ledger_inputs import EvidenceRecordInput
from app.domain.models.evidence import (
    EvidenceBackedFactProjection,
    EvidenceKind,
    EvidenceQualityStatus,
    EvidenceRecord,
    EvidenceResultRef,
    EvidenceReusePolicy,
    EvidenceScope,
    EvidenceSourceRef,
    EvidenceSourceType,
    EvidenceStalenessPolicy,
    EvidenceSubjectRef,
    EvidenceSupportLevel,
    build_evidence_idempotency_key,
    build_evidence_payload_hash,
    build_evidence_result_refs_hash,
    classify_evidence_data,
    validate_evidence_payload,
)
from app.domain.models import Step
from app.domain.models.event import EvidenceEvent, EvidenceEventRef
from app.domain.models.sandbox_fact import SandboxFactRecord, SandboxFactScope
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.evidence_runtime_ports import EvidenceStepProjectionPort

logger = logging.getLogger(__name__)


MAX_EVIDENCE_TEXT_CHARS = 4000
MAX_EVIDENCE_SUMMARY_CHARS = 500
REDACTED = "[REDACTED]"

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(\b(?:access|refresh|id)?_?token\b\s*[:=]\s*)['\"]?[^'\"\s,}]{8,}"),
    re.compile(r"(?i)(\b(?:api[_-]?key|secret[_-]?key)\b\s*[:=]\s*)['\"]?[^'\"\s,}]{8,}"),
    re.compile(r"(?i)(\bpassword\b\s*[:=]\s*)['\"]?[^'\"\s,}]{4,}"),
    re.compile(r"(?i)(\bcookie\b\s*[:=]\s*)['\"]?[^'\"\n,}]{8,}"),
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{8,}"),
)
_EVENT_SUMMARY_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_EVENT_SUMMARY_RAW_MARKER_PATTERN = re.compile(
    r"(?i)\b(raw[_\s-]?stdout|raw[_\s-]?stderr|stdout|stderr|full[_\s-]?text|file[_\s-]?content|"
    r"page[_\s-]?content|document[_\s-]?text|html)\b"
)
_EVIDENCE_EVENT_SUMMARY_MAX_CHARS = 160
_EVIDENCE_EVENT_SUMMARY_OMITTED = "该 evidence 已持久化，事件摘要包含敏感或原始内容，已省略；请通过 evidence_refs 回查。"
_EVIDENCE_EVENT_SUMMARY_TOO_LONG = "该 evidence 已持久化，摘要过长未注入事件；请通过 evidence_refs 回查。"


class EvidenceAssemblyResultLike(Protocol):
    """Evidence assembler 返回对象的最小结构。"""

    evidence_inputs: list[EvidenceRecordInput]
    gap_inputs: list[EvidenceRecordInput]


class EvidenceStepAssembler(Protocol):
    """EvidenceLedgerService 所需的 fact 组织器接口。"""

    def assemble_step(self, *, step: Step, facts: list[SandboxFactRecord]) -> EvidenceAssemblyResultLike:
        ...


class EvidenceLedgerError(RuntimeError):
    """Evidence Ledger 应用服务错误。"""


class EvidenceScopeMismatchError(EvidenceLedgerError):
    """写入或读取对象不属于当前 access scope。"""


class EvidenceSourceMissingError(EvidenceLedgerError):
    """source fact/artifact/event 缺失或不可归属。"""


class EvidenceLedgerService:
    """Evidence 写入和查询的应用层唯一入口。"""

    def __init__(
            self,
            *,
            uow_factory,
            assembler: EvidenceStepAssembler,
            step_projection: EvidenceStepProjectionPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._assembler = assembler
        self._step_projection = step_projection

    async def record_evidence(
            self,
            *,
            scope: AccessScopeResult,
            evidence_input: EvidenceRecordInput,
    ) -> EvidenceRecord:
        self._validate_scope_basics(scope)
        run_id, step_id, source_step_id = self._resolve_evidence_scope(
            scope=scope,
            evidence_input=evidence_input,
        )
        sanitized_payload = _sanitize_payload(evidence_input.payload)
        normalized_payload = validate_evidence_payload(
            evidence_kind=evidence_input.evidence_kind,
            payload=sanitized_payload,
        ).model_dump(mode="json")
        result_refs = [
            ref if isinstance(ref, EvidenceResultRef) else EvidenceResultRef.model_validate(ref)
            for ref in list(evidence_input.result_refs or [])
        ]
        payload_hash = build_evidence_payload_hash(normalized_payload)
        result_refs_hash = build_evidence_result_refs_hash(result_refs)
        classification = classify_evidence_data(
            evidence_kind=evidence_input.evidence_kind,
            source_type=evidence_input.source_ref.source_type,
        )
        source_ref = evidence_input.source_ref
        subject_ref = evidence_input.subject_ref
        idempotency_key = build_evidence_idempotency_key(
            user_id=scope.user_id,
            session_id=str(scope.session_id),
            run_id=run_id,
            step_id=step_id,
            evidence_scope=evidence_input.evidence_scope,
            evidence_kind=evidence_input.evidence_kind,
            source_event_id=source_ref.source_event_id,
            primary_fact_id=_first_non_empty(source_ref.fact_ids),
            primary_artifact_id=_first_non_empty(source_ref.artifact_ids),
            action_key=evidence_input.action_key,
            claim_key=evidence_input.claim_key,
            payload_hash=payload_hash,
            result_refs_hash=result_refs_hash,
        )
        evidence = EvidenceRecord(
            id=str(uuid.uuid4()),
            user_id=scope.user_id,
            session_id=str(scope.session_id),
            workspace_id=str(scope.workspace_id),
            run_id=run_id,
            step_id=step_id,
            evidence_scope=evidence_input.evidence_scope,
            evidence_kind=evidence_input.evidence_kind,
            action_key=evidence_input.action_key,
            claim_key=evidence_input.claim_key,
            claim_text=evidence_input.claim_text,
            source_step_id=source_step_id,
            support_level=evidence_input.support_level,
            quality_status=evidence_input.quality_status,
            source_ref=source_ref,
            subject_ref=subject_ref,
            summary=_sanitize_text(evidence_input.summary, max_chars=MAX_EVIDENCE_SUMMARY_CHARS),
            payload=normalized_payload,
            payload_hash=payload_hash,
            idempotency_key=idempotency_key,
            confidence=evidence_input.confidence,
            reusable=evidence_input.reusable,
            reuse_policy=evidence_input.reuse_policy,
            staleness_policy=evidence_input.staleness_policy,
            visibility=evidence_input.visibility,
            origin=classification[0],
            trust_level=classification[1],
            privacy_level=classification[2],
            retention_policy=classification[3],
            result_refs=result_refs,
            result_refs_hash=result_refs_hash,
            related_evidence_ids=evidence_input.related_evidence_ids,
            supersedes_evidence_id=evidence_input.supersedes_evidence_id,
            created_at=datetime.now(),
        )
        async with self._uow_factory() as uow:
            await self._validate_sources(
                uow=uow,
                scope=scope,
                evidence=evidence,
            )
            saved = await uow.evidence.save_once(evidence)
            logger.info(
                "evidence_record_saved user_id=%s session_id=%s run_id=%s step_id=%s evidence_id=%s evidence_kind=%s",
                scope.user_id,
                scope.session_id,
                run_id,
                step_id,
                saved.id,
                saved.evidence_kind.value,
            )
            return saved

    async def list_reusable_by_run(
            self,
            *,
            scope: AccessScopeResult,
            run_id: str | None = None,
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        self._validate_scope_basics(scope)
        actual_run_id = self._resolve_run_id(scope=scope, run_id=run_id)
        async with self._uow_factory() as uow:
            return await uow.evidence.list_reusable_by_run(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                run_id=actual_run_id,
                limit=limit,
            )

    async def list_by_action_subject(
            self,
            *,
            scope: AccessScopeResult,
            query: EvidenceQueryInput,
    ) -> list[EvidenceRecord]:
        self._validate_scope_basics(scope)
        actual_run_id = self._resolve_run_id(scope=scope, run_id=query.run_id)
        if not query.action_key or not query.subject_key:
            raise EvidenceScopeMismatchError("action_key 和 subject_key 不能为空")
        async with self._uow_factory() as uow:
            return await uow.evidence.list_by_action_subject(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                run_id=actual_run_id,
                action_key=query.action_key,
                subject_key=query.subject_key,
                limit=query.limit,
            )

    async def reconcile_step_evidence(
            self,
            *,
            scope: AccessScopeResult,
            step: Step,
    ) -> list[EvidenceRecord]:
        """在 step completed 前把当前 step facts 对账成 evidence 或 gap。"""
        self._validate_scope_basics(scope)
        step_id = str(step.id or scope.current_step_id or "").strip()
        if not step_id:
            raise EvidenceScopeMismatchError("reconcile_step_evidence 必须包含 step_id")
        run_id = self._resolve_run_id(scope=scope, run_id=scope.run_id)
        saved: list[EvidenceRecord] = []
        try:
            async with self._uow_factory() as uow:
                facts = await uow.sandbox_fact.list_by_scope(
                    user_id=scope.user_id,
                    session_id=str(scope.session_id),
                    fact_scope=SandboxFactScope.STEP,
                    run_id=run_id,
                    step_id=step_id,
                    limit=100,
                )
            assembled = self._assembler.assemble_step(step=step, facts=facts)
            for evidence_input in [*assembled.evidence_inputs, *assembled.gap_inputs]:
                saved.append(await self.record_evidence(scope=scope, evidence_input=evidence_input))
        except Exception as exc:
            logger.warning(
                "evidence_reconcile_failed user_id=%s session_id=%s run_id=%s step_id=%s error_type=%s",
                scope.user_id,
                scope.session_id,
                run_id,
                step_id,
                exc.__class__.__name__,
            )
            gap_input = _reconcile_failed_gap_input(step=step, run_id=run_id)
            saved.append(await self.record_evidence(scope=scope, evidence_input=gap_input))
        return saved

    async def record_reconcile_failed_gap(
            self,
            *,
            scope: AccessScopeResult,
            step: Step,
    ) -> EvidenceRecord:
        """runner 兜底入口：只写 evidence_reconcile_failed gap。"""
        self._validate_scope_basics(scope)
        run_id = self._resolve_run_id(scope=scope, run_id=scope.run_id)
        return await self.record_evidence(
            scope=scope,
            evidence_input=_reconcile_failed_gap_input(step=step, run_id=run_id),
        )

    async def build_step_evidence_backed_facts(
            self,
            *,
            scope: AccessScopeResult,
            step: Step,
    ) -> list[EvidenceBackedFactProjection]:
        """基于当前 step 已落库 evidence 生成 StepOutcome 可读事实投影。"""
        self._validate_scope_basics(scope)
        step_id = str(step.id or scope.current_step_id or "").strip()
        if not step_id:
            return []
        if self._step_projection is None:
            return []
        return await self._step_projection.build_step_evidence_backed_facts(scope=scope, step=step)

    async def build_step_evidence_event(
            self,
            *,
            scope: AccessScopeResult,
            step: Step,
            records: list[EvidenceRecord],
    ) -> EvidenceEvent | None:
        """基于本次 step 对账结果生成轻量 EvidenceEvent，不携带 raw payload。"""
        self._validate_scope_basics(scope)
        step_id = str(step.id or scope.current_step_id or "").strip()
        if not step_id:
            return None
        if not _evidence_event_records_match_scope(scope=scope, step_id=step_id, records=records):
            logger.error(
                "evidence_event_projection_failed",
                extra={
                    "user_id": scope.user_id,
                    "session_id": str(scope.session_id),
                    "workspace_id": str(scope.workspace_id),
                    "run_id": scope.run_id,
                    "step_id": step_id,
                    "reason_code": "evidence_event_scope_mismatch",
                },
            )
            return None
        return build_step_evidence_event(
            step_id=step_id,
            run_id=scope.run_id,
            records=records,
        )

    async def reconcile_previous_steps_evidence(
            self,
            *,
            scope: AccessScopeResult,
            completed_step_ids: list[str],
    ) -> list[EvidenceRecord]:
        """下一个 step execute 前补齐前序 completed step 的 evidence gap。"""
        self._validate_scope_basics(scope)
        run_id = self._resolve_run_id(scope=scope, run_id=scope.run_id)
        saved: list[EvidenceRecord] = []
        missing_step_ids: list[str] = []
        async with self._uow_factory() as uow:
            for step_id in _normalize_text_list(completed_step_ids):
                existing = await uow.evidence.list_by_step(
                    user_id=scope.user_id,
                    session_id=str(scope.session_id),
                    run_id=run_id,
                    step_id=step_id,
                    limit=1,
                )
                if existing:
                    continue
                missing_step_ids.append(step_id)
        for step_id in missing_step_ids:
            gap_input = _previous_step_missing_gap_input(step_id=step_id, run_id=run_id)
            saved.append(await self.record_evidence(scope=scope, evidence_input=gap_input))
        return saved

    @staticmethod
    def _validate_scope_basics(scope: AccessScopeResult) -> None:
        if not str(scope.user_id or "").strip():
            raise EvidenceScopeMismatchError("evidence scope 必须包含 user_id")
        if not str(scope.session_id or "").strip():
            raise EvidenceScopeMismatchError("evidence scope 必须包含 session_id")
        if not str(scope.workspace_id or "").strip():
            raise EvidenceScopeMismatchError("evidence scope 必须包含 workspace_id")

    @staticmethod
    def _resolve_run_id(*, scope: AccessScopeResult, run_id: str | None) -> str:
        actual_run_id = str(run_id or scope.run_id or "").strip()
        if not actual_run_id:
            raise EvidenceScopeMismatchError("evidence 查询必须包含 run_id")
        if scope.run_id and actual_run_id != scope.run_id:
            raise EvidenceScopeMismatchError("run_id 与 access scope 不一致")
        return actual_run_id

    @staticmethod
    def _resolve_evidence_scope(
            *,
            scope: AccessScopeResult,
            evidence_input: EvidenceRecordInput,
    ) -> tuple[str | None, str | None, str | None]:
        run_id = evidence_input.run_id if evidence_input.run_id is not None else scope.run_id
        current_step_id = scope.current_step_id
        step_id = evidence_input.step_id if evidence_input.step_id is not None else current_step_id
        if evidence_input.evidence_scope == EvidenceScope.STEP:
            if not run_id:
                raise EvidenceScopeMismatchError("STEP scope evidence 必须包含 run_id")
            if scope.run_id and run_id != scope.run_id:
                raise EvidenceScopeMismatchError("evidence run_id 与 access scope 不一致")
            if not current_step_id:
                raise EvidenceScopeMismatchError("STEP scope evidence 必须包含 current_step_id")
            is_previous_gap = (
                    evidence_input.evidence_kind == EvidenceKind.EVIDENCE_GAP
                    and str((evidence_input.payload or {}).get("reason_code") or "").strip()
                    == "previous_step_evidence_missing"
            )
            if not step_id or (step_id != current_step_id and not is_previous_gap):
                raise EvidenceScopeMismatchError("STEP scope evidence 的 step_id 必须等于当前 step")
            source_step_id = evidence_input.source_step_id or step_id
            if source_step_id != step_id:
                raise EvidenceScopeMismatchError("STEP scope evidence 的 source_step_id 必须等于 step_id")
            return run_id, step_id, source_step_id
        if evidence_input.evidence_scope == EvidenceScope.RUN:
            if evidence_input.step_id is not None:
                raise EvidenceScopeMismatchError("RUN scope evidence 禁止传入 step_id")
            if not run_id:
                raise EvidenceScopeMismatchError("RUN scope evidence 必须包含 run_id")
            if scope.run_id and run_id != scope.run_id:
                raise EvidenceScopeMismatchError("evidence run_id 与 access scope 不一致")
            return run_id, None, evidence_input.source_step_id
        if run_id is not None or evidence_input.step_id is not None:
            raise EvidenceScopeMismatchError("WORKSPACE scope evidence 禁止绑定 run_id 或 step_id")
        return None, None, evidence_input.source_step_id

    async def _validate_sources(
            self,
            *,
            uow,
            scope: AccessScopeResult,
            evidence: EvidenceRecord,
    ) -> None:
        if evidence.evidence_kind not in {EvidenceKind.EVIDENCE_GAP, EvidenceKind.CORRECTION, EvidenceKind.SUPERSEDED}:
            await self._validate_source_event(uow=uow, scope=scope, evidence=evidence)
        await self._validate_fact_refs(uow=uow, scope=scope, evidence=evidence)
        await self._validate_artifact_refs(uow=uow, scope=scope, evidence=evidence)

    async def _validate_source_event(
            self,
            *,
            uow,
            scope: AccessScopeResult,
            evidence: EvidenceRecord,
    ) -> None:
        event_ids = [
            event_id
            for event_id in (evidence.source_ref.source_event_id, evidence.source_ref.message_event_id)
            if str(event_id or "").strip()
        ]
        if not event_ids or not evidence.run_id:
            logger.warning(
                "evidence_source_event_missing user_id=%s session_id=%s run_id=%s evidence_kind=%s",
                scope.user_id,
                scope.session_id,
                evidence.run_id,
                evidence.evidence_kind.value,
            )
            raise EvidenceSourceMissingError("successful evidence 必须绑定已持久化 source event")
        for event_id in event_ids:
            record = await uow.workflow_run.get_event_record_by_event_id(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                run_id=evidence.run_id,
                event_id=event_id,
            )
            if record is None:
                raise EvidenceSourceMissingError("source event 不存在或不属于当前 scope")

    async def _validate_fact_refs(
            self,
            *,
            uow,
            scope: AccessScopeResult,
            evidence: EvidenceRecord,
    ) -> None:
        payload_fact_ids = _extract_payload_fact_ids(evidence)
        fact_ids = set(evidence.source_ref.fact_ids)
        result_fact_ids = {ref.source_fact_id for ref in evidence.result_refs if ref.source_fact_id}
        if payload_fact_ids:
            _ensure_payload_refs_covered(
                payload_refs=payload_fact_ids,
                source_refs=set(evidence.source_ref.fact_ids),
                result_refs=result_fact_ids,
                message="payload fact refs 必须与 source_ref/result_refs 一致",
            )
        fact_ids.update(result_fact_ids)
        fact_ids.update(payload_fact_ids)
        normalized_ids = _normalize_text_list(list(fact_ids))
        if not normalized_ids:
            return
        facts = await uow.sandbox_fact.list_by_ids(
            user_id=scope.user_id,
            session_id=str(scope.session_id),
            fact_ids=normalized_ids,
            limit=max(100, len(normalized_ids)),
        )
        found_by_id = {fact.id: fact for fact in facts}
        if set(normalized_ids) != set(found_by_id.keys()):
            raise EvidenceSourceMissingError("source fact 不存在或不属于当前用户会话")
        for fact in facts:
            if fact.workspace_id != scope.workspace_id:
                raise EvidenceScopeMismatchError("source fact workspace 与 access scope 不一致")
            if evidence.run_id and fact.run_id != evidence.run_id:
                raise EvidenceScopeMismatchError("source fact run 与 evidence run 不一致")

    async def _validate_artifact_refs(
            self,
            *,
            uow,
            scope: AccessScopeResult,
            evidence: EvidenceRecord,
    ) -> None:
        payload_artifact_ids = _extract_payload_artifact_ids(evidence)
        artifact_ids = set(evidence.source_ref.artifact_ids)
        result_artifact_ids = {ref.artifact_id for ref in evidence.result_refs if ref.artifact_id}
        if payload_artifact_ids:
            _ensure_payload_refs_covered(
                payload_refs=payload_artifact_ids,
                source_refs=set(evidence.source_ref.artifact_ids),
                result_refs=result_artifact_ids,
                message="payload artifact refs 必须与 source_ref/result_refs 一致",
            )
        artifact_ids.update(result_artifact_ids)
        artifact_ids.update(payload_artifact_ids)
        for artifact_id in _normalize_text_list(list(artifact_ids)):
            artifact = await uow.workspace_artifact.get_by_user_workspace_id_and_id(
                user_id=scope.user_id,
                workspace_id=str(scope.workspace_id),
                artifact_id=artifact_id,
            )
            if artifact is None:
                raise EvidenceSourceMissingError("artifact 不存在或不属于当前 workspace scope")
            expected_session_id = str(scope.session_id)
            if artifact.session_id != expected_session_id:
                raise EvidenceScopeMismatchError("artifact session 与 access scope 不一致")
            if evidence.run_id and artifact.run_id != evidence.run_id:
                raise EvidenceScopeMismatchError("artifact run 与 evidence run 不一致")
            if evidence.source_step_id and artifact.source_step_id != evidence.source_step_id:
                raise EvidenceScopeMismatchError("artifact source_step_id 与 evidence 不一致")


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value, max_chars=MAX_EVIDENCE_TEXT_CHARS)
    return value


def _sanitize_text(value: str, *, max_chars: int) -> str:
    sanitized = str(value or "")
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(lambda match: f"{match.group(1)}{REDACTED}", sanitized)
    if len(sanitized) > max_chars:
        return sanitized[:max_chars]
    return sanitized


def _first_non_empty(values: list[str] | None) -> str | None:
    for value in list(values or []):
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return None


def _extract_payload_fact_ids(evidence: EvidenceRecord) -> set[str]:
    payload = evidence.payload or {}
    keys_by_kind = {
        EvidenceKind.ACTION_EVIDENCE: ("source_fact_ids",),
        EvidenceKind.CLAIM_SUPPORT: ("supporting_fact_ids",),
        EvidenceKind.ARTIFACT_EVIDENCE: ("source_fact_ids",),
        EvidenceKind.DOCUMENT_EVIDENCE: ("source_fact_id",),
        EvidenceKind.SEARCH_EVIDENCE: ("source_fact_id",),
        EvidenceKind.PAGE_EVIDENCE: ("source_fact_id",),
        EvidenceKind.FILE_EVIDENCE: ("source_fact_id",),
        EvidenceKind.BROWSER_EVIDENCE: ("source_fact_id",),
        EvidenceKind.TOOL_FAILURE_EVIDENCE: ("source_fact_id",),
    }
    return _extract_payload_refs(payload, keys_by_kind.get(evidence.evidence_kind, ()))


def _extract_payload_artifact_ids(evidence: EvidenceRecord) -> set[str]:
    payload = evidence.payload or {}
    keys_by_kind = {
        EvidenceKind.CLAIM_SUPPORT: ("supporting_artifact_ids",),
        EvidenceKind.ARTIFACT_EVIDENCE: ("artifact_id",),
        EvidenceKind.BROWSER_EVIDENCE: ("screenshot_artifact_id",),
    }
    return _extract_payload_refs(payload, keys_by_kind.get(evidence.evidence_kind, ()))


def _extract_payload_refs(payload: dict[str, Any], keys: tuple[str, ...]) -> set[str]:
    refs: set[str] = set()
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            refs.update(str(item or "").strip() for item in value if str(item or "").strip())
            continue
        normalized = str(value or "").strip()
        if normalized:
            refs.add(normalized)
    return refs


def _ensure_payload_refs_covered(
        *,
        payload_refs: set[str],
        source_refs: set[str],
        result_refs: set[str],
        message: str,
) -> None:
    normalized_payload_refs = set(_normalize_text_list(list(payload_refs)))
    normalized_source_refs = set(_normalize_text_list(list(source_refs)))
    normalized_result_refs = set(_normalize_text_list(list(result_refs)))
    if normalized_source_refs and not normalized_payload_refs.issubset(normalized_source_refs):
        raise EvidenceScopeMismatchError(message)
    if normalized_result_refs and not normalized_payload_refs.issubset(normalized_result_refs):
        raise EvidenceScopeMismatchError(message)


def _normalize_text_list(values: list[str]) -> list[str]:
    return [
        str(value or "").strip()
        for value in list(values or [])
        if str(value or "").strip()
    ]


def _evidence_event_records_match_scope(
        *,
        scope: AccessScopeResult,
        step_id: str,
        records: list[EvidenceRecord],
) -> bool:
    expected_user_id = str(scope.user_id or "").strip()
    expected_session_id = str(scope.session_id or "").strip()
    expected_workspace_id = str(scope.workspace_id or "").strip()
    expected_run_id = str(scope.run_id or "").strip()
    expected_step_id = str(step_id or "").strip()
    if not expected_run_id or not expected_step_id:
        return False
    for record in list(records or []):
        if not isinstance(record, EvidenceRecord):
            return False
        if str(record.user_id or "").strip() != expected_user_id:
            return False
        if str(record.session_id or "").strip() != expected_session_id:
            return False
        if str(record.workspace_id or "").strip() != expected_workspace_id:
            return False
        if str(record.run_id or "").strip() != expected_run_id:
            return False
        if str(record.step_id or record.source_step_id or "").strip() != expected_step_id:
            return False
        if str(record.source_step_id or record.step_id or "").strip() != expected_step_id:
            return False
    return True


def build_step_evidence_event(
        *,
        step_id: str,
        run_id: str | None,
        records: list[EvidenceRecord],
) -> EvidenceEvent | None:
    """把本次对账 evidence 聚合为单条 runtime event。"""
    evidence_records = [record for record in list(records or []) if isinstance(record, EvidenceRecord)]
    if len(evidence_records) == 0:
        return None
    evidence_refs: list[EvidenceEventRef] = []
    source_event_ids: list[str] = []
    quality_status_counts: dict[str, int] = {}
    support_level_counts: dict[str, int] = {}
    gap_count = 0
    for record in evidence_records:
        quality_status = str(record.quality_status.value)
        support_level = str(record.support_level.value)
        quality_status_counts[quality_status] = quality_status_counts.get(quality_status, 0) + 1
        support_level_counts[support_level] = support_level_counts.get(support_level, 0) + 1
        if record.evidence_kind == EvidenceKind.EVIDENCE_GAP:
            gap_count += 1
        for source_event_id in [
            record.source_ref.source_event_id,
            record.source_event_id,
        ]:
            normalized_source_event_id = str(source_event_id or "").strip()
            if normalized_source_event_id and normalized_source_event_id not in source_event_ids:
                source_event_ids.append(normalized_source_event_id)
        evidence_refs.append(
            EvidenceEventRef(
                evidence_id=record.id,
                evidence_kind=str(record.evidence_kind.value),
                quality_status=quality_status,
                support_level=support_level,
                summary=_safe_evidence_event_summary(record.summary),
            )
        )
    return EvidenceEvent(
        step_id=step_id,
        evidence_refs=evidence_refs,
        source_event_ids=source_event_ids,
        quality_status_counts=quality_status_counts,
        support_level_counts=support_level_counts,
        gap_count=gap_count,
        summary=_build_evidence_event_summary(
            evidence_count=len(evidence_refs),
            gap_count=gap_count,
        ),
        runtime_metadata={
            "run_id": str(run_id or ""),
            "event_scope": "step_evidence_reconcile",
        },
    )


def _safe_evidence_event_summary(value: str) -> str:
    summary = str(value or "").strip()
    if not summary:
        return ""
    if _event_summary_contains_sensitive_content(summary):
        return _EVIDENCE_EVENT_SUMMARY_OMITTED
    sanitized = _sanitize_text(summary, max_chars=_EVIDENCE_EVENT_SUMMARY_MAX_CHARS + 1)
    if len(sanitized) > _EVIDENCE_EVENT_SUMMARY_MAX_CHARS:
        return _EVIDENCE_EVENT_SUMMARY_TOO_LONG
    return sanitized


def _event_summary_contains_sensitive_content(value: str) -> bool:
    if _EVENT_SUMMARY_URL_PATTERN.search(value):
        return True
    if _EVENT_SUMMARY_RAW_MARKER_PATTERN.search(value):
        return True
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def _build_evidence_event_summary(*, evidence_count: int, gap_count: int) -> str:
    if gap_count > 0:
        return f"step evidence 对账完成：{evidence_count} 条记录，其中 {gap_count} 条缺口。"
    return f"step evidence 对账完成：{evidence_count} 条记录。"


def _reconcile_failed_gap_input(*, step: Step, run_id: str) -> EvidenceRecordInput:
    return EvidenceRecordInput(
        evidence_scope=EvidenceScope.STEP,
        evidence_kind=EvidenceKind.EVIDENCE_GAP,
        run_id=run_id,
        step_id=step.id,
        source_step_id=step.id,
        source_ref=EvidenceSourceRef(source_type=EvidenceSourceType.SYSTEM_PROJECTION),
        subject_ref=EvidenceSubjectRef(subject_type="step", subject_key=f"step:{step.id}"),
        support_level=EvidenceSupportLevel.GAP,
        quality_status=EvidenceQualityStatus.MISSING_SOURCE,
        summary="evidence reconcile failed",
        payload={
            "gap_type": "step_evidence",
            "missing_source_types": [EvidenceSourceType.SYSTEM_PROJECTION.value],
            "claim_text": str(step.description or step.title or ""),
            "reason_code": "evidence_reconcile_failed",
            "required_for": "step_completion",
        },
    )


def _previous_step_missing_gap_input(*, step_id: str, run_id: str) -> EvidenceRecordInput:
    return EvidenceRecordInput(
        evidence_scope=EvidenceScope.STEP,
        evidence_kind=EvidenceKind.EVIDENCE_GAP,
        run_id=run_id,
        step_id=step_id,
        source_step_id=step_id,
        source_ref=EvidenceSourceRef(source_type=EvidenceSourceType.SYSTEM_PROJECTION),
        subject_ref=EvidenceSubjectRef(subject_type="step", subject_key=f"step:{step_id}"),
        support_level=EvidenceSupportLevel.GAP,
        quality_status=EvidenceQualityStatus.MISSING_SOURCE,
        summary="previous step evidence missing",
        payload={
            "gap_type": "previous_step_evidence",
            "missing_source_types": [EvidenceSourceType.SYSTEM_PROJECTION.value],
            "claim_text": "",
            "reason_code": "previous_step_evidence_missing",
            "required_for": "execute_context",
        },
    )
