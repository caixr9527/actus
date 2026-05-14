#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence digest projector。

Projector 只从已持久化 EvidenceRecord/result_refs 构建运行期 digest 和
ResultHandle，不临时查询 fact/artifact/document 仓储拼 handle。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from app.application.service.artifact_revision_resolver import ArtifactRevisionResolver
from app.domain.models import Step
from app.domain.models.evidence import (
    EvidenceAvailableArtifactResult,
    EvidenceBackedFactProjection,
    EvidenceCompletedActionResult,
    EvidenceDigestResult,
    EvidenceDoNotRepeatResult,
    EvidenceDuplicateDecision,
    EvidenceGapResult,
    EvidenceKind,
    EvidenceQualityStatus,
    EvidenceRecord,
    EvidenceReusePolicy,
    EvidenceReuseSnapshot,
    EvidenceScope,
    EvidenceSourceType,
    EvidenceStalenessPolicy,
    EvidenceSupportLevel,
    RuntimeEvidenceContextResult,
    build_evidence_result_handle,
    build_evidence_result_refs_hash,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactRevisionResolveCommand,
    ArtifactRevisionResolveStatus,
)
from app.domain.services.runtime.contracts.evidence_runtime_ports import EvidenceRuntimeContextProviderPort

_DIGEST_LIMIT = 200
_SUMMARY_LIMIT = 1200
_SUMMARY_COMPLETED_ACTION_LIMIT = 8
_SUMMARY_DO_NOT_REPEAT_LIMIT = 8
_SUMMARY_GAP_LIMIT = 6
_EVIDENCE_BACKED_FACT_TEXT_LIMIT = 300
_SUMMARY_TOO_LONG_PROJECTION_TEXT = (
    "该 step 已形成可审计 evidence，摘要过长未注入 prompt；"
    "请通过 evidence refs/result handle 读取。"
)


@dataclass(frozen=True)
class EvidenceDigestBuildInput:
    scope: AccessScopeResult
    current_step_id: str | None
    completed_step_ids: list[str]
    stage: str = "execute"


class EvidenceDigestProjector(EvidenceRuntimeContextProviderPort):
    """基于 evidence repository 构造 digest/context。"""

    def __init__(self, *, uow_factory) -> None:
        self._uow_factory = uow_factory
        self._artifact_revision_resolver = ArtifactRevisionResolver(uow_factory=uow_factory)

    async def build_context(
            self,
            *,
            stage: str,
            scope: AccessScopeResult,
            completed_step_ids: list[str],
            step: Step | None = None,
            task_mode: str = "",
    ) -> RuntimeEvidenceContextResult | None:
        if stage not in {"execute", "replan", "summary"}:
            return None
        digest = await self.build_digest(
            scope=scope,
            current_step_id=str(getattr(step, "id", None) or scope.current_step_id or ""),
            completed_step_ids=completed_step_ids,
            stage=stage,
        )
        if digest is None:
            return None
        snapshot = None
        has_previous_completed_steps = bool(completed_step_ids)
        if has_previous_completed_steps:
            snapshot = EvidenceReuseSnapshot(
                run_id=digest.run_id,
                current_step_id=digest.current_step_id,
                source_step_ids=digest.source_step_ids,
                cursor=digest.cursor,
                do_not_repeat=digest.do_not_repeat,
                completed_actions=digest.completed_actions,
                available_artifacts=digest.available_artifacts,
                verified_claims=digest.verified_claims,
                result_handles=digest.result_handles,
            )
        result_handle_index = {handle.result_handle_id: handle for handle in digest.result_handles}
        return RuntimeEvidenceContextResult(
            run_id=digest.run_id,
            current_step_id=digest.current_step_id,
            source_step_ids=digest.source_step_ids,
            has_previous_completed_steps=has_previous_completed_steps,
            prompt_digest=digest.summary_for_prompt,
            evidence_reuse_snapshot=snapshot,
            result_handles=digest.result_handles,
            result_handle_index=result_handle_index,
            evidence_gaps=digest.evidence_gaps,
            cursor=digest.cursor,
        )

    async def build_step_evidence_backed_facts(
            self,
            *,
            scope: AccessScopeResult,
            step: Step,
    ) -> list[EvidenceBackedFactProjection]:
        step_id = str(step.id or scope.current_step_id or "").strip()
        if not step_id:
            return []
        digest = await self.build_digest(
            scope=scope,
            current_step_id=step_id,
            completed_step_ids=[step_id],
            stage="execute",
        )
        if digest is None:
            return []
        return list(digest.evidence_backed_facts or [])

    async def build_digest(
            self,
            *,
            scope: AccessScopeResult,
            current_step_id: str | None,
            completed_step_ids: list[str],
            stage: str = "execute",
    ) -> EvidenceDigestResult | None:
        run_id = str(scope.run_id or "").strip()
        session_id = str(scope.session_id or "").strip()
        if not run_id or not session_id:
            return None
        async with self._uow_factory() as uow:
            records = await uow.evidence.list_by_run(
                user_id=scope.user_id,
                session_id=session_id,
                run_id=run_id,
                evidence_scope=EvidenceScope.STEP,
                limit=_DIGEST_LIMIT,
            )
        source_step_ids = _normalize_step_ids(completed_step_ids)
        if source_step_ids:
            records = [
                record
                for record in records
                if str(record.source_step_id or record.step_id or "").strip() in set(source_step_ids)
            ]
        resolved_artifact_handle_ids = await self._resolve_artifact_handle_ids(
            scope=scope,
            records=records,
        )
        return self._project_records(
            records=records,
            run_id=run_id,
            current_step_id=current_step_id,
            source_step_ids=source_step_ids,
            resolved_artifact_handle_ids=resolved_artifact_handle_ids,
        )

    async def _resolve_artifact_handle_ids(
            self,
            *,
            scope: AccessScopeResult,
            records: list[EvidenceRecord],
    ) -> set[str]:
        resolved: set[str] = set()
        for record in records:
            if record.evidence_kind != EvidenceKind.ARTIFACT_EVIDENCE:
                continue
            for ref in list(record.result_refs or []):
                handle = build_evidence_result_handle(ref)
                if not handle.artifact_id or not handle.revision_id or not handle.content_hash:
                    continue
                result = await self._artifact_revision_resolver.resolve(
                    ArtifactRevisionResolveCommand(
                        user_id=scope.user_id,
                        workspace_id=str(scope.workspace_id),
                        session_id=str(scope.session_id),
                        artifact_id=handle.artifact_id,
                        revision_id=handle.revision_id,
                        content_hash=handle.content_hash,
                        run_id=scope.run_id,
                    )
                )
                if result.status == ArtifactRevisionResolveStatus.RESOLVED:
                    resolved.add(handle.result_handle_id)
        return resolved

    def _project_records(
            self,
            *,
            records: list[EvidenceRecord],
            run_id: str,
            current_step_id: str | None,
            source_step_ids: list[str],
            resolved_artifact_handle_ids: set[str] | None = None,
    ) -> EvidenceDigestResult:
        valid_records = [record for record in records if _result_refs_hash_matches(record)]
        handles_by_id = {}
        completed_actions: list[EvidenceCompletedActionResult] = []
        available_artifacts: list[EvidenceAvailableArtifactResult] = []
        gaps: list[EvidenceGapResult] = []
        evidence_backed_facts: list[EvidenceBackedFactProjection] = []
        do_not_repeat: list[EvidenceDoNotRepeatResult] = []

        for record in valid_records:
            record_handles = [build_evidence_result_handle(ref) for ref in record.result_refs]
            for handle in record_handles:
                handles_by_id.setdefault(handle.result_handle_id, handle)
            primary_handle = record_handles[0] if record_handles else None

            if record.evidence_kind == EvidenceKind.EVIDENCE_GAP:
                gaps.append(_gap_result(record))
                continue

            projection = _evidence_backed_fact_projection(record)
            if projection is not None:
                evidence_backed_facts.append(projection)

            if (
                    record.evidence_kind == EvidenceKind.ARTIFACT_EVIDENCE
                    and primary_handle is not None
                    and primary_handle.result_handle_id in (resolved_artifact_handle_ids or set())
            ):
                available_artifacts.append(_available_artifact(record, primary_handle))

            if record.action_key and record.subject_key:
                completed_actions.append(_completed_action(record, primary_handle))
                decision = _do_not_repeat(record, primary_handle)
                if decision is not None:
                    do_not_repeat.append(decision)

        result_handles = list(handles_by_id.values())
        cursor = _build_cursor(
            run_id=run_id,
            current_step_id=current_step_id,
            records=valid_records,
            result_handle_ids=[handle.result_handle_id for handle in result_handles],
        )
        summary_for_prompt = _build_summary(
            completed_actions=completed_actions,
            gaps=gaps,
            do_not_repeat=do_not_repeat,
        )
        return EvidenceDigestResult(
            run_id=run_id,
            current_step_id=current_step_id,
            source_step_ids=source_step_ids,
            completed_actions=completed_actions,
            available_artifacts=available_artifacts,
            evidence_gaps=gaps,
            evidence_backed_facts=evidence_backed_facts,
            do_not_repeat=do_not_repeat,
            requires_verification=[
                gap for gap in gaps if gap.reason_code in {"verification_action_missing", "result_ref_missing"}
            ],
            result_handles=result_handles,
            summary_for_prompt=summary_for_prompt,
            cursor=cursor,
        )


def _completed_action(
        record: EvidenceRecord,
        result_handle,
) -> EvidenceCompletedActionResult:
    return EvidenceCompletedActionResult(
        step_id=str(record.source_step_id or record.step_id or ""),
        action_key=str(record.action_key or ""),
        action_type=str((record.payload or {}).get("action_type") or record.evidence_kind.value),
        function_name=str((record.payload or {}).get("function_name") or ""),
        subject_key=str(record.subject_key or record.subject_ref.subject_key),
        result_status="successful" if record.quality_status == EvidenceQualityStatus.VALID else "partial",
        support_level=record.support_level,
        quality_status=record.quality_status,
        evidence_ids=[record.id],
        fact_ids=list(record.source_ref.fact_ids or []),
        result_refs=list(record.result_refs or []),
        result_handle=result_handle,
    )


def _available_artifact(
        record: EvidenceRecord,
        result_handle,
) -> EvidenceAvailableArtifactResult:
    payload = dict(record.payload or {})
    storage_ref = payload.get("storage_ref") or getattr(result_handle, "storage_ref", None)
    return EvidenceAvailableArtifactResult(
        artifact_id=str(payload.get("artifact_id") or record.primary_artifact_id or result_handle.artifact_id or ""),
        revision_id=str(payload.get("revision_id") or result_handle.revision_id or ""),
        content_hash=str(payload.get("content_hash") or result_handle.content_hash or ""),
        storage_ref=storage_ref,
        path=str(payload.get("artifact_path") or result_handle.artifact_path or ""),
        artifact_type=payload.get("artifact_type") or "file",
        delivery_state=payload.get("delivery_state") or "candidate",
        session_id=str(payload.get("session_id") or record.session_id or ""),
        run_id=str(payload.get("run_id") or record.run_id or "") or None,
        source_run_id=str(payload.get("source_run_id") or record.run_id or "") or None,
        source_event_id=str(payload.get("source_event_id") or record.source_event_id or result_handle.source_event_id or "") or None,
        source_fact_ids=_unique_strings([
            *list(payload.get("source_fact_ids") or []),
            *list(record.source_ref.fact_ids or []),
            result_handle.source_fact_id,
        ]),
        source_step_id=str(payload.get("source_step_id") or record.source_step_id or record.step_id or "") or None,
        source_kind=payload.get("source_kind") or "manual_registration",
        source_evidence_ids=[record.id],
        delivery_candidate=bool(payload.get("delivery_candidate")),
        version_locked=True,
        reuse_policy=record.reuse_policy,
        result_handle=result_handle,
    )


def _gap_result(record: EvidenceRecord) -> EvidenceGapResult:
    payload = dict(record.payload or {})
    return EvidenceGapResult(
        claim_key=record.claim_key,
        claim_text=str(payload.get("claim_text") or record.claim_text or record.summary or ""),
        source_step_id=record.source_step_id or record.step_id,
        reason_code=str(payload.get("reason_code") or "evidence_gap"),
        required_for=str(payload.get("required_for") or "execute"),
        missing_source_types=[
            item if isinstance(item, EvidenceSourceType) else EvidenceSourceType(str(item))
            for item in list(payload.get("missing_source_types") or [])
        ],
    )


def _evidence_backed_fact_projection(record: EvidenceRecord) -> EvidenceBackedFactProjection | None:
    text = _build_evidence_backed_projection_text(record)
    if text is None:
        return None
    payload = dict(record.payload or {})
    artifact_ids = _unique_strings(
        [
            *list(record.source_ref.artifact_ids or []),
            *list(payload.get("supporting_artifact_ids") or []),
            payload.get("artifact_id"),
            record.primary_artifact_id,
        ]
    )
    source_event_ids = _unique_strings(
        [
            record.source_ref.source_event_id,
            record.source_event_id,
            payload.get("source_event_id"),
        ]
    )
    user_confirmation_event_ids = source_event_ids if record.source_ref.source_type == EvidenceSourceType.USER_CONFIRMATION else []
    try:
        return EvidenceBackedFactProjection(
            text=text,
            evidence_ids=[record.id],
            fact_ids=_unique_strings(
                [
                    *list(record.source_ref.fact_ids or []),
                    *list(payload.get("source_fact_ids") or []),
                    *list(payload.get("supporting_fact_ids") or []),
                    payload.get("source_fact_id"),
                    record.primary_fact_id,
                ]
            ),
            artifact_ids=artifact_ids,
            source_event_ids=source_event_ids,
            user_confirmation_event_ids=user_confirmation_event_ids,
        )
    except ValueError:
        return None


def _build_evidence_backed_projection_text(record: EvidenceRecord) -> str | None:
    """构造可注入 prompt 的 evidence-backed 短投影文本。

    Projection 只承载短摘要和 refs，不通过扩大 text 传递完整证据内容。
    完整结果读取必须走 result handle / evidence refs。
    """
    summary = str(record.summary or "").strip()
    if not summary:
        return None
    if len(summary) <= _EVIDENCE_BACKED_FACT_TEXT_LIMIT:
        return summary
    return _SUMMARY_TOO_LONG_PROJECTION_TEXT


def _do_not_repeat(
        record: EvidenceRecord,
        result_handle,
) -> EvidenceDoNotRepeatResult | None:
    if not record.action_key or not record.subject_key:
        return None
    duplicate_decision = _duplicate_decision(record=record, has_handle=result_handle is not None)
    reuse_result_ref = record.result_refs[0] if record.result_refs and result_handle is not None else None
    return EvidenceDoNotRepeatResult(
        action_key=str(record.action_key),
        subject_key=str(record.subject_key),
        reason_code=_decision_reason_code(duplicate_decision),
        source_step_id=str(record.source_step_id or record.step_id or ""),
        evidence_ids=[record.id],
        reuse_policy=record.reuse_policy,
        staleness_policy=record.staleness_policy,
        support_level=record.support_level,
        quality_status=record.quality_status,
        result_status="successful" if record.quality_status == EvidenceQualityStatus.VALID else "partial",
        duplicate_decision=duplicate_decision,
        reuse_result_ref=reuse_result_ref,
        result_handle_id=result_handle.result_handle_id if result_handle is not None else None,
        reuse_summary=str(record.summary or "")[:300],
        message_for_model=str(record.summary or "")[:300],
    )


def _duplicate_decision(*, record: EvidenceRecord, has_handle: bool) -> EvidenceDuplicateDecision:
    if (
            has_handle
            and record.reusable
            and record.reuse_policy == EvidenceReusePolicy.REUSE_ALLOWED
            and record.staleness_policy in {EvidenceStalenessPolicy.RUN_SCOPED, EvidenceStalenessPolicy.STABLE}
            and record.support_level == EvidenceSupportLevel.STRONG
            and record.quality_status == EvidenceQualityStatus.VALID
    ):
        return EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION
    if record.reuse_policy == EvidenceReusePolicy.VERIFY_BEFORE_REUSE or record.staleness_policy == EvidenceStalenessPolicy.EXTERNAL_MAY_CHANGE:
        return EvidenceDuplicateDecision.REQUIRE_VERIFICATION
    return EvidenceDuplicateDecision.BLOCK_DUPLICATE_ACTION


def _decision_reason_code(decision: EvidenceDuplicateDecision) -> str:
    if decision == EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION:
        return "evidence_reuse_pending_resolution"
    if decision == EvidenceDuplicateDecision.REQUIRE_VERIFICATION:
        return "evidence_reuse_requires_verification"
    return "evidence_duplicate_blocked"


def _build_summary(
        *,
        completed_actions: list[EvidenceCompletedActionResult],
        gaps: list[EvidenceGapResult],
        do_not_repeat: list[EvidenceDoNotRepeatResult],
) -> str:
    builder = _PromptSummaryBuilder(limit=_SUMMARY_LIMIT)
    omitted: dict[str, int] = {
        "completed_actions": max(len(completed_actions) - _SUMMARY_COMPLETED_ACTION_LIMIT, 0),
        "do_not_repeat": max(len(do_not_repeat) - _SUMMARY_DO_NOT_REPEAT_LIMIT, 0),
        "evidence_gaps": max(len(gaps) - _SUMMARY_GAP_LIMIT, 0),
    }
    if completed_actions:
        builder.append("completed_actions:")
        for index, item in enumerate(completed_actions[:_SUMMARY_COMPLETED_ACTION_LIMIT]):
            if not builder.append(f"- {item.action_type} {item.subject_key} ({item.result_status})"):
                omitted["completed_actions"] += len(completed_actions[:_SUMMARY_COMPLETED_ACTION_LIMIT]) - index
                break
    if do_not_repeat:
        if builder.append("do_not_repeat:"):
            for index, item in enumerate(do_not_repeat[:_SUMMARY_DO_NOT_REPEAT_LIMIT]):
                if not builder.append(f"- {item.subject_key}: {item.duplicate_decision.value}"):
                    omitted["do_not_repeat"] += len(do_not_repeat[:_SUMMARY_DO_NOT_REPEAT_LIMIT]) - index
                    break
        else:
            omitted["do_not_repeat"] += min(len(do_not_repeat), _SUMMARY_DO_NOT_REPEAT_LIMIT)
    if gaps:
        if builder.append("evidence_gaps:"):
            for index, item in enumerate(gaps[:_SUMMARY_GAP_LIMIT]):
                if not builder.append(f"- {item.source_step_id or ''}: {item.reason_code}"):
                    omitted["evidence_gaps"] += len(gaps[:_SUMMARY_GAP_LIMIT]) - index
                    break
        else:
            omitted["evidence_gaps"] += min(len(gaps), _SUMMARY_GAP_LIMIT)
    _append_omitted_summary(builder=builder, omitted=omitted)
    return builder.render()


class _PromptSummaryBuilder:
    """按完整行构造 prompt 摘要，禁止整体切片产生半行语义。"""

    def __init__(self, *, limit: int) -> None:
        self._limit = limit
        self._lines: list[str] = []
        self._length = 0

    def append(self, line: str) -> bool:
        normalized_line = str(line or "")
        next_length = self._length + len(normalized_line) + (1 if self._lines else 0)
        if next_length > self._limit:
            return False
        self._lines.append(normalized_line)
        self._length = next_length
        return True

    def render(self) -> str:
        return "\n".join(self._lines)

    def pop_last(self) -> bool:
        if not self._lines:
            return False
        self._length -= len(self._lines.pop()) + (1 if self._lines else 0)
        return True

    def snapshot(self) -> tuple[list[str], int]:
        return list(self._lines), self._length

    def restore(self, snapshot: tuple[list[str], int]) -> None:
        self._lines, self._length = list(snapshot[0]), int(snapshot[1])


def _append_omitted_summary(*, builder: _PromptSummaryBuilder, omitted: dict[str, int]) -> None:
    if not any(count > 0 for count in omitted.values()):
        return
    footer_lines = [
        f"... omitted_completed_actions={max(int(omitted.get('completed_actions') or 0), 0)}",
        f"... omitted_do_not_repeat={max(int(omitted.get('do_not_repeat') or 0), 0)}",
        f"... omitted_evidence_gaps={max(int(omitted.get('evidence_gaps') or 0), 0)}",
        "... full_structured_context_available=true",
    ]
    while footer_lines:
        snapshot = builder.snapshot()
        if all(builder.append(line) for line in footer_lines):
            return
        builder.restore(snapshot)
        if not builder.pop_last():
            footer_lines.pop(0)


def _result_refs_hash_matches(record: EvidenceRecord) -> bool:
    return record.result_refs_hash == build_evidence_result_refs_hash(record.result_refs)


def _normalize_step_ids(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _unique_strings(values: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _build_cursor(
        *,
        run_id: str,
        current_step_id: str | None,
        records: list[EvidenceRecord],
        result_handle_ids: list[str],
) -> str:
    payload = "|".join(
        [
            run_id,
            current_step_id or "",
            *sorted(record.id for record in records),
            *sorted(result_handle_ids),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
