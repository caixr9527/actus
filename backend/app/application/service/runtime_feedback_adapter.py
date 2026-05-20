#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PR4 运行反馈来源到 Feedback Ledger command 的适配器。"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

from app.domain.models import ToolEvent
from app.domain.models.evidence import EvidenceKind, EvidenceRecord
from app.domain.models.feedback import (
    FeedbackCategory,
    FeedbackClassificationResult,
    FeedbackDataOrigin,
    FeedbackGapKind,
    FeedbackGapResult,
    FeedbackKind,
    FeedbackPromptSafeSummaryResult,
    FeedbackReasonCode,
    FeedbackRecorderPort,
    RuntimeFeedbackGapSinkPort,
    FeedbackSnapshotStage,
    FeedbackSourceConfidence,
    FeedbackSourceKind,
    FeedbackSourceRefResult,
    FeedbackSummaryKind,
    FeedbackSummaryResult,
    FeedbackTargetRefResult,
    FeedbackTargetType,
    FeedbackWriteResult,
    RuntimeFeedbackCommand,
    build_evidence_snapshot_missing_gap,
)
from app.domain.models.safety_audit import SafetyAuditDecision, SafetyAuditRecord
from app.domain.models.sandbox_fact import SandboxFactKind, SandboxFactRecord
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)

logger = logging.getLogger(__name__)

_SUMMARY_MAX_CHARS = 300
_RUNTIME_DECAY_POLICY = "run_window"
_RUNTIME_TTL_SCOPE = "run"


class RuntimeFeedbackAdapter:
    """把受控运行期来源转换为 `RuntimeFeedbackCommand` 并写入反馈账本。"""

    def __init__(
            self,
            *,
            feedback_recorder: FeedbackRecorderPort,
            feedback_gap_sink: RuntimeFeedbackGapSinkPort | None = None,
    ) -> None:
        self._feedback_recorder = feedback_recorder
        self._feedback_gap_sink = feedback_gap_sink

    async def record_tool_event_feedback(
            self,
            *,
            scope: AccessScopeResult,
            event: ToolEvent,
            source_event_id: str,
            facts: list[SandboxFactRecord],
    ) -> list[FeedbackWriteResult]:
        """ToolEvent 持久化后记录工具失败强/弱运行反馈。"""

        if not _is_failed_tool_event(event):
            return []
        tool_call_id = _normalize_text(event.tool_call_id)
        if not tool_call_id:
            result = self._source_incomplete_gap(
                diagnostic_summary="工具失败事件缺少 tool_call_id，无法绑定反馈目标。",
                source_event_id=source_event_id,
            )
            self._append_runtime_gap(result.gap)
            return [
                result
            ]

        failure_facts = [
            fact
            for fact in list(facts or [])
            if fact.fact_kind == SandboxFactKind.TOOL_FAILURE
        ]
        if failure_facts:
            results: list[FeedbackWriteResult] = []
            for fact in failure_facts:
                results.append(await self.record_sandbox_fact_tool_failure(
                    scope=scope,
                    fact=fact,
                ))
            return results

        return [
            await self.record_weak_tool_failure(
                scope=scope,
                event=event,
                source_event_id=source_event_id,
            )
        ]

    async def record_sandbox_fact_tool_failure(
            self,
            *,
            scope: AccessScopeResult,
            fact: SandboxFactRecord,
    ) -> FeedbackWriteResult:
        """已持久化 TOOL_FAILURE fact 生成 strong runtime feedback。"""

        source_event_id = _normalize_text(fact.source_ref.source_event_id)
        tool_call_id = _normalize_text(fact.source_ref.tool_call_id)
        if not source_event_id or not tool_call_id:
            result = self._source_incomplete_gap(
                diagnostic_summary="TOOL_FAILURE fact 缺少 source_event_id 或 tool_call_id。",
                source_event_id=source_event_id,
            )
            self._append_runtime_gap(result.gap)
            return result

        command = self._runtime_command(
            scope=scope,
            source_ref=FeedbackSourceRefResult(
                source_kind=FeedbackSourceKind.SANDBOX_FACT,
                source_event_id=source_event_id,
                source_record_refs=[
                    {
                        "fact_id": fact.id,
                        "source_event_id": source_event_id,
                        "tool_call_id": tool_call_id,
                    }
                ],
                source_run_id=fact.run_id,
                source_step_id=fact.step_id,
                source_summary=_safe_summary(fact.summary or "工具调用失败。"),
            ),
            target_ref=FeedbackTargetRefResult(
                target_type=FeedbackTargetType.TOOL_CALL,
                target_id=tool_call_id,
                target_run_id=_run_id_from_scope_or_record(scope, fact.run_id),
            ),
            category=FeedbackCategory.TOOL_FAILURE,
            reason_code=FeedbackReasonCode.TOOL_FAILED,
            source_confidence=FeedbackSourceConfidence.STRONG,
            summary_text=_safe_summary(fact.summary or "工具调用失败。"),
            profile_hash=_normalize_text(fact.profile_ref.profile_hash),
        )
        return await self._record(command)

    async def record_weak_tool_failure(
            self,
            *,
            scope: AccessScopeResult,
            event: ToolEvent,
            source_event_id: str,
    ) -> FeedbackWriteResult:
        """缺少 SandboxFact 时，仅以合法 ToolEvent 写入弱反馈。"""

        event_id = _normalize_text(source_event_id or event.id)
        tool_call_id = _normalize_text(event.tool_call_id)
        if not event_id or not tool_call_id:
            result = self._source_incomplete_gap(
                diagnostic_summary="工具失败弱反馈缺少 source_event_id 或 tool_call_id。",
                source_event_id=event_id,
            )
            self._append_runtime_gap(result.gap)
            return result

        command = self._runtime_command(
            scope=scope,
            source_ref=FeedbackSourceRefResult(
                source_kind=FeedbackSourceKind.TOOL_EVENT,
                source_event_id=event_id,
                source_record_refs=[
                    {
                        "event_id": event_id,
                        "tool_call_id": tool_call_id,
                        "function_name": _normalize_text(event.function_name),
                    }
                ],
                source_run_id=str(scope.run_id or "") or None,
                source_step_id=str(scope.current_step_id or event.step_id or "") or None,
                source_summary=_safe_summary(f"工具 {event.function_name} 调用失败，但缺少可用 SandboxFact。"),
            ),
            target_ref=FeedbackTargetRefResult(
                target_type=FeedbackTargetType.TOOL_CALL,
                target_id=tool_call_id,
                target_run_id=_run_id_from_scope_or_record(scope, None),
            ),
            category=FeedbackCategory.TOOL_FAILURE,
            reason_code=FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE,
            source_confidence=FeedbackSourceConfidence.WEAK,
            summary_text=_safe_summary("工具调用失败，但结构化失败 fact 暂不可用，后续步骤需要重新验证。"),
        )
        return await self._record(command)

    async def record_evidence_gap(
            self,
            *,
            scope: AccessScopeResult,
            evidence: EvidenceRecord,
    ) -> FeedbackWriteResult:
        """已持久化 EvidenceRecord(evidence_kind=gap) 生成 evidence gap 运行反馈。"""

        if evidence.evidence_kind != EvidenceKind.EVIDENCE_GAP:
            result = self._source_incomplete_gap(
                diagnostic_summary="非 evidence gap 记录不能生成 evidence_gap 运行反馈。",
                source_event_id=evidence.source_event_id,
            )
            self._append_runtime_gap(result.gap)
            return result
        source_event_id = _normalize_text(evidence.source_event_id)
        if not source_event_id:
            result = self._source_incomplete_gap(
                diagnostic_summary="Evidence gap 缺少 source_event_id。",
                source_event_id=None,
            )
            self._append_runtime_gap(result.gap)
            return result

        summary = _safe_summary(evidence.summary or evidence.claim_text or "存在未满足的 evidence gap。")
        command = self._runtime_command(
            scope=scope,
            source_ref=FeedbackSourceRefResult(
                source_kind=FeedbackSourceKind.EVIDENCE_GAP,
                source_event_id=source_event_id,
                source_record_refs=[{"evidence_id": evidence.id}],
                source_run_id=evidence.run_id,
                source_step_id=evidence.step_id,
                source_summary=summary,
            ),
            target_ref=FeedbackTargetRefResult(
                target_type=FeedbackTargetType.EVIDENCE_GAP,
                target_id=evidence.id,
                target_run_id=_run_id_from_scope_or_record(scope, evidence.run_id),
            ),
            category=FeedbackCategory.EVIDENCE_GAP,
            reason_code=FeedbackReasonCode.EVIDENCE_GAP_DETECTED,
            source_confidence=FeedbackSourceConfidence.STRONG,
            summary_text=summary,
        )
        return await self._record(command)

    def evidence_snapshot_missing_gap(self, *, diagnostic_summary: str | None = None) -> FeedbackWriteResult:
        """PR4 首版 snapshot missing 只作为 transient gap，不落库。"""

        return FeedbackWriteResult(
            success=False,
            created=False,
            reused=False,
            reason_code=FeedbackReasonCode.EVIDENCE_SNAPSHOT_MISSING,
            feedback_id=None,
            record_ref=None,
            scope=None,
            source_ref=None,
            target_ref=None,
            resolution=None,
            gap=build_evidence_snapshot_missing_gap(
                stage=FeedbackSnapshotStage.EXECUTE,
                diagnostic_summary=(
                    diagnostic_summary
                    or "Evidence snapshot missing 缺少合法 source event/source record，未落库。"
                ),
            ),
            created_at=None,
        )

    async def record_safety_audit_feedbacks(
            self,
            *,
            scope: AccessScopeResult,
            audits: Iterable[SafetyAuditRecord],
    ) -> list[FeedbackWriteResult]:
        results: list[FeedbackWriteResult] = []
        for audit in list(audits or []):
            result = await self.record_safety_audit_feedback(scope=scope, audit=audit)
            if result is not None:
                results.append(result)
        return results

    async def record_safety_audit_feedback(
            self,
            *,
            scope: AccessScopeResult,
            audit: SafetyAuditRecord,
    ) -> FeedbackWriteResult | None:
        """Safety Audit block/rewrite/confirmation missing 生成运行反馈。"""

        mapped = _map_safety_decision(audit.decision)
        if mapped is None:
            return None
        category, reason_code = mapped
        source_event_id = _normalize_text(audit.tool_event_source_event_id or audit.decision_event_id)
        if not source_event_id:
            result = self._source_incomplete_gap(
                diagnostic_summary="Safety Audit 缺少可回链 source event。",
                source_event_id=None,
            )
            self._append_runtime_gap(result.gap)
            return result

        target_type = FeedbackTargetType.TOOL_CALL if _normalize_text(audit.tool_call_id) else FeedbackTargetType.RUN
        target_id = _normalize_text(audit.tool_call_id) or audit.run_id
        command = self._runtime_command(
            scope=scope,
            source_ref=FeedbackSourceRefResult(
                source_kind=FeedbackSourceKind.SAFETY_AUDIT,
                source_event_id=source_event_id,
                source_record_refs=[{"audit_id": audit.id}],
                source_run_id=audit.run_id,
                source_step_id=audit.step_id,
                source_summary=_safe_summary(audit.reason_code or f"Safety Audit {audit.decision.value}。"),
            ),
            target_ref=FeedbackTargetRefResult(
                target_type=target_type,
                target_id=target_id,
                target_run_id=audit.run_id,
            ),
            category=category,
            reason_code=reason_code,
            source_confidence=FeedbackSourceConfidence.STRONG,
            summary_text=_safe_summary(audit.reason_code or f"Safety Audit {audit.decision.value}。"),
        )
        return await self._record(command)

    async def _record(self, command: RuntimeFeedbackCommand) -> FeedbackWriteResult:
        try:
            result = await self._feedback_recorder.record_runtime_feedback(command)
        except Exception:
            logger.exception(
                "runtime_feedback_record_failed",
                extra={
                    "user_id": command.access_scope.user_id,
                    "session_id": str(command.access_scope.session_id),
                    "run_id": command.access_scope.run_id,
                    "source_event_id": command.source_ref.source_event_id,
                    "category": command.category.value,
                    "reason_code": "feedback_record_failed",
                },
            )
            result = FeedbackWriteResult(
                success=False,
                created=False,
                reused=False,
                reason_code=FeedbackReasonCode.FEEDBACK_RECORD_FAILED,
                feedback_id=None,
                record_ref=None,
                scope=None,
                source_ref=command.source_ref,
                target_ref=command.target_ref,
                resolution=None,
                gap=FeedbackGapResult(
                    gap_kind=FeedbackGapKind.RECORD_FAILED,
                    reason_code=FeedbackReasonCode.FEEDBACK_RECORD_FAILED,
                    source_ref=command.source_ref,
                    target_ref=command.target_ref,
                    diagnostic_summary="运行反馈写入失败。",
                    stage=FeedbackSnapshotStage.EXECUTE,
                    created_at=datetime.now(),
                ),
                created_at=None,
            )
            self._append_runtime_gap(result.gap)
            return result
        if result.success is False and result.gap is not None:
            self._append_runtime_gap(result.gap)
            logger.warning(
                "runtime_feedback_gap user_id=%s session_id=%s run_id=%s reason_code=%s",
                command.access_scope.user_id,
                command.access_scope.session_id,
                command.access_scope.run_id,
                result.gap.reason_code.value,
            )
        return result

    def _append_runtime_gap(self, gap: FeedbackGapResult | None) -> None:
        if gap is None or self._feedback_gap_sink is None:
            return
        self._feedback_gap_sink.append_feedback_gap(gap)

    def _runtime_command(
            self,
            *,
            scope: AccessScopeResult,
            source_ref: FeedbackSourceRefResult,
            target_ref: FeedbackTargetRefResult,
            category: FeedbackCategory,
            reason_code: FeedbackReasonCode,
            source_confidence: FeedbackSourceConfidence,
            summary_text: str,
            profile_hash: str | None = None,
    ) -> RuntimeFeedbackCommand:
        classification = FeedbackClassificationResult(
            privacy_level=PrivacyLevel.INTERNAL,
            retention_policy=RetentionPolicyKind.SESSION_BOUND,
            trust_level=DataTrustLevel.SYSTEM_GENERATED,
            source_confidence=source_confidence,
            data_origin=FeedbackDataOrigin.RUNTIME,
        )
        run_id = str(scope.run_id or target_ref.target_run_id or source_ref.source_run_id or "").strip() or None
        return RuntimeFeedbackCommand(
            access_scope=scope,
            source_ref=source_ref,
            target_ref=target_ref,
            kind=FeedbackKind.RUNTIME_FEEDBACK,
            category=category,
            reason_code=reason_code,
            feedback_summary=FeedbackSummaryResult(
                summary_text=_safe_summary(summary_text),
                summary_kind=FeedbackSummaryKind.RUNTIME_DIAGNOSTIC,
                is_truncated=False,
                truncation_reason=None,
                language="zh-CN",
            ),
            prompt_safe_summary=FeedbackPromptSafeSummaryResult(
                summary_text=_safe_summary(summary_text),
                is_truncated=False,
                sanitization_applied=False,
                sanitization_reasons=[],
                prompt_visible=True,
            ),
            classification=classification,
            requested_feedback_scope_kind=None,
            requested_scope_id=None,
            current_run_id_at_record_time=run_id,
            step_id=str(scope.current_step_id or source_ref.source_step_id or "") or None,
            profile_hash=profile_hash,
            decay_policy=_RUNTIME_DECAY_POLICY,
            ttl_scope=_RUNTIME_TTL_SCOPE,
            origin=DataOrigin.SYSTEM_OPERATIONAL,
            trust_level=classification.trust_level,
            privacy_level=classification.privacy_level,
            retention_policy=classification.retention_policy,
        )

    @staticmethod
    def _source_incomplete_gap(
            *,
            diagnostic_summary: str,
            source_event_id: str | None,
    ) -> FeedbackWriteResult:
        source_ref = None
        if _normalize_text(source_event_id):
            source_ref = FeedbackSourceRefResult(
                source_kind=FeedbackSourceKind.TOOL_EVENT,
                source_event_id=str(source_event_id),
                source_record_refs=[{"event_id": str(source_event_id)}],
                source_run_id=None,
                source_step_id=None,
                source_summary=_safe_summary(diagnostic_summary),
            )
        return FeedbackWriteResult(
            success=False,
            created=False,
            reused=False,
            reason_code=FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE,
            feedback_id=None,
            record_ref=None,
            scope=None,
            source_ref=source_ref,
            target_ref=None,
            resolution=None,
            gap=FeedbackGapResult(
                gap_kind=FeedbackGapKind.SOURCE_INCOMPLETE,
                reason_code=FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE,
                source_ref=source_ref,
                target_ref=None,
                diagnostic_summary=_safe_summary(diagnostic_summary),
                stage=FeedbackSnapshotStage.EXECUTE,
                created_at=datetime.now(),
            ),
            created_at=None,
        )


def _is_failed_tool_event(event: ToolEvent) -> bool:
    result = event.function_result
    return result is not None and result.success is False


def _map_safety_decision(
        decision: SafetyAuditDecision,
) -> tuple[FeedbackCategory, FeedbackReasonCode] | None:
    if decision == SafetyAuditDecision.BLOCK:
        return FeedbackCategory.SAFETY_BLOCKED, FeedbackReasonCode.SAFETY_BLOCKED
    if decision == SafetyAuditDecision.REWRITE:
        return FeedbackCategory.SAFETY_REWRITE, FeedbackReasonCode.SAFETY_REWRITE_APPLIED
    return None


def _run_id_from_scope_or_record(scope: AccessScopeResult, record_run_id: str | None) -> str:
    run_id = _normalize_text(record_run_id) or _normalize_text(scope.run_id)
    if not run_id:
        raise ValueError("runtime feedback target 必须包含 run_id")
    return run_id


def _safe_summary(value: str) -> str:
    normalized = " ".join(str(value or "").split())
    if not normalized:
        normalized = "运行反馈诊断。"
    if len(normalized) <= _SUMMARY_MAX_CHARS:
        return normalized
    return normalized[:_SUMMARY_MAX_CHARS - 1] + "…"


def _normalize_text(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None
