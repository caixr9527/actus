#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback snapshot builder。"""

from __future__ import annotations

from datetime import datetime

from app.application.service.feedback_ledger_common import (
    CANDIDATE_SCAN_LIMIT,
    DO_NOT_REPEAT_CATEGORIES,
    EXCLUDED_LIST_LIMIT,
    FINAL_GATE_ONLY_CATEGORIES,
    OPEN_LIST_LIMIT,
    PROMPT_LIST_LIMIT,
    QUALITY_SUMMARY_CATEGORIES,
    REPLAN_HINT_CATEGORIES,
    RESOLVED_LIST_LIMIT,
    SEVERITY_PRIORITY,
    STATUS_PRIORITY,
    SUMMARY_USER_CATEGORIES,
    USER_CONSTRAINT_CATEGORIES,
    FeedbackSanitizer,
)
from app.domain.models.feedback import (
    ExcludedFeedbackRefResult,
    FeedbackCategory,
    FeedbackExcludedBy,
    FeedbackGapKind,
    FeedbackGapResult,
    FeedbackKind,
    FeedbackReasonCode,
    FeedbackRecord,
    FeedbackResolutionReasonCode,
    FeedbackScopeKind,
    FeedbackSeverity,
    FeedbackSnapshotCursorResult,
    FeedbackSnapshotItemResult,
    FeedbackSnapshotResult,
    FeedbackSnapshotScopeResult,
    FeedbackSnapshotStage,
    FeedbackStatus,
)


class FeedbackSnapshotPolicy:
    """snapshot 窗口/排序/去重集中策略。"""

    candidate_scan_limit = CANDIDATE_SCAN_LIMIT
    prompt_list_limit = PROMPT_LIST_LIMIT
    open_list_limit = OPEN_LIST_LIMIT
    resolved_list_limit = RESOLVED_LIST_LIMIT
    excluded_list_limit = EXCLUDED_LIST_LIMIT
    feedback_gaps_limit = 20

    @staticmethod
    def status_priority(status: FeedbackStatus) -> int:
        return STATUS_PRIORITY[status]

    @staticmethod
    def severity_priority(severity: FeedbackSeverity) -> int:
        return SEVERITY_PRIORITY[severity]

    @staticmethod
    def dedupe_key(record: FeedbackRecord) -> tuple[str, str, str, str, str, str]:
        return (
            record.kind.value,
            record.category.value,
            record.target_type.value,
            record.target_id,
            record.target_revision_id or "",
            record.target_content_hash or "",
        )


class FeedbackSnapshotBuilder:
    """按 stage 构建只读 snapshot。"""

    def __init__(
            self,
            *,
            policy: FeedbackSnapshotPolicy | None = None,
            sanitizer: FeedbackSanitizer | None = None,
    ) -> None:
        self._policy = policy or FeedbackSnapshotPolicy()
        self._sanitizer = sanitizer or FeedbackSanitizer()

    def build(
            self,
            *,
            scope: FeedbackSnapshotScopeResult,
            stage: FeedbackSnapshotStage,
            records: list[FeedbackRecord],
            runtime_gaps: list[FeedbackGapResult] | None = None,
            now: datetime,
    ) -> FeedbackSnapshotResult:
        if not isinstance(stage, FeedbackSnapshotStage):
            raise ValueError("stage 必须是 FeedbackSnapshotStage 枚举值")
        scanned_records = sorted(records, key=self._sort_key, reverse=True)[: self._policy.candidate_scan_limit]
        winners: list[FeedbackRecord] = []
        excluded_refs: list[ExcludedFeedbackRefResult] = []
        dedupe_winners: dict[tuple[str, str, str, str, str, str], FeedbackRecord] = {}
        for record in scanned_records:
            exclusion = self._evaluate_pre_window_exclusion(scope=scope, stage=stage, record=record)
            if exclusion is not None:
                excluded_refs.append(exclusion)
                continue
            dedupe_key = self._policy.dedupe_key(record)
            if dedupe_key in dedupe_winners:
                excluded_refs.append(self._exclude(record, stage, scope, FeedbackExcludedBy.DEDUPE))
                continue
            dedupe_winners[dedupe_key] = record
            winners.append(record)

        open_candidates = [record for record in winners if record.status == FeedbackStatus.OPEN]
        resolved_candidates = [record for record in winners if record.status != FeedbackStatus.OPEN]
        open_records = open_candidates[: self._policy.open_list_limit]
        resolved_candidates.sort(
            key=lambda record: (record.updated_at, self._policy.severity_priority(record.severity), record.id),
            reverse=True,
        )
        resolved_records = resolved_candidates[: self._policy.resolved_list_limit]
        for record in open_candidates[self._policy.open_list_limit:]:
            excluded_refs.append(self._exclude(record, stage, scope, FeedbackExcludedBy.WINDOW))
        for record in resolved_candidates[self._policy.resolved_list_limit:]:
            excluded_refs.append(self._exclude(record, stage, scope, FeedbackExcludedBy.WINDOW))

        included_records = open_records + resolved_records
        active_user = self._limit_items([self._to_item(record) for record in open_records if record.kind == FeedbackKind.USER_FEEDBACK])
        active_runtime = self._limit_items([self._to_item(record) for record in open_records if record.kind == FeedbackKind.RUNTIME_FEEDBACK])
        active_quality = self._limit_items([self._to_item(record) for record in open_records if record.kind == FeedbackKind.QUALITY_FEEDBACK])
        open_items = [self._to_item(record) for record in open_records]
        resolved_items = [self._to_item(record) for record in resolved_records]
        do_not_repeat = self._limit_items(
            [
                self._to_item(record)
                for record in open_records
                if record.category in DO_NOT_REPEAT_CATEGORIES and not self._is_weak_incomplete_feedback(record)
            ]
        )
        user_constraints = self._limit_items(
            [
                self._to_item(record)
                for record in open_records
                if record.kind == FeedbackKind.USER_FEEDBACK and record.category in USER_CONSTRAINT_CATEGORIES
            ]
        )
        replan_hints = self._limit_items(
            [
                self._to_item(record)
                for record in open_records
                if record.category in REPLAN_HINT_CATEGORIES or self._is_weak_incomplete_feedback(record)
            ]
        )
        review_hints = self._limit_items([self._to_item(record) for record in open_records if record.kind == FeedbackKind.QUALITY_FEEDBACK])
        final_gate_hints = self._limit_items(
            [self._to_item(record) for record in open_records if record.category in FINAL_GATE_ONLY_CATEGORIES]
        )
        gaps = self._merge_runtime_gaps(runtime_gaps=runtime_gaps, scope=scope, stage=stage, now=now)
        latest_feedback_id = None
        if scanned_records:
            latest_record = max(scanned_records, key=lambda record: (record.created_at, record.id))
            latest_feedback_id = latest_record.id
        return FeedbackSnapshotResult(
            scope=scope,
            snapshot_id=f"feedback-snapshot:{scope.feedback_scope_kind.value}:{scope.scope_id}:{stage.value}:{int(now.timestamp())}",
            source_run_id=scope.current_run_id_at_snapshot_time,
            stage=stage,
            active_user_feedback=active_user,
            active_runtime_feedback=active_runtime,
            active_quality_feedback=active_quality,
            open_feedback_items=open_items,
            resolved_feedback_items=resolved_items,
            do_not_repeat_feedback=do_not_repeat,
            user_constraints=user_constraints,
            replan_hints=replan_hints,
            review_hints=review_hints,
            final_gate_hints=final_gate_hints,
            feedback_gaps=gaps,
            included_feedback_ids=[record.id for record in included_records],
            excluded_feedback_refs=excluded_refs[: self._policy.excluded_list_limit],
            cursor=FeedbackSnapshotCursorResult(
                latest_feedback_id=latest_feedback_id,
                source_record_ids=[record.id for record in scanned_records],
            ),
            created_at=now,
        )

    def _sort_key(self, record: FeedbackRecord) -> tuple[int, int, datetime, str]:
        return (
            self._policy.status_priority(record.status),
            self._policy.severity_priority(record.severity),
            record.created_at,
            record.id,
        )

    def _evaluate_pre_window_exclusion(
            self,
            *,
            scope: FeedbackSnapshotScopeResult,
            stage: FeedbackSnapshotStage,
            record: FeedbackRecord,
    ) -> ExcludedFeedbackRefResult | None:
        if scope.feedback_scope_kind != record.feedback_scope_kind or scope.scope_id != record.scope_id:
            return self._exclude(record, stage, scope, FeedbackExcludedBy.SCOPE)
        if self._is_ttl_excluded(scope=scope, stage=stage, record=record):
            return self._exclude(record, stage, scope, FeedbackExcludedBy.TTL)
        if not self._matches_stage(scope=scope, stage=stage, record=record):
            return self._exclude(record, stage, scope, FeedbackExcludedBy.STAGE_POLICY)
        return None

    def _is_ttl_excluded(
            self,
            *,
            scope: FeedbackSnapshotScopeResult,
            stage: FeedbackSnapshotStage,
            record: FeedbackRecord,
    ) -> bool:
        if stage == FeedbackSnapshotStage.EVALUATION:
            return False
        if record.decay_policy != "run_window":
            return False
        if scope.feedback_scope_kind != FeedbackScopeKind.SESSION:
            return False
        current_run_id = scope.current_run_id_at_snapshot_time
        if current_run_id is None:
            return True
        return record.target_run_id != current_run_id

    def _matches_stage(
            self,
            *,
            scope: FeedbackSnapshotScopeResult,
            stage: FeedbackSnapshotStage,
            record: FeedbackRecord,
    ) -> bool:
        if scope.feedback_scope_kind != record.feedback_scope_kind or scope.scope_id != record.scope_id:
            return False
        if record.status != FeedbackStatus.OPEN:
            return stage != FeedbackSnapshotStage.DIRECT_ANSWER
        if stage == FeedbackSnapshotStage.PLANNER:
            return record.kind == FeedbackKind.USER_FEEDBACK
        if stage == FeedbackSnapshotStage.EXECUTE:
            return record.kind in {FeedbackKind.USER_FEEDBACK, FeedbackKind.RUNTIME_FEEDBACK}
        if stage == FeedbackSnapshotStage.REPLAN:
            if record.kind == FeedbackKind.RUNTIME_FEEDBACK:
                return True
            if record.kind == FeedbackKind.USER_FEEDBACK:
                return record.category in {
                    FeedbackCategory.CORRECTION,
                    FeedbackCategory.DISSATISFACTION,
                    FeedbackCategory.CLARIFICATION,
                    FeedbackCategory.CONFIRMATION,
                    FeedbackCategory.SELECTION,
                    FeedbackCategory.PREFERENCE,
                }
            return record.category in FINAL_GATE_ONLY_CATEGORIES or record.category in {
                FeedbackCategory.SELF_REVIEW_FAILED,
                FeedbackCategory.UNMET_REQUIREMENT,
                FeedbackCategory.MISSING_EVIDENCE,
            }
        if stage == FeedbackSnapshotStage.SUMMARY:
            if record.kind == FeedbackKind.USER_FEEDBACK:
                return record.category in SUMMARY_USER_CATEGORIES
            if record.kind == FeedbackKind.QUALITY_FEEDBACK:
                return record.category in QUALITY_SUMMARY_CATEGORIES
            return False
        if stage == FeedbackSnapshotStage.FUTURE_REVIEW:
            return True
        if stage == FeedbackSnapshotStage.FINAL_GATE:
            return record.kind in {FeedbackKind.USER_FEEDBACK, FeedbackKind.QUALITY_FEEDBACK}
        if stage == FeedbackSnapshotStage.EVALUATION:
            return True
        if stage == FeedbackSnapshotStage.DIRECT_ANSWER:
            return record.kind == FeedbackKind.USER_FEEDBACK and record.category == FeedbackCategory.PREFERENCE
        return False

    def _to_item(self, record: FeedbackRecord) -> FeedbackSnapshotItemResult:
        return FeedbackSnapshotItemResult(
            feedback_id=record.id,
            kind=record.kind,
            category=record.category,
            status=record.status,
            severity=record.severity,
            target_ref=record.target_ref,
            source_kind=record.source_kind,
            source_event_id=record.source_event_id,
            source_run_id=record.source_run_id,
            target_run_id=record.target_run_id,
            prompt_safe_summary=record.prompt_safe_summary.summary_text,
            reason_code=record.reason_code,
            resolution_reason_code=record.resolution_reason_code,
            created_at=record.created_at,
        )

    def _exclude(
            self,
            record: FeedbackRecord,
            stage: FeedbackSnapshotStage,
            scope: FeedbackSnapshotScopeResult,
            excluded_by: FeedbackExcludedBy,
    ) -> ExcludedFeedbackRefResult:
        if excluded_by in {FeedbackExcludedBy.STATUS, FeedbackExcludedBy.TTL} and record.resolution_reason_code is not None:
            reason_code: FeedbackReasonCode | FeedbackResolutionReasonCode = record.resolution_reason_code
        else:
            reason_code = FeedbackReasonCode.FEEDBACK_PROJECTION_GAP
        return ExcludedFeedbackRefResult(
            feedback_id=record.id,
            reason_code=reason_code,
            stage=stage,
            scope=scope,
            status=record.status,
            excluded_by=excluded_by,
            created_at=record.updated_at,
        )

    def _is_weak_incomplete_feedback(self, record: FeedbackRecord) -> bool:
        return (
            record.kind == FeedbackKind.RUNTIME_FEEDBACK
            and record.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE
            and record.severity == FeedbackSeverity.WARNING
        )

    def _limit_items(self, items: list[FeedbackSnapshotItemResult]) -> list[FeedbackSnapshotItemResult]:
        return items[: self._policy.prompt_list_limit]

    def _merge_runtime_gaps(
            self,
            *,
            runtime_gaps: list[FeedbackGapResult] | None,
            scope: FeedbackSnapshotScopeResult,
            stage: FeedbackSnapshotStage,
            now: datetime,
    ) -> list[FeedbackGapResult]:
        gaps = list(runtime_gaps or [])
        if len(gaps) <= self._policy.feedback_gaps_limit:
            return gaps
        kept = gaps[: self._policy.feedback_gaps_limit - 1]
        kept.append(
            FeedbackGapResult(
                gap_kind=FeedbackGapKind.PROJECTION_MISSING,
                reason_code=FeedbackReasonCode.FEEDBACK_PROJECTION_GAP,
                source_ref=None,
                target_ref=None,
                stage=stage,
                scope=scope,
                diagnostic_summary=self._sanitizer.sanitize_diagnostic_summary(
                    f"feedback gaps 超出上限，已截断 {len(gaps) - len(kept)} 条。"
                ),
                created_at=now,
            )
        )
        return kept
