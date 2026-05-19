#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback Ledger PR2 公共策略与常量。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from app.domain.models.feedback import (
    FeedbackCategory,
    FeedbackClassificationResult,
    FeedbackGapKind,
    FeedbackKind,
    FeedbackPromptSafeSummaryResult,
    FeedbackReasonCode,
    FeedbackScopeKind,
    FeedbackSeverity,
    FeedbackSourceConfidence,
    FeedbackSourceKind,
    FeedbackSourceRefResult,
    FeedbackSummaryKind,
    FeedbackSummaryResult,
    FeedbackTargetRefResult,
    FeedbackTargetType,
    FeedbackStatus,
)

_REDACTED = "[REDACTED]"
_MAX_SUMMARY_CHARS = 240
_MAX_PROMPT_SUMMARY_CHARS = 160
_MAX_DIAGNOSTIC_SUMMARY_CHARS = 160

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(\b(?:access|refresh|id)?_?token\b\s*[:=]\s*)['\"]?[^'\"\s,}]{8,}"),
    re.compile(r"(?i)(\b(?:api[_-]?key|secret[_-]?key)\b\s*[:=]\s*)['\"]?[^'\"\s,}]{8,}"),
    re.compile(r"(?i)(\bpassword\b\s*[:=]\s*)['\"]?[^'\"\s,}]{4,}"),
    re.compile(r"(?i)(\bcookie\b\s*[:=]\s*)['\"]?[^'\"\n,}]{8,}"),
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{8,}"),
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s-]{8,}\d)\b")

STATUS_PRIORITY = {
    FeedbackStatus.OPEN: 5,
    FeedbackStatus.RESOLVED: 4,
    FeedbackStatus.IGNORED: 3,
    FeedbackStatus.SUPERSEDED: 2,
    FeedbackStatus.EXPIRED: 1,
}
SEVERITY_PRIORITY = {
    FeedbackSeverity.CRITICAL: 4,
    FeedbackSeverity.ERROR: 3,
    FeedbackSeverity.WARNING: 2,
    FeedbackSeverity.INFO: 1,
}
PROMPT_LIST_LIMIT = 20
OPEN_LIST_LIMIT = 50
RESOLVED_LIST_LIMIT = 20
EXCLUDED_LIST_LIMIT = 100
CANDIDATE_SCAN_LIMIT = 200

SESSION_PERSISTENT_CATEGORIES = frozenset(
    {
        FeedbackCategory.CORRECTION,
        FeedbackCategory.SATISFACTION,
        FeedbackCategory.DISSATISFACTION,
        FeedbackCategory.ARTIFACT_UNUSABLE,
        FeedbackCategory.FINAL_ANSWER_MISMATCH,
    }
)
SESSION_WINDOW_CATEGORIES = frozenset(
    {
        FeedbackCategory.PREFERENCE,
        FeedbackCategory.CONFIRMATION,
        FeedbackCategory.SELECTION,
        FeedbackCategory.CLARIFICATION,
        FeedbackCategory.CANCEL,
        FeedbackCategory.CONTINUE_CANCELLED,
        FeedbackCategory.TAKEOVER,
    }
)
RUN_WINDOW_CATEGORIES = frozenset(
    {
        FeedbackCategory.TOOL_FAILURE,
        FeedbackCategory.FETCH_FAILED,
        FeedbackCategory.FILE_MISSING,
        FeedbackCategory.PATH_ERROR,
        FeedbackCategory.REPEAT_CALL,
        FeedbackCategory.NO_PROGRESS,
        FeedbackCategory.SEARCH_QUALITY_INSUFFICIENT,
        FeedbackCategory.SANDBOX_RESOURCE_LIMITED,
        FeedbackCategory.SANDBOX_PROFILE_STALE,
    }
)
RUN_PERSISTENT_CATEGORIES = frozenset(
    {
        FeedbackCategory.SAFETY_BLOCKED,
        FeedbackCategory.SAFETY_REWRITE,
        FeedbackCategory.CONFIRMATION_MISSING,
        FeedbackCategory.SELF_REVIEW_FAILED,
        FeedbackCategory.FINAL_GATE_BLOCKED,
        FeedbackCategory.UNMET_REQUIREMENT,
        FeedbackCategory.MISSING_EVIDENCE,
        FeedbackCategory.CITATION_INSUFFICIENT,
        FeedbackCategory.PARTIAL_GOAL_SATISFIED,
        FeedbackCategory.EVALUATION_FAILURE,
        FeedbackCategory.REGRESSION_FEEDBACK,
    }
)
RUN_OR_SESSION_CATEGORIES = frozenset(
    {
        FeedbackCategory.EVIDENCE_GAP,
        FeedbackCategory.EVIDENCE_SNAPSHOT_MISSING,
    }
)
FINAL_GATE_ONLY_CATEGORIES = frozenset(
    {
        FeedbackCategory.ARTIFACT_UNUSABLE,
        FeedbackCategory.FINAL_ANSWER_MISMATCH,
        FeedbackCategory.FINAL_GATE_BLOCKED,
    }
)
USER_CONSTRAINT_CATEGORIES = frozenset(
    {
        FeedbackCategory.PREFERENCE,
        FeedbackCategory.CLARIFICATION,
        FeedbackCategory.SELECTION,
        FeedbackCategory.CONFIRMATION,
    }
)
REPLAN_HINT_CATEGORIES = frozenset(
    {
        FeedbackCategory.CORRECTION,
        FeedbackCategory.TOOL_FAILURE,
        FeedbackCategory.REPEAT_CALL,
        FeedbackCategory.NO_PROGRESS,
        FeedbackCategory.EVIDENCE_GAP,
        FeedbackCategory.EVIDENCE_SNAPSHOT_MISSING,
        FeedbackCategory.SAFETY_BLOCKED,
        FeedbackCategory.ARTIFACT_UNUSABLE,
        FeedbackCategory.FINAL_ANSWER_MISMATCH,
    }
)
DO_NOT_REPEAT_CATEGORIES = frozenset(
    {
        FeedbackCategory.REPEAT_CALL,
        FeedbackCategory.NO_PROGRESS,
    }
)
SUMMARY_USER_CATEGORIES = frozenset(
    {
        FeedbackCategory.PREFERENCE,
        FeedbackCategory.SATISFACTION,
        FeedbackCategory.DISSATISFACTION,
        FeedbackCategory.CORRECTION,
        FeedbackCategory.CLARIFICATION,
        FeedbackCategory.CONFIRMATION,
        FeedbackCategory.SELECTION,
    }
)
QUALITY_SUMMARY_CATEGORIES = frozenset(
    {
        FeedbackCategory.SELF_REVIEW_FAILED,
        FeedbackCategory.FINAL_GATE_BLOCKED,
        FeedbackCategory.UNMET_REQUIREMENT,
        FeedbackCategory.MISSING_EVIDENCE,
        FeedbackCategory.CITATION_INSUFFICIENT,
        FeedbackCategory.ARTIFACT_UNUSABLE,
        FeedbackCategory.FINAL_ANSWER_MISMATCH,
        FeedbackCategory.PARTIAL_GOAL_SATISFIED,
    }
)
SOURCE_EVENT_TYPE_BY_KIND: dict[FeedbackSourceKind, str] = {
    FeedbackSourceKind.FEEDBACK_INPUT: "feedback_input",
    FeedbackSourceKind.MESSAGE_EVENT: "message",
    FeedbackSourceKind.WAIT_EVENT: "wait",
    FeedbackSourceKind.TOOL_EVENT: "tool",
    FeedbackSourceKind.SANDBOX_FACT: "sandbox_fact",
    FeedbackSourceKind.EVIDENCE: "evidence",
    FeedbackSourceKind.EVIDENCE_GAP: "evidence",
    FeedbackSourceKind.SAFETY_AUDIT: "safety_audit",
    FeedbackSourceKind.ARTIFACT_REVISION: "artifact",
}
TARGET_EVENT_TYPE_BY_TYPE: dict[FeedbackTargetType, str] = {
    FeedbackTargetType.MESSAGE_EVENT: "message",
    FeedbackTargetType.WAIT_EVENT: "wait",
}


class FeedbackLedgerError(RuntimeError):
    """Feedback Ledger 应用服务错误。"""


@dataclass(frozen=True)
class FeedbackValidationIssue:
    reason_code: FeedbackReasonCode
    gap_kind: FeedbackGapKind
    diagnostic_summary: str
    source_ref: FeedbackSourceRefResult | None = None
    target_ref: FeedbackTargetRefResult | None = None


class FeedbackScopeValidationError(FeedbackLedgerError):
    """source/target/scope owned 校验失败。"""

    def __init__(self, issue: FeedbackValidationIssue) -> None:
        super().__init__(issue.diagnostic_summary)
        self.issue = issue


class FeedbackRequiredRecordError(FeedbackLedgerError):
    """required feedback 写入失败，必须 fail closed。"""


@dataclass(frozen=True)
class FeedbackRetentionDecision:
    decay_policy: str
    ttl_scope: str
    resolved_scope_kind: FeedbackScopeKind
    expires_at: datetime | None


class FeedbackSanitizer:
    """生成 feedback summary 和 prompt-safe summary。"""

    def summarize(
            self,
            *,
            kind: FeedbackKind,
            summary_hint: str | None,
            source_summary: str,
            reason_code: FeedbackReasonCode,
    ) -> tuple[FeedbackSummaryResult, FeedbackPromptSafeSummaryResult]:
        base_text = str(summary_hint or "").strip() or str(source_summary or "").strip()
        if not base_text:
            base_text = reason_code.value
        summary_text, summary_truncated = self._sanitize_text(base_text, max_chars=_MAX_SUMMARY_CHARS)
        prompt_text, prompt_truncated = self._sanitize_text(base_text, max_chars=_MAX_PROMPT_SUMMARY_CHARS)
        summary_kind = {
            FeedbackKind.USER_FEEDBACK: FeedbackSummaryKind.USER_STATED,
            FeedbackKind.RUNTIME_FEEDBACK: FeedbackSummaryKind.RUNTIME_DIAGNOSTIC,
            FeedbackKind.QUALITY_FEEDBACK: FeedbackSummaryKind.SYSTEM_QUALITY,
        }[kind]
        sanitization_reasons: list[str] = []
        if summary_truncated or prompt_truncated:
            sanitization_reasons.append("length_limited")
        if summary_text != base_text or prompt_text != base_text:
            sanitization_reasons.append("pii_redacted")
        sanitization_applied = bool(sanitization_reasons)
        return (
            FeedbackSummaryResult(
                summary_text=summary_text,
                summary_kind=summary_kind,
                is_truncated=summary_truncated,
                truncation_reason="length_limited" if summary_truncated else None,
                language="zh-CN",
            ),
            FeedbackPromptSafeSummaryResult(
                summary_text=prompt_text,
                is_truncated=prompt_truncated,
                sanitization_applied=sanitization_applied,
                sanitization_reasons=sanitization_reasons,
                prompt_visible=True,
            ),
        )

    @staticmethod
    def sanitize_diagnostic_summary(summary: str, *, max_chars: int = _MAX_DIAGNOSTIC_SUMMARY_CHARS) -> str:
        text, _ = FeedbackSanitizer._sanitize_text(summary, max_chars=max_chars)
        return text

    @staticmethod
    def _sanitize_text(text: str, *, max_chars: int) -> tuple[str, bool]:
        sanitized = str(text or "").strip()
        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub(r"\1" + _REDACTED, sanitized)
        sanitized = _EMAIL_RE.sub(_REDACTED, sanitized)
        sanitized = _PHONE_RE.sub(_REDACTED, sanitized)
        truncated = len(sanitized) > max_chars
        if truncated:
            sanitized = sanitized[: max_chars - 1].rstrip() + "…"
        return sanitized or "已脱敏反馈摘要", truncated


class FeedbackSeverityPolicy:
    """集中 severity 计算。"""

    def classify(
            self,
            *,
            command_kind: FeedbackKind,
            category: FeedbackCategory,
            reason_code: FeedbackReasonCode,
            classification: FeedbackClassificationResult,
            target_ref: FeedbackTargetRefResult,
            source_ref: FeedbackSourceRefResult,
    ) -> FeedbackSeverity:
        if category in {FeedbackCategory.PREFERENCE, FeedbackCategory.SATISFACTION}:
            return FeedbackSeverity.INFO
        if category in {
            FeedbackCategory.CORRECTION,
            FeedbackCategory.EVIDENCE_GAP,
            FeedbackCategory.EVIDENCE_SNAPSHOT_MISSING,
            FeedbackCategory.FINAL_GATE_BLOCKED,
            FeedbackCategory.ARTIFACT_UNUSABLE,
            FeedbackCategory.FINAL_ANSWER_MISMATCH,
            FeedbackCategory.SELF_REVIEW_FAILED,
            FeedbackCategory.UNMET_REQUIREMENT,
            FeedbackCategory.MISSING_EVIDENCE,
            FeedbackCategory.CITATION_INSUFFICIENT,
        }:
            return FeedbackSeverity.ERROR
        if category == FeedbackCategory.DISSATISFACTION:
            if target_ref.target_type == FeedbackTargetType.ARTIFACT_REVISION:
                return FeedbackSeverity.CRITICAL
            return FeedbackSeverity.ERROR
        if category in {FeedbackCategory.SAFETY_BLOCKED, FeedbackCategory.SAFETY_REWRITE}:
            return FeedbackSeverity.ERROR
        if (
                command_kind == FeedbackKind.RUNTIME_FEEDBACK
                and classification.source_confidence == FeedbackSourceConfidence.WEAK
                and reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE
        ):
            return FeedbackSeverity.WARNING
        if category in {
            FeedbackCategory.TOOL_FAILURE,
            FeedbackCategory.SEARCH_QUALITY_INSUFFICIENT,
            FeedbackCategory.FETCH_FAILED,
            FeedbackCategory.FILE_MISSING,
            FeedbackCategory.PATH_ERROR,
            FeedbackCategory.NO_PROGRESS,
            FeedbackCategory.REPEAT_CALL,
            FeedbackCategory.SANDBOX_RESOURCE_LIMITED,
            FeedbackCategory.SANDBOX_PROFILE_STALE,
            FeedbackCategory.CONFIRMATION_MISSING,
        }:
            return FeedbackSeverity.WARNING
        if source_ref.source_kind == FeedbackSourceKind.SAFETY_AUDIT:
            return FeedbackSeverity.ERROR
        return FeedbackSeverity.INFO


class FeedbackRetentionPolicy:
    """集中 TTL/衰减规则。"""

    def decide(
            self,
            *,
            kind: FeedbackKind,
            category: FeedbackCategory,
            requested_scope_kind: FeedbackScopeKind | None,
            current_run_id: str | None,
            now: datetime,
    ) -> FeedbackRetentionDecision:
        if category in SESSION_PERSISTENT_CATEGORIES:
            return FeedbackRetentionDecision("session_persistent", "session", FeedbackScopeKind.SESSION, None)
        if category in SESSION_WINDOW_CATEGORIES:
            return FeedbackRetentionDecision("session_window", "session", FeedbackScopeKind.SESSION, None)
        if category in RUN_WINDOW_CATEGORIES:
            return FeedbackRetentionDecision(
                "run_window",
                "run",
                requested_scope_kind or FeedbackScopeKind.RUN,
                None if current_run_id else now,
            )
        if category in RUN_PERSISTENT_CATEGORIES:
            return FeedbackRetentionDecision(
                "run_persistent",
                "run",
                requested_scope_kind or FeedbackScopeKind.RUN,
                None if current_run_id else now,
            )
        if category in RUN_OR_SESSION_CATEGORIES:
            scope_kind = requested_scope_kind or FeedbackScopeKind.RUN
            return FeedbackRetentionDecision(
                f"{scope_kind.value}_persistent",
                scope_kind.value,
                scope_kind,
                None,
            )
        if kind == FeedbackKind.QUALITY_FEEDBACK:
            scope_kind = FeedbackScopeKind.SESSION if category in {
                FeedbackCategory.ARTIFACT_UNUSABLE,
                FeedbackCategory.FINAL_ANSWER_MISMATCH,
            } else FeedbackScopeKind.RUN
            return FeedbackRetentionDecision(
                f"{scope_kind.value}_persistent",
                scope_kind.value,
                scope_kind,
                None,
            )
        return FeedbackRetentionDecision(
            "run_window",
            "run",
            requested_scope_kind or FeedbackScopeKind.RUN,
            None if current_run_id else now,
        )
