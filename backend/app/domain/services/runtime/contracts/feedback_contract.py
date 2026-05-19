#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback Ledger 纯领域契约。

P2-2 的反馈账本只记录结构化反馈信号，不解析 message/tool 文案，不保存
raw message、raw args、stdout、文件正文或网页正文。PR1 只落领域契约、
持久化字段和仓储边界，不承担 runtime 写入编排。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)


class FeedbackKind(str, Enum):
    USER_FEEDBACK = "user_feedback"
    RUNTIME_FEEDBACK = "runtime_feedback"
    QUALITY_FEEDBACK = "quality_feedback"


class FeedbackCategory(str, Enum):
    CONFIRMATION = "confirmation"
    SELECTION = "selection"
    CLARIFICATION = "clarification"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    SATISFACTION = "satisfaction"
    DISSATISFACTION = "dissatisfaction"
    CANCEL = "cancel"
    CONTINUE_CANCELLED = "continue_cancelled"
    TAKEOVER = "takeover"
    TOOL_FAILURE = "tool_failure"
    REPEAT_CALL = "repeat_call"
    NO_PROGRESS = "no_progress"
    SEARCH_QUALITY_INSUFFICIENT = "search_quality_insufficient"
    FETCH_FAILED = "fetch_failed"
    FILE_MISSING = "file_missing"
    PATH_ERROR = "path_error"
    SANDBOX_RESOURCE_LIMITED = "sandbox_resource_limited"
    SANDBOX_PROFILE_STALE = "sandbox_profile_stale"
    EVIDENCE_GAP = "evidence_gap"
    EVIDENCE_SNAPSHOT_MISSING = "evidence_snapshot_missing"
    SAFETY_BLOCKED = "safety_blocked"
    SAFETY_REWRITE = "safety_rewrite"
    CONFIRMATION_MISSING = "confirmation_missing"
    SELF_REVIEW_FAILED = "self_review_failed"
    FINAL_GATE_BLOCKED = "final_gate_blocked"
    UNMET_REQUIREMENT = "unmet_requirement"
    MISSING_EVIDENCE = "missing_evidence"
    CITATION_INSUFFICIENT = "citation_insufficient"
    ARTIFACT_UNUSABLE = "artifact_unusable"
    FINAL_ANSWER_MISMATCH = "final_answer_mismatch"
    PARTIAL_GOAL_SATISFIED = "partial_goal_satisfied"
    EVALUATION_FAILURE = "evaluation_failure"
    REGRESSION_FEEDBACK = "regression_feedback"


class FeedbackStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    IGNORED = "ignored"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class FeedbackSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class FeedbackScopeKind(str, Enum):
    RUN = "run"
    SESSION = "session"


class FeedbackSourceKind(str, Enum):
    FEEDBACK_INPUT = "feedback_input"
    MESSAGE_EVENT = "message_event"
    WAIT_EVENT = "wait_event"
    TOOL_EVENT = "tool_event"
    SANDBOX_FACT = "sandbox_fact"
    EVIDENCE = "evidence"
    EVIDENCE_GAP = "evidence_gap"
    ARTIFACT_REVISION = "artifact_revision"
    SAFETY_AUDIT = "safety_audit"
    SELF_REVIEW = "self_review"
    FINAL_DELIVERY = "final_delivery"
    EVALUATION = "evaluation"
    RUNTIME_SIGNAL = "runtime_signal"


class FeedbackTargetType(str, Enum):
    RUN = "run"
    STEP = "step"
    TOOL_CALL = "tool_call"
    MESSAGE_EVENT = "message_event"
    WAIT_EVENT = "wait_event"
    EVIDENCE = "evidence"
    EVIDENCE_GAP = "evidence_gap"
    SANDBOX_FACT = "sandbox_fact"
    ARTIFACT_REVISION = "artifact_revision"
    SAFETY_AUDIT = "safety_audit"
    SELF_REVIEW = "self_review"
    FINAL_DELIVERY = "final_delivery"
    USER_GOAL = "user_goal"


class FeedbackSnapshotStage(str, Enum):
    PLANNER = "planner"
    EXECUTE = "execute"
    REPLAN = "replan"
    SUMMARY = "summary"
    FUTURE_REVIEW = "future_review"
    FINAL_GATE = "final_gate"
    EVALUATION = "evaluation"
    DIRECT_ANSWER = "direct_answer"


class FeedbackGapKind(str, Enum):
    RECORD_FAILED = "record_failed"
    SOURCE_MISSING = "source_missing"
    SOURCE_INCOMPLETE = "source_incomplete"
    PROJECTION_MISSING = "projection_missing"
    CURSOR_MISMATCH = "cursor_mismatch"


class FeedbackExcludedBy(str, Enum):
    TTL = "ttl"
    WINDOW = "window"
    STATUS = "status"
    SCOPE = "scope"
    DEDUPE = "dedupe"
    STAGE_POLICY = "stage_policy"


class FeedbackSummaryKind(str, Enum):
    USER_STATED = "user_stated"
    RUNTIME_DIAGNOSTIC = "runtime_diagnostic"
    SYSTEM_QUALITY = "system_quality"


class FeedbackSourceConfidence(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    DIAGNOSTIC = "diagnostic"


class FeedbackDataOrigin(str, Enum):
    USER = "user"
    RUNTIME = "runtime"
    SYSTEM_QUALITY = "system_quality"
    EVALUATION = "evaluation"


class UserFeedbackIntentKind(str, Enum):
    CORRECTION = "correction"
    PREFERENCE = "preference"
    CLARIFICATION = "clarification"
    SATISFACTION = "satisfaction"
    DISSATISFACTION = "dissatisfaction"
    CONFIRMATION = "confirmation"
    SELECTION = "selection"
    CANCEL = "cancel"
    CONTINUE_CANCELLED = "continue_cancelled"
    TAKEOVER = "takeover"


class FeedbackReasonCode(str, Enum):
    USER_CONFIRMED = "user_confirmed"
    USER_REJECTED = "user_rejected"
    USER_SELECTED_OPTION = "user_selected_option"
    USER_PROVIDED_CLARIFICATION = "user_provided_clarification"
    USER_CORRECTED_REQUIREMENT = "user_corrected_requirement"
    USER_SET_PREFERENCE = "user_set_preference"
    USER_CANCELLED = "user_cancelled"
    USER_CONTINUED_CANCELLED = "user_continued_cancelled"
    USER_REPORTED_SATISFACTION = "user_reported_satisfaction"
    USER_REPORTED_DISSATISFACTION = "user_reported_dissatisfaction"
    TOOL_FAILED = "tool_failed"
    TOOL_REPEATED = "tool_repeated"
    EXECUTION_NO_PROGRESS = "execution_no_progress"
    SEARCH_QUALITY_INSUFFICIENT = "search_quality_insufficient"
    FETCH_FAILED = "fetch_failed"
    FILE_MISSING = "file_missing"
    PATH_ERROR = "path_error"
    SANDBOX_RESOURCE_LIMITED = "sandbox_resource_limited"
    SANDBOX_PROFILE_STALE = "sandbox_profile_stale"
    EVIDENCE_GAP_DETECTED = "evidence_gap_detected"
    EVIDENCE_SNAPSHOT_MISSING = "evidence_snapshot_missing"
    SAFETY_BLOCKED = "safety_blocked"
    SAFETY_REWRITE_APPLIED = "safety_rewrite_applied"
    CONFIRMATION_MISSING = "confirmation_missing"
    SELF_REVIEW_FAILED = "self_review_failed"
    FINAL_GATE_BLOCKED = "final_gate_blocked"
    UNMET_REQUIREMENT = "unmet_requirement"
    MISSING_EVIDENCE = "missing_evidence"
    CITATION_INSUFFICIENT = "citation_insufficient"
    ARTIFACT_UNUSABLE = "artifact_unusable"
    FINAL_ANSWER_MISMATCH = "final_answer_mismatch"
    PARTIAL_GOAL_SATISFIED = "partial_goal_satisfied"
    EVALUATION_FAILURE = "evaluation_failure"
    REGRESSION_FEEDBACK = "regression_feedback"
    FEEDBACK_PAYLOAD_INVALID = "feedback_payload_invalid"
    FEEDBACK_SCOPE_MISMATCH = "feedback_scope_mismatch"
    FEEDBACK_SOURCE_EVENT_MISSING = "feedback_source_event_missing"
    FEEDBACK_SOURCE_RECORD_MISSING = "feedback_source_record_missing"
    FEEDBACK_SOURCE_EVENT_MISMATCH = "feedback_source_event_mismatch"
    FEEDBACK_SOURCE_INCOMPLETE = "feedback_source_incomplete"
    FEEDBACK_TARGET_MISSING = "feedback_target_missing"
    FEEDBACK_TARGET_REVISION_MISSING = "feedback_target_revision_missing"
    FEEDBACK_TARGET_SCOPE_MISMATCH = "feedback_target_scope_mismatch"
    FEEDBACK_REQUIRED_RECORD_FAILED = "feedback_required_record_failed"
    FEEDBACK_RECORD_FAILED = "feedback_record_failed"
    FEEDBACK_PROJECTION_GAP = "feedback_projection_gap"
    FEEDBACK_EVENT_PROJECTION_FAILED = "feedback_event_projection_failed"
    FEEDBACK_SNAPSHOT_CURSOR_MISMATCH = "feedback_snapshot_cursor_mismatch"


class FeedbackResolutionReasonCode(str, Enum):
    RESOLVED_BY_USER_CONFIRMATION = "resolved_by_user_confirmation"
    RESOLVED_BY_REPLAN = "resolved_by_replan"
    RESOLVED_BY_SUCCESSFUL_STEP = "resolved_by_successful_step"
    RESOLVED_BY_SELF_REVIEW_PASSED = "resolved_by_self_review_passed"
    RESOLVED_BY_FINAL_GATE_PASSED = "resolved_by_final_gate_passed"
    IGNORED_NOT_APPLICABLE = "ignored_not_applicable"
    SUPERSEDED_BY_NEW_FEEDBACK = "superseded_by_new_feedback"
    EXPIRED_BY_TTL = "expired_by_ttl"
    EXPIRED_BY_WINDOW = "expired_by_window"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_text(value: str | None, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


USER_FEEDBACK_CATEGORIES = frozenset(
    {
        FeedbackCategory.CONFIRMATION,
        FeedbackCategory.SELECTION,
        FeedbackCategory.CLARIFICATION,
        FeedbackCategory.CORRECTION,
        FeedbackCategory.PREFERENCE,
        FeedbackCategory.SATISFACTION,
        FeedbackCategory.DISSATISFACTION,
        FeedbackCategory.CANCEL,
        FeedbackCategory.CONTINUE_CANCELLED,
        FeedbackCategory.TAKEOVER,
    }
)
RUNTIME_FEEDBACK_CATEGORIES = frozenset(
    {
        FeedbackCategory.TOOL_FAILURE,
        FeedbackCategory.REPEAT_CALL,
        FeedbackCategory.NO_PROGRESS,
        FeedbackCategory.SEARCH_QUALITY_INSUFFICIENT,
        FeedbackCategory.FETCH_FAILED,
        FeedbackCategory.FILE_MISSING,
        FeedbackCategory.PATH_ERROR,
        FeedbackCategory.SANDBOX_RESOURCE_LIMITED,
        FeedbackCategory.SANDBOX_PROFILE_STALE,
        FeedbackCategory.EVIDENCE_GAP,
        FeedbackCategory.EVIDENCE_SNAPSHOT_MISSING,
        FeedbackCategory.SAFETY_BLOCKED,
        FeedbackCategory.SAFETY_REWRITE,
        FeedbackCategory.CONFIRMATION_MISSING,
    }
)
QUALITY_FEEDBACK_CATEGORIES = frozenset(
    {
        FeedbackCategory.SELF_REVIEW_FAILED,
        FeedbackCategory.FINAL_GATE_BLOCKED,
        FeedbackCategory.UNMET_REQUIREMENT,
        FeedbackCategory.MISSING_EVIDENCE,
        FeedbackCategory.CITATION_INSUFFICIENT,
        FeedbackCategory.ARTIFACT_UNUSABLE,
        FeedbackCategory.FINAL_ANSWER_MISMATCH,
        FeedbackCategory.PARTIAL_GOAL_SATISFIED,
        FeedbackCategory.EVALUATION_FAILURE,
        FeedbackCategory.REGRESSION_FEEDBACK,
    }
)


def _validate_kind_category_pair(*, kind: FeedbackKind, category: FeedbackCategory, field_name: str) -> None:
    allowed_categories = {
        FeedbackKind.USER_FEEDBACK: USER_FEEDBACK_CATEGORIES,
        FeedbackKind.RUNTIME_FEEDBACK: RUNTIME_FEEDBACK_CATEGORIES,
        FeedbackKind.QUALITY_FEEDBACK: QUALITY_FEEDBACK_CATEGORIES,
    }[kind]
    if category not in allowed_categories:
        raise ValueError(f"{field_name} 不允许 {kind.value}/{category.value} 组合")


def _validate_runtime_signal_usage(*, source_kind: FeedbackSourceKind, field_name: str) -> None:
    if source_kind == FeedbackSourceKind.RUNTIME_SIGNAL:
        raise ValueError(f"{field_name} 不允许使用 runtime_signal 作为落库或投影视图 source_kind")


class FeedbackScopeResult(_StrictModel):
    user_id: str
    session_id: str
    workspace_id: str
    feedback_scope_kind: FeedbackScopeKind
    scope_id: str
    run_id: str | None = None
    source_run_id: str | None = None
    target_run_id: str | None = None
    current_run_id_at_record_time: str | None = None

    @field_validator("user_id", "session_id", "workspace_id", "scope_id")
    @classmethod
    def _required_text_fields(cls, value: str) -> str:
        return _require_text(value, "scope 字段")

    @model_validator(mode="after")
    def _validate_scope_kind(self) -> "FeedbackScopeResult":
        if self.feedback_scope_kind == FeedbackScopeKind.RUN:
            if not self.run_id:
                raise ValueError("run scope 必须提供 run_id")
            if self.scope_id != self.run_id:
                raise ValueError("run scope 的 scope_id 必须等于 run_id")
            if not self.source_run_id or not self.target_run_id or not self.current_run_id_at_record_time:
                raise ValueError("run scope 必须提供 source_run_id、target_run_id 和 current_run_id_at_record_time")
            if len({self.run_id, self.source_run_id, self.target_run_id, self.current_run_id_at_record_time}) != 1:
                raise ValueError("run scope 要求 scope/run/source/target/current run 一致")
            return self
        if self.scope_id != self.session_id:
            raise ValueError("session scope 的 scope_id 必须等于 session_id")
        return self


class FeedbackSnapshotScopeResult(_StrictModel):
    user_id: str
    session_id: str
    workspace_id: str
    feedback_scope_kind: FeedbackScopeKind
    scope_id: str
    current_run_id_at_snapshot_time: str | None = None

    @field_validator("user_id", "session_id", "workspace_id", "scope_id")
    @classmethod
    def _required_text_fields(cls, value: str) -> str:
        return _require_text(value, "snapshot scope 字段")

    @model_validator(mode="after")
    def _validate_scope(self) -> "FeedbackSnapshotScopeResult":
        if self.feedback_scope_kind == FeedbackScopeKind.RUN:
            if self.current_run_id_at_snapshot_time is None:
                raise ValueError("run snapshot 必须提供 current_run_id_at_snapshot_time")
            if self.scope_id != self.current_run_id_at_snapshot_time:
                raise ValueError("run snapshot 的 scope_id 必须等于 current_run_id_at_snapshot_time")
        if self.feedback_scope_kind == FeedbackScopeKind.SESSION and self.scope_id != self.session_id:
            raise ValueError("session snapshot 的 scope_id 必须等于 session_id")
        return self


class FeedbackSnapshotCursorResult(_StrictModel):
    latest_feedback_id: str | None = None
    source_record_ids: list[str] = Field(default_factory=list)

    @field_validator("latest_feedback_id")
    @classmethod
    def _normalize_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "latest_feedback_id")

    @field_validator("source_record_ids")
    @classmethod
    def _normalize_record_ids(cls, values: list[str]) -> list[str]:
        return [_require_text(value, "source_record_ids") for value in list(values or [])]


class FeedbackSummaryResult(_StrictModel):
    summary_text: str
    summary_kind: FeedbackSummaryKind
    is_truncated: bool
    truncation_reason: str | None = None
    language: str

    @field_validator("summary_text", "language")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _require_text(value, "summary 字段")

    @field_validator("truncation_reason")
    @classmethod
    def _normalize_optional_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "truncation_reason")

    @model_validator(mode="after")
    def _validate_truncation(self) -> "FeedbackSummaryResult":
        if self.is_truncated and not self.truncation_reason:
            raise ValueError("截断摘要必须提供 truncation_reason")
        if not self.is_truncated and self.truncation_reason is not None:
            raise ValueError("未截断摘要不能提供 truncation_reason")
        return self


class FeedbackPromptSafeSummaryResult(_StrictModel):
    summary_text: str
    is_truncated: bool
    sanitization_applied: bool
    sanitization_reasons: list[str] = Field(default_factory=list)
    prompt_visible: bool

    @field_validator("summary_text")
    @classmethod
    def _summary_text_must_not_be_empty(cls, value: str) -> str:
        return _require_text(value, "prompt_safe_summary.summary_text")

    @field_validator("sanitization_reasons")
    @classmethod
    def _sanitize_reason_list(cls, values: list[str]) -> list[str]:
        return [_require_text(value, "sanitization_reasons") for value in list(values or [])]

    @model_validator(mode="after")
    def _validate_sanitization(self) -> "FeedbackPromptSafeSummaryResult":
        if self.sanitization_applied and not self.sanitization_reasons:
            raise ValueError("发生脱敏/裁剪时必须记录 sanitization_reasons")
        if not self.sanitization_applied and self.sanitization_reasons:
            raise ValueError("未脱敏时不允许保存 sanitization_reasons")
        return self


class FeedbackClassificationResult(_StrictModel):
    privacy_level: PrivacyLevel
    retention_policy: RetentionPolicyKind
    trust_level: DataTrustLevel
    source_confidence: FeedbackSourceConfidence
    data_origin: FeedbackDataOrigin


class FeedbackSourceRefResult(_StrictModel):
    source_kind: FeedbackSourceKind
    source_event_id: str
    source_record_refs: list[dict[str, str | None]] = Field(default_factory=list)
    source_run_id: str | None = None
    source_step_id: str | None = None
    source_summary: str

    @field_validator("source_event_id", "source_summary")
    @classmethod
    def _required_text_fields(cls, value: str) -> str:
        return _require_text(value, "source_ref 字段")

    @field_validator("source_run_id", "source_step_id")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "source_ref 可选字段")

    @field_validator("source_record_refs")
    @classmethod
    def _normalize_source_record_refs(cls, values: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
        normalized: list[dict[str, str | None]] = []
        for item in list(values or []):
            if not item:
                raise ValueError("source_record_refs 不能包含空引用")
            normalized_item: dict[str, str | None] = {}
            for key, value in item.items():
                normalized_key = _require_text(key, "source_record_refs key")
                normalized_item[normalized_key] = None if value is None else _require_text(value, normalized_key)
            normalized.append(normalized_item)
        return normalized


class FeedbackTargetRefResult(_StrictModel):
    target_type: FeedbackTargetType
    target_id: str
    target_run_id: str | None = None
    target_revision_id: str | None = None
    target_content_hash: str | None = None

    @field_validator("target_id")
    @classmethod
    def _target_id_must_not_be_empty(cls, value: str) -> str:
        return _require_text(value, "target_id")

    @field_validator("target_run_id", "target_revision_id", "target_content_hash")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "target_ref 可选字段")

    @model_validator(mode="after")
    def _validate_target_matrix(self) -> "FeedbackTargetRefResult":
        run_scoped_types = {
            FeedbackTargetType.RUN,
            FeedbackTargetType.STEP,
            FeedbackTargetType.TOOL_CALL,
            FeedbackTargetType.MESSAGE_EVENT,
            FeedbackTargetType.WAIT_EVENT,
            FeedbackTargetType.EVIDENCE,
            FeedbackTargetType.EVIDENCE_GAP,
            FeedbackTargetType.SANDBOX_FACT,
            FeedbackTargetType.SAFETY_AUDIT,
            FeedbackTargetType.SELF_REVIEW,
            FeedbackTargetType.FINAL_DELIVERY,
            FeedbackTargetType.USER_GOAL,
        }
        if self.target_type == FeedbackTargetType.ARTIFACT_REVISION:
            if not self.target_revision_id or not self.target_content_hash:
                raise ValueError("artifact_revision target 必须同时提供 target_revision_id 和 target_content_hash")
            return self
        if self.target_revision_id is not None or self.target_content_hash is not None:
            raise ValueError("只有 artifact_revision target 允许 revision/content hash")
        if self.target_type in run_scoped_types and not self.target_run_id:
            raise ValueError(f"{self.target_type.value} target 必须提供 target_run_id")
        return self


class FeedbackResolutionResult(_StrictModel):
    status: FeedbackStatus
    resolution_reason_code: FeedbackResolutionReasonCode | None = None
    resolved_by_ref: dict[str, str | None] | None = None
    resolved_at: datetime | None = None
    superseded_by_feedback_id: str | None = None
    resolution_summary: str | None = None

    @field_validator("superseded_by_feedback_id", "resolution_summary")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "resolution 字段")

    @field_validator("resolved_by_ref")
    @classmethod
    def _normalize_resolved_by_ref(
            cls,
            value: dict[str, str | None] | None,
    ) -> dict[str, str | None] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("resolved_by_ref 不能为空对象")
        normalized: dict[str, str | None] = {}
        for key, item in value.items():
            normalized_key = _require_text(key, "resolved_by_ref key")
            normalized[normalized_key] = None if item is None else _require_text(item, normalized_key)
        return normalized

    @model_validator(mode="after")
    def _validate_resolution_status(self) -> "FeedbackResolutionResult":
        if self.status == FeedbackStatus.OPEN:
            if any(
                value is not None
                for value in (
                    self.resolution_reason_code,
                    self.resolved_by_ref,
                    self.resolved_at,
                    self.superseded_by_feedback_id,
                    self.resolution_summary,
                )
            ):
                raise ValueError("open 状态不允许携带 resolution 字段")
            return self
        if self.resolution_reason_code is None or self.resolved_at is None or self.resolved_by_ref is None:
            raise ValueError("非 open 状态必须提供 reason_code、resolved_by_ref 和 resolved_at")
        if self.status == FeedbackStatus.SUPERSEDED and self.superseded_by_feedback_id is None:
            raise ValueError("superseded 状态必须提供 superseded_by_feedback_id")
        return self


class FeedbackRecordResult(_StrictModel):
    feedback_id: str
    scope: FeedbackScopeResult
    source_ref: FeedbackSourceRefResult
    target_ref: FeedbackTargetRefResult
    kind: FeedbackKind
    category: FeedbackCategory
    status: FeedbackStatus
    severity: FeedbackSeverity
    reason_code: FeedbackReasonCode
    feedback_summary: FeedbackSummaryResult
    prompt_safe_summary: FeedbackPromptSafeSummaryResult
    classification: FeedbackClassificationResult
    resolution: FeedbackResolutionResult
    created_at: datetime
    updated_at: datetime

    @field_validator("feedback_id")
    @classmethod
    def _feedback_id_must_not_be_empty(cls, value: str) -> str:
        return _require_text(value, "feedback_id")

    @model_validator(mode="after")
    def _validate_record_result_semantics(self) -> "FeedbackRecordResult":
        _validate_runtime_signal_usage(source_kind=self.source_ref.source_kind, field_name="FeedbackRecordResult")
        _validate_kind_category_pair(kind=self.kind, category=self.category, field_name="FeedbackRecordResult")
        return self


class FeedbackRecord(_StrictModel):
    id: str
    scope: FeedbackScopeResult
    source_ref: FeedbackSourceRefResult
    target_ref: FeedbackTargetRefResult
    resolution: FeedbackResolutionResult
    feedback_summary: FeedbackSummaryResult
    prompt_safe_summary: FeedbackPromptSafeSummaryResult
    classification: FeedbackClassificationResult
    user_id: str
    session_id: str
    workspace_id: str
    run_id: str | None = None
    feedback_scope_kind: FeedbackScopeKind
    scope_id: str
    source_run_id: str | None = None
    target_run_id: str | None = None
    step_id: str | None = None
    kind: FeedbackKind
    category: FeedbackCategory
    status: FeedbackStatus
    severity: FeedbackSeverity
    source_kind: FeedbackSourceKind
    source_event_id: str
    source_record_refs: list[dict[str, str | None]] = Field(default_factory=list)
    target_type: FeedbackTargetType
    target_id: str
    target_revision_id: str | None = None
    target_content_hash: str | None = None
    dedupe_key: str
    feedback_key: str
    reason_code: FeedbackReasonCode
    resolution_reason_code: FeedbackResolutionReasonCode | None = None
    resolved_by_ref: dict[str, str | None] | None = None
    decay_policy: str
    expires_at: datetime | None = None
    ttl_scope: str
    profile_hash: str | None = None
    origin: DataOrigin
    trust_level: DataTrustLevel
    privacy_level: PrivacyLevel
    retention_policy: RetentionPolicyKind
    created_at: datetime
    updated_at: datetime

    @field_validator(
        "id",
        "user_id",
        "session_id",
        "workspace_id",
        "scope_id",
        "source_event_id",
        "dedupe_key",
        "feedback_key",
        "decay_policy",
        "ttl_scope",
        "target_id",
    )
    @classmethod
    def _required_text_fields(cls, value: str) -> str:
        return _require_text(value, "FeedbackRecord 字段")

    @field_validator("run_id", "source_run_id", "target_run_id", "step_id", "profile_hash")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "FeedbackRecord 可选字段")

    @model_validator(mode="after")
    def _validate_nested_and_flat_fields(self) -> "FeedbackRecord":
        if self.scope.user_id != self.user_id or self.scope.session_id != self.session_id or self.scope.workspace_id != self.workspace_id:
            raise ValueError("scope 与 flat scope 字段不一致")
        if self.scope.feedback_scope_kind != self.feedback_scope_kind or self.scope.scope_id != self.scope_id:
            raise ValueError("scope.kind/scope_id 与 flat 字段不一致")
        if self.scope.run_id != self.run_id or self.scope.source_run_id != self.source_run_id or self.scope.target_run_id != self.target_run_id:
            raise ValueError("scope run 字段与 flat 字段不一致")
        if self.source_ref.source_kind != self.source_kind or self.source_ref.source_event_id != self.source_event_id:
            raise ValueError("source_ref 与 source flat 字段不一致")
        if self.source_ref.source_run_id != self.source_run_id:
            raise ValueError("source_ref.source_run_id 与 flat 字段不一致")
        if self.target_ref.target_type != self.target_type or self.target_ref.target_id != self.target_id:
            raise ValueError("target_ref 与 target flat 字段不一致")
        if self.target_ref.target_run_id != self.target_run_id:
            raise ValueError("target_ref.target_run_id 与 flat 字段不一致")
        if self.target_ref.target_revision_id != self.target_revision_id:
            raise ValueError("target_revision_id 与 target_ref 不一致")
        if self.target_ref.target_content_hash != self.target_content_hash:
            raise ValueError("target_content_hash 与 target_ref 不一致")
        if self.resolution.status != self.status or self.resolution.resolution_reason_code != self.resolution_reason_code:
            raise ValueError("resolution 与 status/resolution_reason_code 不一致")
        if self.resolution.resolved_by_ref != self.resolved_by_ref:
            raise ValueError("resolved_by_ref 与 resolution 不一致")
        if self.classification.privacy_level != self.privacy_level:
            raise ValueError("classification.privacy_level 与 flat 字段不一致")
        if self.classification.retention_policy != self.retention_policy:
            raise ValueError("classification.retention_policy 与 flat 字段不一致")
        if self.classification.trust_level != self.trust_level:
            raise ValueError("classification.trust_level 与 flat 字段不一致")
        _validate_runtime_signal_usage(source_kind=self.source_kind, field_name="FeedbackRecord")
        _validate_kind_category_pair(kind=self.kind, category=self.category, field_name="FeedbackRecord")
        return self


class UserFeedbackIntent(_StrictModel):
    intent_kind: UserFeedbackIntentKind
    target_ref: FeedbackTargetRefResult
    reason_code: FeedbackReasonCode
    summary_hint: str | None = None

    @field_validator("summary_hint")
    @classmethod
    def _normalize_summary_hint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "summary_hint")


class UserFeedbackCaptureResult(_StrictModel):
    captured: bool
    intent: UserFeedbackIntent | None = None
    reason_code: FeedbackReasonCode | None = None

    @model_validator(mode="after")
    def _validate_capture_result(self) -> "UserFeedbackCaptureResult":
        if self.captured and self.intent is None:
            raise ValueError("captured=true 时必须提供 intent")
        if not self.captured and self.intent is not None:
            raise ValueError("captured=false 时不能提供 intent")
        return self


class FeedbackRecordCommand(_StrictModel):
    access_scope: AccessScopeResult
    source_ref: FeedbackSourceRefResult
    target_ref: FeedbackTargetRefResult
    kind: FeedbackKind
    category: FeedbackCategory
    reason_code: FeedbackReasonCode
    feedback_summary: FeedbackSummaryResult
    prompt_safe_summary: FeedbackPromptSafeSummaryResult
    classification: FeedbackClassificationResult
    requested_feedback_scope_kind: FeedbackScopeKind | None = None
    requested_scope_id: str | None = None
    current_run_id_at_record_time: str | None = None
    step_id: str | None = None
    profile_hash: str | None = None
    decay_policy: str
    ttl_scope: str
    origin: DataOrigin
    trust_level: DataTrustLevel
    privacy_level: PrivacyLevel
    retention_policy: RetentionPolicyKind

    @field_validator("requested_scope_id", "current_run_id_at_record_time", "step_id", "profile_hash", "decay_policy", "ttl_scope")
    @classmethod
    def _normalize_optional_or_required_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "FeedbackRecordCommand 字段")

    @model_validator(mode="after")
    def _validate_command_semantics(self) -> "FeedbackRecordCommand":
        _validate_runtime_signal_usage(source_kind=self.source_ref.source_kind, field_name="FeedbackRecordCommand")
        _validate_kind_category_pair(kind=self.kind, category=self.category, field_name="FeedbackRecordCommand")
        if self.classification.privacy_level != self.privacy_level:
            raise ValueError("classification.privacy_level 与 flat 字段不一致")
        if self.classification.retention_policy != self.retention_policy:
            raise ValueError("classification.retention_policy 与 flat 字段不一致")
        if self.classification.trust_level != self.trust_level:
            raise ValueError("classification.trust_level 与 flat 字段不一致")
        return self


class UserFeedbackCommand(FeedbackRecordCommand):
    kind: Literal[FeedbackKind.USER_FEEDBACK] = FeedbackKind.USER_FEEDBACK
    intent: UserFeedbackIntent


class RuntimeFeedbackCommand(FeedbackRecordCommand):
    kind: Literal[FeedbackKind.RUNTIME_FEEDBACK] = FeedbackKind.RUNTIME_FEEDBACK


class QualityFeedbackCommand(FeedbackRecordCommand):
    kind: Literal[FeedbackKind.QUALITY_FEEDBACK] = FeedbackKind.QUALITY_FEEDBACK


class FeedbackResolutionCommand(_StrictModel):
    access_scope: AccessScopeResult
    feedback_id: str
    requested_feedback_scope_kind: FeedbackScopeKind | None = None
    requested_scope_id: str | None = None
    resolution: FeedbackResolutionResult
    updated_at: datetime
    resolution_source_event_id: str | None = None
    resolution_batch_id: str | None = None

    @field_validator("feedback_id", "requested_scope_id", "resolution_source_event_id", "resolution_batch_id")
    @classmethod
    def _normalize_required_or_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "FeedbackResolutionCommand 字段")

    @model_validator(mode="after")
    def _validate_resolution_source_fields(self) -> "FeedbackResolutionCommand":
        has_source_event = self.resolution_source_event_id is not None
        has_batch_id = self.resolution_batch_id is not None
        if has_source_event == has_batch_id:
            raise ValueError("resolution_source_event_id 与 resolution_batch_id 必须且只能提供一个")
        return self


class FeedbackGapResult(_StrictModel):
    gap_kind: FeedbackGapKind
    reason_code: FeedbackReasonCode
    source_ref: FeedbackSourceRefResult | None = None
    target_ref: FeedbackTargetRefResult | None = None
    stage: FeedbackSnapshotStage
    scope: FeedbackSnapshotScopeResult | None = None
    diagnostic_summary: str
    created_at: datetime

    @field_validator("diagnostic_summary")
    @classmethod
    def _diagnostic_summary_required(cls, value: str) -> str:
        return _require_text(value, "diagnostic_summary")

    @model_validator(mode="after")
    def _validate_gap_invariants(self) -> "FeedbackGapResult":
        if self.gap_kind == FeedbackGapKind.SOURCE_MISSING:
            if self.reason_code != FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISSING:
                raise ValueError("source_missing gap 的 reason_code 必须是 feedback_source_event_missing")
            if self.source_ref is not None:
                raise ValueError("source_missing gap 不允许携带 source_ref")
        return self


class FeedbackWriteResult(_StrictModel):
    success: bool
    created: bool
    reused: bool
    reason_code: FeedbackReasonCode
    feedback_id: str | None = None
    record_ref: FeedbackRecordResult | None = None
    scope: FeedbackScopeResult | None = None
    source_ref: FeedbackSourceRefResult | None = None
    target_ref: FeedbackTargetRefResult | None = None
    resolution: FeedbackResolutionResult | None = None
    gap: FeedbackGapResult | None = None
    created_at: datetime | None = None

    @field_validator("feedback_id")
    @classmethod
    def _normalize_optional_feedback_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "feedback_id")

    @model_validator(mode="after")
    def _validate_state_matrix(self) -> "FeedbackWriteResult":
        if self.created and self.reused:
            raise ValueError("created 与 reused 不能同时为 true")
        if self.success:
            if self.feedback_id is None or self.record_ref is None or self.scope is None or self.source_ref is None or self.target_ref is None or self.created_at is None:
                raise ValueError("success=true 时必须提供完整 record/scope/source/target/created_at")
            if self.gap is not None:
                raise ValueError("success=true 时不允许携带 gap")
            is_created = self.created and not self.reused and self.resolution is None
            is_reused = self.reused and not self.created and self.resolution is None
            is_resolution = self.resolution is not None and not self.created and not self.reused
            if not any((is_created, is_reused, is_resolution)):
                raise ValueError("success=true 时必须且只能命中 created/reused/resolution 三类成功语义之一")
            return self
        if self.created or self.reused:
            raise ValueError("success=false 时 created/reused 必须为 false")
        if self.feedback_id is not None or self.record_ref is not None or self.resolution is not None:
            raise ValueError("success=false 时不允许携带 feedback_id/record_ref/resolution")
        if self.gap is None:
            raise ValueError("success=false 时必须提供 gap")
        return self


class FeedbackSnapshotItemResult(_StrictModel):
    feedback_id: str
    kind: FeedbackKind
    category: FeedbackCategory
    status: FeedbackStatus
    severity: FeedbackSeverity
    target_ref: FeedbackTargetRefResult
    source_kind: FeedbackSourceKind
    source_event_id: str
    source_run_id: str | None = None
    target_run_id: str | None = None
    prompt_safe_summary: str
    reason_code: FeedbackReasonCode
    resolution_reason_code: FeedbackResolutionReasonCode | None = None
    created_at: datetime

    @field_validator("feedback_id", "source_event_id", "prompt_safe_summary")
    @classmethod
    def _required_text_fields(cls, value: str) -> str:
        return _require_text(value, "FeedbackSnapshotItemResult 字段")

    @field_validator("source_run_id", "target_run_id")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "FeedbackSnapshotItemResult 可选字段")

    @model_validator(mode="after")
    def _validate_snapshot_item_semantics(self) -> "FeedbackSnapshotItemResult":
        _validate_runtime_signal_usage(source_kind=self.source_kind, field_name="FeedbackSnapshotItemResult")
        _validate_kind_category_pair(kind=self.kind, category=self.category, field_name="FeedbackSnapshotItemResult")
        return self


class ExcludedFeedbackRefResult(_StrictModel):
    feedback_id: str
    reason_code: FeedbackReasonCode | FeedbackResolutionReasonCode
    stage: FeedbackSnapshotStage
    scope: FeedbackSnapshotScopeResult
    status: FeedbackStatus
    excluded_by: FeedbackExcludedBy
    created_at: datetime

    @field_validator("feedback_id")
    @classmethod
    def _feedback_id_required(cls, value: str) -> str:
        return _require_text(value, "feedback_id")


class FeedbackSnapshotResult(_StrictModel):
    scope: FeedbackSnapshotScopeResult
    snapshot_id: str
    source_run_id: str | None = None
    stage: FeedbackSnapshotStage
    active_user_feedback: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    active_runtime_feedback: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    active_quality_feedback: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    open_feedback_items: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    resolved_feedback_items: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    do_not_repeat_feedback: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    user_constraints: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    replan_hints: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    review_hints: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    final_gate_hints: list[FeedbackSnapshotItemResult] = Field(default_factory=list)
    feedback_gaps: list[FeedbackGapResult] = Field(default_factory=list)
    included_feedback_ids: list[str] = Field(default_factory=list)
    excluded_feedback_refs: list[ExcludedFeedbackRefResult] = Field(default_factory=list)
    cursor: FeedbackSnapshotCursorResult
    created_at: datetime

    @field_validator("snapshot_id")
    @classmethod
    def _snapshot_id_required(cls, value: str) -> str:
        return _require_text(value, "snapshot_id")

    @field_validator("source_run_id")
    @classmethod
    def _normalize_source_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "source_run_id")

    @field_validator("included_feedback_ids")
    @classmethod
    def _normalize_feedback_ids(cls, values: list[str]) -> list[str]:
        return [_require_text(value, "included_feedback_ids") for value in list(values or [])]


class FeedbackEventPayloadResult(_StrictModel):
    feedback_refs: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    kind_counts: dict[str, int] = Field(default_factory=dict)
    summary: str | None = None
    source_event_ids: list[str] = Field(default_factory=list)
    runtime_metadata: dict[str, str | int | bool | None] = Field(default_factory=dict)

    @field_validator("feedback_refs", "source_event_ids")
    @classmethod
    def _normalize_id_lists(cls, values: list[str]) -> list[str]:
        return [_require_text(value, "event payload id") for value in list(values or [])]

    @field_validator("summary")
    @classmethod
    def _normalize_optional_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "summary")


class FeedbackInputEventPayloadResult(_StrictModel):
    source_action: str
    intent_kind: UserFeedbackIntentKind
    target_ref: FeedbackTargetRefResult
    reason_code: FeedbackReasonCode
    sanitized_summary: str | None = None
    input_hash: str
    runtime_metadata: dict[str, str | int | bool | None] = Field(default_factory=dict)

    @field_validator("source_action", "input_hash")
    @classmethod
    def _required_text_fields(cls, value: str) -> str:
        return _require_text(value, "feedback_input 字段")

    @field_validator("sanitized_summary")
    @classmethod
    def _normalize_optional_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "sanitized_summary")


def build_feedback_record_result(record: FeedbackRecord) -> FeedbackRecordResult:
    """从领域记录构造完整 Result，供 ORM round-trip 与测试复用。"""
    return FeedbackRecordResult(
        feedback_id=record.id,
        scope=record.scope,
        source_ref=record.source_ref,
        target_ref=record.target_ref,
        kind=record.kind,
        category=record.category,
        status=record.status,
        severity=record.severity,
        reason_code=record.reason_code,
        feedback_summary=record.feedback_summary,
        prompt_safe_summary=record.prompt_safe_summary,
        classification=record.classification,
        resolution=record.resolution,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def map_feedback_data_origin_to_data_origin(value: FeedbackDataOrigin) -> DataOrigin:
    """将 feedback 分类来源映射回 P0-4 DataOrigin。"""
    if value == FeedbackDataOrigin.USER:
        return DataOrigin.USER_MESSAGE
    if value in {FeedbackDataOrigin.RUNTIME, FeedbackDataOrigin.SYSTEM_QUALITY, FeedbackDataOrigin.EVALUATION}:
        return DataOrigin.SYSTEM_OPERATIONAL
    raise ValueError(f"未知 FeedbackDataOrigin: {value}")


def build_feedback_record_from_result(
        *,
        record: FeedbackRecordResult,
        user_id: str,
        session_id: str,
        workspace_id: str,
        run_id: str | None,
        source_run_id: str | None,
        target_run_id: str | None,
        step_id: str | None,
        source_record_refs: list[dict[str, str | None]] | None,
        dedupe_key: str,
        feedback_key: str,
        decay_policy: str,
        ttl_scope: str,
        expires_at: datetime | None,
        profile_hash: str | None,
        origin: DataOrigin,
) -> FeedbackRecord:
    """测试/持久化需要时，从 Result 反建领域记录。"""
    return FeedbackRecord(
        id=record.feedback_id,
        scope=record.scope,
        source_ref=record.source_ref,
        target_ref=record.target_ref,
        resolution=record.resolution,
        feedback_summary=record.feedback_summary,
        prompt_safe_summary=record.prompt_safe_summary,
        classification=record.classification,
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        run_id=run_id,
        feedback_scope_kind=record.scope.feedback_scope_kind,
        scope_id=record.scope.scope_id,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        step_id=step_id,
        kind=record.kind,
        category=record.category,
        status=record.status,
        severity=record.severity,
        source_kind=record.source_ref.source_kind,
        source_event_id=record.source_ref.source_event_id,
        source_record_refs=list(source_record_refs or record.source_ref.source_record_refs),
        target_type=record.target_ref.target_type,
        target_id=record.target_ref.target_id,
        target_revision_id=record.target_ref.target_revision_id,
        target_content_hash=record.target_ref.target_content_hash,
        dedupe_key=dedupe_key,
        feedback_key=feedback_key,
        reason_code=record.reason_code,
        resolution_reason_code=record.resolution.resolution_reason_code,
        resolved_by_ref=record.resolution.resolved_by_ref,
        decay_policy=decay_policy,
        expires_at=expires_at,
        ttl_scope=ttl_scope,
        profile_hash=profile_hash,
        origin=origin,
        trust_level=record.classification.trust_level,
        privacy_level=record.classification.privacy_level,
        retention_policy=record.classification.retention_policy,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
