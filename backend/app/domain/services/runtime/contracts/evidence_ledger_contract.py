#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence Ledger 纯领域契约。
Evidence 是 step/action/claim 组织层：它引用 sandbox fact、持久事件、
artifact 或用户确认，说明这些来源能证明什么。它不重新采集工具结果，
也不保存 raw stdout、完整网页正文或完整文件正文。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)

from app.domain.services.runtime.contracts.document_input_contract import DocumentParseStatus
from app.domain.services.runtime.contracts.artifact_governance_contract import ArtifactStorageRef


class EvidenceScope(str, Enum):
    STEP = "step"
    RUN = "run"
    WORKSPACE = "workspace"


class EvidenceKind(str, Enum):
    ACTION_EVIDENCE = "action_evidence"
    CLAIM_SUPPORT = "claim_support"
    ARTIFACT_EVIDENCE = "artifact_evidence"
    DOCUMENT_EVIDENCE = "document_evidence"
    SEARCH_EVIDENCE = "search_evidence"
    PAGE_EVIDENCE = "page_evidence"
    FILE_EVIDENCE = "file_evidence"
    BROWSER_EVIDENCE = "browser_evidence"
    TOOL_FAILURE_EVIDENCE = "tool_failure_evidence"
    HUMAN_CONFIRMATION_EVIDENCE = "human_confirmation_evidence"
    EVIDENCE_GAP = "evidence_gap"
    CORRECTION = "correction"
    SUPERSEDED = "superseded"


class EvidenceSourceType(str, Enum):
    SANDBOX_FACT = "sandbox_fact"
    WORKFLOW_EVENT = "workflow_event"
    ARTIFACT = "artifact"
    USER_CONFIRMATION = "user_confirmation"
    SYSTEM_PROJECTION = "system_projection"


class EvidenceSupportLevel(str, Enum):
    """证据支持强度，不等同于证据质量。

    support level 表达来源对 claim 的证明力度；quality status 表达来源是否
    截断、过期、失败或缺失。下游必须同时消费两者。
    """

    STRONG = "strong"
    PARTIAL = "partial"
    WEAK = "weak"
    CONTRADICTS = "contradicts"
    GAP = "gap"


class EvidenceQualityStatus(str, Enum):
    VALID = "valid"
    PARTIAL = "partial"
    TRUNCATED = "truncated"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    MISSING_SOURCE = "missing_source"
    STALE = "stale"
    SUPERSEDED = "superseded"


class EvidenceReusePolicy(str, Enum):
    REUSE_ALLOWED = "reuse_allowed"
    VERIFY_BEFORE_REUSE = "verify_before_reuse"
    DO_NOT_REUSE = "do_not_reuse"


class EvidenceStalenessPolicy(str, Enum):
    STABLE = "stable"
    RUN_SCOPED = "run_scoped"
    STEP_SCOPED = "step_scoped"
    EXTERNAL_MAY_CHANGE = "external_may_change"


class EvidenceVisibility(str, Enum):
    INTERNAL = "internal"
    PROMPT_SUMMARY = "prompt_summary"
    AUDIT = "audit"


class EvidenceResultRefType(str, Enum):
    ARTIFACT_REF = "artifact_ref"
    FACT_REF = "fact_ref"
    SOURCE_EVENT_REF = "source_event_ref"
    DOCUMENT_SOURCE_REF = "document_source_ref"
    USER_CONFIRMATION_REF = "user_confirmation_ref"
    VERIFICATION_REF = "verification_ref"


class EvidenceReadStrategy(str, Enum):
    USE_DIGEST_SUMMARY = "use_digest_summary"
    READ_ARTIFACT = "read_artifact"
    READ_FACT_PAYLOAD = "read_fact_payload"
    READ_DOCUMENT_SOURCE = "read_document_source"
    VERIFY_BEFORE_USE = "verify_before_use"
    NOT_READABLE = "not_readable"


class EvidenceResolvedStatus(str, Enum):
    RESOLVED = "resolved"
    REQUIRES_VERIFICATION = "requires_verification"
    NOT_READABLE = "not_readable"
    MISSING = "missing"
    SCOPE_MISMATCH = "scope_mismatch"
    STALE = "stale"


FORBIDDEN_RESOLVED_PAYLOAD_RAW_KEYS = frozenset(
    {
        "raw_stdout",
        "stdout",
        "full_text",
        "file_content",
        "page_content",
        "document_text",
        "html",
    }
)
FORBIDDEN_PROJECTION_TEXT_TOKENS = frozenset(
    {
        "raw_stdout",
        "stdout",
        "full_text",
        "file_content",
        "page_content",
        "document_text",
        "html",
    }
)


class EvidenceDuplicateDecision(str, Enum):
    REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION = "reuse_existing_evidence_pending_resolution"
    REQUIRE_VERIFICATION = "require_verification"
    BLOCK_DUPLICATE_ACTION = "block_duplicate_action"


class EvidenceSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_type: EvidenceSourceType
    source_event_id: str | None = None
    fact_ids: list[str] = Field(default_factory=list)
    tool_call_id: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    message_event_id: str | None = None
    profile_hash: str | None = None


class EvidenceSubjectRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_type: str
    subject_key: str
    path: str | None = None
    url_hash: str | None = None
    query_hash: str | None = None
    artifact_path: str | None = None

    @field_validator("subject_type", "subject_key")
    @classmethod
    def _required_text_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("evidence subject 必填文本字段不能为空")
        return normalized


class _StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActionEvidencePayload(_StrictPayload):
    action_type: str
    function_name: str
    source_fact_ids: list[str]
    source_event_id: str
    result_status: str
    reason_code: str | None = None
    excerpt: str = ""
    is_truncated: bool


class ClaimSupportPayload(_StrictPayload):
    claim_text: str
    supporting_fact_ids: list[str] = Field(default_factory=list)
    supporting_artifact_ids: list[str] = Field(default_factory=list)
    support_level: EvidenceSupportLevel
    quality_status: EvidenceQualityStatus
    source_excerpt: str = ""
    limitations: list[str] = Field(default_factory=list)


class ArtifactEvidencePayload(_StrictPayload):
    artifact_id: str
    revision_id: str
    content_hash: str
    storage_ref: ArtifactStorageRef
    artifact_path: str
    artifact_type: str
    source_fact_ids: list[str] = Field(default_factory=list)
    source_event_id: str | None = None
    delivery_candidate: bool
    version_locked: Literal[True] = True


class DocumentEvidencePayload(_StrictPayload):
    file_id: str
    parse_status: DocumentParseStatus
    reason_code: str | None = None
    full_file_sha256: str | None = None
    read_content_sha256: str | None = None
    is_truncated: bool
    excerpt_char_count: int
    source_fact_id: str


class SearchEvidencePayload(_StrictPayload):
    query_excerpt: str
    query_hash: str | None = None
    verification_reason_code: str | None = None
    result_count: int
    top_result_origins: list[str]
    source_fact_id: str
    snippet_quality: str
    needs_fetch: bool


class PageEvidencePayload(_StrictPayload):
    origin: str
    url_hash: str | None = None
    verification_reason_code: str | None = None
    title: str
    status_code: int | None = None
    source_fact_id: str
    excerpt: str = ""
    is_truncated: bool


class FileEvidencePayload(_StrictPayload):
    path: str
    operation: str
    mutation_intent_hash: str | None = None
    exists: bool
    content_sha256: str | None = None
    content_sha256_kind: str
    source_fact_id: str
    excerpt: str = ""
    is_truncated: bool


class BrowserEvidencePayload(_StrictPayload):
    action: str | None = None
    origin: str | None = None
    title: str = ""
    screenshot_artifact_id: str | None = None
    source_fact_id: str
    excerpt: str = ""
    is_truncated: bool = False
    reason_code: str | None = None


class ToolFailureEvidencePayload(_StrictPayload):
    function_name: str
    reason_code: str
    source_fact_id: str
    message_excerpt: str = ""
    retry_count: int = 0
    timeout: bool = False


class HumanConfirmationEvidencePayload(_StrictPayload):
    interaction_id: str
    interaction_type: str
    message_event_id: str
    confirmed: bool | None = None
    summary: str
    reason_code: str | None = None


class EvidenceGapPayload(_StrictPayload):
    gap_type: str
    missing_source_types: list[EvidenceSourceType]
    claim_text: str
    reason_code: str
    required_for: str


class CorrectionEvidencePayload(_StrictPayload):
    corrected_evidence_ids: list[str]
    reason_code: str
    message_excerpt: str
    supersedes_evidence_id: str


class SupersededEvidencePayload(_StrictPayload):
    supersedes_evidence_id: str
    reason_code: str
    message_excerpt: str


EVIDENCE_PAYLOAD_SCHEMAS: dict[EvidenceKind, type[BaseModel]] = {
    EvidenceKind.ACTION_EVIDENCE: ActionEvidencePayload,
    EvidenceKind.CLAIM_SUPPORT: ClaimSupportPayload,
    EvidenceKind.ARTIFACT_EVIDENCE: ArtifactEvidencePayload,
    EvidenceKind.DOCUMENT_EVIDENCE: DocumentEvidencePayload,
    EvidenceKind.SEARCH_EVIDENCE: SearchEvidencePayload,
    EvidenceKind.PAGE_EVIDENCE: PageEvidencePayload,
    EvidenceKind.FILE_EVIDENCE: FileEvidencePayload,
    EvidenceKind.BROWSER_EVIDENCE: BrowserEvidencePayload,
    EvidenceKind.TOOL_FAILURE_EVIDENCE: ToolFailureEvidencePayload,
    EvidenceKind.HUMAN_CONFIRMATION_EVIDENCE: HumanConfirmationEvidencePayload,
    EvidenceKind.EVIDENCE_GAP: EvidenceGapPayload,
    EvidenceKind.CORRECTION: CorrectionEvidencePayload,
    EvidenceKind.SUPERSEDED: SupersededEvidencePayload,
}


class EvidenceResultRef(BaseModel):
    """ResultRef 是持久化结果引用，ResultHandle 是运行期读取入口。"""

    model_config = ConfigDict(extra="forbid")
    result_ref_type: EvidenceResultRefType
    ref_id: str
    source_step_id: str | None = None
    source_evidence_id: str | None = None
    source_fact_id: str | None = None
    source_event_id: str | None = None
    artifact_id: str | None = None
    revision_id: str | None = None
    artifact_path: str | None = None
    artifact_version_locked: bool = False
    artifact_hash_kind: str | None = None
    storage_ref: ArtifactStorageRef | None = None
    document_file_id: str | None = None
    subject_key: str | None = None
    payload_hash: str | None = None
    content_hash: str | None = None
    quality_status: EvidenceQualityStatus = EvidenceQualityStatus.MISSING_SOURCE
    support_level: EvidenceSupportLevel = EvidenceSupportLevel.WEAK
    reuse_policy: EvidenceReusePolicy = EvidenceReusePolicy.DO_NOT_REUSE
    staleness_policy: EvidenceStalenessPolicy = EvidenceStalenessPolicy.STEP_SCOPED
    read_strategy: EvidenceReadStrategy = EvidenceReadStrategy.NOT_READABLE
    reason_code: str | None = None
    allowed_verification_actions: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("ref_id")
    @classmethod
    def _ref_id_must_not_be_empty(cls, value: str) -> str:
        return _required_text(value, "result ref id 不能为空")

    @field_validator("summary")
    @classmethod
    def _summary_must_be_short(cls, value: str) -> str:
        return _short_text(value, 300, "ResultHandle.summary 不能超过 300 字符")

    @model_validator(mode="after")
    def _validate_ref_recoverability(self) -> "EvidenceResultRef":
        _validate_result_ref_recoverability(
            result_ref_type=self.result_ref_type,
            artifact_id=self.artifact_id,
            source_fact_id=self.source_fact_id,
            source_event_id=self.source_event_id,
            document_file_id=self.document_file_id,
            payload_hash=self.payload_hash,
            content_hash=self.content_hash,
            revision_id=self.revision_id,
            read_strategy=self.read_strategy,
            reason_code=self.reason_code,
            allowed_verification_actions=self.allowed_verification_actions,
        )
        return self


class EvidenceResultHandle(BaseModel):
    """ResultHandle 是后续 step 获取前序结果的唯一结构化入口。"""

    model_config = ConfigDict(extra="forbid")
    result_handle_id: str
    result_ref_type: EvidenceResultRefType
    ref_id: str
    source_step_id: str | None = None
    source_evidence_id: str | None = None
    source_fact_id: str | None = None
    source_event_id: str | None = None
    artifact_id: str | None = None
    revision_id: str | None = None
    artifact_path: str | None = None
    artifact_version_locked: bool = False
    artifact_hash_kind: str | None = None
    storage_ref: ArtifactStorageRef | None = None
    document_file_id: str | None = None
    subject_key: str | None = None
    payload_hash: str | None = None
    content_hash: str | None = None
    quality_status: EvidenceQualityStatus
    support_level: EvidenceSupportLevel
    reuse_policy: EvidenceReusePolicy
    staleness_policy: EvidenceStalenessPolicy
    read_strategy: EvidenceReadStrategy
    reason_code: str | None = None
    allowed_verification_actions: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("result_handle_id", "ref_id")
    @classmethod
    def _required_text_must_not_be_empty(cls, value: str) -> str:
        return _required_text(value, "result handle 必填文本字段不能为空")

    @field_validator("summary")
    @classmethod
    def _summary_must_be_short(cls, value: str) -> str:
        return _short_text(value, 300, "ResultHandle.summary 不能超过 300 字符")

    @model_validator(mode="after")
    def _handle_id_must_match_formula(self) -> "EvidenceResultHandle":
        _validate_result_ref_recoverability(
            result_ref_type=self.result_ref_type,
            artifact_id=self.artifact_id,
            source_fact_id=self.source_fact_id,
            source_event_id=self.source_event_id,
            document_file_id=self.document_file_id,
            payload_hash=self.payload_hash,
            content_hash=self.content_hash,
            revision_id=self.revision_id,
            read_strategy=self.read_strategy,
            reason_code=self.reason_code,
            allowed_verification_actions=self.allowed_verification_actions,
        )
        expected = build_evidence_result_handle_id_from_parts(
            result_ref_type=self.result_ref_type,
            ref_id=self.ref_id,
            source_evidence_id=self.source_evidence_id,
            source_fact_id=self.source_fact_id,
            source_event_id=self.source_event_id,
            artifact_id=self.artifact_id,
            revision_id=self.revision_id,
            document_file_id=self.document_file_id,
            subject_key=self.subject_key,
            payload_hash=self.payload_hash,
            content_hash=self.content_hash,
            read_strategy=self.read_strategy,
        )
        if self.result_handle_id != expected:
            raise ValueError("result_handle_id 与固定公式不一致")
        return self


class EvidenceResolvedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: EvidenceResolvedStatus
    result_ref_type: EvidenceResultRefType
    source_evidence_id: str | None = None
    source_fact_id: str | None = None
    source_event_id: str | None = None
    artifact_id: str | None = None
    revision_id: str | None = None
    document_file_id: str | None = None
    subject_key: str | None = None
    read_strategy: EvidenceReadStrategy
    summary: str = ""
    resolved_payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str | None = None
    content_hash: str | None = None
    reason_code: str | None = None
    allowed_verification_actions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_resolved_result_auditability(self) -> "EvidenceResolvedResult":
        has_reason_code = bool(str(self.reason_code or "").strip())
        has_payload_hash = bool(str(self.payload_hash or "").strip())
        has_content_hash = bool(str(self.content_hash or "").strip())
        if _resolved_payload_has_forbidden_raw_key(self.resolved_payload):
            raise ValueError("resolved_payload 禁止携带 raw/full content 字段")
        if self.read_strategy == EvidenceReadStrategy.NOT_READABLE and not has_reason_code:
            raise ValueError("not_readable resolved result 必须携带 reason_code")
        if self.status != EvidenceResolvedStatus.RESOLVED:
            if not has_reason_code:
                raise ValueError("非 resolved result 必须携带稳定 reason_code")
            return self
        if not self.resolved_payload:
            raise ValueError("resolved result 必须携带非空 resolved_payload")
        if not has_payload_hash and not has_content_hash:
            raise ValueError("resolved result 必须携带 payload_hash 或 content_hash")
        return self


class EvidenceVerificationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result_ref_type: EvidenceResultRefType
    subject_key: str | None = None
    allowed_verification_actions: list[str] = Field(default_factory=list)
    blocked_duplicate_actions: list[str] = Field(default_factory=list)
    failure_reason_codes: list[str] = Field(default_factory=list)


class EvidenceCompletedActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    action_key: str
    action_type: str
    function_name: str
    subject_key: str
    result_status: str
    support_level: EvidenceSupportLevel
    quality_status: EvidenceQualityStatus
    evidence_ids: list[str]
    fact_ids: list[str]
    result_refs: list[EvidenceResultRef] = Field(default_factory=list)
    result_handle: EvidenceResultHandle | None = None


class EvidenceAvailableArtifactResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    revision_id: str
    content_hash: str
    storage_ref: ArtifactStorageRef
    path: str
    artifact_type: str
    source_event_id: str | None = None
    source_fact_ids: list[str] = Field(default_factory=list)
    source_step_id: str
    source_evidence_ids: list[str]
    delivery_candidate: bool
    version_locked: Literal[True]
    reuse_policy: EvidenceReusePolicy
    result_handle: EvidenceResultHandle | None = None


class EvidenceVerifiedClaimResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_key: str
    claim_text: str
    source_step_id: str
    support_level: EvidenceSupportLevel
    quality_status: EvidenceQualityStatus
    evidence_ids: list[str]
    supporting_result_refs: list[EvidenceResultRef] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class EvidenceGapResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_key: str | None = None
    claim_text: str
    source_step_id: str | None = None
    reason_code: str
    required_for: str
    missing_source_types: list[EvidenceSourceType] = Field(default_factory=list)


class EvidenceBackedFactProjection(BaseModel):
    """由 Evidence digest 投影出的短可读事实，保留可审计 refs。"""

    model_config = ConfigDict(extra="forbid")
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    user_confirmation_event_ids: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _text_must_be_short_and_safe(cls, value: str) -> str:
        normalized = _required_text(value, "evidence-backed fact text 不能为空")
        normalized = _short_text(normalized, 300, "evidence-backed fact text 不能超过 300 字符")
        _reject_raw_projection_text(normalized)
        return normalized


class EvidenceDoNotRepeatResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_key: str
    subject_key: str
    reason_code: str
    source_step_id: str
    evidence_ids: list[str]
    reuse_policy: EvidenceReusePolicy
    staleness_policy: EvidenceStalenessPolicy
    support_level: EvidenceSupportLevel
    quality_status: EvidenceQualityStatus
    result_status: str
    duplicate_decision: EvidenceDuplicateDecision
    reuse_result_ref: EvidenceResultRef | None = None
    result_handle_id: str | None = None
    reuse_summary: str = ""
    message_for_model: str = ""

    @field_validator("reuse_summary")
    @classmethod
    def _reuse_summary_must_be_short(cls, value: str) -> str:
        normalized = str(value or "")
        if len(normalized) > 300:
            raise ValueError("reuse_summary 不能超过 300 字符")
        return normalized


class EvidenceDigestResult(BaseModel):
    """下一步执行决策输入，不是 UI 展示文案。

    execute/replan 只能消费这里的结构化字段判断已完成动作、可用产物、
    可信 claim、缺口和 do_not_repeat；summary_for_prompt 只是裁剪说明。
    """

    model_config = ConfigDict(extra="forbid")
    run_id: str
    current_step_id: str | None = None
    source_step_ids: list[str] = Field(default_factory=list)
    completed_actions: list[EvidenceCompletedActionResult] = Field(default_factory=list)
    available_artifacts: list[EvidenceAvailableArtifactResult] = Field(default_factory=list)
    verified_claims: list[EvidenceVerifiedClaimResult] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGapResult] = Field(default_factory=list)
    evidence_backed_facts: list[EvidenceBackedFactProjection] = Field(default_factory=list)
    do_not_repeat: list[EvidenceDoNotRepeatResult] = Field(default_factory=list)
    requires_verification: list[EvidenceGapResult] = Field(default_factory=list)
    result_handles: list[EvidenceResultHandle] = Field(default_factory=list)
    summary_for_prompt: str = ""
    cursor: str

    @field_validator("summary_for_prompt")
    @classmethod
    def _summary_for_prompt_must_be_short(cls, value: str) -> str:
        normalized = str(value or "")
        if len(normalized) > 1200:
            raise ValueError("summary_for_prompt 默认不能超过 1200 字符")
        return normalized


class EvidenceReuseSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    current_step_id: str | None = None
    source_step_ids: list[str] = Field(default_factory=list)
    cursor: str
    do_not_repeat: list[EvidenceDoNotRepeatResult] = Field(default_factory=list)
    completed_actions: list[EvidenceCompletedActionResult] = Field(default_factory=list)
    available_artifacts: list[EvidenceAvailableArtifactResult] = Field(default_factory=list)
    verified_claims: list[EvidenceVerifiedClaimResult] = Field(default_factory=list)
    result_handles: list[EvidenceResultHandle] = Field(default_factory=list)

    @model_validator(mode="after")
    def _do_not_repeat_handles_must_exist(self) -> "EvidenceReuseSnapshot":
        handle_ids = {handle.result_handle_id for handle in self.result_handles}
        for item in self.do_not_repeat:
            if item.result_handle_id and item.result_handle_id not in handle_ids:
                raise ValueError("do_not_repeat.result_handle_id 必须指向 snapshot 内已有 handle")
            if item.duplicate_decision == EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION:
                if item.reuse_result_ref is None or not item.result_handle_id:
                    raise ValueError("pending resolution 必须携带 reuse_result_ref 和 result_handle_id")
                if item.result_handle_id not in handle_ids:
                    raise ValueError("pending resolution result_handle_id 必须可解析")
        return self


class RuntimeEvidenceContextResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    current_step_id: str | None = None
    source_step_ids: list[str] = Field(default_factory=list)
    has_previous_completed_steps: bool
    prompt_digest: str = ""
    evidence_reuse_snapshot: EvidenceReuseSnapshot | None = None
    result_handles: list[EvidenceResultHandle] = Field(default_factory=list)
    result_handle_index: dict[str, EvidenceResultHandle] = Field(default_factory=dict)
    evidence_gaps: list[EvidenceGapResult] = Field(default_factory=list)
    cursor: str

    @field_validator("prompt_digest")
    @classmethod
    def _prompt_digest_must_be_short(cls, value: str) -> str:
        normalized = str(value or "")
        if len(normalized) > 1200:
            raise ValueError("prompt_digest 默认不能超过 1200 字符")
        return normalized

    @field_validator("cursor")
    @classmethod
    def _cursor_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("evidence context cursor 不能为空")
        return normalized

    @model_validator(mode="after")
    def _validate_snapshot_and_handle_index(self) -> "RuntimeEvidenceContextResult":
        if self.has_previous_completed_steps and self.evidence_reuse_snapshot is None:
            raise ValueError("存在前序 completed step 时必须携带 evidence_reuse_snapshot")
        if self.evidence_reuse_snapshot is not None and self.evidence_reuse_snapshot.cursor != self.cursor:
            raise ValueError("prompt 通道与结构化通道 cursor 不一致")
        handle_ids = {handle.result_handle_id for handle in self.result_handles}
        if self.evidence_reuse_snapshot is not None:
            snapshot_handle_ids = {
                handle.result_handle_id
                for handle in self.evidence_reuse_snapshot.result_handles
            }
            if handle_ids != snapshot_handle_ids:
                raise ValueError("context.result_handles 必须与 snapshot.result_handles 保持一致")
        index_ids = set(self.result_handle_index.keys())
        if index_ids != handle_ids:
            raise ValueError("result_handle_index 必须只以 result_handle_id 为主键")
        for key, handle in self.result_handle_index.items():
            if key != handle.result_handle_id:
                raise ValueError("result_handle_index key 必须等于 handle.result_handle_id")
        return self


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    workspace_id: str
    run_id: str | None = None
    step_id: str | None = None
    evidence_scope: EvidenceScope
    evidence_kind: EvidenceKind
    action_key: str | None = None
    claim_key: str | None = None
    claim_text: str | None = None
    subject_key: str | None = None
    source_step_id: str | None = None
    support_level: EvidenceSupportLevel
    quality_status: EvidenceQualityStatus
    source_ref: EvidenceSourceRef
    subject_ref: EvidenceSubjectRef
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str
    idempotency_key: str
    confidence: float = 0.0
    reusable: bool = False
    reuse_policy: EvidenceReusePolicy = EvidenceReusePolicy.DO_NOT_REUSE
    staleness_policy: EvidenceStalenessPolicy = EvidenceStalenessPolicy.STEP_SCOPED
    visibility: EvidenceVisibility = EvidenceVisibility.INTERNAL
    origin: DataOrigin = DataOrigin.SYSTEM_OPERATIONAL
    trust_level: DataTrustLevel = DataTrustLevel.SYSTEM_GENERATED
    privacy_level: PrivacyLevel = PrivacyLevel.INTERNAL
    retention_policy: RetentionPolicyKind = RetentionPolicyKind.SESSION_BOUND
    result_refs: list[EvidenceResultRef] = Field(default_factory=list)
    result_refs_hash: str
    related_evidence_ids: list[str] = Field(default_factory=list)
    supersedes_evidence_id: str | None = None
    source_event_id: str | None = None
    tool_call_id: str | None = None
    primary_fact_id: str | None = None
    primary_artifact_id: str | None = None
    profile_hash: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("user_id", "session_id", "workspace_id", "payload_hash", "idempotency_key", "result_refs_hash")
    @classmethod
    def _required_text_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("evidence 必填文本字段不能为空")
        return normalized

    @field_validator("summary")
    @classmethod
    def _summary_must_be_short(cls, value: str) -> str:
        normalized = str(value or "")
        if len(normalized) > 500:
            raise ValueError("summary 不能超过 500 字符")
        return normalized

    @field_validator("confidence")
    @classmethod
    def _confidence_must_be_ratio(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence 必须在 0 到 1 之间")
        return value

    @model_validator(mode="after")
    def _validate_record_contract(self) -> "EvidenceRecord":
        self._fill_index_fields_from_refs()
        self._validate_scope()
        self._validate_source_invariants()
        self._validate_reuse_and_result_refs()
        payload_model = validate_evidence_payload(
            evidence_kind=self.evidence_kind,
            payload=self.payload,
        )
        normalized_payload = payload_model.model_dump(mode="json")
        expected_payload_hash = build_evidence_payload_hash(normalized_payload)
        if self.payload_hash != expected_payload_hash:
            raise ValueError("payload_hash 与 normalized evidence payload 不一致")
        expected_result_refs_hash = build_evidence_result_refs_hash(self.result_refs)
        if self.result_refs_hash != expected_result_refs_hash:
            raise ValueError("result_refs_hash 与 normalized result_refs 不一致")
        expected_idempotency_key = build_evidence_idempotency_key(
            user_id=self.user_id,
            session_id=self.session_id,
            run_id=self.run_id,
            step_id=self.step_id,
            evidence_scope=self.evidence_scope,
            evidence_kind=self.evidence_kind,
            source_event_id=self.source_event_id,
            primary_fact_id=self.primary_fact_id,
            primary_artifact_id=self.primary_artifact_id,
            action_key=self.action_key,
            claim_key=self.claim_key,
            payload_hash=expected_payload_hash,
            result_refs_hash=expected_result_refs_hash,
        )
        if self.idempotency_key != expected_idempotency_key:
            raise ValueError("idempotency_key 与 evidence 幂等字段不一致")
        self.payload = normalized_payload
        return self

    def _fill_index_fields_from_refs(self) -> None:
        self.source_event_id = self.source_event_id or self.source_ref.source_event_id
        self.tool_call_id = self.tool_call_id or self.source_ref.tool_call_id
        self.primary_fact_id = self.primary_fact_id or _first_non_empty(self.source_ref.fact_ids)
        self.primary_artifact_id = self.primary_artifact_id or _first_non_empty(self.source_ref.artifact_ids)
        self.profile_hash = self.profile_hash or self.source_ref.profile_hash
        self.subject_key = self.subject_key or self.subject_ref.subject_key
        if self.source_step_id is None and self.evidence_scope == EvidenceScope.STEP:
            self.source_step_id = self.step_id

    def _validate_scope(self) -> None:
        if self.evidence_scope == EvidenceScope.STEP:
            if not self.run_id or not self.step_id:
                raise ValueError("STEP scope evidence 必须包含 run_id 和 step_id")
            if self.source_step_id != self.step_id:
                raise ValueError("STEP scope evidence 的 source_step_id 必须等于 step_id")
            return
        if self.evidence_scope == EvidenceScope.RUN:
            if not self.run_id:
                raise ValueError("RUN scope evidence 必须包含 run_id")
            if self.step_id is not None:
                raise ValueError("RUN scope evidence 的 step_id 必须为空")
            return
        if self.evidence_scope == EvidenceScope.WORKSPACE:
            if self.run_id is not None or self.step_id is not None:
                raise ValueError("WORKSPACE scope evidence 禁止绑定 run_id 或 step_id")

    def _validate_source_invariants(self) -> None:
        if self.evidence_kind in {EvidenceKind.CORRECTION, EvidenceKind.SUPERSEDED} and not self.supersedes_evidence_id:
            raise ValueError("CORRECTION/SUPERSEDED evidence 必须包含 supersedes_evidence_id")
        if self.evidence_kind not in {EvidenceKind.CORRECTION, EvidenceKind.SUPERSEDED} and self.supersedes_evidence_id:
            raise ValueError("supersedes_evidence_id 只能用于 CORRECTION/SUPERSEDED evidence")
        if self.evidence_kind == EvidenceKind.EVIDENCE_GAP:
            if not str(self.payload.get("reason_code") or "").strip():
                raise ValueError("EVIDENCE_GAP 必须包含稳定 reason_code")
            return
        if self.evidence_kind in {EvidenceKind.CORRECTION, EvidenceKind.SUPERSEDED}:
            return
        has_source_ref = bool(
            self.source_ref.source_event_id
            or self.source_ref.fact_ids
            or self.source_ref.artifact_ids
            or self.source_ref.message_event_id
        )
        if not has_source_ref:
            raise ValueError("successful evidence 必须包含可审计 source ref")
        if not self.source_ref.source_event_id:
            raise ValueError("successful evidence 必须绑定 source_event_id")

    def _validate_reuse_and_result_refs(self) -> None:
        if self.reusable and not self.result_refs:
            raise ValueError("reusable evidence 必须持久化 result_refs")
        if self.reusable and self.reuse_policy != EvidenceReusePolicy.REUSE_ALLOWED:
            raise ValueError("reusable=true 只能用于 reuse_allowed evidence")


def validate_evidence_payload(*, evidence_kind: EvidenceKind, payload: Mapping[str, Any]) -> BaseModel:
    """按 EvidenceKind 校验 payload strict schema。"""
    schema = EVIDENCE_PAYLOAD_SCHEMAS[evidence_kind]
    return schema.model_validate(dict(payload or {}))


def stable_json_dumps(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    """按稳定 JSON 形态序列化 evidence payload/ref。"""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def build_evidence_payload_hash(payload: Mapping[str, Any]) -> str:
    """对已脱敏、已归一的 evidence payload 计算 sha256。"""
    digest = hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


RESULT_REF_HASH_FIELDS = (
    "result_ref_type",
    "ref_id",
    "source_event_id",
    "source_fact_id",
    "artifact_id",
    "artifact_path",
    "document_file_id",
    "subject_key",
    "payload_hash",
    "content_hash",
    "read_strategy",
    "quality_status",
    "reuse_policy",
    "staleness_policy",
)


def _result_ref_sort_key(ref: EvidenceResultRef) -> tuple[str, str, str, str, str, str, str]:
    return (
        ref.result_ref_type.value,
        ref.ref_id,
        ref.source_event_id or "",
        ref.source_fact_id or "",
        ref.artifact_id or "",
        ref.document_file_id or "",
        ref.subject_key or "",
    )


def _result_ref_hash_payload(ref: EvidenceResultRef) -> dict[str, str]:
    dumped = ref.model_dump(mode="json")
    return {
        field: str(dumped.get(field) or "")
        for field in RESULT_REF_HASH_FIELDS
    }


def build_evidence_result_refs_hash(result_refs: Sequence[EvidenceResultRef | Mapping[str, Any]]) -> str:
    """对 result_refs 计算稳定 hash。
    摘要、列表顺序和 dict 输入顺序都不能影响 hash；固定字段缺失时按空字符串参与。
    """
    normalized_refs = [
        ref if isinstance(ref, EvidenceResultRef) else EvidenceResultRef.model_validate(ref)
        for ref in list(result_refs or [])
    ]
    hash_payload = [
        _result_ref_hash_payload(ref)
        for ref in sorted(normalized_refs, key=_result_ref_sort_key)
    ]
    digest = hashlib.sha256(stable_json_dumps(hash_payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_evidence_idempotency_key(
    *,
    user_id: str,
    session_id: str,
    run_id: str | None,
    step_id: str | None,
    evidence_scope: EvidenceScope,
    evidence_kind: EvidenceKind,
    source_event_id: str | None,
    primary_fact_id: str | None,
    primary_artifact_id: str | None,
    action_key: str | None,
    claim_key: str | None,
    payload_hash: str,
    result_refs_hash: str,
) -> str:
    """按 PR1 固定字段构造 evidence 幂等键。"""
    parts = [
        user_id,
        session_id,
        run_id or "",
        step_id or "",
        evidence_scope.value,
        evidence_kind.value,
        source_event_id or "",
        primary_fact_id or "",
        primary_artifact_id or "",
        action_key or "",
        claim_key or "",
        payload_hash,
        result_refs_hash,
    ]
    normalized = stable_json_dumps([str(part or "").strip() for part in parts])
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_evidence_result_handle_id_from_parts(
    *,
    result_ref_type: EvidenceResultRefType,
    ref_id: str,
    source_evidence_id: str | None,
    source_fact_id: str | None,
    source_event_id: str | None,
    artifact_id: str | None,
    revision_id: str | None = None,
    document_file_id: str | None,
    subject_key: str | None,
    payload_hash: str | None,
    content_hash: str | None,
    read_strategy: EvidenceReadStrategy,
) -> str:
    parts = [
        result_ref_type.value,
        ref_id,
        source_evidence_id or "",
        source_fact_id or "",
        source_event_id or "",
        artifact_id or "",
        revision_id or "",
        document_file_id or "",
        subject_key or "",
        payload_hash or "",
        content_hash or "",
        read_strategy.value,
    ]
    normalized = stable_json_dumps([str(part or "").strip() for part in parts])
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_evidence_result_handle(result_ref: EvidenceResultRef) -> EvidenceResultHandle:
    """从持久化 ResultRef 稳定重建运行期 ResultHandle。"""
    result_handle_id = build_evidence_result_handle_id_from_parts(
        result_ref_type=result_ref.result_ref_type,
        ref_id=result_ref.ref_id,
        source_evidence_id=result_ref.source_evidence_id,
        source_fact_id=result_ref.source_fact_id,
        source_event_id=result_ref.source_event_id,
        artifact_id=result_ref.artifact_id,
        revision_id=result_ref.revision_id,
        document_file_id=result_ref.document_file_id,
        subject_key=result_ref.subject_key,
        payload_hash=result_ref.payload_hash,
        content_hash=result_ref.content_hash,
        read_strategy=result_ref.read_strategy,
    )
    return EvidenceResultHandle(result_handle_id=result_handle_id, **result_ref.model_dump(mode="python"))


def classify_evidence_data(
    *,
    evidence_kind: EvidenceKind,
    source_type: EvidenceSourceType,
) -> tuple[DataOrigin, DataTrustLevel, PrivacyLevel, RetentionPolicyKind]:
    """按 Evidence 来源返回 PR1 固定数据分类枚举。"""
    if evidence_kind == EvidenceKind.DOCUMENT_EVIDENCE:
        return (
            DataOrigin.USER_UPLOAD,
            DataTrustLevel.USER_PROVIDED,
            PrivacyLevel.PRIVATE,
            RetentionPolicyKind.WORKSPACE_BOUND,
        )
    if evidence_kind in {EvidenceKind.SEARCH_EVIDENCE, EvidenceKind.PAGE_EVIDENCE}:
        return (
            DataOrigin.EXTERNAL_WEB,
            DataTrustLevel.EXTERNAL_UNTRUSTED,
            PrivacyLevel.PRIVATE,
            RetentionPolicyKind.WORKSPACE_BOUND,
        )
    if evidence_kind == EvidenceKind.HUMAN_CONFIRMATION_EVIDENCE or source_type == EvidenceSourceType.USER_CONFIRMATION:
        return (
            DataOrigin.USER_MESSAGE,
            DataTrustLevel.USER_PROVIDED,
            PrivacyLevel.PRIVATE,
            RetentionPolicyKind.SESSION_BOUND,
        )
    if evidence_kind == EvidenceKind.EVIDENCE_GAP:
        return (
            DataOrigin.SYSTEM_OPERATIONAL,
            DataTrustLevel.SYSTEM_GENERATED,
            PrivacyLevel.INTERNAL,
            RetentionPolicyKind.SESSION_BOUND,
        )
    return (
        DataOrigin.SANDBOX_STATE,
        DataTrustLevel.SYSTEM_GENERATED,
        PrivacyLevel.PRIVATE,
        RetentionPolicyKind.WORKSPACE_BOUND,
    )


def _first_non_empty(values: Sequence[str] | None) -> str | None:
    for value in list(values or []):
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return None


def _required_text(value: str, message: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(message)
    return normalized


def _short_text(value: str, limit: int, message: str) -> str:
    normalized = str(value or "")
    if len(normalized) > limit:
        raise ValueError(message)
    return normalized


def _resolved_payload_has_forbidden_raw_key(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) in FORBIDDEN_RESOLVED_PAYLOAD_RAW_KEYS:
                return True
            if _resolved_payload_has_forbidden_raw_key(value):
                return True
        return False
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return any(_resolved_payload_has_forbidden_raw_key(item) for item in payload)
    return False


def _reject_raw_projection_text(text: str) -> None:
    normalized = str(text or "").lower()
    if any(token in normalized for token in FORBIDDEN_PROJECTION_TEXT_TOKENS):
        raise ValueError("evidence-backed fact text 禁止包含 raw/full content 字段名")


def _validate_result_ref_recoverability(
    *,
    result_ref_type: EvidenceResultRefType,
    artifact_id: str | None,
    revision_id: str | None,
    source_fact_id: str | None,
    source_event_id: str | None,
    document_file_id: str | None,
    payload_hash: str | None,
    content_hash: str | None,
    read_strategy: EvidenceReadStrategy,
    reason_code: str | None,
    allowed_verification_actions: Sequence[str],
) -> None:
    if read_strategy == EvidenceReadStrategy.NOT_READABLE and not str(reason_code or "").strip():
        raise ValueError("not_readable result ref/handle 必须携带 reason_code")
    if result_ref_type == EvidenceResultRefType.ARTIFACT_REF:
        if not artifact_id or not revision_id or not content_hash:
            raise ValueError("artifact_ref 必须包含 artifact_id/revision_id/content_hash")
    if result_ref_type == EvidenceResultRefType.FACT_REF and not source_fact_id:
        raise ValueError("fact_ref 必须包含 source_fact_id")
    if result_ref_type == EvidenceResultRefType.SOURCE_EVENT_REF and not source_event_id:
        raise ValueError("source_event_ref 必须包含 source_event_id")
    if result_ref_type == EvidenceResultRefType.DOCUMENT_SOURCE_REF and not document_file_id:
        raise ValueError("document_source_ref 必须包含 document_file_id")
    if result_ref_type == EvidenceResultRefType.USER_CONFIRMATION_REF and not source_event_id:
        raise ValueError("user_confirmation_ref 必须绑定 confirmation event")
    if result_ref_type == EvidenceResultRefType.VERIFICATION_REF:
        if not str(reason_code or "").strip():
            raise ValueError("verification_ref 必须携带 reason_code")
        if read_strategy == EvidenceReadStrategy.VERIFY_BEFORE_USE and not list(allowed_verification_actions or []):
            raise ValueError("verify_before_use verification_ref 必须携带 allowed_verification_actions")
