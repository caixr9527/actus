#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safety Audit Ledger 纯领域契约。

Safety Audit 只记录运行时安全决策。参数摘要必须是脱敏、裁剪后的
结构化描述：保留字段形状、类型、长度、hash、路径/URL 的安全摘要，
禁止保存 raw args、文件正文、网页正文、raw stdout 或 secret。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping, Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.sensitive_data_policy import detect_sensitive_text


class SafetyAuditDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REWRITE = "rewrite"
    REQUIRE_CONFIRMATION = "require_confirmation"
    CONFIRMATION_APPROVED = "confirmation_approved"
    CONFIRMATION_REJECTED = "confirmation_rejected"
    CORRECTION = "correction"
    SUPERSEDED = "superseded"


class SafetyAuditRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyAuditWriteStatus(str, Enum):
    CREATED = "created"
    REUSED = "reused"


class SafetyAuditNonToolActionKind(str, Enum):
    ARTIFACT_DOWNLOAD = "artifact_download"
    ARTIFACT_PREVIEW = "artifact_preview"
    DOCUMENT_PREFLIGHT = "document_preflight"
    EXTERNAL_CAPABILITY_GOVERNANCE = "external_capability_governance"


class SafetyAuditPolicyTraceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_name: str
    action: str
    reason_code: str

    @field_validator("policy_name", "action", "reason_code")
    @classmethod
    def _text_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("policy_trace 文本字段不能为空")
        return normalized


class SafetyAuditRelatedArtifactRevisionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    revision_id: str
    content_hash: str | None = None


class SafetyAuditRelatedRefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    artifact_revisions: list[SafetyAuditRelatedArtifactRevisionRef] = Field(default_factory=list)


class SafetyAuditUrlDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheme: str
    host: str
    normalized_url_hash: str
    has_query: bool
    query_stripped: bool
    is_external_network: bool


class SafetyAuditPathDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basename: str
    extension: str
    path_hash: str
    is_absolute: bool
    scope_hint: Literal["workspace", "tmp", "home", "absolute", "relative"]
    scope_status: Literal["unverified", "verified"]
    truncated: bool
    reason_code: str | None = None
    relative_path: str | None = None

    @model_validator(mode="after")
    def _relative_path_only_for_verified_safe_scope(self) -> "SafetyAuditPathDigest":
        if self.relative_path and self.scope_status != "verified":
            raise ValueError("relative_path 只能用于已校验安全路径")
        if self.relative_path and self.relative_path.startswith("/"):
            raise ValueError("relative_path 禁止保存绝对路径")
        if self.scope_status == "verified" and not self.relative_path:
            raise ValueError("已校验安全路径必须保存相对路径")
        return self


class SafetyAuditArgValueDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    is_empty: bool
    hash: str | None = None
    length: int | None = None
    field_count: int | None = None
    fields: dict[str, "SafetyAuditArgValueDigest"] | None = None
    items: list["SafetyAuditArgValueDigest"] | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None
    has_secret: bool | None = None
    has_pii: bool | None = None
    sensitive_categories: list[str] = Field(default_factory=list)
    path: SafetyAuditPathDigest | None = None
    url: SafetyAuditUrlDigest | None = None

    @model_validator(mode="after")
    def _reject_raw_payload_shapes(self) -> "SafetyAuditArgValueDigest":
        if self.type == "string" and self.hash is None:
            raise ValueError("字符串摘要必须只保存 hash，不允许保存原文")
        if self.path is not None and self.type != "string":
            raise ValueError("path 摘要只能挂载在 string 类型字段下")
        if self.url is not None and self.type != "string":
            raise ValueError("url 摘要只能挂载在 string 类型字段下")
        return self


class SafetyAuditArgsDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["object"] = "object"
    field_count: int
    fields: dict[str, SafetyAuditArgValueDigest]
    hash: str

    @model_validator(mode="after")
    def _field_count_must_match_fields(self) -> "SafetyAuditArgsDigest":
        if self.field_count != len(self.fields):
            raise ValueError("field_count 与 fields 数量不一致")
        return self


class SafetyAuditRewriteMetadataDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["object"] = "object"
    field_count: int = 0
    fields: dict[str, SafetyAuditArgValueDigest] = Field(default_factory=dict)
    hash: str = Field(default_factory=lambda: build_hash({}))

    @model_validator(mode="after")
    def _field_count_must_match_fields(self) -> "SafetyAuditRewriteMetadataDigest":
        if self.field_count != len(self.fields):
            raise ValueError("rewrite metadata field_count 与 fields 数量不一致")
        return self


class SafetyAuditDataClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: DataOrigin
    trust_level: DataTrustLevel
    privacy_level: PrivacyLevel
    retention_policy: RetentionPolicyKind
    has_sensitive_refs: bool = False
    data_categories: list[str] = Field(default_factory=list)


class SafetyAuditExternalCapabilityGovernanceDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_provider: str | None = None
    manifest_ref: str | None = None
    permission_claims: list[str] = Field(default_factory=list)
    network_required: bool = False
    filesystem_access: Literal["none", "read", "write", "delete", "overwrite", "read_write"] = "none"

    @field_validator("external_provider", "manifest_ref")
    @classmethod
    def _optional_text_must_be_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value or "").strip()
        return normalized or None

    @field_validator("permission_claims")
    @classmethod
    def _permission_claims_must_be_normalized(cls, value: list[str]) -> list[str]:
        normalized = sorted({str(item or "").strip().lower() for item in value if str(item or "").strip()})
        return normalized


class SafetyAuditRiskClassificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: SafetyAuditDecision
    normalized_function_name: str | None = None
    tool_family: str | None = None
    capability_id: str | None = None
    action_kind: str | None = None
    reason_code: str | None = None
    scope_result: str | None = None
    artifact_delivery_state: str | None = None
    external_provider: str | None = None
    external_capability: SafetyAuditExternalCapabilityGovernanceDigest | None = None


class SafetyAuditRiskClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: SafetyAuditRiskLevel
    matched_rule: str


class SafetyAuditRiskClassificationDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: SafetyAuditRiskLevel
    matched_rule: str

    @field_validator("matched_rule")
    @classmethod
    def _matched_rule_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("matched_rule 不能为空")
        return normalized


class SafetyAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    workspace_id: str
    run_id: str
    step_id: str | None = None
    action_id: str
    tool_call_id: str | None = None
    capability_id: str
    tool_family: str
    function_name: str
    normalized_function_name: str
    requested_args_digest: SafetyAuditArgsDigest
    final_function_name: str
    final_normalized_function_name: str
    final_args_digest: SafetyAuditArgsDigest
    decision: SafetyAuditDecision
    reason_code: str
    risk_level: SafetyAuditRiskLevel
    policy_trace: list[SafetyAuditPolicyTraceEntry] = Field(default_factory=list)
    winning_policy: str
    tool_call_fingerprint: str
    rewrite_applied: bool = False
    rewrite_reason: str | None = None
    rewrite_metadata_digest: SafetyAuditRewriteMetadataDigest = Field(default_factory=SafetyAuditRewriteMetadataDigest)
    confirmation_id: str | None = None
    decision_event_id: str | None = None
    tool_event_source_event_id: str | None = None
    confirmation_event_id: str | None = None
    source_event_type: str | None = None
    source_linked_at: datetime | None = None
    related_fact_ids: list[str] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    related_artifact_revisions: list[SafetyAuditRelatedArtifactRevisionRef] = Field(default_factory=list)
    external_capability_governance: SafetyAuditExternalCapabilityGovernanceDigest | None = None
    profile_hash: str | None = None
    origin: DataOrigin = DataOrigin.SYSTEM_OPERATIONAL
    trust_level: DataTrustLevel = DataTrustLevel.SYSTEM_GENERATED
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    retention_policy: RetentionPolicyKind = RetentionPolicyKind.WORKSPACE_BOUND
    classification: SafetyAuditDataClassification | None = None
    risk_classification_digest: SafetyAuditRiskClassificationDigest | None = None
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator(
        "user_id",
        "session_id",
        "workspace_id",
        "run_id",
        "action_id",
        "capability_id",
        "tool_family",
        "function_name",
        "normalized_function_name",
        "final_function_name",
        "final_normalized_function_name",
        "reason_code",
        "winning_policy",
        "tool_call_fingerprint",
    )
    @classmethod
    def _required_text_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("Safety Audit 必填文本字段不能为空")
        return normalized

    @model_validator(mode="after")
    def _validate_action_id_and_rewrite(self) -> "SafetyAuditRecord":
        expected_action_id = build_safety_audit_action_id(
            run_id=self.run_id,
            step_id=self.step_id,
            tool_call_id=self.tool_call_id,
            tool_call_fingerprint=self.tool_call_fingerprint,
            decision=self.decision,
            reason_code=self.reason_code,
        )
        if self.action_id != expected_action_id:
            raise ValueError("action_id 与 Safety Audit 幂等字段不一致")
        if self.decision == SafetyAuditDecision.REWRITE and not self.rewrite_applied:
            raise ValueError("rewrite 决策必须标记 rewrite_applied")
        if self.decision != SafetyAuditDecision.REWRITE and self.rewrite_applied and not self.rewrite_reason:
            raise ValueError("非 rewrite 决策如标记 rewrite_applied 必须提供 rewrite_reason")
        return self


class SafetyAuditRecordCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scope: AccessScopeResult
    user_id: str
    session_id: str
    workspace_id: str
    run_id: str
    step_id: str | None = None
    tool_call_id: str | None = None
    capability_id: str
    tool_family: str
    function_name: str
    normalized_function_name: str
    requested_args: dict[str, Any] = Field(default_factory=dict)
    final_function_name: str
    final_normalized_function_name: str
    final_args: dict[str, Any] = Field(default_factory=dict)
    decision: SafetyAuditDecision
    reason_code: str
    policy_trace: list[SafetyAuditPolicyTraceEntry] = Field(default_factory=list)
    winning_policy: str
    tool_call_fingerprint: str
    rewrite_applied: bool = False
    rewrite_reason: str | None = None
    rewrite_metadata: dict[str, Any] = Field(default_factory=dict)
    confirmation_id: str | None = None
    related_fact_ids: list[str] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    related_artifact_revisions: list[SafetyAuditRelatedArtifactRevisionRef] = Field(default_factory=list)
    profile_hash: str | None = None
    data_classification: SafetyAuditDataClassification | None = None
    risk_input: SafetyAuditRiskClassificationInput | None = None
    safe_path_roots: list[str] = Field(default_factory=list)
    external_capability_governance: SafetyAuditExternalCapabilityGovernanceDigest | None = None


class NonToolSafetyAuditCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scope: AccessScopeResult
    action_kind: SafetyAuditNonToolActionKind
    user_id: str
    session_id: str
    workspace_id: str
    run_id: str
    step_id: str | None = None
    action_id_hint: str | None = None
    capability_id: str
    tool_family: str = "non_tool"
    function_name: str
    requested_args: dict[str, Any] = Field(default_factory=dict)
    final_args: dict[str, Any] = Field(default_factory=dict)
    decision: SafetyAuditDecision
    reason_code: str
    artifact_delivery_state: str | None = None
    external_provider: str | None = None
    external_capability: SafetyAuditExternalCapabilityGovernanceDigest | None = None
    related_artifact_revisions: list[SafetyAuditRelatedArtifactRevisionRef] = Field(default_factory=list)
    data_classification: SafetyAuditDataClassification | None = None
    safe_path_roots: list[str] = Field(default_factory=list)


class SafetyAuditRecordResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    action_id: str
    decision: SafetyAuditDecision
    risk_level: SafetyAuditRiskLevel
    reason_code: str
    run_id: str
    step_id: str | None = None
    tool_call_id: str | None = None


class SafetyAuditWriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    record: SafetyAuditRecordResult
    status: SafetyAuditWriteStatus
    reason_code: str


class SafetyAuditDecisionCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow: int = 0
    block: int = 0
    rewrite: int = 0
    require_confirmation: int = 0
    confirmation_approved: int = 0
    confirmation_rejected: int = 0
    correction: int = 0
    superseded: int = 0


class SafetyAuditRiskCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0


class SafetyAuditSnapshotRecordRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    decision: SafetyAuditDecision
    risk_level: SafetyAuditRiskLevel
    reason_code: str
    step_id: str | None = None
    tool_call_id: str | None = None
    function_name: str
    created_at: datetime


class SafetyAuditSnapshotResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    audit_cursor: str | None = None
    decision_counts: SafetyAuditDecisionCounts
    risk_counts: SafetyAuditRiskCounts
    blocked_actions: list[SafetyAuditSnapshotRecordRef] = Field(default_factory=list)
    rewritten_actions: list[SafetyAuditSnapshotRecordRef] = Field(default_factory=list)
    confirmation_decisions: list[SafetyAuditSnapshotRecordRef] = Field(default_factory=list)
    critical_findings: list[SafetyAuditSnapshotRecordRef] = Field(default_factory=list)
    latest_records: list[SafetyAuditSnapshotRecordRef] = Field(default_factory=list)


class SafetyAuditEventRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    decision: SafetyAuditDecision
    risk_level: SafetyAuditRiskLevel
    reason_code: str
    step_id: str | None = None
    tool_call_id: str | None = None
    function_name: str


class SafetyAuditEventRuntimeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visibility: Literal["hidden"] = "hidden"
    projection_key: str
    schema_version: Literal["safety_audit_event.v1"] = "safety_audit_event.v1"


class SafetyAuditEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_refs: list[SafetyAuditEventRef] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    decision_counts: SafetyAuditDecisionCounts
    risk_counts: SafetyAuditRiskCounts
    blocked_count: int = 0
    rewrite_count: int = 0
    confirmation_count: int = 0
    summary: str
    runtime_metadata: SafetyAuditEventRuntimeMetadata


class SafetyAuditEventProjectResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str | None = None
    projected: bool = False
    audit_ids: list[str] = Field(default_factory=list)
    reason_code: str = ""


class SafetyAuditRecorderPort(Protocol):
    async def record_constraint_decision(self, command: SafetyAuditRecordCommand) -> SafetyAuditWriteResult:
        ...

    async def attach_tool_event_source(
            self,
            audit_id: str,
            tool_event_source_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        ...

    async def attach_decision_event(
            self,
            audit_id: str,
            decision_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        ...

    async def attach_confirmation_event(
            self,
            audit_id: str,
            confirmation_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        ...

    async def record_confirmation_decision(self, command: SafetyAuditRecordCommand) -> SafetyAuditWriteResult:
        ...

    async def record_non_tool_action(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        ...


class NonToolSafetyAuditRecorderPort(Protocol):
    async def record_artifact_download_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        ...

    async def record_artifact_preview_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        ...

    async def record_document_preflight_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        ...

    async def record_external_capability_governance_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        ...


class SafetyAuditEventProjectorPort(Protocol):
    async def project_tool_event_source(
            self,
            *,
            scope: AccessScopeResult,
            tool_event_source_event_id: str,
    ) -> SafetyAuditEventProjectResult:
        ...

    async def project_single_audit(
            self,
            *,
            scope: AccessScopeResult,
            audit_id: str,
    ) -> SafetyAuditEventProjectResult:
        ...


def stable_json_dumps(payload: Any) -> str:
    """按稳定 JSON 形态序列化安全摘要，供 hash 和幂等计算使用。"""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def build_hash(value: Any) -> str:
    digest = hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_safety_audit_action_id(
        *,
        run_id: str,
        step_id: str | None,
        tool_call_id: str | None,
        tool_call_fingerprint: str,
        decision: SafetyAuditDecision | str,
        reason_code: str,
) -> str:
    """按 PR1 固定字段构造业务幂等身份。"""

    decision_value = decision.value if isinstance(decision, SafetyAuditDecision) else str(decision or "")
    parts = [
        run_id,
        step_id or "",
        tool_call_id or "",
        tool_call_fingerprint,
        decision_value,
        reason_code,
    ]
    normalized = "\n".join(str(part or "").strip() for part in parts)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"audit_action:{digest}"


def build_args_digest(
        args: Mapping[str, Any] | None,
        *,
        safe_path_roots: list[str] | None = None,
        max_text_length: int = 120,
        max_items: int = 20,
) -> SafetyAuditArgsDigest:
    """构造安全参数摘要。

    字符串只保存长度、hash、敏感类型和路径/URL 安全摘要；长文本、大数组
    会裁剪并记录原因。未传入安全根目录或校验失败时，路径不保存完整原文。
    """

    normalized_args = dict(args or {})
    fields = {
        str(key): _digest_value(
            value,
            safe_path_roots=safe_path_roots or [],
            max_text_length=max_text_length,
            max_items=max_items,
            depth=0,
        )
        for key, value in sorted(normalized_args.items(), key=lambda item: str(item[0]))
    }
    return SafetyAuditArgsDigest.model_validate(
        {
            "type": "object",
            "field_count": len(fields),
            "fields": fields,
            "hash": build_hash(fields),
        }
    )


def _digest_value(
        value: Any,
        *,
        safe_path_roots: list[str],
        max_text_length: int,
        max_items: int,
        depth: int,
) -> dict[str, Any]:
    if value is None:
        return {"type": "null", "is_empty": True}
    if isinstance(value, bool):
        return {"type": "bool", "is_empty": False, "hash": build_hash(value)}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "int", "is_empty": False, "hash": build_hash(value)}
    if isinstance(value, float):
        return {"type": "float", "is_empty": False, "hash": build_hash(value)}
    if isinstance(value, str):
        return _digest_string(value, safe_path_roots=safe_path_roots, max_text_length=max_text_length)
    if isinstance(value, Mapping):
        if depth >= 3:
            return _container_digest("object", value, truncated=True, reason="max_depth")
        items = list(value.items())
        truncated = len(items) > max_items
        visible_items = items[:max_items]
        return {
            "type": "object",
            "is_empty": len(items) == 0,
            "field_count": len(items),
            "truncated": truncated,
            "truncation_reason": "max_items" if truncated else None,
            "fields": {
                str(key): _digest_value(
                    child,
                    safe_path_roots=safe_path_roots,
                    max_text_length=max_text_length,
                    max_items=max_items,
                    depth=depth + 1,
                )
                for key, child in sorted(visible_items, key=lambda item: str(item[0]))
            },
            "hash": build_hash(value),
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        truncated = len(items) > max_items
        visible_items = items[:max_items]
        return {
            "type": "array",
            "is_empty": len(items) == 0,
            "length": len(items),
            "truncated": truncated,
            "truncation_reason": "max_items" if truncated else None,
            "items": [
                _digest_value(
                    item,
                    safe_path_roots=safe_path_roots,
                    max_text_length=max_text_length,
                    max_items=max_items,
                    depth=depth + 1,
                )
                for item in visible_items
            ],
            "hash": build_hash(items),
        }
    return {"type": type(value).__name__, "is_empty": False, "hash": build_hash(str(value))}


def _container_digest(kind: str, value: Any, *, truncated: bool, reason: str) -> dict[str, Any]:
    return {
        "type": kind,
        "is_empty": False,
        "truncated": truncated,
        "truncation_reason": reason,
        "hash": build_hash(value),
    }


def _digest_string(value: str, *, safe_path_roots: list[str], max_text_length: int) -> dict[str, Any]:
    normalized = str(value or "")
    sensitive = detect_sensitive_text(normalized)
    base = {
        "type": "string",
        "is_empty": normalized == "",
        "length": len(normalized),
        "hash": build_hash(normalized),
        "has_secret": sensitive.has_secret,
        "has_pii": sensitive.has_pii,
        "sensitive_categories": sorted(set(sensitive.categories)),
        "truncated": len(normalized) > max_text_length,
        "truncation_reason": "max_text_length" if len(normalized) > max_text_length else None,
    }
    if _looks_like_url(normalized):
        base["url"] = _digest_url(normalized)
        return base
    if _looks_like_path(normalized):
        base["path"] = _digest_path(normalized, safe_path_roots=safe_path_roots)
        return base
    return base


def _looks_like_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https", "ftp"} and bool(parsed.netloc)


def _digest_url(value: str) -> dict[str, Any]:
    parsed = urlsplit(value)
    normalized_without_query = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", ""))
    return {
        "scheme": parsed.scheme.lower(),
        "host": parsed.hostname or "",
        "normalized_url_hash": build_hash(normalized_without_query),
        "has_query": bool(parsed.query),
        "query_stripped": bool(parsed.query),
        "is_external_network": parsed.scheme in {"http", "https", "ftp"},
    }


def _looks_like_path(value: str) -> bool:
    if not value or "\n" in value:
        return False
    if value.startswith(("/", "./", "../", "~")):
        return True
    suffix = PurePosixPath(value.replace("\\", "/")).suffix
    return bool(suffix) and "/" in value.replace("\\", "/")


def _digest_path(value: str, *, safe_path_roots: list[str]) -> dict[str, Any]:
    expanded = os.path.expanduser(value)
    is_absolute = os.path.isabs(expanded)
    normalized_abs = os.path.abspath(expanded)
    path = PurePosixPath(value.replace("\\", "/"))
    digest = {
        "basename": path.name,
        "extension": path.suffix,
        "path_hash": build_hash(normalized_abs),
        "is_absolute": is_absolute,
        "scope_hint": _path_scope_hint(value, normalized_abs=normalized_abs, is_absolute=is_absolute),
        "scope_status": "unverified",
        "truncated": True,
        "reason_code": "safe_root_missing",
    }
    for root in safe_path_roots:
        root_abs = os.path.abspath(os.path.expanduser(root))
        try:
            if os.path.commonpath([normalized_abs, root_abs]) == root_abs:
                digest["relative_path"] = os.path.relpath(normalized_abs, root_abs)
                digest["scope_status"] = "verified"
                digest["truncated"] = False
                digest["reason_code"] = None
                break
        except ValueError:
            continue
    return digest


def _path_scope_hint(
        original_value: str,
        *,
        normalized_abs: str,
        is_absolute: bool,
) -> Literal["workspace", "tmp", "home", "absolute", "relative"]:
    if not is_absolute:
        return "relative"
    if normalized_abs.startswith("/tmp/") or normalized_abs.startswith("/private/tmp/"):
        return "tmp"
    if "/workspace/" in normalized_abs:
        return "workspace"
    if normalized_abs.startswith(os.path.expanduser("~")):
        return "home"
    if os.path.isabs(normalized_abs):
        return "absolute"
    return "relative"


CRITICAL_SCOPE_RESULTS = {
    "cross_user",
    "cross_session",
    "cross_workspace",
    "cross_run",
    "cursor_mismatch",
}

HIGH_REASON_TOKENS = {
    "permission",
    "unauthorized",
    "scope",
    "hash",
    "source",
    "contract",
    "missing",
    "credential",
    "secret",
}

MEDIUM_REASON_CODES = {
    "task_mode_tool_blocked",
    "research_file_context_required",
    "web_reading_file_tool_blocked",
    "file_processing_shell_explicit_required",
    "file_processing_shell_auxiliary_blocked",
    "file_write_intent_read_tool_blocked",
    "browser_route_blocked",
    "read_only_file_intent_write_blocked",
    "artifact_policy_file_output_blocked",
    "human_wait_non_interrupt_tool_blocked",
    "ask_user_not_allowed",
    "research_query_style_blocked",
    "research_route_search_required",
    "research_route_fetch_required",
    "research_route_cross_domain_fetch_limit",
    "research_search_to_fetch_rewrite",
    "web_reading_low_value_fetch_repeat",
    "search_repeat",
    "research_route_fingerprint_repeat",
    "repeat_tool_call",
    "repeat_tool_call_success_fallback",
    "browser_click_target_blocked",
    "browser_high_level_retry_blocked",
    "evidence_duplicate_blocked",
    "rewrite_chain_blocked",
}

LOW_REASON_CODES = {
    "allow",
    "evidence_reuse_pending_resolution",
    "evidence_reuse_allowed",
    "research_tool_fact_ready",
    "file_processing_facts_ready",
    "file_processing_raw_content_ready",
    "general_file_observation_ready",
    "research_evidence_ready",
    "web_reading_page_evidence_ready",
}

HIGH_REASON_CODES = {
    "invalid_tool",
    "evidence_reuse_requires_verification",
    "evidence_reuse_snapshot_missing",
    "evidence_context_missing",
    "evidence_context_cursor_mismatch",
    "evidence_context_invalid_schema",
    "result_handle_missing",
    "result_handle_resolve_failed",
    "result_handle_stale",
    "artifact_revision_scope_mismatch",
    "artifact_hash_changed",
    "artifact_storage_not_deliverable",
    "unsupported_media_image",
    "unsupported_media_audio",
    "unsupported_media_video",
}

EXISTING_CONSTRAINT_REASON_CODES = LOW_REASON_CODES | MEDIUM_REASON_CODES | HIGH_REASON_CODES | {
    "constraint_engine_error",
}


class SafetyAuditRiskClassifier:
    """稳定风险分类器，禁止读取 prompt 文案、ToolResult message 或前端展示文本。"""

    @staticmethod
    def classify(input_data: SafetyAuditRiskClassificationInput) -> SafetyAuditRiskClassificationResult:
        reason_code = str(input_data.reason_code or "").strip()
        function_name = str(input_data.normalized_function_name or "").lower()
        tool_family = str(input_data.tool_family or "").lower()
        capability_id = str(input_data.capability_id or "").lower()
        action_kind = str(input_data.action_kind or "").lower()
        scope_result = str(input_data.scope_result or "").strip()
        external_capability = input_data.external_capability

        if reason_code == "constraint_engine_error":
            return _classification(SafetyAuditRiskLevel.CRITICAL, "constraint_engine_error")
        if scope_result in CRITICAL_SCOPE_RESULTS:
            return _classification(SafetyAuditRiskLevel.CRITICAL, "critical_scope_result")
        if _has_secret_or_cross_scope_reason(reason_code):
            return _classification(SafetyAuditRiskLevel.CRITICAL, "secret_or_cross_scope_reason")
        if external_capability is not None and _has_sensitive_permission_claims(external_capability.permission_claims):
            return _classification(SafetyAuditRiskLevel.CRITICAL, "sensitive_external_permission_claim")
        if _has_unknown_identity(function_name, tool_family, capability_id, action_kind):
            return _classification(SafetyAuditRiskLevel.HIGH, "unknown_identity")
        if _is_external_capability_high_risk(
                tool_family=tool_family,
                external_provider=input_data.external_provider,
                external_capability=external_capability,
        ):
            return _classification(SafetyAuditRiskLevel.HIGH, "external_capability_high_risk")
        if _is_high_risk_action(function_name, tool_family, capability_id, action_kind):
            return _classification(SafetyAuditRiskLevel.HIGH, "high_risk_action")
        if input_data.decision == SafetyAuditDecision.REQUIRE_CONFIRMATION:
            return _classification(SafetyAuditRiskLevel.HIGH, "confirmation_high_risk_default")
        if reason_code in HIGH_REASON_CODES:
            return _classification(SafetyAuditRiskLevel.HIGH, "high_reason_code")
        if input_data.decision == SafetyAuditDecision.BLOCK and _contains_any(reason_code, HIGH_REASON_TOKENS):
            return _classification(SafetyAuditRiskLevel.HIGH, "block_security_reason")
        if _is_medium_risk_action(function_name, tool_family, capability_id, action_kind):
            return _classification(SafetyAuditRiskLevel.MEDIUM, "medium_risk_action")
        if reason_code in MEDIUM_REASON_CODES:
            return _classification(SafetyAuditRiskLevel.MEDIUM, "medium_reason_code")
        if reason_code in LOW_REASON_CODES:
            return _classification(SafetyAuditRiskLevel.LOW, "low_reason_code")
        return _classification(SafetyAuditRiskLevel.HIGH, "unknown_default")


def classify_safety_audit_risk(
        input_data: SafetyAuditRiskClassificationInput,
) -> SafetyAuditRiskClassificationResult:
    return SafetyAuditRiskClassifier.classify(input_data)


def _classification(
        risk_level: SafetyAuditRiskLevel,
        matched_rule: str,
) -> SafetyAuditRiskClassificationResult:
    return SafetyAuditRiskClassificationResult(risk_level=risk_level, matched_rule=matched_rule)


def _has_secret_or_cross_scope_reason(reason_code: str) -> bool:
    return _contains_any(reason_code, {"secret", "credential", "cross_user", "cross_session", "cross_workspace", "cross_run"})


def _contains_any(value: str, tokens: set[str]) -> bool:
    normalized = value.lower()
    return any(token in normalized for token in tokens)


def _has_sensitive_permission_claims(permission_claims: list[str]) -> bool:
    sensitive_tokens = {
        "user_data_export",
        "user data export",
        "secret",
        "credential",
        "external_send",
        "external send",
        "remote_execution",
        "remote execution",
    }
    return any(_contains_any(str(claim or ""), sensitive_tokens) for claim in permission_claims)


def _is_external_capability_high_risk(
        *,
        tool_family: str,
        external_provider: str | None,
        external_capability: SafetyAuditExternalCapabilityGovernanceDigest | None,
) -> bool:
    if tool_family in {"mcp", "a2a"}:
        return True
    if str(external_provider or "").strip():
        return True
    if external_capability is None:
        return False
    if str(external_capability.external_provider or "").strip():
        return True
    if external_capability.network_required:
        return True
    if external_capability.filesystem_access in {"write", "delete", "overwrite", "read_write"}:
        return True
    return False


def _has_unknown_identity(function_name: str, tool_family: str, capability_id: str, action_kind: str) -> bool:
    values = [function_name, tool_family, capability_id, action_kind]
    unknown_markers = {"unknown", "unsupported", "invalid"}
    if any(not value for value in values):
        return True
    return any(value in unknown_markers or _contains_any(value, unknown_markers) for value in values)


def _is_high_risk_action(function_name: str, tool_family: str, capability_id: str, action_kind: str) -> bool:
    joined = " ".join([function_name, tool_family, capability_id, action_kind])
    high_tokens = {
        "shell",
        "execute",
        "command",
        "install",
        "delete",
        "overwrite",
        "write_file",
        "replace_file",
        "external_api",
        "mcp",
        "a2a",
        "download",
        "preview",
        "skill_install",
        "skill_enable",
    }
    return _contains_any(joined, high_tokens)


def _is_medium_risk_action(function_name: str, tool_family: str, capability_id: str, action_kind: str) -> bool:
    joined = " ".join([function_name, tool_family, capability_id, action_kind])
    medium_tokens = {
        "search",
        "fetch",
        "browser",
        "web",
        "network",
        "read_file",
        "document",
        "artifact_read",
        "verification",
    }
    return _contains_any(joined, medium_tokens)
