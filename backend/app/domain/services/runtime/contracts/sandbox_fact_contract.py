#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Fact Ledger 纯领域契约。

Fact 是 sandbox、工具适配器或受控投影器产生的低层可验证事实，
不是 Evidence、Artifact 生命周期对象，也不是最终回答。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)


class SandboxFactKind(str, Enum):
    COMMAND_EXECUTION = "command_execution"
    SHELL_OUTPUT = "shell_output"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_LIST = "file_list"
    FILE_SEARCH = "file_search"
    FILE_SNAPSHOT = "file_snapshot"
    BROWSER_SNAPSHOT = "browser_snapshot"
    BROWSER_ACTION = "browser_action"
    SEARCH_RESULT = "search_result"
    FETCHED_PAGE = "fetched_page"
    DOCUMENT_CONTEXT = "document_context"
    PROFILE_REFERENCE = "profile_reference"
    TOOL_FAILURE = "tool_failure"
    HUMAN_INTERACTION = "human_interaction"
    CORRECTION = "correction"
    SUPERSEDED = "superseded"


class SandboxFactSourceType(str, Enum):
    TOOL_EVENT = "tool_event"
    SANDBOX_API = "sandbox_api"
    PROFILE = "profile"
    DOCUMENT_INPUT = "document_input"
    USER_CONFIRMATION = "user_confirmation"
    SYSTEM_PROJECTION = "system_projection"


class SandboxFactVisibility(str, Enum):
    INTERNAL = "internal"
    TIMELINE_SUMMARY = "timeline_summary"
    EVIDENCE_INPUT = "evidence_input"


class SandboxFactScope(str, Enum):
    WORKSPACE = "workspace"
    RUN = "run"
    STEP = "step"


class SandboxFactDataClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: DataOrigin
    trust_level: DataTrustLevel
    privacy_level: PrivacyLevel
    retention_policy: RetentionPolicyKind


class SandboxFactProfileRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str | None = None
    profile_hash: str | None = None
    sandbox_id: str | None = None
    generated_at: datetime | None = None
    status: Literal["available", "missing", "invalid"] = "missing"


class SandboxFactSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: SandboxFactSourceType
    source_event_id: str | None = None
    source_event_status: Literal["available", "missing"] = "missing"
    tool_event_id: str | None = None
    tool_call_id: str | None = None
    function_name: str | None = None
    artifact_id: str | None = None
    file_id: str | None = None
    document_source_id: str | None = None

    @model_validator(mode="after")
    def _source_event_status_must_match_id(self) -> "SandboxFactSourceRef":
        if self.source_event_id and self.source_event_status != "available":
            raise ValueError("source_event_id 存在时 source_event_status 必须为 available")
        if not self.source_event_id and self.source_event_status != "missing":
            raise ValueError("source_event_id 缺失时 source_event_status 必须为 missing")
        return self


class SandboxFactSubjectRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_type: Literal["command", "file", "browser", "search", "page", "document", "profile", "interaction"]
    subject_key: str
    path: str | None = None
    url_hash: str | None = None
    artifact_path: str | None = None

    @field_validator("subject_key")
    @classmethod
    def _subject_key_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("subject_key 不能为空")
        return normalized


class _StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CommandExecutionPayload(_StrictPayload):
    command_fingerprint: str
    cwd: str
    exit_code: int | None
    duration_ms: int | None
    stdout_excerpt: str
    stderr_excerpt: str
    stdout_truncated: bool
    stderr_truncated: bool
    changed_paths: list[str]
    timeout: bool
    command_excerpt: str | None = None
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class ShellOutputPayload(_StrictPayload):
    session_ref: str
    output_excerpt: str
    output_truncated: bool
    console_record_count: int
    process_status: str
    exit_code: int | None = None
    duration_ms: int | None = None
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class FileReadPayload(_StrictPayload):
    path: str
    exists: bool
    size: int | None
    content_sha256: str | None
    read_content_sha256: str | None = None
    content_sha256_kind: Literal["full_file_sha256", "read_content_sha256", "unknown"]
    mime_type: str
    line_range: dict[str, Any] | None
    excerpt: str
    is_truncated: bool
    mtime: datetime | None = None
    missing_fields: list[str] | None = None
    reason_code: str | None = None

    @model_validator(mode="after")
    def _canonicalize_hash_fields(self) -> "FileReadPayload":
        if self.read_content_sha256 and not self.content_sha256:
            self.content_sha256 = self.read_content_sha256
        if self.content_sha256 and not self.read_content_sha256:
            self.read_content_sha256 = self.content_sha256
        return self


class FileMutationPayload(_StrictPayload):
    path: str
    operation: Literal["write", "delete", "snapshot"]
    mutation_intent_hash: str
    exists: bool
    before_content_sha256: str | None
    after_content_sha256: str | None
    content_sha256_kind: Literal["full_file_sha256", "read_content_sha256", "unknown"]
    size_after: int | None
    changed: bool
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class FileListEntryPayload(_StrictPayload):
    name: str
    type: str
    size: int | None = None
    mtime: datetime | None = None


class FileListPayload(_StrictPayload):
    dir_path: str
    entry_count: int
    entries: list[FileListEntryPayload]
    is_truncated: bool
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class FileSearchMatchPayload(_StrictPayload):
    path: str
    line_number: int | None = None
    excerpt: str


class FileSearchPayload(_StrictPayload):
    path: str
    regex_hash: str
    match_count: int
    matches: list[FileSearchMatchPayload]
    is_truncated: bool
    regex_excerpt: str | None = None
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class BrowserSnapshotPayload(_StrictPayload):
    url_hash: str
    url_origin: str
    title: str
    screenshot_artifact_id: str | None
    screenshot_artifact_path: str | None
    structured_summary: str
    actionable_element_count: int
    degrade_reason: str | None
    screenshot_file_id: str | None = None
    screenshot_filename: str | None = None
    screenshot_filepath: str | None = None
    screenshot_key: str | None = None
    screenshot_mime_type: str | None = None
    screenshot_size: int | None = None
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class BrowserActionPayload(_StrictPayload):
    action: str
    target_summary: str
    url_hash_before: str | None
    url_hash_after: str | None
    success: bool
    degrade_reason: str | None
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class SearchResultItemPayload(_StrictPayload):
    title: str
    origin: str
    url_hash: str
    snippet_excerpt: str


class SearchResultPayload(_StrictPayload):
    query_hash: str
    query_excerpt: str
    result_count: int
    top_results: list[SearchResultItemPayload]
    is_truncated: bool
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class FetchedPagePayload(_StrictPayload):
    fetched_url_hash: str
    final_url_origin: str
    status_code: int | None
    content_type: str | None
    title: str
    excerpt: str
    is_truncated: bool
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class DocumentContextPayload(_StrictPayload):
    file_id: str
    filename_extension: str
    mime_type: str
    parse_status: str
    reason_code: str | None
    full_file_sha256: str | None
    read_content_sha256: str | None
    is_truncated: bool
    excerpt_char_count: int
    missing_fields: list[str] | None = None


class ToolFailurePayload(_StrictPayload):
    function_name: str
    reason_code: str
    message_excerpt: str
    retry_count: int
    timeout: bool
    diagnostic_type: str
    missing_fields: list[str] | None = None


class ProfileReferencePayload(_StrictPayload):
    profile_id: str | None
    profile_hash: str | None
    sandbox_id: str | None
    generated_at: datetime | None
    status: Literal["available", "missing", "invalid"]
    reason_code: str | None = None
    missing_fields: list[str] | None = None


class HumanInteractionPayload(_StrictPayload):
    interaction_type: str
    message_excerpt: str
    confirmed: bool | None = None
    reason_code: str | None = None
    missing_fields: list[str] | None = None


class CorrectionPayload(_StrictPayload):
    corrected_fact_ids: list[str]
    reason_code: str
    message_excerpt: str


class SupersededPayload(_StrictPayload):
    supersedes_fact_id: str
    reason_code: str
    message_excerpt: str


class SandboxFactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    workspace_id: str
    fact_scope: SandboxFactScope
    run_id: str | None = None
    step_id: str | None = None
    sandbox_id: str | None = None
    fact_kind: SandboxFactKind
    source_ref: SandboxFactSourceRef
    subject_ref: SandboxFactSubjectRef
    profile_ref: SandboxFactProfileRef = Field(default_factory=SandboxFactProfileRef)
    related_fact_ids: list[str] = Field(default_factory=list)
    supersedes_fact_id: str | None = None
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str
    idempotency_key: str
    visibility: SandboxFactVisibility = SandboxFactVisibility.EVIDENCE_INPUT
    origin: DataOrigin = DataOrigin.SANDBOX_STATE
    trust_level: DataTrustLevel = DataTrustLevel.SYSTEM_GENERATED
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    retention_policy: RetentionPolicyKind = RetentionPolicyKind.WORKSPACE_BOUND
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("user_id", "session_id", "workspace_id", "payload_hash", "idempotency_key")
    @classmethod
    def _required_text_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("sandbox fact 必填文本字段不能为空")
        return normalized

    @field_validator("summary")
    @classmethod
    def _summary_must_be_short(cls, value: str) -> str:
        normalized = str(value or "")
        if len(normalized) > 500:
            raise ValueError("summary 不能超过 500 字符")
        return normalized

    @model_validator(mode="after")
    def _validate_scope_and_append_only_refs(self) -> "SandboxFactRecord":
        if self.fact_scope == SandboxFactScope.STEP and (not self.run_id or not self.step_id):
            raise ValueError("STEP scope fact 必须包含 run_id 和 step_id")
        if self.fact_scope == SandboxFactScope.RUN:
            if not self.run_id:
                raise ValueError("RUN scope fact 必须包含 run_id")
            if self.step_id is not None:
                raise ValueError("RUN scope fact 的 step_id 必须为空")
        if self.fact_scope == SandboxFactScope.WORKSPACE and self.step_id is not None:
            raise ValueError("WORKSPACE scope fact 的 step_id 必须为空")

        if self.fact_kind == SandboxFactKind.SUPERSEDED and not self.supersedes_fact_id:
            raise ValueError("SUPERSEDED fact 必须包含 supersedes_fact_id")
        if self.fact_kind != SandboxFactKind.SUPERSEDED and self.supersedes_fact_id is not None:
            raise ValueError("supersedes_fact_id 只能用于 SUPERSEDED fact")
        if self.fact_kind not in {SandboxFactKind.CORRECTION, SandboxFactKind.SUPERSEDED} and self.related_fact_ids:
            raise ValueError("related_fact_ids 只能用于 CORRECTION 或 SUPERSEDED fact")
        payload_model = validate_sandbox_fact_payload(
            fact_kind=self.fact_kind,
            payload=self.payload,
        )
        normalized_payload = payload_model.model_dump(mode="json")
        expected_payload_hash = build_sandbox_fact_payload_hash(normalized_payload)
        if self.payload_hash != expected_payload_hash:
            raise ValueError("payload_hash 与 normalized payload 不一致")
        expected_idempotency_key = build_sandbox_fact_idempotency_key(
            user_id=self.user_id,
            session_id=self.session_id,
            workspace_id=self.workspace_id,
            fact_scope=self.fact_scope,
            run_id=self.run_id,
            step_id=self.step_id,
            fact_kind=self.fact_kind,
            source_event_id=self.source_ref.source_event_id,
            tool_call_id=self.source_ref.tool_call_id,
            subject_key=self.subject_ref.subject_key,
            payload_hash=expected_payload_hash,
        )
        if self.idempotency_key != expected_idempotency_key:
            raise ValueError("idempotency_key 与 sandbox fact 幂等字段不一致")
        self.payload = normalized_payload
        return self


SANDBOX_FACT_PAYLOAD_SCHEMAS: dict[SandboxFactKind, type[BaseModel]] = {
    SandboxFactKind.COMMAND_EXECUTION: CommandExecutionPayload,
    SandboxFactKind.SHELL_OUTPUT: ShellOutputPayload,
    SandboxFactKind.FILE_READ: FileReadPayload,
    SandboxFactKind.FILE_WRITE: FileMutationPayload,
    SandboxFactKind.FILE_DELETE: FileMutationPayload,
    SandboxFactKind.FILE_LIST: FileListPayload,
    SandboxFactKind.FILE_SEARCH: FileSearchPayload,
    SandboxFactKind.FILE_SNAPSHOT: FileMutationPayload,
    SandboxFactKind.BROWSER_SNAPSHOT: BrowserSnapshotPayload,
    SandboxFactKind.BROWSER_ACTION: BrowserActionPayload,
    SandboxFactKind.SEARCH_RESULT: SearchResultPayload,
    SandboxFactKind.FETCHED_PAGE: FetchedPagePayload,
    SandboxFactKind.DOCUMENT_CONTEXT: DocumentContextPayload,
    SandboxFactKind.PROFILE_REFERENCE: ProfileReferencePayload,
    SandboxFactKind.TOOL_FAILURE: ToolFailurePayload,
    SandboxFactKind.HUMAN_INTERACTION: HumanInteractionPayload,
    SandboxFactKind.CORRECTION: CorrectionPayload,
    SandboxFactKind.SUPERSEDED: SupersededPayload,
}


def validate_sandbox_fact_payload(
        *,
        fact_kind: SandboxFactKind,
        payload: Mapping[str, Any],
) -> BaseModel:
    """按 fact kind 校验 payload strict schema。"""

    schema = SANDBOX_FACT_PAYLOAD_SCHEMAS[fact_kind]
    return schema.model_validate(dict(payload or {}))


def stable_json_dumps(payload: Mapping[str, Any]) -> str:
    """按稳定 JSON 形态序列化 fact payload。"""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def build_sandbox_fact_payload_hash(payload: Mapping[str, Any]) -> str:
    """对已脱敏、已归一的 payload 计算 sha256。"""

    normalized_payload = _normalize_payload_for_hash(payload)
    digest = hashlib.sha256(stable_json_dumps(normalized_payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _normalize_payload_for_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload or {})
    if _looks_like_file_read_payload(normalized):
        content_hash = str(
            normalized.get("content_sha256")
            or normalized.get("read_content_sha256")
            or ""
        ).strip()
        if content_hash:
            normalized["content_sha256"] = content_hash
            normalized["read_content_sha256"] = content_hash
        else:
            normalized.pop("content_sha256", None)
            normalized.pop("read_content_sha256", None)
    return normalized


def _looks_like_file_read_payload(payload: Mapping[str, Any]) -> bool:
    keys = set(payload.keys())
    return {
        "path",
        "exists",
        "size",
        "content_sha256_kind",
        "mime_type",
        "line_range",
        "excerpt",
        "is_truncated",
    }.issubset(keys) and "operation" not in keys and "mutation_intent_hash" not in keys


def build_sandbox_fact_idempotency_key(
        *,
        user_id: str,
        session_id: str,
        workspace_id: str,
        fact_scope: SandboxFactScope,
        run_id: str | None,
        step_id: str | None,
        fact_kind: SandboxFactKind,
        source_event_id: str | None,
        tool_call_id: str | None,
        subject_key: str,
        payload_hash: str,
) -> str:
    """按 PR1 固定字段构造 fact 幂等键。"""

    parts = [
        user_id,
        session_id,
        workspace_id,
        fact_scope.value,
        run_id or "",
        step_id or "",
        fact_kind.value,
        source_event_id or "",
        tool_call_id or "",
        subject_key,
        payload_hash,
    ]
    normalized = "\n".join(str(part or "").strip() for part in parts)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def classify_sandbox_fact_data(
        *,
        fact_kind: SandboxFactKind,
        source_type: SandboxFactSourceType,
) -> SandboxFactDataClassificationResult:
    """按 fact kind 和 source type 返回数据分类字段。"""

    if fact_kind == SandboxFactKind.DOCUMENT_CONTEXT and source_type == SandboxFactSourceType.DOCUMENT_INPUT:
        return SandboxFactDataClassificationResult(
            origin=DataOrigin.USER_UPLOAD,
            trust_level=DataTrustLevel.USER_PROVIDED,
            privacy_level=PrivacyLevel.PRIVATE,
            retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        )
    if fact_kind in {SandboxFactKind.SEARCH_RESULT, SandboxFactKind.FETCHED_PAGE}:
        return SandboxFactDataClassificationResult(
            origin=DataOrigin.EXTERNAL_WEB,
            trust_level=DataTrustLevel.EXTERNAL_UNTRUSTED,
            privacy_level=PrivacyLevel.PRIVATE,
            retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        )
    if fact_kind in {
        SandboxFactKind.COMMAND_EXECUTION,
        SandboxFactKind.SHELL_OUTPUT,
        SandboxFactKind.FILE_READ,
        SandboxFactKind.FILE_WRITE,
        SandboxFactKind.FILE_DELETE,
        SandboxFactKind.FILE_LIST,
        SandboxFactKind.FILE_SEARCH,
        SandboxFactKind.FILE_SNAPSHOT,
        SandboxFactKind.BROWSER_SNAPSHOT,
        SandboxFactKind.BROWSER_ACTION,
    }:
        return SandboxFactDataClassificationResult(
            origin=DataOrigin.SANDBOX_STATE,
            trust_level=DataTrustLevel.SYSTEM_GENERATED,
            privacy_level=PrivacyLevel.PRIVATE,
            retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        )
    if fact_kind == SandboxFactKind.PROFILE_REFERENCE:
        return SandboxFactDataClassificationResult(
            origin=DataOrigin.SYSTEM_OPERATIONAL,
            trust_level=DataTrustLevel.SYSTEM_GENERATED,
            privacy_level=PrivacyLevel.INTERNAL,
            retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        )
    if fact_kind == SandboxFactKind.HUMAN_INTERACTION:
        return SandboxFactDataClassificationResult(
            origin=DataOrigin.USER_MESSAGE,
            trust_level=DataTrustLevel.USER_PROVIDED,
            privacy_level=PrivacyLevel.PRIVATE,
            retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        )
    if fact_kind == SandboxFactKind.TOOL_FAILURE:
        return SandboxFactDataClassificationResult(
            origin=DataOrigin.SYSTEM_OPERATIONAL,
            trust_level=DataTrustLevel.SYSTEM_GENERATED,
            privacy_level=PrivacyLevel.PRIVATE,
            retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        )
    return SandboxFactDataClassificationResult(
        origin=DataOrigin.SYSTEM_OPERATIONAL,
        trust_level=DataTrustLevel.SYSTEM_GENERATED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
    )
