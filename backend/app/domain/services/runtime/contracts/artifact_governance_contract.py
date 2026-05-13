#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-4 Artifact Governance 领域契约。"""

from __future__ import annotations

import re
import uuid
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


class ArtifactRevisionSourceKind(str, Enum):
    TOOL_WRITE_FILE = "tool_write_file"
    TOOL_REPLACE_FILE = "tool_replace_file"
    BROWSER_SCREENSHOT = "browser_screenshot"
    BROWSER_SNAPSHOT = "browser_snapshot"
    PAGE_SNAPSHOT = "page_snapshot"
    DOCUMENT_INPUT = "document_input"
    USER_UPLOAD = "user_upload"
    FINAL_ANSWER_SNAPSHOT = "final_answer_snapshot"
    DERIVED_EXPORT = "derived_export"
    RAG_CHUNK_INDEX = "rag_chunk_index"
    MANUAL_REGISTRATION = "manual_registration"


class ArtifactType(str, Enum):
    FILE = "file"
    SCREENSHOT = "screenshot"
    BROWSER_SNAPSHOT = "browser_snapshot"
    PAGE_SNAPSHOT = "page_snapshot"
    DATASET = "dataset"
    REPORT = "report"
    LOG_EXCERPT = "log_excerpt"
    FINAL_ANSWER_SNAPSHOT = "final_answer_snapshot"
    RAG_CHUNK_INDEX = "rag_chunk_index"


class ArtifactDeliveryState(str, Enum):
    CANDIDATE = "candidate"
    SELECTED = "selected"
    DELIVERED = "delivered"
    REJECTED = "rejected"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class ArtifactStatus(str, Enum):
    ACTIVE = "active"
    DELETED = "deleted"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class ArtifactStorageBackend(str, Enum):
    FILE_STORAGE = "file_storage"
    SANDBOX = "sandbox"
    INLINE_SNAPSHOT = "inline_snapshot"


_URL_OR_FILE_RE = re.compile(r"^(?:https?://|file://)", re.IGNORECASE)
_WINDOWS_OR_UNC_RE = re.compile(r"^(?:[a-zA-Z]:\\|\\\\)")
_SENSITIVE_TOKEN_RE = re.compile(
    r"(?:authorization|bearer\s+|token=|cookie=|api[_-]?key|password=)",
    re.IGNORECASE,
)
_HOST_ABSOLUTE_PREFIXES = (
    "/Users/",
    "/private/",
    "/var/",
    "/System/",
    "/Applications/",
    "/Library/",
    "/Volumes/",
)
_ALLOWED_SANDBOX_ABSOLUTE_PREFIXES = (
    "/home/ubuntu/",
    "/workspace/",
    "/tmp/",
)


def _has_parent_traversal(value: str) -> bool:
    parts = value.replace("\\", "/").split("/")
    return any(part == ".." for part in parts)


def _is_host_absolute_path(value: str) -> bool:
    return any(value.startswith(prefix) for prefix in _HOST_ABSOLUTE_PREFIXES)


def _validate_storage_locator(value: str | None, *, allow_sandbox_path: bool) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if _URL_OR_FILE_RE.search(normalized):
        raise ValueError("storage ref 不得保存 URL 或 file:// 地址")
    if _WINDOWS_OR_UNC_RE.search(normalized):
        raise ValueError("storage ref 不得保存 Windows 或 UNC 路径")
    if _SENSITIVE_TOKEN_RE.search(normalized):
        raise ValueError("storage ref 不得保存敏感读取凭证")
    if _has_parent_traversal(normalized):
        raise ValueError("storage ref 路径不得包含父目录跳转")
    if _is_host_absolute_path(normalized):
        raise ValueError("storage ref 不得保存宿主机绝对路径")
    if allow_sandbox_path and normalized.startswith("/"):
        if not normalized.startswith(_ALLOWED_SANDBOX_ABSOLUTE_PREFIXES):
            raise ValueError("sandbox_path 只允许受控 sandbox 绝对路径")
    return normalized


class ArtifactStorageRef(BaseModel):
    """Artifact revision 的严格存储引用。"""

    model_config = ConfigDict(extra="forbid")

    storage_backend: ArtifactStorageBackend
    object_key: str | None = None
    file_id: str | None = None
    sandbox_path: str | None = None
    storage_hash: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    reason_code: str | None = None

    @field_validator("object_key", "file_id")
    @classmethod
    def _validate_storage_keys(cls, value: str | None) -> str | None:
        return _validate_storage_locator(value, allow_sandbox_path=False)

    @field_validator("sandbox_path")
    @classmethod
    def _validate_sandbox_path(cls, value: str | None) -> str | None:
        return _validate_storage_locator(value, allow_sandbox_path=True)

    @field_validator("storage_hash", "mime_type", "reason_code")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("missing_fields")
    @classmethod
    def _normalize_missing_fields(cls, value: list[str]) -> list[str]:
        return [field for field in [str(item or "").strip() for item in value or []] if field]

    @model_validator(mode="after")
    def _validate_backend_contract(self) -> "ArtifactStorageRef":
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes 不能为负数")

        if self.storage_backend == ArtifactStorageBackend.FILE_STORAGE:
            if not self.file_id:
                raise ValueError("file_storage storage_ref 必须包含 file_id")
            if self.size_bytes is None or self.size_bytes <= 0:
                raise ValueError("file_storage storage_ref 必须包含正数 size_bytes")
            if not self.mime_type:
                raise ValueError("file_storage storage_ref 必须包含 mime_type")
            return self

        if self.storage_backend == ArtifactStorageBackend.SANDBOX:
            if not self.sandbox_path:
                raise ValueError("sandbox storage_ref 必须包含 sandbox_path")
            if self.object_key:
                raise ValueError("sandbox storage_ref 不得包含 object_key")
            if self.file_id:
                raise ValueError("sandbox storage_ref 不得包含 file_id")
            if self.size_bytes is None or self.size_bytes <= 0:
                raise ValueError("sandbox storage_ref 必须包含正数 size_bytes")
            if not self.mime_type:
                raise ValueError("sandbox storage_ref 必须包含 mime_type")
            return self

        if self.storage_backend == ArtifactStorageBackend.INLINE_SNAPSHOT:
            expected_missing = {"storage_hash", "size_bytes", "mime_type"}
            actual_missing = set(self.missing_fields)
            if self.object_key or self.file_id or self.sandbox_path:
                raise ValueError("inline_snapshot 不得包含 object_key/file_id/sandbox_path")
            if self.storage_hash is not None or self.size_bytes is not None or self.mime_type is not None:
                raise ValueError("inline_snapshot 不得保存 materialized storage 字段")
            if not expected_missing.issubset(actual_missing):
                raise ValueError("inline_snapshot 必须记录缺失的 storage_hash/size_bytes/mime_type")
            if self.reason_code != "inline_snapshot_no_materialized_storage":
                raise ValueError("inline_snapshot reason_code 固定为 inline_snapshot_no_materialized_storage")
            return self

        return self


class WorkspaceArtifactRevision(BaseModel):
    """Artifact revision 是产物版本历史和交付锁真相源。"""

    model_config = ConfigDict(extra="forbid")

    revision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_id: str
    revision_no: int
    user_id: str
    session_id: str
    workspace_id: str
    run_id: str | None = None
    step_id: str | None = None
    path: str
    storage_ref: ArtifactStorageRef
    content_hash: str
    storage_hash: str | None = None
    hash_algorithm: Literal["sha256"] = "sha256"
    size_bytes: int | None = None
    mime_type: str | None = None
    artifact_type: ArtifactType
    delivery_state: ArtifactDeliveryState = ArtifactDeliveryState.CANDIDATE
    source_kind: ArtifactRevisionSourceKind
    source_event_id: str | None = None
    source_run_id: str | None = None
    source_message_event_id: str | None = None
    source_revision_id: str | None = None
    source_fact_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    source_final_answer_hash: str | None = None
    derived_content_hash: str | None = None
    tool_call_id: str | None = None
    function_name: str | None = None
    profile_hash: str | None = None
    profile_status: str = "missing"
    origin: DataOrigin = DataOrigin.AGENT_GENERATED
    trust_level: DataTrustLevel = DataTrustLevel.AGENT_GENERATED
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    retention_policy: RetentionPolicyKind = RetentionPolicyKind.WORKSPACE_BOUND
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator(
        "revision_id",
        "artifact_id",
        "user_id",
        "session_id",
        "workspace_id",
        "path",
        "content_hash",
        "profile_status",
    )
    @classmethod
    def _required_text_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("artifact revision 必填文本字段不能为空")
        return normalized

    @field_validator(
        "run_id",
        "step_id",
        "storage_hash",
        "mime_type",
        "source_event_id",
        "source_run_id",
        "source_message_event_id",
        "source_revision_id",
        "source_final_answer_hash",
        "derived_content_hash",
        "tool_call_id",
        "function_name",
        "profile_hash",
    )
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("source_fact_ids", "source_evidence_ids")
    @classmethod
    def _normalize_id_list(cls, value: list[str]) -> list[str]:
        return [item for item in [str(item or "").strip() for item in value or []] if item]

    @model_validator(mode="after")
    def _validate_revision_contract(self) -> "WorkspaceArtifactRevision":
        if self.revision_no < 1:
            raise ValueError("revision_no 必须从 1 开始")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes 不能为负数")

        if self.source_kind in {
            ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
            ArtifactRevisionSourceKind.TOOL_REPLACE_FILE,
        }:
            expected_function_name = (
                "write_file"
                if self.source_kind == ArtifactRevisionSourceKind.TOOL_WRITE_FILE
                else "replace_in_file"
            )
            if not self.tool_call_id:
                raise ValueError("工具 artifact revision 必须包含 tool_call_id")
            if self.function_name != expected_function_name:
                raise ValueError("工具 artifact revision function_name 与 source_kind 不匹配")
            if not self.source_fact_ids:
                raise ValueError("工具 artifact revision 必须包含 source_fact_ids")
            if self.size_bytes is None or self.size_bytes <= 0 or not self.mime_type:
                raise ValueError("工具 artifact revision 必须包含正数 size_bytes 和 mime_type")
            if self.storage_ref.storage_backend == ArtifactStorageBackend.INLINE_SNAPSHOT:
                raise ValueError("工具 artifact revision 不得使用 inline_snapshot storage backend")

        if self.source_kind == ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT:
            if not self.source_final_answer_hash:
                raise ValueError("final_answer_snapshot 必须包含 source_final_answer_hash")
            if self.content_hash != self.source_final_answer_hash:
                raise ValueError("final_answer_snapshot content_hash 必须等于 source_final_answer_hash")
            if self.storage_ref.storage_backend != ArtifactStorageBackend.INLINE_SNAPSHOT:
                raise ValueError("final_answer_snapshot 必须使用 inline_snapshot storage backend")

        if self.source_kind == ArtifactRevisionSourceKind.DERIVED_EXPORT:
            if not self.source_revision_id or not self.source_final_answer_hash or not self.derived_content_hash:
                raise ValueError("derived_export 必须包含 source_revision_id/source_final_answer_hash/derived_content_hash")
            if self.derived_content_hash != self.content_hash:
                raise ValueError("derived_export derived_content_hash 必须等于 content_hash")
            if self.storage_ref.storage_backend != ArtifactStorageBackend.FILE_STORAGE:
                raise ValueError("derived_export 必须落地到 file_storage")

        if self.storage_ref.storage_backend == ArtifactStorageBackend.SANDBOX and self.delivery_state != ArtifactDeliveryState.CANDIDATE:
            raise ValueError("sandbox storage revision 只能保持 candidate")

        return self


class ArtifactRevisionRegistrationCommand(BaseModel):
    """PR2 受控 artifact revision 注册命令。"""

    model_config = ConfigDict(extra="forbid")

    scope: AccessScopeResult
    path: str
    storage_ref: ArtifactStorageRef
    content_hash: str
    storage_hash: str | None = None
    hash_algorithm: Literal["sha256"] = "sha256"
    size_bytes: int | None = None
    mime_type: str | None = None
    artifact_type: ArtifactType
    delivery_state: ArtifactDeliveryState = ArtifactDeliveryState.CANDIDATE
    source_kind: ArtifactRevisionSourceKind
    source_event_id: str
    source_run_id: str | None = None
    source_message_event_id: str | None = None
    source_revision_id: str | None = None
    source_fact_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    source_final_answer_hash: str | None = None
    derived_content_hash: str | None = None
    tool_call_id: str | None = None
    function_name: str | None = None
    profile_hash: str | None = None
    profile_status: str = "missing"
    origin: DataOrigin = DataOrigin.AGENT_GENERATED
    trust_level: DataTrustLevel = DataTrustLevel.AGENT_GENERATED
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    retention_policy: RetentionPolicyKind = RetentionPolicyKind.WORKSPACE_BOUND
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path", "content_hash", "source_event_id")
    @classmethod
    def _required_command_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("artifact revision registration command 必填文本字段不能为空")
        return normalized

    @field_validator(
        "storage_hash",
        "mime_type",
        "source_run_id",
        "source_message_event_id",
        "source_revision_id",
        "source_final_answer_hash",
        "derived_content_hash",
        "tool_call_id",
        "function_name",
        "profile_hash",
        "profile_status",
    )
    @classmethod
    def _normalize_optional_command_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("source_fact_ids", "source_evidence_ids")
    @classmethod
    def _normalize_command_id_list(cls, value: list[str]) -> list[str]:
        return [item for item in [str(item or "").strip() for item in value or []] if item]

    @model_validator(mode="after")
    def _validate_command_contract(self) -> "ArtifactRevisionRegistrationCommand":
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes 不能为负数")
        if self.source_kind in {
            ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
            ArtifactRevisionSourceKind.TOOL_REPLACE_FILE,
        }:
            expected_function_name = (
                "write_file"
                if self.source_kind == ArtifactRevisionSourceKind.TOOL_WRITE_FILE
                else "replace_in_file"
            )
            if not self.tool_call_id:
                raise ValueError("工具 artifact revision 注册命令必须包含 tool_call_id")
            if self.function_name != expected_function_name:
                raise ValueError("工具 artifact revision 注册命令 function_name 与 source_kind 不匹配")
            if not self.source_fact_ids:
                raise ValueError("工具 artifact revision 注册命令必须包含 source_fact_ids")
            if self.size_bytes is None or self.size_bytes <= 0 or not self.mime_type:
                raise ValueError("工具 artifact revision 注册命令必须包含正数 size_bytes 和 mime_type")
            if self.storage_ref.storage_backend == ArtifactStorageBackend.INLINE_SNAPSHOT:
                raise ValueError("工具 artifact revision 注册命令不得使用 inline_snapshot storage backend")
        if self.source_kind == ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT:
            if self.content_hash != self.source_final_answer_hash:
                raise ValueError("final_answer_snapshot content_hash 必须等于 source_final_answer_hash")
        if self.source_kind == ArtifactRevisionSourceKind.DERIVED_EXPORT:
            if not self.source_revision_id or not self.source_final_answer_hash or not self.derived_content_hash:
                raise ValueError("derived_export 必须包含 source_revision_id/source_final_answer_hash/derived_content_hash")
            if self.derived_content_hash != self.content_hash:
                raise ValueError("derived_export derived_content_hash 必须等于 content_hash")
        return self


class ArtifactRevisionIdentity(BaseModel):
    """revision-aware 状态迁移身份锁。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    revision_id: str
    content_hash: str


class ResolvedArtifactRevisionResult(BaseModel):
    """path current projection 解析出的 version-locked revision。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    revision_id: str
    content_hash: str
    path: str
    artifact_type: ArtifactType
    delivery_state: ArtifactDeliveryState
    session_id: str
    run_id: str | None = None
    source_run_id: str | None = None
    source_step_id: str | None = None
    source_event_id: str | None = None
    source_kind: ArtifactRevisionSourceKind


class SelectedArtifactRevisionResult(BaseModel):
    """Summary 选择的 revision-aware 最终附件候选。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    revision_id: str
    content_hash: str
    path: str
    artifact_type: ArtifactType
    delivery_state: ArtifactDeliveryState
    session_id: str
    run_id: str | None = None
    source_run_id: str | None = None
    source_step_id: str | None = None
    source_event_id: str | None = None
    source_kind: ArtifactRevisionSourceKind
    selected_reason: str
    selected_at: datetime


class ArtifactRevisionEventRef(BaseModel):
    """Artifact event 中最小 revision 引用。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    revision_id: str
    content_hash: str
    path: str
    artifact_type: ArtifactType
    delivery_state: ArtifactDeliveryState
    source_event_id: str | None = None


class ArtifactEventArtifactRef(BaseModel):
    """Artifact event 中最小 artifact 引用。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    path: str
    artifact_type: ArtifactType
    delivery_state: ArtifactDeliveryState
    current_revision_id: str | None = None
    latest_content_hash: str | None = None


class ArtifactEventPayload(BaseModel):
    """workflow_run_events.event_type='artifact' 的轻量 payload。"""

    model_config = ConfigDict(extra="forbid")

    artifact_refs: list[ArtifactEventArtifactRef] = Field(default_factory=list)
    revision_refs: list[ArtifactRevisionEventRef] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    summary: str = ""
    source_event_ids: list[str] = Field(default_factory=list)
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ArtifactDeliveryState",
    "ArtifactEventArtifactRef",
    "ArtifactEventPayload",
    "ArtifactRevisionIdentity",
    "ArtifactRevisionEventRef",
    "ArtifactRevisionRegistrationCommand",
    "ArtifactRevisionSourceKind",
    "ArtifactStatus",
    "ArtifactStorageBackend",
    "ArtifactStorageRef",
    "ArtifactType",
    "ResolvedArtifactRevisionResult",
    "SelectedArtifactRevisionResult",
    "WorkspaceArtifactRevision",
]
