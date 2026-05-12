#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Capability Profile 领域契约。

该模块只描述 sandbox 环境事实的纯领域形态，不依赖数据库、FastAPI、
Docker、LangGraph 或应用层服务。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SANDBOX_CAPABILITY_PROFILE_SCHEMA_VERSION = "sandbox_capability_profile.v1"
SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY = "sandbox_capability_profile"


class SandboxCapabilityStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class SandboxCapabilityKind(str, Enum):
    OS = "os"
    FILESYSTEM = "filesystem"
    SHELL = "shell"
    PYTHON = "python"
    NODE = "node"
    BROWSER = "browser"
    VNC = "vnc"
    SEARCH = "search"
    NETWORK = "network"
    PROXY = "proxy"
    MCP = "mcp"
    A2A = "a2a"
    SKILL = "skill"
    RESOURCE_LIMIT = "resource_limit"


class SandboxProfileRefreshReason(str, Enum):
    SANDBOX_CREATED = "sandbox_created"
    SANDBOX_RESUMED = "sandbox_resumed"
    PERIODIC = "periodic"
    TOOL_ENV_ERROR = "tool_env_error"
    SKILL_CHANGED = "skill_changed"
    REMOTE_TOOL_CHANGED = "remote_tool_changed"
    USER_REQUESTED = "user_requested"


class SandboxCapabilityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: SandboxCapabilityKind
    name: str
    status: SandboxCapabilityStatus
    version: str = ""
    path: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    reason_code: str = ""
    requires_confirmation: bool = False
    disabled: bool = False

    @field_validator("name")
    @classmethod
    def _name_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("capability name 不能为空")
        return normalized


class SandboxResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_file_read_bytes: int | None = None
    max_command_seconds: int | None = None
    max_tool_iterations: int | None = None
    writable_dirs: list[str] = Field(default_factory=list)
    readable_dirs: list[str] = Field(default_factory=list)
    network_policy: str = "unknown"


class SandboxCapabilityProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_profile: dict[str, Any] = Field(default_factory=dict)
    reason_code: str = ""
    probe_status: SandboxCapabilityStatus = SandboxCapabilityStatus.UNKNOWN


class RuntimeToolCapabilitySnapshotItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str
    tool_family: str
    enabled: bool = True
    source: Literal["local", "mcp", "a2a", "custom"] = "local"

    @field_validator("capability_id", "tool_family")
    @classmethod
    def _required_text_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("runtime tool capability 字段不能为空")
        return normalized


class RuntimeToolCapabilitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RuntimeToolCapabilitySnapshotItem] = Field(default_factory=list)


class SandboxCapabilityRefreshError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str
    stage: Literal["probe", "normalize", "write", "refresh"] = "refresh"
    message: str = ""
    occurred_at: datetime
    retryable: bool = True
    probe_status: SandboxCapabilityStatus = SandboxCapabilityStatus.UNKNOWN

    @field_validator("reason_code")
    @classmethod
    def _reason_code_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("reason_code 不能为空")
        return normalized


class SandboxCapabilityPromptSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sandbox_capability_profile.v1"] = SANDBOX_CAPABILITY_PROFILE_SCHEMA_VERSION
    health_status: SandboxCapabilityStatus
    cwd: str = ""
    available_runtime: dict[str, str] = Field(default_factory=dict)
    available_tools: list[str] = Field(default_factory=list)
    unavailable_capabilities: list[str] = Field(default_factory=list)
    requires_confirmation: list[str] = Field(default_factory=list)
    resource_limits: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime
    sandbox_profile_stale: bool = False


class SandboxCapabilityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sandbox_capability_profile.v1"] = SANDBOX_CAPABILITY_PROFILE_SCHEMA_VERSION
    profile_id: str
    user_id: str
    session_id: str
    workspace_id: str
    run_id: str | None = None
    sandbox_id: str
    generated_at: datetime
    expires_at: datetime | None = None
    refresh_reason: SandboxProfileRefreshReason
    profile_hash: str
    health_status: SandboxCapabilityStatus
    cwd: str = ""
    capabilities: list[SandboxCapabilityItem] = Field(default_factory=list)
    resource_limits: SandboxResourceLimits = Field(default_factory=SandboxResourceLimits)
    disabled_capabilities: list[str] = Field(default_factory=list)
    confirmation_required_capabilities: list[str] = Field(default_factory=list)
    runtime_tool_capabilities: RuntimeToolCapabilitySnapshot = Field(default_factory=RuntimeToolCapabilitySnapshot)
    prompt_summary: SandboxCapabilityPromptSummary
    last_refresh_error: SandboxCapabilityRefreshError | None = None

    @field_validator("profile_id", "user_id", "session_id", "workspace_id", "sandbox_id", "profile_hash")
    @classmethod
    def _scope_text_must_not_be_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("sandbox profile scope 字段不能为空")
        return normalized

    @field_validator("profile_hash")
    @classmethod
    def _profile_hash_must_use_sha256_prefix(cls, value: str) -> str:
        normalized = str(value or "").strip()
        digest = normalized.removeprefix("sha256:")
        if (
                not normalized.startswith("sha256:")
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("profile_hash 必须使用 sha256:<hex> 格式")
        return normalized

    @model_validator(mode="after")
    def _profile_hash_must_match_stable_payload(self) -> "SandboxCapabilityProfile":
        expected_hash = build_profile_hash_from_payload(self.model_dump(mode="json"))
        if self.profile_hash != expected_hash:
            raise ValueError("profile_hash 与 sandbox capability profile 稳定字段不一致")
        return self


def build_profile_hash_from_payload(payload: Mapping[str, Any]) -> str:
    """按稳定环境事实计算 profile hash。

    hash 只消费 schema、sandbox_id、health、cwd、capabilities、resource_limits、
    disabled/confirmation 列表和 runtime_tool_capabilities；必须排除 scope、
    时间、refresh reason、last_refresh_error 和 prompt_summary。
    """

    normalized_payload = _normalize_hash_payload(payload)
    serialized_payload = json.dumps(
        normalized_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_sandbox_capability_profile_from_draft(payload: Mapping[str, Any]) -> SandboxCapabilityProfile:
    """从不含 profile_hash 的 draft 构造最终 profile，固定 hash 写入顺序。"""

    draft_payload = dict(payload or {})
    if "profile_hash" in draft_payload:
        raise ValueError("sandbox capability profile draft 禁止包含 profile_hash")
    profile_hash = build_profile_hash_from_payload(draft_payload)
    return SandboxCapabilityProfile.model_validate({
        **draft_payload,
        "profile_hash": profile_hash,
    })


def validate_sandbox_capability_profile_payload(payload: Mapping[str, Any]) -> SandboxCapabilityProfile:
    """校验 workspace 已存 profile payload，未知字段和旧格式由调用方按非法处理。"""

    return SandboxCapabilityProfile.model_validate(payload)


def _normalize_hash_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_payload = dict(payload or {})
    runtime_snapshot = _coerce_model_dump(raw_payload.get("runtime_tool_capabilities"))
    resource_limits = _coerce_model_dump(raw_payload.get("resource_limits"))
    return {
        "schema_version": str(
            raw_payload.get("schema_version") or SANDBOX_CAPABILITY_PROFILE_SCHEMA_VERSION
        ),
        "sandbox_id": str(raw_payload.get("sandbox_id") or ""),
        "health_status": _enum_or_value(raw_payload.get("health_status")),
        "cwd": str(raw_payload.get("cwd") or ""),
        "capabilities": _normalize_capabilities(raw_payload.get("capabilities")),
        "resource_limits": _normalize_resource_limits(resource_limits),
        "disabled_capabilities": _dedupe_sorted_strings(raw_payload.get("disabled_capabilities")),
        "confirmation_required_capabilities": _dedupe_sorted_strings(
            raw_payload.get("confirmation_required_capabilities")
        ),
        "runtime_tool_capabilities": {
            "items": _normalize_runtime_tool_items(runtime_snapshot.get("items") if runtime_snapshot else [])
        },
    }


def _normalize_capabilities(raw_capabilities: Any) -> list[dict[str, Any]]:
    normalized_capabilities: list[dict[str, Any]] = []
    if not isinstance(raw_capabilities, list):
        return normalized_capabilities
    for item in raw_capabilities:
        raw_item = _coerce_model_dump(item)
        normalized_item = {
            "kind": _enum_or_value(raw_item.get("kind")),
            "name": str(raw_item.get("name") or ""),
            "status": _enum_or_value(raw_item.get("status")),
            "version": str(raw_item.get("version") or ""),
            "path": str(raw_item.get("path") or ""),
            "details": _normalize_json_value(raw_item.get("details") if raw_item.get("details") is not None else {}),
            "reason_code": str(raw_item.get("reason_code") or ""),
            "requires_confirmation": bool(raw_item.get("requires_confirmation", False)),
            "disabled": bool(raw_item.get("disabled", False)),
        }
        normalized_capabilities.append(normalized_item)
    return sorted(
        normalized_capabilities,
        key=lambda item: (
            str(item.get("kind") or ""),
            str(item.get("name") or ""),
            str(item.get("path") or ""),
            str(item.get("version") or ""),
            _stable_json_dumps(item),
        ),
    )


def _normalize_resource_limits(raw_limits: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "max_file_read_bytes": raw_limits.get("max_file_read_bytes"),
        "max_command_seconds": raw_limits.get("max_command_seconds"),
        "max_tool_iterations": raw_limits.get("max_tool_iterations"),
        "writable_dirs": list(raw_limits.get("writable_dirs") or []),
        "readable_dirs": list(raw_limits.get("readable_dirs") or []),
        "network_policy": str(raw_limits.get("network_policy") or "unknown"),
    }


def _normalize_runtime_tool_items(raw_items: Any) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    if not isinstance(raw_items, list):
        return normalized_items
    for item in raw_items:
        raw_item = _coerce_model_dump(item)
        normalized_items.append(
            {
                "capability_id": str(raw_item.get("capability_id") or ""),
                "tool_family": str(raw_item.get("tool_family") or ""),
                "enabled": bool(raw_item.get("enabled", True)),
                "source": str(raw_item.get("source") or "local"),
            }
        )
    return sorted(
        normalized_items,
        key=lambda item: (
            str(item.get("source") or ""),
            str(item.get("capability_id") or ""),
            str(item.get("tool_family") or ""),
            _stable_json_dumps(item),
        ),
    )


def _dedupe_sorted_strings(raw_values: Any) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    return sorted({
        str(value).strip()
        for value in raw_values
        if str(value).strip()
    })


def _coerce_model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _enum_or_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value or "")


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    return value


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(
        _normalize_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
