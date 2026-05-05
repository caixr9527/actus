#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox raw capability payload 严格归一。"""

from __future__ import annotations

from typing import Any, Mapping

from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    SandboxCapabilityItem,
    SandboxCapabilityStatus,
    SandboxResourceLimits,
)


ALLOWED_RAW_PROFILE_KEYS = {
    "health_status",
    "cwd",
    "capabilities",
    "resource_limits",
    "disabled_capabilities",
    "confirmation_required_capabilities",
}
FORBIDDEN_RAW_PROFILE_KEYS = {
    "schema_version",
    "user_id",
    "session_id",
    "workspace_id",
    "run_id",
    "sandbox_id",
    "profile_id",
    "profile_hash",
    "prompt_summary",
    "generated_at",
    "expires_at",
    "refresh_reason",
    "runtime_tool_capabilities",
    "last_refresh_error",
}


class NormalizedSandboxRawProfile:
    def __init__(
            self,
            *,
            health_status: SandboxCapabilityStatus,
            cwd: str,
            capabilities: list[SandboxCapabilityItem],
            resource_limits: SandboxResourceLimits,
            disabled_capabilities: list[str],
            confirmation_required_capabilities: list[str],
    ) -> None:
        self.health_status = health_status
        self.cwd = cwd
        self.capabilities = capabilities
        self.resource_limits = resource_limits
        self.disabled_capabilities = disabled_capabilities
        self.confirmation_required_capabilities = confirmation_required_capabilities


def normalize_sandbox_raw_profile(raw_profile: Mapping[str, Any]) -> NormalizedSandboxRawProfile:
    raw_payload = dict(raw_profile or {})
    raw_keys = set(raw_payload.keys())
    forbidden_keys = raw_keys & FORBIDDEN_RAW_PROFILE_KEYS
    if forbidden_keys:
        raise ValueError(f"sandbox raw profile 包含后端归一字段: {sorted(forbidden_keys)[0]}")
    unknown_keys = raw_keys - ALLOWED_RAW_PROFILE_KEYS
    if unknown_keys:
        raise ValueError(f"sandbox raw profile 包含未知字段: {sorted(unknown_keys)[0]}")
    raw_resource_limits = raw_payload.get("resource_limits") or {}
    if not isinstance(raw_resource_limits, Mapping):
        raise ValueError("sandbox raw resource_limits 必须是对象")
    if "max_tool_iterations" in raw_resource_limits:
        raise ValueError("sandbox raw resource_limits 禁止包含 max_tool_iterations")
    return NormalizedSandboxRawProfile(
        health_status=_parse_required_status(raw_payload.get("health_status")),
        cwd=str(raw_payload.get("cwd") or "").strip(),
        capabilities=_parse_capabilities(raw_payload.get("capabilities")),
        resource_limits=SandboxResourceLimits.model_validate(dict(raw_resource_limits)),
        disabled_capabilities=_parse_string_list(raw_payload.get("disabled_capabilities")),
        confirmation_required_capabilities=_parse_string_list(
            raw_payload.get("confirmation_required_capabilities")
        ),
    )


def _parse_required_status(raw_status: Any) -> SandboxCapabilityStatus:
    try:
        return SandboxCapabilityStatus(raw_status)
    except ValueError as exc:
        raise ValueError("sandbox raw health_status 非法") from exc


def _parse_capabilities(raw_capabilities: Any) -> list[SandboxCapabilityItem]:
    if not isinstance(raw_capabilities, list):
        raise ValueError("sandbox raw capabilities 必须是数组")
    return [
        SandboxCapabilityItem.model_validate(raw_item)
        for raw_item in raw_capabilities
    ]


def _parse_string_list(raw_values: Any) -> list[str]:
    if raw_values is None:
        return []
    if not isinstance(raw_values, list):
        raise ValueError("sandbox raw capability 列表字段必须是数组")
    return sorted({
        str(value).strip()
        for value in raw_values
        if str(value).strip()
    })
