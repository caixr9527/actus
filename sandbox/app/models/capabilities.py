#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox 能力探测模型。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SandboxCapabilityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str
    status: Literal["available", "unavailable", "degraded", "unknown"]
    version: str = ""
    path: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    reason_code: str = ""
    requires_confirmation: bool = False
    disabled: bool = False


class SandboxResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_file_read_bytes: int = 10000
    max_command_seconds: int = 600
    writable_dirs: list[str] = Field(default_factory=list)
    readable_dirs: list[str] = Field(default_factory=list)
    network_policy: Literal["proxy_configured", "restricted", "unknown"] = "unknown"
    proxy_host_categories: list[str] = Field(default_factory=list)


class SandboxCapabilityProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_profile: dict[str, Any] = Field(default_factory=dict)
    reason_code: str = ""
    probe_status: Literal["available", "unavailable", "degraded", "unknown"] = "unknown"
