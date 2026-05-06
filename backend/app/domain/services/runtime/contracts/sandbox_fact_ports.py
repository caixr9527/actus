#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Fact Ledger 应用端口契约。"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from app.application.service.runtime_access_control_service import AccessScopeResult
from app.domain.models import ToolEvent
from app.domain.models.sandbox_fact import SandboxFactRecord
from app.domain.services.runtime.contracts.sandbox_fact_contract import (
    SandboxFactProfileRef,
)


class SandboxFactProjectionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: AccessScopeResult
    profile_ref: SandboxFactProfileRef
    sandbox_id: str | None = None
    source_event_id: str | None = None
    current_step_id: str | None = None


class SandboxFactRecorderPort(Protocol):
    async def record_from_tool_event(
            self,
            *,
            context: SandboxFactProjectionContext,
            event: ToolEvent,
    ) -> list[SandboxFactRecord]:
        """PR3 接入 ToolEvent 投影；PR2 不实现工具事件分发。"""
        ...
