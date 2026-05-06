#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Fact Ledger 应用端口契约。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from app.domain.models import SandboxFactEvent, ToolEvent
from app.domain.models.sandbox_fact import SandboxFactRecord
from app.domain.services.runtime.contracts.sandbox_fact_contract import (
    SandboxFactProfileRef,
)

if TYPE_CHECKING:
    from app.application.service.runtime_access_control_service import AccessScopeResult
else:
    AccessScopeResult = Any


class SandboxFactProjectionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: AccessScopeResult
    profile_ref: SandboxFactProfileRef
    sandbox_id: str | None = None
    source_event_id: str | None = None
    current_step_id: str | None = None


class SandboxFactProjectionContextBuilderPort(Protocol):
    async def build_for_tool_event(
            self,
            *,
            source_event_id: str,
    ) -> SandboxFactProjectionContext:
        """由 runner/runtime 上游集中构造 ToolEvent fact 投影上下文。"""
        ...

    async def build_for_document_input(
            self,
            *,
            source_event_id: str,
            scope: AccessScopeResult,
    ) -> SandboxFactProjectionContext:
        """由 runtime 文档输入链路集中构造 DOCUMENT_CONTEXT fact 投影上下文。"""
        ...


@runtime_checkable
class SandboxFactRecorderPort(Protocol):
    async def record_from_tool_event(
            self,
            *,
            context: SandboxFactProjectionContext,
            event: ToolEvent,
    ) -> list[SandboxFactRecord]:
        """PR3 接入 ToolEvent 投影；PR2 不实现工具事件分发。"""
        ...


class SandboxFactEventProjectorPort(Protocol):
    async def project_tool_event_facts(
            self,
            *,
            context: SandboxFactProjectionContext,
            facts: list[SandboxFactRecord],
    ) -> SandboxFactEvent | None:
        """PR4 将已入库 fact 转换为轻量 runtime event。"""
        ...
