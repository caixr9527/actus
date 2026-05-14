#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Artifact governance runtime 侧端口契约。"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from app.domain.models import ToolEvent
from app.domain.models.sandbox_fact import SandboxFactRecord
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionIdentity,
    ArtifactRevisionRegistrationCommand,
    ResolvedArtifactRevisionResult,
)


class ArtifactRevisionProjectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision_count: int = 0
    skipped_count: int = 0
    reason_codes: list[str] = []


class ArtifactLedgerPort(Protocol):
    """Runtime 依赖的 artifact ledger 端口，避免 domain 反向导入 application。"""

    async def register_revision(
            self,
            *,
            command: ArtifactRevisionRegistrationCommand,
    ) -> ResolvedArtifactRevisionResult:
        ...

    async def resolve_authoritative_artifact_revisions(
            self,
            *,
            scope: AccessScopeResult,
            paths: list[str],
    ) -> list[ResolvedArtifactRevisionResult]:
        ...

    async def mark_artifact_revisions_delivery_state(
            self,
            *,
            scope: AccessScopeResult,
            revisions: list[ArtifactRevisionIdentity],
            delivery_state: ArtifactDeliveryState,
    ) -> list[ResolvedArtifactRevisionResult]:
        ...


class ArtifactRevisionProjectorPort(Protocol):
    async def project_from_tool_event_facts(
            self,
            *,
            scope: AccessScopeResult,
            event: ToolEvent,
            facts: list[SandboxFactRecord],
    ) -> ArtifactRevisionProjectionResult:
        """在 source event 和 SandboxFact 均已持久化后登记 artifact revision。"""
        ...

    async def project_from_document_facts(
            self,
            *,
            scope: AccessScopeResult,
            facts: list[SandboxFactRecord],
    ) -> ArtifactRevisionProjectionResult:
        """在 document input source event 和 DOCUMENT_CONTEXT fact 均已持久化后登记 revision。"""
        ...
