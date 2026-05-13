#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Artifact governance runtime 侧端口契约。"""

from __future__ import annotations

from typing import Protocol

from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionIdentity,
    ArtifactRevisionRegistrationCommand,
    ResolvedArtifactRevisionResult,
)


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
