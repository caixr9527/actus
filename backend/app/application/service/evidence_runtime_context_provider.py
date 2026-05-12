#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence runtime context provider。

负责 PR3 运行期编排：先补齐前序 step evidence gap，再委托 projector 构造
strict runtime evidence context。Projector 保持只读投影，不承担写库职责。
"""

from __future__ import annotations

from app.application.service.evidence_digest_projector import EvidenceDigestProjector
from app.application.service.evidence_ledger_service import EvidenceLedgerService
from app.domain.models import Step
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.evidence_ledger_contract import RuntimeEvidenceContextResult
from app.domain.services.runtime.contracts.evidence_runtime_ports import EvidenceRuntimeContextProviderPort


class EvidenceRuntimeContextProvider(EvidenceRuntimeContextProviderPort):
    """运行期 evidence context 编排器。"""

    def __init__(
            self,
            *,
            ledger_service: EvidenceLedgerService,
            projector: EvidenceDigestProjector,
    ) -> None:
        self._ledger_service = ledger_service
        self._projector = projector

    async def build_context(
            self,
            *,
            stage: str,
            scope: AccessScopeResult,
            completed_step_ids: list[str],
            step: Step | None = None,
            task_mode: str = "",
    ) -> RuntimeEvidenceContextResult | None:
        if stage in {"execute", "replan", "summary"} and completed_step_ids:
            await self._ledger_service.reconcile_previous_steps_evidence(
                scope=scope,
                completed_step_ids=completed_step_ids,
            )
        return await self._projector.build_context(
            stage=stage,
            scope=scope,
            completed_step_ids=completed_step_ids,
            step=step,
            task_mode=task_mode,
        )
