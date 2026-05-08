#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence runtime 接入端口。

这些端口只描述 domain runtime 侧需要消费的能力，具体读取 DB、构建
digest 的实现留在 application/infrastructure 层，避免 domain 反向依赖。
"""

from __future__ import annotations

from typing import Protocol

from app.domain.models import Step
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.evidence_ledger_contract import (
    EvidenceBackedFactProjection,
    EvidenceResolvedResult,
    EvidenceResultHandle,
    RuntimeEvidenceContextResult,
)


class EvidenceRuntimeContextProviderPort(Protocol):
    """为 runtime context 提供 PR1 strict evidence context。"""

    async def build_context(
            self,
            *,
            stage: str,
            scope: AccessScopeResult,
            completed_step_ids: list[str],
            step: Step | None = None,
            task_mode: str = "",
    ) -> RuntimeEvidenceContextResult | None:
        """按 runtime stage 构造 evidence context；不返回普通 dict。"""
        ...


class EvidenceStepReconcilerPort(Protocol):
    """Step completed 前的 evidence 对账端口。"""

    async def reconcile_step_evidence(
            self,
            *,
            scope: AccessScopeResult,
            step: Step,
    ) -> list[object]:
        """在 StepEvent(COMPLETED) 持久化前完成当前 step evidence/gap 落库。"""
        ...

    async def record_reconcile_failed_gap(
            self,
            *,
            scope: AccessScopeResult,
            step: Step,
    ) -> object:
        """runner fail-safe 兜底写入 evidence_reconcile_failed gap。"""
        ...

    async def build_step_evidence_backed_facts(
            self,
            *,
            scope: AccessScopeResult,
            step: Step,
    ) -> list[EvidenceBackedFactProjection]:
        """基于已落库 evidence 为 StepOutcome 生成可读事实投影。"""
        ...


class EvidenceResultHandleResolverPort(Protocol):
    """解析 PR1 result handle 的端口。"""

    async def resolve(
            self,
            *,
            scope: AccessScopeResult,
            handle: EvidenceResultHandle,
    ) -> EvidenceResolvedResult:
        """按 handle 稳定读取前序结果；失败时返回 strict resolved result。"""
        ...
