#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence Ledger 应用层输入契约。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.evidence import (
    EvidenceKind,
    EvidenceQualityStatus,
    EvidenceResultRef,
    EvidenceReusePolicy,
    EvidenceScope,
    EvidenceSourceRef,
    EvidenceStalenessPolicy,
    EvidenceSubjectRef,
    EvidenceSupportLevel,
    EvidenceVisibility,
)


class EvidenceRecordInput(BaseModel):
    """Evidence 显式写入命令。

    调用方只能提供业务字段；hash 和 idempotency key 必须由 service 生成。
    """

    model_config = ConfigDict(extra="forbid")

    evidence_scope: EvidenceScope = EvidenceScope.STEP
    evidence_kind: EvidenceKind
    source_ref: EvidenceSourceRef
    subject_ref: EvidenceSubjectRef
    support_level: EvidenceSupportLevel
    quality_status: EvidenceQualityStatus
    payload: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    step_id: str | None = None
    action_key: str | None = None
    claim_key: str | None = None
    claim_text: str | None = None
    source_step_id: str | None = None
    summary: str = ""
    confidence: float = 0.0
    reusable: bool = False
    reuse_policy: EvidenceReusePolicy = EvidenceReusePolicy.DO_NOT_REUSE
    staleness_policy: EvidenceStalenessPolicy = EvidenceStalenessPolicy.STEP_SCOPED
    visibility: EvidenceVisibility = EvidenceVisibility.INTERNAL
    result_refs: list[EvidenceResultRef] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    supersedes_evidence_id: str | None = None


class EvidenceQueryInput(BaseModel):
    """PR2 强过滤查询输入。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    action_key: str | None = None
    subject_key: str | None = None
    limit: int = 100
