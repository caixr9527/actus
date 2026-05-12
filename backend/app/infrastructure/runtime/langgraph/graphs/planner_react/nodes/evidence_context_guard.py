#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""replan/summary 节点共享的 Evidence context fail-closed guard。"""

from dataclasses import dataclass
from typing import Any, Dict, Literal

from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import (
    REASON_EVIDENCE_CONTEXT_CURSOR_MISMATCH,
    REASON_EVIDENCE_CONTEXT_INVALID_SCHEMA,
    REASON_EVIDENCE_CONTEXT_MISSING,
)

EvidenceContextGuardStage = Literal["replan", "summary"]


@dataclass(frozen=True)
class EvidenceContextGuardResult:
    """Evidence context 是否允许当前节点继续调用 LLM。"""

    blocked: bool
    reason_code: str = ""


def validate_stage_evidence_context_packet(
        *,
        stage: EvidenceContextGuardStage,
        context_packet: Dict[str, Any],
) -> EvidenceContextGuardResult:
    """统一校验 replan/summary 的 evidence context 双通道契约。

    这里不做业务决策，只判断节点是否必须 fail closed：
    - missing: RuntimeContextService 已明确标记 context 缺失；
    - invalid_schema: 当前 stage 需要的 prompt-safe evidence view 缺失或类型错误；
    - cursor_mismatch: guard-only cursor 与 prompt-safe view cursor 不一致。
    """
    error = context_packet.get("evidence_context_error")
    if isinstance(error, dict):
        reason_code = str(error.get("reason_code") or "").strip()
        return EvidenceContextGuardResult(
            blocked=True,
            reason_code=reason_code or REASON_EVIDENCE_CONTEXT_MISSING,
        )
    if error is not None:
        return EvidenceContextGuardResult(
            blocked=True,
            reason_code=REASON_EVIDENCE_CONTEXT_INVALID_SCHEMA,
        )

    stage_context_key = "evidence_replan_context" if stage == "replan" else "summary_evidence_context"
    stage_context = context_packet.get(stage_context_key)
    if not isinstance(stage_context, dict):
        return EvidenceContextGuardResult(
            blocked=True,
            reason_code=REASON_EVIDENCE_CONTEXT_INVALID_SCHEMA,
        )

    if stage == "replan":
        evidence_context = context_packet.get("evidence_context")
        if not isinstance(evidence_context, dict):
            return EvidenceContextGuardResult(
                blocked=True,
                reason_code=REASON_EVIDENCE_CONTEXT_INVALID_SCHEMA,
            )
        evidence_cursor = str(evidence_context.get("cursor") or "").strip()
    else:
        evidence_cursor = str(context_packet.get("evidence_context_cursor") or "").strip()

    stage_cursor = str(stage_context.get("cursor") or "").strip()
    if not evidence_cursor or not stage_cursor:
        return EvidenceContextGuardResult(
            blocked=True,
            reason_code=REASON_EVIDENCE_CONTEXT_INVALID_SCHEMA,
        )
    if evidence_cursor != stage_cursor:
        return EvidenceContextGuardResult(
            blocked=True,
            reason_code=REASON_EVIDENCE_CONTEXT_CURSOR_MISMATCH,
        )
    return EvidenceContextGuardResult(blocked=False)
