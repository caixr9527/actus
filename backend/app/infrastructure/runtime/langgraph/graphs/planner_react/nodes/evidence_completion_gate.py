#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""execute step 完成前的 Evidence 主链路门禁。"""

from __future__ import annotations

import logging
from typing import Any

from app.domain.models import Step
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.evidence_runtime_ports import EvidenceStepReconcilerPort
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState

logger = logging.getLogger(__name__)


async def reconcile_step_evidence_before_state_return(
        *,
        state: PlannerReActLangGraphState,
        step: Step,
        reconciler: EvidenceStepReconcilerPort | None,
) -> list[object]:
    """在 graph 返回当前 step 完成状态前完成 evidence 对账。

    这里是 P1-3 跨 step reuse 的主写入路径。runner 事件旁路只能作为兼容防御，
    不能再承担“下一步 context 可见 evidence”的时序保证。
    """
    if reconciler is None:
        return []
    scope = _build_step_scope(state=state, step=step)
    if scope is None:
        log_runtime(
            logger,
            logging.ERROR,
            "evidence_reconcile_scope_missing",
            state=state,
            step_id=str(getattr(step, "id", "") or ""),
            reason_code="evidence_reconcile_scope_missing",
        )
        return []
    try:
        records = await reconciler.reconcile_step_evidence(scope=scope, step=step)
        if records is None:
            log_runtime(
                logger,
                logging.ERROR,
                "evidence_reconcile_contract_violation",
                state=state,
                step_id=str(getattr(step, "id", "") or ""),
                reason_code="evidence_reconcile_return_missing",
            )
            records = []
        await _project_step_evidence_event(
            state=state,
            reconciler=reconciler,
            scope=scope,
            step=step,
            records=list(records or []),
        )
        await _overwrite_step_outcome_with_evidence_projection(
            state=state,
            reconciler=reconciler,
            scope=scope,
            step=step,
        )
        _mark_step_graph_evidence_reconciled(step)
        log_runtime(
            logger,
            logging.INFO,
            "evidence_step_completion_gate_reconciled",
            state=state,
            step_id=str(getattr(step, "id", "") or ""),
            evidence_record_count=len(list(records or [])),
            reason_code="evidence_step_completion_gate_reconciled",
        )
        return list(records or [])
    except Exception as exc:
        log_runtime(
            logger,
            logging.ERROR,
            "evidence_reconcile_failed",
            state=state,
            step_id=str(getattr(step, "id", "") or ""),
            error_type=exc.__class__.__name__,
            reason_code="evidence_reconcile_failed",
        )
        try:
            await reconciler.record_reconcile_failed_gap(scope=scope, step=step)
        except Exception as gap_exc:
            log_runtime(
                logger,
                logging.ERROR,
                "evidence_reconcile_gap_write_failed",
                state=state,
                step_id=str(getattr(step, "id", "") or ""),
                error_type=gap_exc.__class__.__name__,
                reason_code="evidence_reconcile_gap_write_failed",
            )
        return []


def _build_step_scope(
        *,
        state: PlannerReActLangGraphState,
        step: Step,
) -> AccessScopeResult | None:
    user_id = str(state.get("user_id") or "").strip()
    session_id = str(state.get("session_id") or "").strip()
    workspace_id = str(state.get("workspace_id") or "").strip()
    run_id = str(state.get("run_id") or "").strip()
    step_id = str(getattr(step, "id", "") or "").strip()
    if not user_id or not session_id or not workspace_id or not run_id or not step_id:
        return None
    return AccessScopeResult(
        tenant_id=user_id,
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        run_id=run_id,
        current_step_id=step_id,
    )


async def _project_step_evidence_event(
        *,
        state: PlannerReActLangGraphState,
        reconciler: EvidenceStepReconcilerPort,
        scope: AccessScopeResult,
        step: Step,
        records: list[object],
) -> None:
    build_event = getattr(reconciler, "build_step_evidence_event", None)
    if not callable(build_event):
        return
    try:
        event = await build_event(scope=scope, step=step, records=records)
        if event is None:
            return
        persist_event = getattr(reconciler, "persist_step_evidence_event", None)
        if callable(persist_event):
            await persist_event(scope=scope, event=event)
    except Exception as exc:
        log_runtime(
            logger,
            logging.ERROR,
            "evidence_event_projection_failed",
            state=state,
            step_id=str(getattr(step, "id", "") or ""),
            error_type=exc.__class__.__name__,
            reason_code="evidence_event_projection_failed",
        )


async def _overwrite_step_outcome_with_evidence_projection(
        *,
        state: PlannerReActLangGraphState,
        reconciler: EvidenceStepReconcilerPort,
        scope: AccessScopeResult,
        step: Step,
) -> None:
    try:
        projections = await reconciler.build_step_evidence_backed_facts(scope=scope, step=step)
    except Exception as exc:
        log_runtime(
            logger,
            logging.ERROR,
            "evidence_backed_projection_failed",
            state=state,
            step_id=str(getattr(step, "id", "") or ""),
            error_type=exc.__class__.__name__,
            reason_code="evidence_backed_projection_failed",
        )
        return
    if getattr(step, "outcome", None) is None:
        return
    step.outcome.evidence_backed_facts = list(projections or [])
    step.outcome.facts_learned = [
        str(item.text or "").strip()
        for item in list(projections or [])
        if str(item.text or "").strip()
    ]


def _mark_step_graph_evidence_reconciled(step: Step) -> None:
    outcome = getattr(step, "outcome", None)
    if outcome is None:
        return
    existing = getattr(outcome, "evidence_reconcile_metadata", None)
    metadata: dict[str, Any] = dict(existing or {}) if isinstance(existing, dict) else {}
    metadata["graph_completion_gate"] = True
    outcome.evidence_reconcile_metadata = metadata
