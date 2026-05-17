#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safety Audit runtime event 投影器。"""

from __future__ import annotations

import hashlib
import logging
from typing import Callable

from app.domain.models import RuntimeEventProjection, SafetyAuditEvent
from app.domain.models.safety_audit import (
    SafetyAuditDecision,
    SafetyAuditDecisionCounts,
    SafetyAuditEventPayload,
    SafetyAuditEventProjectResult,
    SafetyAuditEventProjectorPort,
    SafetyAuditEventRef,
    SafetyAuditEventRuntimeMetadata,
    SafetyAuditRecorderPort,
    SafetyAuditRecord,
    SafetyAuditRiskCounts,
    SafetyAuditRiskLevel,
    stable_json_dumps,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult

logger = logging.getLogger(__name__)


class SafetyAuditEventProjectionError(RuntimeError):
    """Safety Audit runtime event 投影失败。"""


class SafetyAuditEventProjector(SafetyAuditEventProjectorPort):
    """把已落库 SafetyAuditRecord 投影成隐藏 runtime event。"""

    def __init__(
            self,
            *,
            uow_factory: Callable[[], IUnitOfWork],
            runtime_state_coordinator,
            recorder: SafetyAuditRecorderPort,
    ) -> None:
        self._uow_factory = uow_factory
        self._runtime_state_coordinator = runtime_state_coordinator
        self._recorder = recorder

    async def project_tool_event_source(
            self,
            *,
            scope: AccessScopeResult,
            tool_event_source_event_id: str,
    ) -> SafetyAuditEventProjectResult:
        self._validate_scope(scope)
        source_event_id = self._require_text(tool_event_source_event_id, "tool_event_source_event_id")
        async with self._uow_factory() as uow:
            records = await uow.safety_audit.list_by_tool_event_source(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                tool_event_source_event_id=source_event_id,
            )
        eligible = self._eligible_records(records=records, scope=scope)
        if not eligible:
            return SafetyAuditEventProjectResult(reason_code="no_unprojected_audit_records")
        return await self._project_records(
            scope=scope,
            records=eligible,
            source_event_ids=[source_event_id],
        )

    async def project_single_audit(
            self,
            *,
            scope: AccessScopeResult,
            audit_id: str,
    ) -> SafetyAuditEventProjectResult:
        self._validate_scope(scope)
        normalized_audit_id = self._require_text(audit_id, "audit_id")
        async with self._uow_factory() as uow:
            record = await uow.safety_audit.get_by_scope(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                audit_id=normalized_audit_id,
            )
        eligible = self._eligible_records(records=[record] if record is not None else [], scope=scope)
        if not eligible:
            return SafetyAuditEventProjectResult(reason_code="no_unprojected_audit_records")
        return await self._project_records(scope=scope, records=eligible, source_event_ids=[])

    async def _project_records(
            self,
            *,
            scope: AccessScopeResult,
            records: list[SafetyAuditRecord],
            source_event_ids: list[str],
    ) -> SafetyAuditEventProjectResult:
        ordered_records = sorted(records, key=lambda item: item.id)
        audit_ids = [record.id for record in ordered_records]
        projection_key = _build_projection_key(
            user_id=scope.user_id,
            session_id=str(scope.session_id),
            run_id=str(scope.run_id),
            audit_ids=audit_ids,
        )
        event = SafetyAuditEvent(
            id=projection_key,
            payload=_build_payload(
                records=ordered_records,
                source_event_ids=source_event_ids,
                projection_key=projection_key,
            ),
        )
        try:
            persist_result = await self._runtime_state_coordinator.persist_runtime_event(
                session_id=str(scope.session_id),
                event=event,
                projection=RuntimeEventProjection(),
                allow_status_transition=False,
            )
        except Exception as exc:
            logger.exception(
                "safety_audit_event_projection_failed",
                extra={
                    "user_id": scope.user_id,
                    "session_id": scope.session_id,
                    "run_id": scope.run_id,
                    "audit_ids": audit_ids,
                    "projection_key": projection_key,
                    "error_type": exc.__class__.__name__,
                    "reason_code": "safety_audit_event_projection_failed",
                },
            )
            raise SafetyAuditEventProjectionError(str(exc) or "safety_audit_event_projection_failed") from exc

        for record in ordered_records:
            try:
                await self._recorder.attach_decision_event(
                    record.id,
                    projection_key,
                    scope=scope,
                )
                logger.info(
                    "safety_audit_decision_event_attached",
                    extra={
                        "user_id": scope.user_id,
                        "session_id": scope.session_id,
                        "run_id": scope.run_id,
                        "audit_id": record.id,
                        "decision_event_id": projection_key,
                    },
                )
            except Exception as exc:
                logger.exception(
                    "safety_audit_decision_event_attach_failed",
                    extra={
                        "user_id": scope.user_id,
                        "session_id": scope.session_id,
                        "run_id": scope.run_id,
                        "audit_id": record.id,
                        "decision_event_id": projection_key,
                        "error_type": exc.__class__.__name__,
                        "reason_code": "safety_audit_decision_event_attach_failed",
                    },
                )
        logger.info(
            "safety_audit_event_projected",
            extra={
                "user_id": scope.user_id,
                "session_id": scope.session_id,
                "run_id": scope.run_id,
                "audit_ids": audit_ids,
                "event_id": projection_key,
                "event_inserted": getattr(persist_result, "event_inserted", False),
            },
        )
        return SafetyAuditEventProjectResult(
            event_id=projection_key,
            projected=True,
            audit_ids=audit_ids,
            reason_code="safety_audit_event_projected",
        )

    @staticmethod
    def _eligible_records(
            *,
            records: list[SafetyAuditRecord],
            scope: AccessScopeResult,
    ) -> list[SafetyAuditRecord]:
        run_id = str(scope.run_id or "").strip()
        return [
            record for record in records
            if record.user_id == scope.user_id
            and record.session_id == str(scope.session_id)
            and record.run_id == run_id
            and record.decision_event_id is None
        ]

    @staticmethod
    def _validate_scope(scope: AccessScopeResult) -> None:
        if not scope or not scope.user_id or not scope.session_id or not scope.run_id:
            raise SafetyAuditEventProjectionError("Safety Audit event 投影缺少 user/session/run scope")

    @staticmethod
    def _require_text(value: str | None, field_name: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise SafetyAuditEventProjectionError(f"{field_name} 不能为空")
        return normalized


def _build_projection_key(
        *,
        user_id: str,
        session_id: str,
        run_id: str,
        audit_ids: list[str],
) -> str:
    digest = hashlib.sha256(stable_json_dumps(sorted(audit_ids)).encode("utf-8")).hexdigest()
    return f"safety_audit:v1:{user_id}:{session_id}:{run_id}:{digest}"


def _build_payload(
        *,
        records: list[SafetyAuditRecord],
        source_event_ids: list[str],
        projection_key: str,
) -> SafetyAuditEventPayload:
    decision_counts = _decision_counts(records)
    risk_counts = _risk_counts(records)
    blocked_count = decision_counts.block
    rewrite_count = decision_counts.rewrite
    confirmation_count = (
            decision_counts.require_confirmation
            + decision_counts.confirmation_approved
            + decision_counts.confirmation_rejected
    )
    return SafetyAuditEventPayload(
        audit_refs=[
            SafetyAuditEventRef(
                audit_id=record.id,
                decision=record.decision,
                risk_level=record.risk_level,
                reason_code=record.reason_code,
                step_id=record.step_id,
                tool_call_id=record.tool_call_id,
                function_name=record.function_name,
            )
            for record in records
        ],
        source_event_ids=list(dict.fromkeys(source_event_ids)),
        decision_counts=decision_counts,
        risk_counts=risk_counts,
        blocked_count=blocked_count,
        rewrite_count=rewrite_count,
        confirmation_count=confirmation_count,
        summary=_build_summary(
            total=len(records),
            blocked_count=blocked_count,
            rewrite_count=rewrite_count,
            confirmation_count=confirmation_count,
            high_or_critical_count=risk_counts.high + risk_counts.critical,
        ),
        runtime_metadata=SafetyAuditEventRuntimeMetadata(projection_key=projection_key),
    )


def _decision_counts(records: list[SafetyAuditRecord]) -> SafetyAuditDecisionCounts:
    counts = SafetyAuditDecisionCounts()
    for record in records:
        if record.decision == SafetyAuditDecision.ALLOW:
            counts.allow += 1
        elif record.decision == SafetyAuditDecision.BLOCK:
            counts.block += 1
        elif record.decision == SafetyAuditDecision.REWRITE:
            counts.rewrite += 1
        elif record.decision == SafetyAuditDecision.REQUIRE_CONFIRMATION:
            counts.require_confirmation += 1
        elif record.decision == SafetyAuditDecision.CONFIRMATION_APPROVED:
            counts.confirmation_approved += 1
        elif record.decision == SafetyAuditDecision.CONFIRMATION_REJECTED:
            counts.confirmation_rejected += 1
        elif record.decision == SafetyAuditDecision.CORRECTION:
            counts.correction += 1
        elif record.decision == SafetyAuditDecision.SUPERSEDED:
            counts.superseded += 1
    return counts


def _risk_counts(records: list[SafetyAuditRecord]) -> SafetyAuditRiskCounts:
    counts = SafetyAuditRiskCounts()
    for record in records:
        if record.risk_level == SafetyAuditRiskLevel.LOW:
            counts.low += 1
        elif record.risk_level == SafetyAuditRiskLevel.MEDIUM:
            counts.medium += 1
        elif record.risk_level == SafetyAuditRiskLevel.HIGH:
            counts.high += 1
        elif record.risk_level == SafetyAuditRiskLevel.CRITICAL:
            counts.critical += 1
    return counts


def _build_summary(
        *,
        total: int,
        blocked_count: int,
        rewrite_count: int,
        confirmation_count: int,
        high_or_critical_count: int,
) -> str:
    parts: list[str] = []
    if high_or_critical_count:
        parts.append(f"{high_or_critical_count} 个高风险动作")
    if blocked_count:
        parts.append(f"{blocked_count} 个阻断")
    if rewrite_count:
        parts.append(f"{rewrite_count} 个改写")
    if confirmation_count:
        parts.append(f"{confirmation_count} 个确认")
    if parts:
        return "安全审计：" + "，".join(parts)
    return f"安全审计：记录 {total} 个动作决策"
