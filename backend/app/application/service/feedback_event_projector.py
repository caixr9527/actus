#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FeedbackEvent runtime observation 投影。"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Callable, Iterable

from app.domain.models import FeedbackEvent
from app.domain.models.feedback import (
    FeedbackEventPayloadResult,
    FeedbackRecord,
    FeedbackStatus,
    FeedbackWriteResult,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.feedback_runtime_ports import FeedbackEventProjectorPort

logger = logging.getLogger(__name__)


class FeedbackEventProjector(FeedbackEventProjectorPort):
    """把已落库 FeedbackRecord 聚合投影成 hidden FeedbackEvent。"""

    def __init__(self, *, uow_factory: Callable[[], IUnitOfWork], projection_limit: int = 100) -> None:
        self._uow_factory = uow_factory
        self._projection_limit = max(1, min(int(projection_limit or 100), 200))

    async def project_record_written(self, record_ref: FeedbackWriteResult) -> None:
        record = record_ref.record_ref
        if record is None or not record_ref.success:
            return
        source_run_id = record.source_ref.source_run_id
        aggregation_key = record.source_ref.source_event_id
        if not source_run_id:
            self._log_projection_failed(
                reason="missing_source_run_id",
                feedback_id=record.feedback_id,
                session_id=record.scope.session_id,
            )
            return
        await self._project_source_event(
            user_id=record.scope.user_id,
            session_id=record.scope.session_id,
            source_run_id=source_run_id,
            aggregation_key=aggregation_key,
        )

    async def project_resolution_updated(self, record_ref: FeedbackWriteResult) -> None:
        record = record_ref.record_ref
        resolution = record.resolution if record is not None else None
        resolved_by_ref = resolution.resolved_by_ref if resolution is not None else None
        if record is None or resolution is None or not record_ref.success or not resolved_by_ref:
            return

        aggregation_key = resolved_by_ref.get("resolution_aggregation_key")
        aggregation_kind = resolved_by_ref.get("resolution_aggregation_kind")
        if not aggregation_key or not aggregation_kind:
            self._log_projection_failed(
                reason="missing_resolution_aggregation_key",
                feedback_id=record.feedback_id,
                session_id=record.scope.session_id,
            )
            return

        if aggregation_kind == "source_event":
            source_event_id = resolved_by_ref.get("resolution_source_event_id")
            if not source_event_id:
                self._log_projection_failed(
                    reason="missing_resolution_source_event_id",
                    feedback_id=record.feedback_id,
                    session_id=record.scope.session_id,
                )
                return
            source_event_run_id = await self._load_resolution_source_run_id(
                user_id=record.scope.user_id,
                session_id=record.scope.session_id,
                source_event_id=source_event_id,
            )
            if not source_event_run_id:
                self._log_projection_failed(
                    reason="missing_resolution_source_event_run_id",
                    feedback_id=record.feedback_id,
                    session_id=record.scope.session_id,
                )
                return

        source_run_id = record.source_ref.source_run_id
        if not source_run_id:
            self._log_projection_failed(
                reason="missing_resolution_source_run_id",
                feedback_id=record.feedback_id,
                session_id=record.scope.session_id,
            )
            return

        await self._project_resolution(
            user_id=record.scope.user_id,
            session_id=record.scope.session_id,
            source_run_id=source_run_id,
            aggregation_key=aggregation_key,
        )

    async def _project_source_event(
            self,
            *,
            user_id: str,
            session_id: str,
            source_run_id: str,
            aggregation_key: str,
    ) -> None:
        try:
            async with self._uow_factory() as uow:
                records = await uow.feedback.list_by_source_event_for_projection(
                    user_id=user_id,
                    session_id=session_id,
                    source_run_id=source_run_id,
                    source_event_id=aggregation_key,
                    limit=self._projection_limit,
                )
                await self._upsert_projection(
                    uow=uow,
                    records=records,
                    session_id=session_id,
                    source_run_id=source_run_id,
                    aggregation_key=aggregation_key,
                    aggregation_kind="source_event",
                )
        except Exception:
            self._log_projection_failed(
                reason="source_event_projection_error",
                feedback_id=None,
                session_id=session_id,
                exc_info=True,
            )

    async def _project_resolution(
            self,
            *,
            user_id: str,
            session_id: str,
            source_run_id: str,
            aggregation_key: str,
    ) -> None:
        try:
            async with self._uow_factory() as uow:
                records = await uow.feedback.list_by_resolution_aggregation_key(
                    user_id=user_id,
                    session_id=session_id,
                    source_run_id=source_run_id,
                    resolution_aggregation_key=aggregation_key,
                    limit=self._projection_limit,
                )
                await self._upsert_projection(
                    uow=uow,
                    records=records,
                    session_id=session_id,
                    source_run_id=source_run_id,
                    aggregation_key=aggregation_key,
                    aggregation_kind="resolution",
                )
        except Exception:
            self._log_projection_failed(
                reason="resolution_projection_error",
                feedback_id=None,
                session_id=session_id,
                exc_info=True,
            )

    async def _load_resolution_source_run_id(
            self,
            *,
            user_id: str,
            session_id: str,
            source_event_id: str,
    ) -> str | None:
        async with self._uow_factory() as uow:
            record = await uow.workflow_run.get_event_record_by_event_id_in_session(
                user_id=user_id,
                session_id=session_id,
                event_id=source_event_id,
            )
            return record.run_id if record is not None else None

    async def _upsert_projection(
            self,
            *,
            uow: IUnitOfWork,
            records: list[FeedbackRecord],
            session_id: str,
            source_run_id: str,
            aggregation_key: str,
            aggregation_kind: str,
    ) -> None:
        if not records:
            return
        event = FeedbackEvent(
            id=f"feedback:{source_run_id}:{aggregation_key}",
            payload=self._build_payload(
                records=records,
                source_run_id=source_run_id,
                aggregation_key=aggregation_key,
                aggregation_kind=aggregation_kind,
            ),
            created_at=datetime.now(),
        )
        projected = await uow.workflow_run.upsert_feedback_event_record(
            session_id=session_id,
            run_id=source_run_id,
            event=event,
        )
        if projected is None:
            self._log_projection_failed(
                reason="feedback_event_upsert_rejected",
                feedback_id=None,
                session_id=session_id,
            )
            return
        logger.info(
            "feedback_event_projected session_id=%s run_id=%s event_id=%s feedback_count=%s aggregation_kind=%s",
            session_id,
            source_run_id,
            projected.event_id,
            len(records),
            aggregation_kind,
        )

    def _build_payload(
            self,
            *,
            records: list[FeedbackRecord],
            source_run_id: str,
            aggregation_key: str,
            aggregation_kind: str,
    ) -> FeedbackEventPayloadResult:
        severity_counts = Counter(record.severity.value for record in records)
        status_counts = Counter(record.status.value for record in records)
        kind_counts = Counter(record.kind.value for record in records)
        source_event_ids = self._unique(record.source_event_id for record in records)
        resolution_reason_codes = self._unique(
            record.resolution_reason_code.value
            for record in records
            if record.resolution_reason_code is not None
        )
        return FeedbackEventPayloadResult(
            feedback_refs=[record.id for record in records],
            counts={"feedback_count": len(records)},
            severity_counts=dict(severity_counts),
            status_counts=dict(status_counts),
            kind_counts=dict(kind_counts),
            summary=self._build_summary(records),
            source_event_ids=source_event_ids,
            runtime_metadata={
                "schema_version": "feedback_event.v1",
                "source_run_id": source_run_id,
                "aggregation_key": aggregation_key,
                "aggregation_kind": aggregation_kind,
                "open_count": status_counts.get(FeedbackStatus.OPEN.value, 0),
                "resolution_reason_codes": ",".join(resolution_reason_codes) if resolution_reason_codes else None,
            },
        )

    @staticmethod
    def _build_summary(records: list[FeedbackRecord]) -> str:
        # FeedbackEvent 是隐藏投影，只给面板和评测提供脱敏摘要；原文必须回到受控 source event。
        summaries = [
            record.prompt_safe_summary.summary_text
            for record in records
            if record.prompt_safe_summary.prompt_visible and record.prompt_safe_summary.summary_text
        ][:3]
        if not summaries:
            return f"Feedback Ledger 投影 {len(records)} 条反馈。"
        return "；".join(summaries)

    @staticmethod
    def _unique(values: Iterable[str | None]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @staticmethod
    def _log_projection_failed(
            *,
            reason: str,
            feedback_id: str | None,
            session_id: str,
            exc_info: bool = False,
    ) -> None:
        logger.warning(
            "feedback_event_projection_failed reason=%s session_id=%s feedback_id=%s",
            reason,
            session_id,
            feedback_id,
            exc_info=exc_info,
        )


__all__ = ["FeedbackEventProjector"]
