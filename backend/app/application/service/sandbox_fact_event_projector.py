#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Fact Runtime 事件投影器。"""

from __future__ import annotations

import logging
from typing import Callable

from app.domain.models import SandboxFactEvent, SandboxFactEventRef
from app.domain.models.sandbox_fact import SandboxFactRecord
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.sandbox_fact_ports import SandboxFactProjectionContext

logger = logging.getLogger(__name__)


class SandboxFactEventProjectionError(RuntimeError):
    """Sandbox Fact event 投影失败，携带稳定 reason_code。"""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class SandboxFactEventProjector:
    """将已入库 fact 投影为轻量 runtime event。"""

    def __init__(self, *, uow_factory: Callable[[], IUnitOfWork]) -> None:
        self._uow_factory = uow_factory

    async def project_tool_event_facts(
            self,
            *,
            context: SandboxFactProjectionContext,
            facts: list[SandboxFactRecord],
    ) -> SandboxFactEvent | None:
        if not facts:
            return None
        fact_ids = [fact.id for fact in facts if str(fact.id or "").strip()]
        if not fact_ids:
            return None

        try:
            source_event_id = str(context.source_event_id or "").strip()
            if not source_event_id:
                raise SandboxFactEventProjectionError("source_event_id_missing")
            if not context.scope.run_id:
                raise SandboxFactEventProjectionError("run_id_missing")

            scoped_facts = await self._load_scoped_facts(context=context, fact_ids=fact_ids)
            if len(scoped_facts) != len(set(fact_ids)):
                raise SandboxFactEventProjectionError("fact_ref_scope_mismatch")

            ordered_facts = _sort_facts_by_input_order(scoped_facts, fact_ids)
            if any(fact.source_ref.source_event_id != source_event_id for fact in ordered_facts):
                raise SandboxFactEventProjectionError("fact_source_event_mismatch")

            event = SandboxFactEvent(
                fact_refs=[
                    SandboxFactEventRef(
                        fact_id=fact.id,
                        fact_kind=fact.fact_kind,
                        summary=fact.summary,
                    )
                    for fact in ordered_facts
                ],
                summary=_build_event_summary(ordered_facts),
                source_event_id=source_event_id,
                step_id=_resolve_event_step_id(ordered_facts),
            )
            return event
        except Exception as exc:
            logger.exception(
                "sandbox_fact_event_projection_failed",
                extra={
                    "user_id": context.scope.user_id,
                    "session_id": context.scope.session_id,
                    "run_id": context.scope.run_id,
                    "step_id": context.current_step_id or context.scope.current_step_id,
                    "source_event_id": context.source_event_id,
                    "fact_ids": fact_ids,
                    "error_type": exc.__class__.__name__,
                    "reason_code": getattr(exc, "reason_code", "event_projection_failed"),
                },
            )
            return None

    async def _load_scoped_facts(
            self,
            *,
            context: SandboxFactProjectionContext,
            fact_ids: list[str],
    ) -> list[SandboxFactRecord]:
        async with self._uow_factory() as uow:
            return await uow.sandbox_fact.list_by_ids(
                user_id=context.scope.user_id,
                session_id=str(context.scope.session_id),
                fact_ids=fact_ids,
                limit=len(fact_ids),
            )


def _sort_facts_by_input_order(
        facts: list[SandboxFactRecord],
        fact_ids: list[str],
) -> list[SandboxFactRecord]:
    order = {fact_id: index for index, fact_id in enumerate(fact_ids)}
    return sorted(facts, key=lambda fact: order.get(fact.id, len(order)))


def _build_event_summary(facts: list[SandboxFactRecord]) -> str:
    summaries = [fact.summary.strip() for fact in facts if fact.summary.strip()]
    if not summaries:
        return f"记录了 {len(facts)} 条事实"
    if len(summaries) == 1:
        return summaries[0][:500]
    return "；".join(summaries)[:500]


def _resolve_event_step_id(facts: list[SandboxFactRecord]) -> str | None:
    step_ids = [fact.step_id for fact in facts if fact.step_id]
    if not step_ids:
        return None
    first_step_id = step_ids[0]
    if all(step_id == first_step_id for step_id in step_ids):
        return first_step_id
    return None
