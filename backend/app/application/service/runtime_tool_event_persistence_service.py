#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Graph 主链路 ToolEvent 持久化与 Sandbox Fact 投影服务。"""

from __future__ import annotations

import logging
from typing import Callable

from app.domain.external import Task
from app.domain.models import RuntimeEventProjection, TaskStreamEventRecord, ToolEvent, ToolEventStatus
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.sandbox_fact_ports import (
    RuntimeToolEventPersistencePort,
    SandboxFactEventProjectorPort,
    SandboxFactProjectionContextBuilderPort,
    SandboxFactRecorderPort,
    ToolEventDisplayProjectorPort,
    ToolEventFactProjectionResult,
)

logger = logging.getLogger(__name__)


class RuntimeToolEventPersistenceService(RuntimeToolEventPersistencePort):
    """在 graph completion gate 前补齐 ToolEvent -> Fact 的主写入链路。"""

    def __init__(
            self,
            *,
            session_id: str,
            task: Task,
            uow_factory: Callable[[], IUnitOfWork],
            runtime_state_coordinator,
            sandbox_fact_recorder: SandboxFactRecorderPort,
            sandbox_fact_context_builder: SandboxFactProjectionContextBuilderPort,
            sandbox_fact_event_projector: SandboxFactEventProjectorPort | None = None,
            tool_event_display_projector: ToolEventDisplayProjectorPort | None = None,
    ) -> None:
        self._session_id = str(session_id or "").strip()
        self._task = task
        self._uow_factory = uow_factory
        self._runtime_state_coordinator = runtime_state_coordinator
        self._sandbox_fact_recorder = sandbox_fact_recorder
        self._sandbox_fact_context_builder = sandbox_fact_context_builder
        self._sandbox_fact_event_projector = sandbox_fact_event_projector
        self._tool_event_display_projector = tool_event_display_projector

    async def persist_tool_event_and_record_facts(
            self,
            *,
            event: ToolEvent,
            run_id: str,
            session_id: str,
            current_step_id: str,
    ) -> ToolEventFactProjectionResult:
        existing_projection = event.runtime_fact_projection if isinstance(event.runtime_fact_projection, dict) else {}
        existing_source_event_id = str(existing_projection.get("source_event_id") or "").strip()
        if existing_projection.get("graph_main_chain") is True and existing_source_event_id:
            event.id = existing_source_event_id
            return ToolEventFactProjectionResult(
                source_event_id=existing_source_event_id,
                fact_count=int(existing_projection.get("fact_count") or 0),
                sandbox_fact_event_persisted=bool(existing_projection.get("sandbox_fact_event_persisted")),
                event_inserted=bool(existing_projection.get("event_inserted")),
            )
        if str(session_id or "").strip() != self._session_id:
            raise ValueError("ToolEvent session_id 与运行实例不一致")
        if not str(current_step_id or "").strip():
            raise ValueError("ToolEvent fact 投影需要 current_step_id")

        await self._project_tool_event_display_content(event)

        stream_event_id = await self._task.output_stream.put(
            TaskStreamEventRecord(event=event).model_dump_json()
        )
        source_event_id = str(stream_event_id or "").strip()
        if not source_event_id:
            raise ValueError("ToolEvent 输出流写入未返回 source_event_id")
        persisted_event = event.model_copy(deep=True)
        persisted_event.id = source_event_id
        db_event_persisted = False
        try:
            persist_result = await self._runtime_state_coordinator.persist_runtime_event(
                session_id=self._session_id,
                event=persisted_event,
                projection=RuntimeEventProjection(),
                allow_status_transition=False,
            )
            db_event_persisted = True
            event.id = source_event_id

            fact_count = 0
            sandbox_fact_event_persisted = False
            if event.status == ToolEventStatus.CALLED:
                context = await self._sandbox_fact_context_builder.build_for_tool_event(
                    source_event_id=source_event_id,
                    current_step_id=str(current_step_id or "").strip(),
                )
                if str(run_id or "").strip() and str(context.scope.run_id or "").strip() != str(run_id or "").strip():
                    raise ValueError("ToolEvent run_id 与投影上下文不一致")
                facts = await self._sandbox_fact_recorder.record_from_tool_event(
                    context=context,
                    event=event,
                )
                fact_count = len(list(facts or []))
                sandbox_fact_event_persisted = await self._project_sandbox_fact_event(
                    context=context,
                    facts=list(facts or []),
                )
            event.runtime_fact_projection = {
                "graph_main_chain": True,
                "source_event_id": source_event_id,
                "fact_count": fact_count,
                "sandbox_fact_event_persisted": sandbox_fact_event_persisted,
                "event_inserted": bool(getattr(persist_result, "event_inserted", False)),
            }
        except Exception:
            if not db_event_persisted:
                try:
                    await self._task.output_stream.delete_message(source_event_id)
                except Exception as exc:
                    logger.exception(
                        "tool_event_output_stream_compensate_failed",
                        extra={
                            "session_id": self._session_id,
                            "source_event_id": source_event_id,
                            "error_type": exc.__class__.__name__,
                        },
                    )
            else:
                logger.exception(
                    "tool_event_fact_projection_contract_failed",
                    extra={
                        "session_id": self._session_id,
                        "source_event_id": source_event_id,
                        "function_name": event.function_name,
                        "tool_call_id": event.tool_call_id,
                        "reason_code": "fact_projection_failed_after_source_event_persisted",
                    },
                )
            raise

        return ToolEventFactProjectionResult(
            source_event_id=source_event_id,
            fact_count=fact_count,
            sandbox_fact_event_persisted=sandbox_fact_event_persisted,
            event_inserted=bool(getattr(persist_result, "event_inserted", False)),
        )

    async def _project_tool_event_display_content(self, event: ToolEvent) -> None:
        if event.status != ToolEventStatus.CALLED:
            return
        if event.tool_content is not None:
            return
        if self._tool_event_display_projector is None:
            return
        await self._tool_event_display_projector.project(event)

    async def _project_sandbox_fact_event(
            self,
            *,
            context,
            facts,
    ) -> bool:
        if self._sandbox_fact_event_projector is None or not facts:
            return False
        try:
            fact_event = await self._sandbox_fact_event_projector.project_tool_event_facts(
                context=context,
                facts=facts,
            )
            if fact_event is None:
                return False
            await self._runtime_state_coordinator.persist_runtime_event(
                session_id=self._session_id,
                event=fact_event,
                projection=RuntimeEventProjection(),
                allow_status_transition=False,
            )
            return True
        except Exception as exc:
            logger.exception(
                "sandbox_fact_event_projection_failed",
                extra={
                    "user_id": getattr(context.scope, "user_id", None),
                    "session_id": getattr(context.scope, "session_id", None),
                    "run_id": getattr(context.scope, "run_id", None),
                    "step_id": getattr(context, "current_step_id", None)
                    or getattr(context.scope, "current_step_id", None),
                    "source_event_id": getattr(context, "source_event_id", None),
                    "fact_ids": [getattr(fact, "id", None) for fact in facts],
                    "error_type": exc.__class__.__name__,
                    "reason_code": "event_stream_persist_failed",
                },
            )
            return False
