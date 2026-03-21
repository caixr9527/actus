#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21 10:35
@Author : caixiaorong01@outlook.com
@File   : db_workflow_run_repository.py
"""
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import (
    BaseEvent,
    DoneEvent,
    ErrorEvent,
    Event,
    File,
    Memory,
    Plan,
    WaitEvent,
    Session,
    WorkflowRun,
    WorkflowRunStatus,
)
from app.domain.repositories import WorkflowRunRepository
from app.infrastructure.models import WorkflowRunModel, WorkflowRunEventModel, WorkflowRunStepModel


class DBWorkflowRunRepository(WorkflowRunRepository):
    """WorkflowRun 仓库 DB 实现"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def create_for_session(
            self,
            session: Session,
            status: WorkflowRunStatus = WorkflowRunStatus.RUNNING,
            thread_id: Optional[str] = None,
    ) -> WorkflowRun:
        run = WorkflowRun(
            session_id=session.id,
            user_id=session.user_id,
            thread_id=thread_id or session.id,
            status=status,
            files_snapshot=list(session.files or []),
            memories_snapshot=dict(session.memories or {}),
            started_at=datetime.now(),
        )
        record = WorkflowRunModel.from_domain(run)
        self.db_session.add(record)
        return run

    async def get_by_id(self, run_id: str) -> Optional[WorkflowRun]:
        stmt = select(WorkflowRunModel).where(WorkflowRunModel.id == run_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def update_checkpoint_ref(
            self,
            run_id: str,
            checkpoint_namespace: Optional[str],
            checkpoint_id: Optional[str],
    ) -> None:
        stmt = select(WorkflowRunModel).where(WorkflowRunModel.id == run_id).with_for_update()
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            raise ValueError(f"运行[{run_id}]不存在，请核实后重试")

        record.checkpoint_namespace = checkpoint_namespace
        record.checkpoint_id = checkpoint_id

    async def _get_record_with_lock(self, run_id: str) -> Optional[WorkflowRunModel]:
        stmt = select(WorkflowRunModel).where(WorkflowRunModel.id == run_id).with_for_update()
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _refresh_run_status_by_event(self, run_id: str, event: BaseEvent) -> None:
        record = await self._get_record_with_lock(run_id)
        if record is None:
            return

        record.last_event_at = event.created_at
        if isinstance(event, DoneEvent):
            record.status = WorkflowRunStatus.COMPLETED.value
            record.finished_at = event.created_at
            return
        if isinstance(event, WaitEvent):
            record.status = WorkflowRunStatus.WAITING.value
            return
        if isinstance(event, ErrorEvent):
            record.status = WorkflowRunStatus.FAILED.value
            record.finished_at = event.created_at
            return
        if record.status not in {
            WorkflowRunStatus.COMPLETED.value,
            WorkflowRunStatus.CANCELLED.value,
            WorkflowRunStatus.FAILED.value,
        }:
            record.status = WorkflowRunStatus.RUNNING.value

    async def add_event_if_absent(
            self,
            session_id: str,
            run_id: Optional[str],
            event: BaseEvent,
    ) -> bool:
        if not run_id:
            return False

        stmt = select(WorkflowRunEventModel.id).where(
            WorkflowRunEventModel.run_id == run_id,
            WorkflowRunEventModel.event_id == str(event.id),
        )
        result = await self.db_session.execute(stmt)
        if result.scalar_one_or_none() is not None:
            return False

        event_record = WorkflowRunEventModel(
            run_id=run_id,
            session_id=session_id,
            event_id=str(event.id),
            event_type=event.type,
            event_payload=event.model_dump(mode="json"),
            created_at=event.created_at,
        )
        self.db_session.add(event_record)
        await self._refresh_run_status_by_event(run_id=run_id, event=event)
        return True

    async def replace_steps_from_plan(self, run_id: str, plan: Plan) -> None:
        run_record = await self._get_record_with_lock(run_id)
        if run_record is None:
            return

        run_record.plan_snapshot = plan.model_dump(mode="json")
        next_step = plan.get_next_step()
        run_record.current_step_id = next_step.id if next_step else None

        await self.db_session.execute(
            delete(WorkflowRunStepModel).where(WorkflowRunStepModel.run_id == run_id)
        )
        for index, step in enumerate(plan.steps):
            self.db_session.add(
                WorkflowRunStepModel(
                    run_id=run_id,
                    step_id=step.id,
                    step_index=index,
                    description=step.description,
                    status=step.status.value,
                    result=step.result,
                    error=step.error,
                    success=step.success,
                    attachments=list(step.attachments or []),
                )
            )

    async def append_file_snapshot(self, run_id: str, file: File) -> None:
        run_record = await self._get_record_with_lock(run_id)
        if run_record is None:
            return

        current_files = list(run_record.files_snapshot or [])
        file_data = file.model_dump(mode="json")
        for current_file in current_files:
            if str(current_file.get("id", "")) == file.id:
                return
        current_files.append(file_data)
        run_record.files_snapshot = current_files

    async def remove_file_snapshot(self, run_id: str, file_id: str) -> None:
        run_record = await self._get_record_with_lock(run_id)
        if run_record is None:
            return
        run_record.files_snapshot = [
            file_data for file_data in list(run_record.files_snapshot or [])
            if str(file_data.get("id", "")) != file_id
        ]

    async def upsert_memory_snapshot(self, run_id: str, agent_name: str, memory: Memory) -> None:
        run_record = await self._get_record_with_lock(run_id)
        if run_record is None:
            return

        current_memories = dict(run_record.memories_snapshot or {})
        current_memories[agent_name] = memory.model_dump(mode="json")
        run_record.memories_snapshot = current_memories

    async def _list_events_by_run_id(self, run_id: str) -> List[Event]:
        stmt = (
            select(WorkflowRunEventModel)
            .where(WorkflowRunEventModel.run_id == run_id)
            .order_by(WorkflowRunEventModel.created_at.asc(), WorkflowRunEventModel.id.asc())
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain().event_payload for record in records]

    async def get_events_with_compat(self, session: Session) -> List[Event]:
        if session.current_run_id:
            run_events = await self._list_events_by_run_id(session.current_run_id)
            if run_events:
                return run_events
        return list(session.events or [])

    async def get_files_with_compat(self, session: Session) -> List[File]:
        if session.current_run_id:
            run = await self.get_by_id(session.current_run_id)
            if run is not None and run.files_snapshot:
                return list(run.files_snapshot)
        return list(session.files or [])

    async def get_memories_with_compat(self, session: Session) -> Dict[str, Memory]:
        if session.current_run_id:
            run = await self.get_by_id(session.current_run_id)
            if run is not None and run.memories_snapshot:
                return dict(run.memories_snapshot)
        return dict(session.memories or {})
