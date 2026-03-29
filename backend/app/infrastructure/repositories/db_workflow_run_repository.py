#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21 10:35
@Author : caixiaorong01@outlook.com
@File   : db_workflow_run_repository.py
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import (
    BaseEvent,
    DoneEvent,
    ErrorEvent,
    Event,
    ExecutionStatus,
    File,
    Plan,
    PlanEvent,
    StepEvent,
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

    async def update_runtime_metadata(
            self,
            run_id: str,
            runtime_metadata: Dict[str, Any],
            current_step_id: Optional[str],
    ) -> None:
        """回写运行时契约元数据，并同步当前步骤指针。"""
        record = await self._get_record_with_lock(run_id)
        if record is None:
            raise ValueError(f"运行[{run_id}]不存在，请核实后重试")

        merged_metadata = dict(record.runtime_metadata or {})
        merged_metadata.update(runtime_metadata or {})
        record.runtime_metadata = merged_metadata
        record.current_step_id = current_step_id

    async def _get_record_with_lock(self, run_id: str) -> Optional[WorkflowRunModel]:
        stmt = select(WorkflowRunModel).where(WorkflowRunModel.id == run_id).with_for_update()
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_step_record_with_lock(self, run_id: str, step_id: str) -> Optional[WorkflowRunStepModel]:
        stmt = (
            select(WorkflowRunStepModel)
            .where(
                WorkflowRunStepModel.run_id == run_id,
                WorkflowRunStepModel.step_id == step_id,
            )
            .with_for_update()
        )
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _resolve_step_index(self, run_record: WorkflowRunModel, step_id: str) -> int:
        """优先按 plan_snapshot 推断步骤顺序，缺失时回退到当前最大索引+1。"""
        plan_snapshot = run_record.plan_snapshot if isinstance(run_record.plan_snapshot, dict) else {}
        raw_steps = plan_snapshot.get("steps") if isinstance(plan_snapshot, dict) else None
        if isinstance(raw_steps, list):
            for index, step in enumerate(raw_steps):
                if isinstance(step, dict) and str(step.get("id", "")) == step_id:
                    return index

        stmt = select(func.max(WorkflowRunStepModel.step_index)).where(WorkflowRunStepModel.run_id == run_record.id)
        result = await self.db_session.execute(stmt)
        max_index = result.scalar_one_or_none()
        return int(max_index or -1) + 1

    async def upsert_step_from_event(self, run_id: str, event: StepEvent) -> None:
        """基于 StepEvent 增量收敛 workflow_run_steps 快照。"""
        run_record = await self._get_record_with_lock(run_id)
        if run_record is None:
            return

        step = event.step
        step_record = await self._get_step_record_with_lock(run_id=run_id, step_id=step.id)
        if step_record is None:
            # 步骤不存在时创建快照，尽量保持与计划中的顺序一致。
            step_record = WorkflowRunStepModel(
                run_id=run_id,
                step_id=step.id,
                step_index=await self._resolve_step_index(run_record=run_record, step_id=step.id),
                description=step.description,
                status=step.status.value,
                result=step.result,
                error=step.error,
                success=step.success,
                attachments=list(step.attachments or []),
            )
            self.db_session.add(step_record)
        else:
            # 步骤已存在时仅做字段收敛，避免覆盖既有顺序。
            step_record.description = step.description
            step_record.status = step.status.value
            step_record.result = step.result
            step_record.error = step.error
            step_record.success = step.success
            step_record.attachments = list(step.attachments or [])

        if step.status in {ExecutionStatus.RUNNING, ExecutionStatus.PENDING}:
            run_record.current_step_id = step.id
        elif step.done and run_record.current_step_id == step.id:
            run_record.current_step_id = None

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
        # BE-LG-08：步骤投影同步策略统一收口到 run 仓库。
        if isinstance(event, PlanEvent):
            await self.replace_steps_from_plan(run_id=run_id, plan=event.plan)
        elif isinstance(event, StepEvent):
            await self.upsert_step_from_event(run_id=run_id, event=event)
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
