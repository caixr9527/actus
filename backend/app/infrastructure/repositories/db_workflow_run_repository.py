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
    Plan,
    PlanEvent,
    StepEvent,
    WaitEvent,
    Session,
    WorkflowRun,
    WorkflowRunEventRecord,
    WorkflowRunStatus,
)
from app.domain.repositories.workflow_run_repository import (
    CurrentStepIdUpdate,
    UNSET_CURRENT_STEP_ID,
    WorkflowRunRepository,
)
from app.domain.services.runtime.contracts.event_delivery_policy import should_persist_event
from app.domain.services.runtime.normalizers import normalize_step_outcome_payload
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

    async def get_by_id_for_update(self, run_id: str) -> Optional[WorkflowRun]:
        record = await self._get_record_with_lock(run_id=run_id)
        return record.to_domain() if record is not None else None

    async def update_status(
            self,
            run_id: str,
            *,
            status: WorkflowRunStatus,
            finished_at: Optional[datetime] = None,
            last_event_at: Optional[datetime] = None,
            current_step_id: CurrentStepIdUpdate = UNSET_CURRENT_STEP_ID,
    ) -> None:
        """原子更新运行状态，不做状态合法性判断。"""
        record = await self._get_record_with_lock(run_id=run_id)
        if record is None:
            raise ValueError(f"运行[{run_id}]不存在，请核实后重试")

        record.status = status.value
        if finished_at is not None:
            record.finished_at = finished_at
        if last_event_at is not None:
            record.last_event_at = last_event_at
        if current_step_id is not UNSET_CURRENT_STEP_ID:
            record.current_step_id = current_step_id

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
        """步骤顺序统一依赖 step 投影表，缺失时回退到当前最大索引+1。"""
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
            # 步骤不存在时创建快照，并把运行时结构化语义一并落库，避免恢复时再丢字段。
            step_record = WorkflowRunStepModel(
                run_id=run_id,
                step_id=step.id,
                step_index=await self._resolve_step_index(run_record=run_record, step_id=step.id),
                title=step.title,
                description=step.description,
                objective_key=step.objective_key,
                success_criteria=list(step.success_criteria or []),
                status=step.status.value,
                task_mode_hint=step.task_mode_hint.value if step.task_mode_hint is not None else None,
                output_mode=step.output_mode.value if step.output_mode is not None else None,
                artifact_policy=step.artifact_policy.value if step.artifact_policy is not None else None,
                outcome=normalize_step_outcome_payload(step.outcome),
                error=step.error,
            )
            self.db_session.add(step_record)
        else:
            # 步骤已存在时仅更新内容与结构化语义，不改既有顺序。
            step_record.title = step.title
            step_record.description = step.description
            step_record.objective_key = step.objective_key
            step_record.success_criteria = list(step.success_criteria or [])
            step_record.status = step.status.value
            step_record.task_mode_hint = step.task_mode_hint.value if step.task_mode_hint is not None else None
            step_record.output_mode = step.output_mode.value if step.output_mode is not None else None
            step_record.artifact_policy = step.artifact_policy.value if step.artifact_policy is not None else None
            step_record.outcome = normalize_step_outcome_payload(step.outcome)
            step_record.error = step.error

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
        if isinstance(event, PlanEvent) and event.plan.status == ExecutionStatus.CANCELLED:
            record.status = WorkflowRunStatus.CANCELLED.value
            record.finished_at = event.created_at
            record.current_step_id = None
            return
        if record.status not in {
            WorkflowRunStatus.COMPLETED.value,
            WorkflowRunStatus.CANCELLED.value,
            WorkflowRunStatus.FAILED.value,
        }:
            record.status = WorkflowRunStatus.RUNNING.value

    async def cancel_run(self, run_id: str) -> None:
        """将运行及其所有未完成步骤统一收敛为 cancelled。"""
        run_record = await self._get_record_with_lock(run_id)
        if run_record is None:
            raise ValueError(f"运行[{run_id}]不存在，请核实后重试")

        cancelled_at = datetime.now()
        run_record.status = WorkflowRunStatus.CANCELLED.value
        run_record.finished_at = cancelled_at
        run_record.last_event_at = cancelled_at
        run_record.current_step_id = None
        await self.mark_unfinished_steps_cancelled(run_id)

    async def mark_unfinished_steps_cancelled(self, run_id: str) -> None:
        """低层步骤投影兜底：只取消未完成 step，不推导 run 状态。"""

        stmt = (
            select(WorkflowRunStepModel)
            .where(WorkflowRunStepModel.run_id == run_id)
            .with_for_update()
        )
        result = await self.db_session.execute(stmt)
        step_records = result.scalars().all()
        for step_record in step_records:
            if step_record.status in {
                ExecutionStatus.COMPLETED.value,
                ExecutionStatus.FAILED.value,
                ExecutionStatus.CANCELLED.value,
            }:
                continue
            step_record.status = ExecutionStatus.CANCELLED.value

    async def add_event_if_absent(
            self,
            session_id: str,
            run_id: Optional[str],
            event: BaseEvent,
    ) -> bool:
        if not should_persist_event(event):
            return False
        if not run_id:
            return False

        inserted = await self.add_event_record_if_absent(
            session_id=session_id,
            run_id=run_id,
            event=event,
        )
        if not inserted:
            return False
        # BE-LG-08：步骤投影同步策略统一收口到 run 仓库。
        if isinstance(event, PlanEvent):
            await self.replace_steps_from_plan(run_id=run_id, plan=event.plan)
        elif isinstance(event, StepEvent):
            await self.upsert_step_from_event(run_id=run_id, event=event)
        return True

    async def add_event_record_if_absent(
            self,
            session_id: str,
            run_id: str,
            event: BaseEvent,
    ) -> bool:
        """按事件ID幂等写入事件记录，不刷新运行状态。"""
        if not should_persist_event(event):
            return False
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
        return True

    async def replace_steps_from_plan(self, run_id: str, plan: Plan) -> None:
        run_record = await self._get_record_with_lock(run_id)
        if run_record is None:
            return

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
                    title=step.title,
                    description=step.description,
                    objective_key=step.objective_key,
                    success_criteria=list(step.success_criteria or []),
                    status=step.status.value,
                    task_mode_hint=step.task_mode_hint.value if step.task_mode_hint is not None else None,
                    output_mode=step.output_mode.value if step.output_mode is not None else None,
                    artifact_policy=step.artifact_policy.value if step.artifact_policy is not None else None,
                    outcome=normalize_step_outcome_payload(step.outcome),
                    error=step.error,
                )
            )

    async def _list_events_by_run_id(self, run_id: str) -> List[Event]:
        stmt = (
            select(WorkflowRunEventModel)
            .where(WorkflowRunEventModel.run_id == run_id)
            .order_by(WorkflowRunEventModel.created_at.asc(), WorkflowRunEventModel.id.asc())
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain().event_payload for record in records]

    async def list_events(self, run_id: Optional[str]) -> List[Event]:
        if not run_id:
            return []
        return await self._list_events_by_run_id(run_id)

    async def list_event_records_by_session(self, session_id: str) -> List[WorkflowRunEventRecord]:
        stmt = (
            select(WorkflowRunEventModel)
            .where(WorkflowRunEventModel.session_id == session_id)
            .order_by(WorkflowRunEventModel.created_at.asc(), WorkflowRunEventModel.id.asc())
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]

    async def list_events_by_session(self, session_id: str) -> List[Event]:
        records = await self.list_event_records_by_session(session_id=session_id)
        return [record.event_payload for record in records]
