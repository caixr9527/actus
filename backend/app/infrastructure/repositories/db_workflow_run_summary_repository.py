#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行摘要仓储 DB 实现。"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import WorkflowRunSummary, WorkflowRunStatus
from app.domain.repositories import WorkflowRunSummaryRepository
from app.infrastructure.models import WorkflowRunSummaryModel


class DBWorkflowRunSummaryRepository(WorkflowRunSummaryRepository):
    """运行摘要仓储 DB 实现。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    @staticmethod
    def _to_insert_values(summary: WorkflowRunSummary) -> dict:
        return {
            "id": summary.id,
            "run_id": summary.run_id,
            "session_id": summary.session_id,
            "user_id": summary.user_id,
            "thread_id": summary.thread_id,
            "goal": summary.goal,
            "title": summary.title,
            "final_answer_summary": summary.final_answer_summary,
            "final_answer_text": summary.final_answer_text,
            "status": summary.status.value,
            "completed_steps": summary.completed_steps,
            "total_steps": summary.total_steps,
            "step_ledger": summary.model_dump(mode="json", include={"step_ledger"})["step_ledger"],
            "artifacts": list(summary.artifacts or []),
            "open_questions": list(summary.open_questions or []),
            "blockers": list(summary.blockers or []),
            "facts_learned": list(summary.facts_learned or []),
            "created_at": summary.created_at,
            "updated_at": summary.updated_at,
        }

    async def get_by_run_id(self, run_id: str) -> Optional[WorkflowRunSummary]:
        stmt = select(WorkflowRunSummaryModel).where(WorkflowRunSummaryModel.run_id == run_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def list_by_session_id(
            self,
            session_id: str,
            limit: int = 10,
            statuses: Optional[List[WorkflowRunStatus]] = None,
    ) -> List[WorkflowRunSummary]:
        stmt = (
            select(WorkflowRunSummaryModel)
            .where(WorkflowRunSummaryModel.session_id == session_id)
        )
        normalized_statuses = [status.value for status in list(statuses or []) if isinstance(status, WorkflowRunStatus)]
        if len(normalized_statuses) > 0:
            stmt = stmt.where(WorkflowRunSummaryModel.status.in_(normalized_statuses))
        stmt = (
            stmt
            .order_by(WorkflowRunSummaryModel.created_at.desc(), WorkflowRunSummaryModel.id.desc())
            .limit(max(int(limit or 10), 1))
        )
        result = await self.db_session.execute(stmt)
        records = list(result.scalars().all())
        return [record.to_domain() for record in records]

    async def upsert(self, summary: WorkflowRunSummary) -> WorkflowRunSummary:
        insert_values = self._to_insert_values(summary)
        update_values = {
            key: value
            for key, value in insert_values.items()
            if key not in {"id", "run_id", "created_at"}
        }
        update_values["updated_at"] = datetime.now()
        stmt = (
            insert(WorkflowRunSummaryModel)
            .values(**insert_values)
            .on_conflict_do_update(
                index_elements=[WorkflowRunSummaryModel.run_id],
                set_=update_values,
            )
        )
        await self.db_session.execute(stmt)
        persisted = await self.get_by_run_id(summary.run_id)
        if persisted is None:
            raise RuntimeError(f"运行摘要[{summary.run_id}]写入后读取失败")
        return persisted
