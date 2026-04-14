#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""会话上下文快照仓储 DB 实现。"""
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import SessionContextSnapshot
from app.domain.repositories import SessionContextSnapshotRepository
from app.infrastructure.models import SessionContextSnapshotModel


class DBSessionContextSnapshotRepository(SessionContextSnapshotRepository):
    """会话上下文快照仓储 DB 实现。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    @staticmethod
    def _to_insert_values(snapshot: SessionContextSnapshot) -> dict:
        return {
            "session_id": snapshot.session_id,
            "user_id": snapshot.user_id,
            "last_run_id": snapshot.last_run_id,
            "summary_text": snapshot.summary_text,
            "recent_run_briefs": snapshot.model_dump(mode="json", include={"recent_run_briefs"})["recent_run_briefs"],
            "open_questions": list(snapshot.open_questions or []),
            "artifact_paths": list(snapshot.artifact_paths or []),
            "created_at": snapshot.created_at,
            "updated_at": snapshot.updated_at,
        }

    async def get_by_session_id(self, session_id: str) -> Optional[SessionContextSnapshot]:
        stmt = select(SessionContextSnapshotModel).where(SessionContextSnapshotModel.session_id == session_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def upsert(self, snapshot: SessionContextSnapshot) -> SessionContextSnapshot:
        insert_values = self._to_insert_values(snapshot)
        update_values = {
            key: value
            for key, value in insert_values.items()
            if key not in {"session_id", "created_at"}
        }
        update_values["updated_at"] = datetime.now()
        stmt = (
            insert(SessionContextSnapshotModel)
            .values(**insert_values)
            .on_conflict_do_update(
                index_elements=[SessionContextSnapshotModel.session_id],
                set_=update_values,
            )
        )
        await self.db_session.execute(stmt)
        persisted = await self.get_by_session_id(snapshot.session_id)
        if persisted is None:
            raise RuntimeError(f"会话快照[{snapshot.session_id}]写入后读取失败")
        return persisted
