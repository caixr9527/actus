#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/12 17:32
@Author : caixiaorong01@outlook.com
@File   : db_workspace_repository.py
"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Workspace
from app.domain.repositories import WorkspaceRepository
from app.infrastructure.models import WorkspaceModel


class DBWorkspaceRepository(WorkspaceRepository):
    """基于数据库的工作区仓库。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def save(self, workspace: Workspace) -> None:
        payload = workspace.model_dump(mode="python")
        update_values = {
            key: value
            for key, value in payload.items()
            if key not in {"id", "created_at"}
        }
        update_values["updated_at"] = datetime.now()
        stmt = (
            insert(WorkspaceModel.__table__)
            .values(**payload)
            .on_conflict_do_update(
                index_elements=[WorkspaceModel.id],
                set_=update_values,
            )
        )
        await self.db_session.execute(stmt)

    async def get_by_id(self, workspace_id: str) -> Optional[Workspace]:
        stmt = select(WorkspaceModel).where(WorkspaceModel.id == workspace_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def get_by_session_id(self, session_id: str) -> Optional[Workspace]:
        stmt = select(WorkspaceModel).where(WorkspaceModel.session_id == session_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def list_by_session_id(self, session_id: str) -> List[Workspace]:
        stmt = select(WorkspaceModel).where(WorkspaceModel.session_id == session_id)
        result = await self.db_session.execute(stmt)
        return [record.to_domain() for record in result.scalars().all()]

    async def delete_by_id(self, workspace_id: str) -> None:
        await self.db_session.execute(
            delete(WorkspaceModel).where(WorkspaceModel.id == workspace_id)
        )
