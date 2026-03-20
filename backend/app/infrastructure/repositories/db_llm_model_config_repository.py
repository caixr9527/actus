#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 16:08
@Author : caixiaorong01@outlook.com
@File   : db_llm_model_config_repository.py
"""
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import LLMModelConfig
from app.domain.repositories import LLMModelConfigRepository
from app.infrastructure.models import LLMModelConfigModel


class DBLLMModelConfigRepository(LLMModelConfigRepository):
    """基于 Postgres 的 LLM 模型配置仓库"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def list_enabled(self) -> List[LLMModelConfig]:
        stmt = (
            select(LLMModelConfigModel)
            .where(LLMModelConfigModel.enabled.is_(True))
            .order_by(
                LLMModelConfigModel.sort_order.asc(),
                LLMModelConfigModel.created_at.asc(),
                LLMModelConfigModel.id.asc(),
            )
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]

    async def get_by_id(self, model_id: str) -> Optional[LLMModelConfig]:
        stmt = select(LLMModelConfigModel).where(LLMModelConfigModel.id == model_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def get_default(self) -> Optional[LLMModelConfig]:
        stmt = (
            select(LLMModelConfigModel)
            .where(
                LLMModelConfigModel.is_default.is_(True),
                LLMModelConfigModel.enabled.is_(True),
            )
            .order_by(LLMModelConfigModel.updated_at.desc(), LLMModelConfigModel.id.asc())
        )
        result = await self.db_session.execute(stmt)
        record = result.scalars().first()
        return record.to_domain() if record is not None else None
