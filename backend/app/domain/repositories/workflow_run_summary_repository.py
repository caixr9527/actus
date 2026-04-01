#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行摘要仓储协议。"""
from typing import List, Optional, Protocol

from app.domain.models import WorkflowRunSummary, WorkflowRunStatus


class WorkflowRunSummaryRepository(Protocol):
    """运行摘要仓储协议定义。"""

    async def get_by_run_id(self, run_id: str) -> Optional[WorkflowRunSummary]:
        """根据 run_id 读取运行摘要。"""
        ...

    async def list_by_session_id(
            self,
            session_id: str,
            limit: int = 10,
            statuses: Optional[List[WorkflowRunStatus]] = None,
    ) -> List[WorkflowRunSummary]:
        """按 session 读取最近的运行摘要列表。"""
        ...

    async def upsert(self, summary: WorkflowRunSummary) -> WorkflowRunSummary:
        """按 run_id 幂等写入运行摘要。"""
        ...
