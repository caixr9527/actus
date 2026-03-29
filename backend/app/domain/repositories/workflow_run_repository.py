#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21 10:30
@Author : caixiaorong01@outlook.com
@File   : workflow_run_repository.py
"""
from typing import Any, Dict, List, Optional, Protocol

from app.domain.models import (
    BaseEvent,
    Event,
    Plan,
    Session,
    StepEvent,
    WorkflowRun,
    WorkflowRunStatus,
)


class WorkflowRunRepository(Protocol):
    """WorkflowRun 仓库协议定义"""

    async def create_for_session(
            self,
            session: Session,
            status: WorkflowRunStatus = WorkflowRunStatus.RUNNING,
            thread_id: Optional[str] = None,
    ) -> WorkflowRun:
        """按会话创建新的运行主记录"""
        ...

    async def get_by_id(self, run_id: str) -> Optional[WorkflowRun]:
        """根据运行ID查询运行主记录"""
        ...

    async def update_checkpoint_ref(
            self,
            run_id: str,
            checkpoint_namespace: Optional[str],
            checkpoint_id: Optional[str],
    ) -> None:
        """更新运行记录的 checkpoint 引用"""
        ...

    async def update_runtime_metadata(
            self,
            run_id: str,
            runtime_metadata: Dict[str, Any],
            current_step_id: Optional[str],
    ) -> None:
        """更新运行记录的 runtime_metadata 与当前步骤引用"""
        ...

    async def add_event_if_absent(
            self,
            session_id: str,
            run_id: Optional[str],
            event: BaseEvent,
    ) -> bool:
        """按事件ID幂等写入运行事件"""
        ...

    async def replace_steps_from_plan(self, run_id: str, plan: Plan) -> None:
        """用计划快照替换运行步骤快照"""
        ...

    async def upsert_step_from_event(self, run_id: str, event: StepEvent) -> None:
        """基于 StepEvent 增量更新运行步骤快照"""
        ...

    async def list_events(self, run_id: Optional[str]) -> List[Event]:
        """按运行ID读取事件列表；缺少运行ID时返回空列表"""
        ...
