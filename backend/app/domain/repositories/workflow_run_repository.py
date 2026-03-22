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
    File,
    Memory,
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

    async def append_file_snapshot(self, run_id: str, file: File) -> None:
        """向运行快照追加文件"""
        ...

    async def remove_file_snapshot(self, run_id: str, file_id: str) -> None:
        """从运行快照移除文件"""
        ...

    async def upsert_memory_snapshot(self, run_id: str, agent_name: str, memory: Memory) -> None:
        """更新运行快照中的指定 Agent 记忆"""
        ...

    async def get_events_with_compat(self, session: Session) -> List[Event]:
        """按兼容策略读取事件（优先运行事件，回退会话事件）"""
        ...

    async def get_files_with_compat(self, session: Session) -> List[File]:
        """按兼容策略读取文件（优先运行快照，回退会话文件）"""
        ...

    async def get_memories_with_compat(self, session: Session) -> Dict[str, Memory]:
        """按兼容策略读取记忆（优先运行快照，回退会话记忆）"""
        ...
