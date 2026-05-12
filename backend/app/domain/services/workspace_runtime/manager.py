#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace 生命周期管理器。"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

from app.domain.models import Session, Workspace
from app.domain.repositories import IUnitOfWork

logger = logging.getLogger(__name__)


def _log_core(level: int, event: str, **fields: object) -> None:
    """核心链路日志统一走 runtime_logging 格式；导入失败时降级普通日志。"""
    try:
        from app.domain.services.runtime.contracts.runtime_logging import log_runtime

        log_runtime(logger, level, event, **fields)
    except Exception:
        logger.log(level, "规划执行 事件=%s 字段=%s", event, fields)


class WorkspaceManager:
    """收口 workspace 生命周期与环境绑定写回。"""

    def __init__(self, *, uow_factory: Callable[[], IUnitOfWork]) -> None:
        self._uow_factory = uow_factory

    async def ensure_workspace(
            self,
            session: Session,
            *,
            uow: Optional[IUnitOfWork] = None,
    ) -> Workspace:
        if uow is not None:
            return await self._ensure_workspace_with_uow(session=session, uow=uow)

        async with self._uow_factory() as managed_uow:
            return await self._ensure_workspace_with_uow(session=session, uow=managed_uow)

    async def get_workspace(
            self,
            session: Session,
            *,
            uow: Optional[IUnitOfWork] = None,
    ) -> Optional[Workspace]:
        if uow is not None:
            return await self._load_workspace(session=session, uow=uow)

        async with self._uow_factory() as managed_uow:
            return await self._load_workspace(session=session, uow=managed_uow)

    async def get_workspace_or_raise(
            self,
            session: Session,
            *,
            uow: Optional[IUnitOfWork] = None,
    ) -> Workspace:
        workspace = await self.get_workspace(session=session, uow=uow)
        if workspace is None:
            raise RuntimeError(f"会话[{session.id}]未绑定 workspace")
        return workspace

    async def resolve_current_run_id(
            self,
            session: Session,
            *,
            uow: Optional[IUnitOfWork] = None,
            require_bound: bool = False,
    ) -> Optional[str]:
        workspace = await self.get_workspace(session=session, uow=uow)
        if workspace is None:
            if require_bound:
                raise RuntimeError(f"会话[{session.id}]未绑定 workspace，无法解析 current_run_id")
            return None

        run_id = str(workspace.current_run_id or "").strip()
        if run_id:
            return run_id

        if require_bound:
            raise RuntimeError(f"会话[{session.id}]所属 workspace 未绑定 current_run_id")
        return None

    async def bind_run(
            self,
            workspace: Workspace,
            run_id: str,
            *,
            uow: Optional[IUnitOfWork] = None,
    ) -> Workspace:
        normalized_run_id = str(run_id or "").strip()
        if not normalized_run_id:
            raise ValueError("run_id 不能为空")

        workspace.current_run_id = normalized_run_id
        return await self.touch_workspace(workspace=workspace, uow=uow)

    async def ensure_environment(
            self,
            workspace: Workspace,
            *,
            sandbox_id: Optional[str] = None,
            task_id: Optional[str] = None,
            shell_session_id: Optional[str] = None,
            uow: Optional[IUnitOfWork] = None,
    ) -> Workspace:
        if sandbox_id is not None:
            workspace.sandbox_id = self._normalize_optional_text(sandbox_id)
        if task_id is not None:
            workspace.task_id = self._normalize_optional_text(task_id)
        if shell_session_id is not None:
            workspace.shell_session_id = self._normalize_optional_text(shell_session_id)

        return await self.touch_workspace(workspace=workspace, uow=uow)

    async def touch_workspace(
            self,
            workspace: Workspace,
            *,
            uow: Optional[IUnitOfWork] = None,
    ) -> Workspace:
        now = datetime.now()
        workspace.last_active_at = now
        workspace.updated_at = now

        if uow is not None:
            await uow.workspace.save(workspace=workspace)
            return workspace

        async with self._uow_factory() as managed_uow:
            await managed_uow.workspace.save(workspace=workspace)
        return workspace

    async def _ensure_workspace_with_uow(
            self,
            *,
            session: Session,
            uow: IUnitOfWork,
    ) -> Workspace:
        workspace = await self._load_workspace(session=session, uow=uow)
        if workspace is None:
            workspace = Workspace(session_id=session.id, user_id=session.user_id)
            await uow.workspace.save(workspace=workspace)
            _log_core(
                logging.INFO,
                "创建 workspace",
                session_id=session.id,
                workspace_id=workspace.id,
            )

        if workspace.user_id != session.user_id:
            workspace.user_id = session.user_id
            workspace.updated_at = datetime.now()
            await uow.workspace.save(workspace=workspace)

        if session.workspace_id != workspace.id:
            previous_workspace_id = str(session.workspace_id or "").strip() or "-"
            session.workspace_id = workspace.id
            session.updated_at = datetime.now()
            await uow.session.save(session=session)
            _log_core(
                logging.INFO,
                "绑定 workspace 到会话",
                session_id=session.id,
                workspace_id=workspace.id,
                previous_workspace_id=previous_workspace_id,
            )

        return workspace

    async def _load_workspace(
            self,
            *,
            session: Session,
            uow: IUnitOfWork,
    ) -> Optional[Workspace]:
        workspace_id = str(session.workspace_id or "").strip()
        if workspace_id:
            workspace = await uow.workspace.get_by_id_for_user(
                workspace_id=workspace_id,
                user_id=session.user_id,
            )
            if workspace is not None:
                return workspace
            _log_core(
                logging.WARNING,
                "workspace_id 未命中或归属不匹配",
                session_id=session.id,
                workspace_id=workspace_id,
            )
            return None

        get_by_session_id = getattr(uow.workspace, "get_by_session_id_for_user", None)
        if callable(get_by_session_id):
            workspace = await get_by_session_id(session_id=session.id, user_id=session.user_id)
            if workspace is not None and workspace_id and workspace.id != workspace_id:
                _log_core(
                    logging.WARNING,
                    "workspace 绑定不一致",
                    session_id=session.id,
                    workspace_id=workspace_id,
                    fallback_workspace_id=workspace.id,
                )
            elif workspace is not None and not workspace_id:
                _log_core(
                    logging.INFO,
                    "通过 session_id 恢复 workspace 绑定",
                    session_id=session.id,
                    workspace_id=workspace.id,
                )
            return workspace
        return None

    @staticmethod
    def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
        normalized = str(value or "").strip()
        return normalized or None
