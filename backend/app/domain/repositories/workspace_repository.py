#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/12 17:31
@Author : caixiaorong01@outlook.com
@File   : workspace_repository.py
"""
from typing import Optional, Protocol

from app.domain.models import Workspace


class WorkspaceRepository(Protocol):
    """Workspace 仓库协议定义。"""

    async def save(self, workspace: Workspace) -> None:
        """保存或更新工作区。"""
        ...

    async def get_by_id(self, workspace_id: str) -> Optional[Workspace]:
        """按工作区 ID 查询。"""
        ...

    async def get_by_session_id(self, session_id: str) -> Optional[Workspace]:
        """按会话 ID 查询工作区。"""
        ...

    async def delete_by_id(self, workspace_id: str) -> None:
        """按工作区 ID 删除。"""
        ...
