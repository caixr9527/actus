#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/12 17:31
@Author : caixiaorong01@outlook.com
@File   : workspace_artifact_repository.py
"""
from typing import List, Optional, Protocol

from app.domain.models import WorkspaceArtifact


class WorkspaceArtifactRepository(Protocol):
    """WorkspaceArtifact 仓库协议定义。"""

    async def save(self, artifact: WorkspaceArtifact) -> None:
        """保存或更新工作区产物。"""
        ...

    async def list_by_workspace_id(self, workspace_id: str) -> List[WorkspaceArtifact]:
        """按工作区 ID 查询全部产物。"""
        ...

    async def list_by_workspace_id_and_paths(
            self,
            workspace_id: str,
            paths: List[str],
    ) -> List[WorkspaceArtifact]:
        """按工作区 ID + 路径列表批量查询产物。"""
        ...

    async def get_by_workspace_id_and_path(
            self,
            workspace_id: str,
            path: str,
    ) -> Optional[WorkspaceArtifact]:
        """按工作区 ID + 路径查询单个产物。"""
        ...

    async def update_delivery_state_by_workspace_id_and_paths(
            self,
            *,
            workspace_id: str,
            paths: List[str],
            delivery_state: str,
    ) -> List[WorkspaceArtifact]:
        """按工作区 ID + 路径列表批量更新交付状态。"""
        ...
