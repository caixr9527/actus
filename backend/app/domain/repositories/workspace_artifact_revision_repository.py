#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace artifact revision 仓库协议。"""

from typing import List, Optional, Protocol

from app.domain.models import WorkspaceArtifactRevision


class WorkspaceArtifactRevisionRepository(Protocol):
    """Artifact revision 仓库协议定义。"""

    async def insert_or_get_existing(self, revision: WorkspaceArtifactRevision) -> WorkspaceArtifactRevision:
        """幂等写入 revision，冲突时返回既有 revision。"""
        ...

    async def get_by_user_workspace_revision_id(
            self,
            *,
            user_id: str,
            workspace_id: str,
            revision_id: str,
    ) -> Optional[WorkspaceArtifactRevision]:
        """按用户 + workspace + revision_id 强过滤查询。"""
        ...

    async def list_by_user_workspace_artifact_id(
            self,
            *,
            user_id: str,
            workspace_id: str,
            artifact_id: str,
    ) -> List[WorkspaceArtifactRevision]:
        """按用户 + workspace + artifact_id 查询全部 revision。"""
        ...
