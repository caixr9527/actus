#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace artifact revision 仓库协议。"""

from typing import List, Optional, Protocol

from app.domain.models import ArtifactDeliveryState, ArtifactRevisionIdentity, WorkspaceArtifactRevision


class WorkspaceArtifactRevisionRepository(Protocol):
    """Artifact revision 仓库协议定义。"""

    async def insert_or_get_existing(self, revision: WorkspaceArtifactRevision) -> WorkspaceArtifactRevision:
        """幂等写入 revision，冲突时返回既有 revision。"""
        ...

    async def append_revision_for_artifact(self, revision: WorkspaceArtifactRevision) -> WorkspaceArtifactRevision:
        """在仓储事务内分配 revision_no、追加 revision，并更新 current projection。"""
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

    async def get_by_identity(
            self,
            *,
            user_id: str,
            workspace_id: str,
            session_id: str,
            artifact_id: str,
            revision_id: str,
            content_hash: str,
    ) -> Optional[WorkspaceArtifactRevision]:
        """按 artifact revision 强身份锁读取。"""
        ...

    async def list_by_source_facts(
            self,
            *,
            user_id: str,
            workspace_id: str,
            session_id: str,
            source_event_id: str,
            source_fact_ids: List[str],
            tool_call_id: str | None = None,
            content_hash: str | None = None,
    ) -> List[WorkspaceArtifactRevision]:
        """按 source event/fact/tool/hash 强过滤查询 revision。"""
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

    async def update_delivery_state_by_identities(
            self,
            *,
            user_id: str,
            workspace_id: str,
            session_id: str,
            identities: List[ArtifactRevisionIdentity],
            delivery_state: ArtifactDeliveryState,
    ) -> List[WorkspaceArtifactRevision]:
        """按 revision identity 强过滤更新 delivery_state。"""
        ...
