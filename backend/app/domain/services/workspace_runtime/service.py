#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace Runtime 统一环境服务。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from app.domain.models import Workspace, WorkspaceArtifact
from app.domain.models.tool_result import ToolResult
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.domain.services.workspace_runtime.manager import WorkspaceManager


@dataclass(frozen=True)
class WorkspaceEnvironmentSnapshot:
    """Workspace 环境快照。"""

    workspace: Workspace
    artifacts: List[WorkspaceArtifact]


class WorkspaceRuntimeService:
    """统一管理 workspace 级环境读取与写回。"""

    _MAX_ENVIRONMENT_ITEMS = 8

    def __init__(
            self,
            *,
            session_id: str,
            uow_factory: Callable[[], IUnitOfWork],
            user_id: str,
    ) -> None:
        self._session_id = session_id
        self._user_id = str(user_id or "").strip()
        if not self._user_id:
            raise ValueError("WorkspaceRuntimeService 必须提供 user_id")
        self._uow_factory = uow_factory
        self._workspace_manager = WorkspaceManager(uow_factory=uow_factory)

    @property
    def session_id(self) -> str:
        return self._session_id

    async def get_workspace(self) -> Optional[Workspace]:
        async with self._uow_factory() as uow:
            return await uow.workspace.get_by_session_id_for_user(
                session_id=self._session_id,
                user_id=self._user_id,
            )

    async def get_workspace_or_raise(self) -> Workspace:
        workspace = await self.get_workspace()
        if workspace is None:
            raise RuntimeError(f"会话[{self._session_id}]未绑定 workspace")
        return workspace

    async def list_artifacts(self) -> List[WorkspaceArtifact]:
        workspace = await self.get_workspace()
        if workspace is None:
            return []
        async with self._uow_factory() as uow:
            return await uow.workspace_artifact.list_by_user_workspace_id(
                user_id=self._user_id,
                workspace_id=workspace.id,
            )

    async def build_environment_snapshot(self) -> Optional[WorkspaceEnvironmentSnapshot]:
        workspace = await self.get_workspace()
        if workspace is None:
            return None
        async with self._uow_factory() as uow:
            artifacts = await uow.workspace_artifact.list_by_user_workspace_id(
                user_id=self._user_id,
                workspace_id=workspace.id,
            )
        return WorkspaceEnvironmentSnapshot(workspace=workspace, artifacts=artifacts)

    async def ensure_shell_session_id(self) -> str:
        workspace = await self.get_workspace_or_raise()
        shell_session_id = str(workspace.shell_session_id or "").strip()
        if shell_session_id:
            return shell_session_id

        shell_session_id = f"shell-{workspace.id}"
        await self._workspace_manager.ensure_environment(
            workspace=workspace,
            shell_session_id=shell_session_id,
        )
        return shell_session_id

    async def record_shell_state(
            self,
            *,
            cwd: Optional[str] = None,
            shell_session_status: Optional[str] = None,
            latest_shell_result: Optional[Dict[str, Any]] = None,
    ) -> Workspace:
        workspace = await self.get_workspace_or_raise()

        next_environment_summary = dict(workspace.environment_summary or {})
        if cwd is not None:
            workspace.cwd = str(cwd or "").strip()
        if shell_session_status is not None:
            next_environment_summary["shell_session_status"] = str(shell_session_status or "").strip()
        if latest_shell_result is not None:
            next_environment_summary["latest_shell_result"] = dict(latest_shell_result)

        workspace.environment_summary = next_environment_summary
        workspace.last_active_at = datetime.now()
        workspace.updated_at = datetime.now()
        await self._save_workspace(workspace)
        return workspace

    async def get_latest_shell_tool_result(self) -> ToolResult:
        workspace = await self.get_workspace()
        if workspace is None:
            return ToolResult(
                success=False,
                message="workspace 未绑定",
                data={"console_records": []},
            )

        latest_shell_result = dict((workspace.environment_summary or {}).get("latest_shell_result") or {})
        console_records = latest_shell_result.get("console_records")
        if not isinstance(console_records, list):
            console_records = []

        return ToolResult(
            success=bool(console_records),
            message=str(latest_shell_result.get("message") or "").strip(),
            data={
                "output": str(latest_shell_result.get("output") or latest_shell_result.get("console") or "").strip(),
                "console_records": console_records,
            },
        )

    async def record_environment_summary(
            self,
            *,
            summary_patch: Dict[str, Any],
    ) -> Workspace:
        workspace = await self.get_workspace_or_raise()
        next_environment_summary = dict(workspace.environment_summary or {})
        next_environment_summary.update(summary_patch)
        workspace.environment_summary = next_environment_summary
        workspace.last_active_at = datetime.now()
        workspace.updated_at = datetime.now()
        await self._save_workspace(workspace)
        return workspace

    async def record_browser_snapshot(
            self,
            *,
            snapshot: Dict[str, Any],
    ) -> Workspace:
        workspace = await self.get_workspace_or_raise()
        cleaned_snapshot = self._clean_mapping(snapshot)
        if not cleaned_snapshot:
            return workspace

        existing_snapshot = dict(workspace.browser_snapshot or {})
        next_snapshot = (
            cleaned_snapshot
            if str(existing_snapshot.get("url") or "").strip() != str(cleaned_snapshot.get("url") or "").strip()
            and str(cleaned_snapshot.get("url") or "").strip()
            else {**existing_snapshot, **cleaned_snapshot}
        )
        workspace.browser_snapshot = next_snapshot
        workspace.last_active_at = datetime.now()
        workspace.updated_at = datetime.now()
        await self._save_workspace(workspace)
        return workspace

    async def record_search_results(
            self,
            *,
            query: str,
            candidate_links: List[Dict[str, Any]],
    ) -> Workspace:
        cleaned_links = [
            self._clean_mapping(item)
            for item in candidate_links
            if isinstance(item, dict)
        ]
        cleaned_links = [item for item in cleaned_links if item]
        return await self.record_environment_summary(
            summary_patch={
                "last_search_query": str(query or "").strip(),
                "candidate_links": cleaned_links[:self._MAX_ENVIRONMENT_ITEMS],
            }
        )

    async def record_fetched_page_summary(
            self,
            *,
            page_summary: Dict[str, Any],
    ) -> Workspace:
        workspace = await self.get_workspace_or_raise()
        cleaned_summary = self._clean_mapping(page_summary)
        if not cleaned_summary:
            return workspace

        existing_pages = [
            self._clean_mapping(item)
            for item in list(workspace.environment_summary.get("read_page_summaries") or [])
            if isinstance(item, dict)
        ]
        merged_pages: List[Dict[str, Any]] = [cleaned_summary]
        for item in existing_pages:
            if str(item.get("url") or "").strip() == str(cleaned_summary.get("url") or "").strip():
                continue
            merged_pages.append(item)
        return await self.record_environment_summary(
            summary_patch={
                "read_page_summaries": merged_pages[:self._MAX_ENVIRONMENT_ITEMS],
            }
        )

    async def record_file_tree_summary(
            self,
            *,
            summary_text: str,
    ) -> Workspace:
        workspace = await self.get_workspace_or_raise()
        normalized_summary = str(summary_text or "").strip()
        if not normalized_summary:
            return workspace

        existing_summaries = [
            str(item).strip()
            for item in list(workspace.environment_summary.get("file_tree_summary") or [])
            if str(item).strip()
        ]
        merged_summaries = [normalized_summary]
        for item in existing_summaries:
            if item == normalized_summary:
                continue
            merged_summaries.append(item)
        return await self.record_environment_summary(
            summary_patch={
                "file_tree_summary": merged_summaries[:self._MAX_ENVIRONMENT_ITEMS],
            }
        )

    async def record_changed_file(
            self,
            *,
            filepath: str,
    ) -> Workspace:
        workspace = await self.get_workspace_or_raise()
        normalized_path = str(filepath or "").strip()
        if not normalized_path:
            return workspace

        existing_files = [
            str(item).strip()
            for item in list(workspace.environment_summary.get("recent_changed_files") or [])
            if str(item).strip()
        ]
        merged_files = [normalized_path]
        for item in existing_files:
            if item == normalized_path:
                continue
            merged_files.append(item)
        return await self.record_environment_summary(
            summary_patch={
                "recent_changed_files": merged_files[:self._MAX_ENVIRONMENT_ITEMS],
            }
        )

    async def upsert_artifact(
            self,
            *,
            path: str,
            artifact_type: str,
            summary: str = "",
            source_capability: Optional[str] = None,
            source_step_id: Optional[str] = None,
            delivery_state: str = "",
            metadata: Optional[Dict[str, Any]] = None,
            record_as_changed_file: bool = True,
    ) -> WorkspaceArtifact:
        workspace = await self.get_workspace_or_raise()
        normalized_path = str(path or "").strip()
        if not normalized_path:
            raise ValueError("artifact path 不能为空")

        async with self._uow_factory() as uow:
            existing = await uow.workspace_artifact.get_by_user_workspace_id_and_path(
                user_id=self._user_id,
                workspace_id=workspace.id,
                path=normalized_path,
            )
            if existing is None:
                artifact = WorkspaceArtifact(
                    workspace_id=workspace.id,
                    user_id=workspace.user_id,
                    session_id=workspace.session_id,
                    run_id=workspace.current_run_id,
                    path=normalized_path,
                    artifact_type=str(artifact_type or "file").strip() or "file",
                    summary=str(summary or "").strip(),
                    source_step_id=source_step_id,
                    source_capability=source_capability,
                    delivery_state=str(delivery_state or "").strip(),
                    origin=DataOrigin.AGENT_GENERATED,
                    trust_level=DataTrustLevel.AGENT_GENERATED,
                    privacy_level=PrivacyLevel.PRIVATE,
                    retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
                    metadata=dict(metadata or {}),
                )
            else:
                artifact = existing.model_copy(deep=True)
                artifact.user_id = artifact.user_id or workspace.user_id
                artifact.session_id = artifact.session_id or workspace.session_id
                artifact.run_id = artifact.run_id or workspace.current_run_id
                artifact.artifact_type = str(artifact_type or artifact.artifact_type or "file").strip() or "file"
                artifact.summary = str(summary or artifact.summary or "").strip()
                artifact.source_step_id = source_step_id or artifact.source_step_id
                artifact.source_capability = source_capability or artifact.source_capability
                artifact.delivery_state = str(delivery_state or artifact.delivery_state or "").strip()
                artifact.metadata = {
                    **dict(artifact.metadata or {}),
                    **dict(metadata or {}),
                }
                artifact.updated_at = datetime.now()
            await uow.workspace_artifact.save(artifact=artifact)
        if record_as_changed_file:
            await self.record_changed_file(filepath=normalized_path)
        return artifact

    async def mark_artifacts_delivery_state(
            self,
            *,
            paths: List[str],
            delivery_state: str,
    ) -> List[WorkspaceArtifact]:
        workspace = await self.get_workspace()
        normalized_paths = [
            str(path or "").strip()
            for path in paths
            if str(path or "").strip()
        ]
        if workspace is None or len(normalized_paths) == 0:
            return []

        updated_artifacts: List[WorkspaceArtifact] = []
        async with self._uow_factory() as uow:
            existing_artifacts = await uow.workspace_artifact.list_by_user_workspace_id_and_paths(
                user_id=self._user_id,
                workspace_id=workspace.id,
                paths=normalized_paths,
            )
            if len(existing_artifacts) == 0:
                return []
            updated_artifacts = await uow.workspace_artifact.update_delivery_state_by_user_workspace_id_and_paths(
                user_id=self._user_id,
                workspace_id=workspace.id,
                paths=normalized_paths,
                delivery_state=delivery_state,
            )
        artifact_by_path = {
            str(artifact.path or "").strip(): artifact
            for artifact in updated_artifacts
            if str(artifact.path or "").strip()
        }
        return [
            artifact_by_path[path]
            for path in normalized_paths
            if path in artifact_by_path
        ]

    async def resolve_authoritative_artifact_paths(
            self,
            *,
            paths: List[str],
    ) -> List[str]:
        workspace = await self.get_workspace()
        normalized_paths: List[str] = []
        for path in paths:
            normalized_path = str(path or "").strip()
            if not normalized_path or normalized_path in normalized_paths:
                continue
            normalized_paths.append(normalized_path)
        if workspace is None or len(normalized_paths) == 0:
            return []

        authoritative_paths: List[str] = []
        async with self._uow_factory() as uow:
            artifacts = await uow.workspace_artifact.list_by_user_workspace_id_and_paths(
                user_id=self._user_id,
                workspace_id=workspace.id,
                paths=normalized_paths,
            )
        indexed_paths = {
            str(artifact.path or "").strip()
            for artifact in artifacts
            if str(artifact.path or "").strip()
        }
        for path in normalized_paths:
            if path in indexed_paths:
                authoritative_paths.append(path)
        return authoritative_paths

    @staticmethod
    def _clean_mapping(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in dict(payload or {}).items()
            if value not in (None, "", [], {})
        }

    async def _save_workspace(self, workspace: Workspace) -> None:
        async with self._uow_factory() as uow:
            await uow.workspace.save(workspace=workspace)
