#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace Runtime 统一环境服务。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from app.domain.models import Workspace, WorkspaceArtifact
from app.domain.models.tool_result import ToolResult
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionIdentity,
    ArtifactRevisionRegistrationCommand,
    ResolvedArtifactRevisionResult,
)
from app.domain.services.runtime.contracts.artifact_governance_ports import ArtifactLedgerPort
from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY,
    SandboxCapabilityProfile,
    validate_sandbox_capability_profile_payload,
)
from app.domain.services.workspace_runtime.manager import WorkspaceManager


logger = logging.getLogger(__name__)


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
            artifact_ledger: ArtifactLedgerPort | None = None,
    ) -> None:
        self._session_id = session_id
        self._user_id = str(user_id or "").strip()
        if not self._user_id:
            raise ValueError("WorkspaceRuntimeService 必须提供 user_id")
        self._uow_factory = uow_factory
        self._workspace_manager = WorkspaceManager(uow_factory=uow_factory)
        self._artifact_ledger = artifact_ledger

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

    async def record_sandbox_capability_profile(self, *, profile: SandboxCapabilityProfile) -> Workspace:
        workspace = await self.get_workspace_or_raise()
        if (
                profile.user_id != self._user_id
                or profile.session_id != self._session_id
                or profile.workspace_id != workspace.id
        ):
            raise ValueError("sandbox capability profile scope 与当前 workspace 不一致")
        next_environment_summary = dict(workspace.environment_summary or {})
        next_environment_summary[SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY] = profile.model_dump(mode="json")
        workspace.environment_summary = next_environment_summary
        workspace.last_active_at = datetime.now()
        workspace.updated_at = datetime.now()
        await self._save_workspace(workspace)
        return workspace

    async def get_sandbox_capability_profile(self) -> SandboxCapabilityProfile | None:
        workspace = await self.get_workspace()
        if workspace is None:
            return None
        raw_profile = dict(workspace.environment_summary or {}).get(SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY)
        if raw_profile is None:
            return None
        if not isinstance(raw_profile, dict):
            logger.warning(
                "sandbox_profile_invalid_payload",
                extra={
                    "user_id": self._user_id,
                    "session_id": self._session_id,
                    "workspace_id": workspace.id,
                    "reason_code": "sandbox_profile_payload_not_mapping",
                },
            )
            return None
        try:
            return validate_sandbox_capability_profile_payload(raw_profile)
        except ValidationError as exc:
            logger.warning(
                "sandbox_profile_invalid_payload",
                extra={
                    "user_id": self._user_id,
                    "session_id": self._session_id,
                    "workspace_id": workspace.id,
                    "reason_code": "sandbox_profile_payload_invalid",
                    "error_count": exc.error_count(),
                },
            )
            return None

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
            command: ArtifactRevisionRegistrationCommand,
    ) -> ResolvedArtifactRevisionResult:
        if self._artifact_ledger is None:
            raise RuntimeError("WorkspaceRuntimeService 未配置 ArtifactLedgerPort")
        return await self._artifact_ledger.register_revision(command=command)

    async def resolve_authoritative_artifact_revisions(
            self,
            *,
            paths: List[str],
    ) -> List[ResolvedArtifactRevisionResult]:
        if self._artifact_ledger is None:
            return []
        scope = await self._build_current_scope()
        if scope is None:
            return []
        return await self._artifact_ledger.resolve_authoritative_artifact_revisions(
            scope=scope,
            paths=paths,
        )

    async def mark_artifact_revisions_delivery_state(
            self,
            *,
            revisions: List[ArtifactRevisionIdentity],
            delivery_state: ArtifactDeliveryState,
    ) -> List[ResolvedArtifactRevisionResult]:
        if self._artifact_ledger is None:
            return []
        scope = await self._build_current_scope()
        if scope is None:
            return []
        return await self._artifact_ledger.mark_artifact_revisions_delivery_state(
            scope=scope,
            revisions=revisions,
            delivery_state=delivery_state,
        )

    async def mark_artifacts_delivery_state(
            self,
            *,
            paths: List[str],
            delivery_state: str,
    ) -> List[ResolvedArtifactRevisionResult]:
        resolved = await self.resolve_authoritative_artifact_revisions(paths=paths)
        identities = [
            ArtifactRevisionIdentity(
                artifact_id=item.artifact_id,
                revision_id=item.revision_id,
                content_hash=item.content_hash,
            )
            for item in resolved
        ]
        return await self.mark_artifact_revisions_delivery_state(
            revisions=identities,
            delivery_state=ArtifactDeliveryState(delivery_state),
        )

    async def resolve_authoritative_artifact_paths(
            self,
            *,
            paths: List[str],
    ) -> List[str]:
        return [
            item.path
            for item in await self.resolve_authoritative_artifact_revisions(paths=paths)
        ]

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

    async def _build_current_scope(self) -> AccessScopeResult | None:
        workspace = await self.get_workspace()
        if workspace is None:
            return None
        return AccessScopeResult(
            tenant_id=self._user_id,
            user_id=self._user_id,
            session_id=workspace.session_id,
            workspace_id=workspace.id,
            run_id=workspace.current_run_id,
            current_step_id=None,
        )
