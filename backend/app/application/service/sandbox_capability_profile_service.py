#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox capability profile 刷新编排应用服务。"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Callable, Sequence, Type

from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.application.service.sandbox_capability_profile_normalizer import (
    normalize_sandbox_raw_profile,
)
from app.application.service.sandbox_capability_profile_sanitizer import (
    sanitize_sandbox_profile_message,
)
from app.domain.external import Sandbox
from app.domain.models import ToolResult, Workspace
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.data_access_contract import (
    DataAccessAction,
    DataResourceKind,
)
from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    RuntimeToolCapabilitySnapshot,
    SandboxCapabilityItem,
    SandboxCapabilityKind,
    SandboxCapabilityProbePayload,
    SandboxCapabilityProfile,
    SandboxCapabilityPromptSummary,
    SandboxCapabilityRefreshError,
    SandboxCapabilityStatus,
    SandboxProfileRefreshReason,
    SandboxResourceLimits,
    build_sandbox_capability_profile_from_draft,
)
from app.domain.services.runtime.contracts.sandbox_capability_profile_ports import (
    RuntimeToolSnapshotRecorderPort,
    SandboxCapabilityProfileRefresherPort,
)
from app.domain.services.workspace_runtime import WorkspaceRuntimeService
from app.application.service.artifact_ledger_service import ArtifactLedgerService


logger = logging.getLogger(__name__)


class SandboxCapabilityProfileScopeMismatchError(RuntimeError):
    """refresher port 入参与当前权威 workspace 绑定不一致。"""


class SandboxCapabilityProfileService(
    SandboxCapabilityProfileRefresherPort,
    RuntimeToolSnapshotRecorderPort,
):
    """统一刷新与记录 sandbox capability profile。"""

    _DEFAULT_TTL = timedelta(minutes=30)

    def __init__(
            self,
            *,
            uow_factory: Callable[[], IUnitOfWork],
            sandbox_cls: Type[Sandbox],
            access_control_service: RuntimeAccessControlService | None = None,
            ttl: timedelta | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._sandbox_cls = sandbox_cls
        self._access_control_service = access_control_service or RuntimeAccessControlService(
            uow_factory=uow_factory,
        )
        self._ttl = ttl or self._DEFAULT_TTL

    async def refresh_after_sandbox_bound(
            self,
            *,
            user_id: str,
            session_id: str,
            workspace_id: str,
            run_id: str,
            sandbox_id: str,
            task_id: str,
            reason: SandboxProfileRefreshReason,
    ) -> None:
        scope: AccessScopeResult | None = None
        workspace: Workspace | None = None
        try:
            scope, workspace = await self._load_refresh_scope(
                user_id=user_id,
                session_id=session_id,
            )
            self._assert_bound_snapshot_matches_workspace(
                scope=scope,
                workspace=workspace,
                workspace_id=workspace_id,
                run_id=run_id,
                sandbox_id=sandbox_id,
                task_id=task_id,
            )
            await self._refresh_profile_for_scope(
                scope=scope,
                workspace=workspace,
                reason=reason,
                section_kinds=None,
                existing_profile=await self.get_profile(user_id=user_id, session_id=session_id),
            )
        except SandboxCapabilityProfileScopeMismatchError as exc:
            await self._record_refresh_mismatch_failure(
                user_id=user_id,
                session_id=session_id,
                workspace_id=workspace_id,
                run_id=run_id,
                sandbox_id=sandbox_id,
                reason=reason,
                exc=exc,
            )
        except Exception as exc:
            await self._record_refresh_execution_failure(
                user_id=user_id,
                session_id=session_id,
                scope=scope,
                workspace=workspace,
                reason=reason,
                exc=exc,
            )

    async def refresh_profile(
            self,
            *,
            user_id: str,
            session_id: str,
            reason: SandboxProfileRefreshReason,
            section_kinds: list[SandboxCapabilityKind] | None = None,
    ) -> SandboxCapabilityProfile:
        """刷新 profile 环境事实；该操作不推进 runtime run/session 状态。"""

        scope, workspace = await self._load_refresh_scope(user_id=user_id, session_id=session_id)
        existing_profile = await self.get_profile(user_id=user_id, session_id=session_id)
        return await self._refresh_profile_for_scope(
            scope=scope,
            workspace=workspace,
            reason=reason,
            section_kinds=section_kinds,
            existing_profile=existing_profile,
        )

    async def get_profile(
            self,
            *,
            user_id: str,
            session_id: str,
    ) -> SandboxCapabilityProfile | None:
        workspace_runtime = self._build_workspace_runtime(user_id=user_id, session_id=session_id)
        return await workspace_runtime.get_sandbox_capability_profile()

    async def ensure_fresh_profile(
            self,
            *,
            user_id: str,
            session_id: str,
            reason: SandboxProfileRefreshReason,
    ) -> SandboxCapabilityProfile:
        scope, workspace = await self._load_refresh_scope(user_id=user_id, session_id=session_id)
        existing_profile = await self.get_profile(user_id=user_id, session_id=session_id)
        if existing_profile is not None and not self._is_expired(existing_profile):
            self._log_refresh_skipped(
                profile=existing_profile,
                reason=reason,
                reason_code="sandbox_profile_ttl_not_expired",
            )
            return existing_profile
        try:
            return await self._refresh_profile_for_scope(
                scope=scope,
                workspace=workspace,
                reason=reason,
                section_kinds=None,
                existing_profile=existing_profile,
            )
        except Exception as exc:
            if existing_profile is not None and not self._is_expired(existing_profile):
                return existing_profile
            return await self._write_minimal_unknown_profile(
                scope=scope,
                workspace=workspace,
                reason=reason,
                reason_code="sandbox_profile_prompt_refresh_failed",
                stage="refresh",
                message=self._safe_error_message(exc),
            )

    async def record_runtime_tool_snapshot(
            self,
            *,
            user_id: str,
            session_id: str,
            snapshot: RuntimeToolCapabilitySnapshot,
    ) -> SandboxCapabilityProfile:
        try:
            scope = await self._access_control_service.resolve_session_scope(
                user_id=user_id,
                session_id=session_id,
            )
            workspace = await self._load_owned_workspace_from_scope(scope=scope, session_id=session_id)
            self._assert_snapshot_scope_is_writable(scope=scope, workspace=workspace)
            existing_profile = await self.get_profile(user_id=user_id, session_id=session_id)
            if existing_profile is None:
                raise RuntimeError("sandbox profile runtime tool snapshot 缺少 existing profile")
            profile = self._merge_runtime_tool_snapshot(
                scope=scope,
                workspace=workspace,
                existing_profile=existing_profile,
                snapshot=snapshot,
            )
            if snapshot.items and not profile.prompt_summary.available_tools:
                logger.error(
                    "sandbox_runtime_tool_snapshot_invalid",
                    extra={
                        "user_id": scope.user_id,
                        "session_id": str(scope.session_id),
                        "workspace_id": str(scope.workspace_id),
                        "run_id": scope.run_id,
                        "runtime_tool_count": len(list(snapshot.items or [])),
                        "reason_code": "sandbox_runtime_tool_snapshot_invalid",
                    },
                )
                raise RuntimeError("runtime tools 非空但 sandbox prompt summary available_tools 为空")
            await self._record_profile(profile=profile)
            self._log_runtime_tool_snapshot_recorded(profile=profile)
            return profile
        except Exception:
            logger.warning(
                "sandbox_profile_runtime_tool_snapshot_rejected",
                extra={
                    "user_id": user_id,
                    "session_id": session_id,
                    "reason_code": "sandbox_profile_runtime_tool_snapshot_rejected",
                },
            )
            raise

    async def _refresh_profile_for_scope(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
            reason: SandboxProfileRefreshReason,
            section_kinds: list[SandboxCapabilityKind] | None,
            existing_profile: SandboxCapabilityProfile | None,
    ) -> SandboxCapabilityProfile:
        started_at = time.perf_counter()
        section_values = self._normalize_section_kinds(section_kinds)
        self._log_refresh_started(
            scope=scope,
            workspace=workspace,
            reason=reason,
            section_kinds=section_values,
        )
        sandbox = await self._get_bound_sandbox(workspace=workspace)
        probe_result = await sandbox.probe_capabilities()
        if not probe_result.success:
            return await self._handle_probe_failure(
                scope=scope,
                workspace=workspace,
                reason=reason,
                section_kinds=section_kinds,
                existing_profile=existing_profile,
                probe_result=probe_result,
                started_at=started_at,
            )
        try:
            profile = self._build_profile_from_probe(
                scope=scope,
                workspace=workspace,
                reason=reason,
                probe_payload=probe_result.data,
                existing_profile=existing_profile,
                section_kinds=section_kinds,
                last_refresh_error=None,
            )
        except Exception as exc:
            return await self._handle_normalize_failure(
                scope=scope,
                workspace=workspace,
                reason=reason,
                section_kinds=section_kinds,
                existing_profile=existing_profile,
                exc=exc,
            )
        await self._record_profile(profile=profile)
        self._log_prompt_summary_built(profile=profile)
        self._log_refresh_finished(
            profile=profile,
            section_kinds=section_values,
            elapsed_ms=self._elapsed_ms(started_at),
        )
        return profile

    async def _handle_probe_failure(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
            reason: SandboxProfileRefreshReason,
            section_kinds: list[SandboxCapabilityKind] | None,
            existing_profile: SandboxCapabilityProfile | None,
            probe_result: ToolResult[SandboxCapabilityProbePayload],
            started_at: float,
    ) -> SandboxCapabilityProfile:
        probe_payload = probe_result.data or SandboxCapabilityProbePayload(
            reason_code="sandbox_profile_probe_failed",
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        )
        reason_code = str(probe_payload.reason_code or "sandbox_profile_probe_failed").strip()
        logger.warning(
            "sandbox_profile_probe_failed",
            extra={
                "user_id": scope.user_id,
                "session_id": scope.session_id,
                "workspace_id": workspace.id,
                "run_id": scope.run_id,
                "sandbox_id": workspace.sandbox_id,
                "refresh_reason": reason.value,
                "section_kinds": self._normalize_section_kinds(section_kinds),
                "reason_code": reason_code,
                "health_status": probe_payload.probe_status.value,
                "elapsed_ms": self._elapsed_ms(started_at),
            },
        )
        refresh_error = self._build_refresh_error(
            reason_code=reason_code,
            stage="probe",
            message="sandbox capability probe failed",
            probe_status=probe_payload.probe_status,
        )
        if section_kinds and existing_profile is not None:
            profile = self._copy_profile_with_error(existing_profile, reason=reason, refresh_error=refresh_error)
            await self._record_profile(profile=profile)
            return profile
        profile = self._build_unknown_profile(
            scope=scope,
            workspace=workspace,
            reason=reason,
            refresh_error=refresh_error,
            probe_status=probe_payload.probe_status,
        )
        await self._record_profile(profile=profile)
        return profile

    async def _handle_normalize_failure(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
            reason: SandboxProfileRefreshReason,
            section_kinds: list[SandboxCapabilityKind] | None,
            existing_profile: SandboxCapabilityProfile | None,
            exc: Exception,
    ) -> SandboxCapabilityProfile:
        refresh_error = self._build_refresh_error(
            reason_code="sandbox_profile_normalize_failed",
            stage="normalize",
            message=self._safe_error_message(exc),
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        )
        if section_kinds and existing_profile is not None:
            profile = self._copy_profile_with_error(existing_profile, reason=reason, refresh_error=refresh_error)
            await self._record_profile(profile=profile)
            return profile
        profile = self._build_unknown_profile(
            scope=scope,
            workspace=workspace,
            reason=reason,
            refresh_error=refresh_error,
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        )
        await self._record_profile(profile=profile)
        return profile

    async def _write_minimal_unknown_profile(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
            reason: SandboxProfileRefreshReason,
            reason_code: str,
            stage: str,
            message: str,
    ) -> SandboxCapabilityProfile:
        refresh_error = self._build_refresh_error(
            reason_code=reason_code,
            stage=stage,
            message=message,
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        )
        profile = self._build_unknown_profile(
            scope=scope,
            workspace=workspace,
            reason=reason,
            refresh_error=refresh_error,
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        )
        await self._record_profile(profile=profile)
        return profile

    async def _load_refresh_scope(
            self,
            *,
            user_id: str,
            session_id: str,
    ) -> tuple[AccessScopeResult, Workspace]:
        scope = await self._access_control_service.assert_sandbox_access(
            user_id=user_id,
            session_id=session_id,
            resource_kind=DataResourceKind.SANDBOX_SHELL,
            action=DataAccessAction.READ,
        )
        workspace = await self._load_owned_workspace_from_scope(scope=scope, session_id=session_id)
        return scope, workspace

    async def _load_owned_workspace_from_scope(
            self,
            *,
            scope: AccessScopeResult,
            session_id: str,
    ) -> Workspace:
        workspace_id = str(scope.workspace_id or "").strip()
        if not workspace_id:
            raise RuntimeError("sandbox profile scope 缺少 workspace_id")
        async with self._uow_factory() as uow:
            workspace = await uow.workspace.get_by_id_for_user(
                workspace_id=workspace_id,
                user_id=scope.user_id,
            )
        if workspace is None or workspace.session_id != session_id:
            raise RuntimeError("sandbox profile workspace scope 不一致")
        return workspace

    def _assert_bound_snapshot_matches_workspace(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
            workspace_id: str,
            run_id: str,
            sandbox_id: str,
            task_id: str,
    ) -> None:
        if (
                str(scope.workspace_id or "").strip() != str(workspace_id or "").strip()
                or str(scope.run_id or "").strip() != str(run_id or "").strip()
                or str(workspace.current_run_id or "").strip() != str(run_id or "").strip()
                or str(workspace.sandbox_id or "").strip() != str(sandbox_id or "").strip()
                or str(workspace.task_id or "").strip() != str(task_id or "").strip()
        ):
            raise SandboxCapabilityProfileScopeMismatchError("sandbox profile bound snapshot 与当前 workspace 不一致")

    def _assert_snapshot_scope_is_writable(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
    ) -> None:
        if not str(scope.workspace_id or "").strip():
            raise RuntimeError("sandbox profile runtime tool snapshot 缺少 workspace")
        if not str(scope.run_id or "").strip():
            raise RuntimeError("sandbox profile runtime tool snapshot 缺少 current run")
        if str(workspace.current_run_id or "").strip() != str(scope.run_id or "").strip():
            raise RuntimeError("sandbox profile runtime tool snapshot run scope 不一致")
        if not str(workspace.sandbox_id or "").strip():
            raise RuntimeError("sandbox profile runtime tool snapshot 缺少 sandbox")

    async def _get_bound_sandbox(self, *, workspace: Workspace) -> Sandbox:
        sandbox_id = str(workspace.sandbox_id or "").strip()
        if not sandbox_id:
            raise RuntimeError("workspace 未绑定 sandbox")
        sandbox = await self._sandbox_cls.get(id=sandbox_id)
        if sandbox is None:
            raise RuntimeError("sandbox 不存在或不可用")
        return sandbox

    def _build_profile_from_probe(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
            reason: SandboxProfileRefreshReason,
            probe_payload: SandboxCapabilityProbePayload | None,
            existing_profile: SandboxCapabilityProfile | None,
            section_kinds: list[SandboxCapabilityKind] | None,
            last_refresh_error: SandboxCapabilityRefreshError | None,
    ) -> SandboxCapabilityProfile:
        raw_profile = dict((probe_payload or SandboxCapabilityProbePayload()).raw_profile or {})
        normalized_raw_profile = normalize_sandbox_raw_profile(raw_profile)
        generated_at = datetime.now()
        capabilities = normalized_raw_profile.capabilities
        resource_limits = normalized_raw_profile.resource_limits
        disabled_capabilities = normalized_raw_profile.disabled_capabilities
        confirmation_required = normalized_raw_profile.confirmation_required_capabilities
        health_status = normalized_raw_profile.health_status
        cwd = normalized_raw_profile.cwd
        runtime_snapshot = existing_profile.runtime_tool_capabilities if existing_profile is not None else (
            RuntimeToolCapabilitySnapshot()
        )

        if section_kinds and existing_profile is not None:
            selected_kinds = set(section_kinds)
            capabilities = self._merge_capabilities_by_kind(
                existing_capabilities=existing_profile.capabilities,
                refreshed_capabilities=capabilities,
                selected_kinds=selected_kinds,
            )
            if SandboxCapabilityKind.RESOURCE_LIMIT not in selected_kinds:
                resource_limits = existing_profile.resource_limits
            disabled_capabilities = existing_profile.disabled_capabilities
            confirmation_required = existing_profile.confirmation_required_capabilities

        prompt_summary = self._build_prompt_summary(
            health_status=health_status,
            cwd=cwd,
            capabilities=capabilities,
            resource_limits=resource_limits,
            generated_at=generated_at,
            sandbox_profile_stale=False,
        )
        draft = {
            "profile_id": str(uuid.uuid4()),
            "user_id": scope.user_id,
            "session_id": str(scope.session_id or ""),
            "workspace_id": workspace.id,
            "run_id": scope.run_id,
            "sandbox_id": str(workspace.sandbox_id or "").strip(),
            "generated_at": generated_at,
            "expires_at": generated_at + self._ttl,
            "refresh_reason": reason,
            "health_status": health_status,
            "cwd": cwd,
            "capabilities": capabilities,
            "resource_limits": resource_limits,
            "disabled_capabilities": disabled_capabilities,
            "confirmation_required_capabilities": confirmation_required,
            "runtime_tool_capabilities": runtime_snapshot,
            "prompt_summary": prompt_summary,
            "last_refresh_error": last_refresh_error,
        }
        return build_sandbox_capability_profile_from_draft(draft)

    def _build_unknown_profile(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
            reason: SandboxProfileRefreshReason,
            refresh_error: SandboxCapabilityRefreshError,
            probe_status: SandboxCapabilityStatus,
    ) -> SandboxCapabilityProfile:
        generated_at = datetime.now()
        health_status = (
            probe_status
            if probe_status in {SandboxCapabilityStatus.DEGRADED, SandboxCapabilityStatus.UNKNOWN}
            else SandboxCapabilityStatus.UNKNOWN
        )
        resource_limits = SandboxResourceLimits(network_policy="unknown")
        prompt_summary = self._build_prompt_summary(
            health_status=health_status,
            cwd="",
            capabilities=[],
            resource_limits=resource_limits,
            generated_at=generated_at,
            sandbox_profile_stale=True,
        )
        return build_sandbox_capability_profile_from_draft({
            "profile_id": str(uuid.uuid4()),
            "user_id": scope.user_id,
            "session_id": str(scope.session_id or ""),
            "workspace_id": workspace.id,
            "run_id": scope.run_id,
            "sandbox_id": str(workspace.sandbox_id or "").strip(),
            "generated_at": generated_at,
            "expires_at": generated_at + self._ttl,
            "refresh_reason": reason,
            "health_status": health_status,
            "cwd": "",
            "capabilities": [],
            "resource_limits": resource_limits,
            "disabled_capabilities": [],
            "confirmation_required_capabilities": [],
            "runtime_tool_capabilities": RuntimeToolCapabilitySnapshot(),
            "prompt_summary": prompt_summary,
            "last_refresh_error": refresh_error,
        })

    def _copy_profile_with_error(
            self,
            existing_profile: SandboxCapabilityProfile,
            *,
            reason: SandboxProfileRefreshReason,
            refresh_error: SandboxCapabilityRefreshError,
    ) -> SandboxCapabilityProfile:
        payload = existing_profile.model_dump()
        payload.update({
            "generated_at": datetime.now(),
            "refresh_reason": reason,
            "last_refresh_error": refresh_error,
        })
        payload.pop("profile_hash", None)
        return build_sandbox_capability_profile_from_draft(payload)

    def _merge_runtime_tool_snapshot(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
            existing_profile: SandboxCapabilityProfile | None,
            snapshot: RuntimeToolCapabilitySnapshot,
    ) -> SandboxCapabilityProfile:
        payload = existing_profile.model_dump()
        prompt_summary = existing_profile.prompt_summary.model_dump()
        runtime_tool_families = sorted({
            str(item.tool_family or "").strip()
            for item in list(snapshot.items or [])
            if item.enabled and str(item.tool_family or "").strip()
        })
        if runtime_tool_families:
            prompt_summary["available_tools"] = sorted(set([
                *list(prompt_summary.get("available_tools") or []),
                *runtime_tool_families,
            ]))
            prompt_summary["sandbox_profile_stale"] = False
        payload.update({
            "runtime_tool_capabilities": snapshot,
            "prompt_summary": prompt_summary,
            "last_refresh_error": None,
        })
        payload.pop("profile_hash", None)
        return build_sandbox_capability_profile_from_draft(payload)

    async def _record_profile(self, *, profile: SandboxCapabilityProfile) -> None:
        workspace_runtime = self._build_workspace_runtime(
            user_id=profile.user_id,
            session_id=profile.session_id,
        )
        await workspace_runtime.record_sandbox_capability_profile(profile=profile)

    async def _record_refresh_mismatch_failure(
            self,
            *,
            user_id: str,
            session_id: str,
            workspace_id: str,
            run_id: str,
            sandbox_id: str,
            reason: SandboxProfileRefreshReason,
            exc: Exception,
    ) -> None:
        refresh_error = self._build_refresh_error(
            reason_code="sandbox_profile_refresh_failed",
            stage="refresh",
            message=self._safe_error_message(exc),
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        )
        try:
            existing_profile = await self.get_profile(user_id=user_id, session_id=session_id)
            if existing_profile is not None:
                profile = self._copy_profile_with_error(
                    existing_profile,
                    reason=reason,
                    refresh_error=refresh_error,
                )
                await self._record_profile(profile=profile)
        except Exception:
            pass
        logger.warning(
            "sandbox_profile_refresh_failed",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "workspace_id": workspace_id,
                "run_id": run_id,
                "sandbox_id": sandbox_id,
                "refresh_reason": reason.value,
                "reason_code": refresh_error.reason_code,
            },
        )

    async def _record_refresh_execution_failure(
            self,
            *,
            user_id: str,
            session_id: str,
            scope: AccessScopeResult | None,
            workspace: Workspace | None,
            reason: SandboxProfileRefreshReason,
            exc: Exception,
    ) -> None:
        logger.warning(
            "sandbox_profile_refresh_failed",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "workspace_id": workspace.id if workspace is not None else None,
                "run_id": scope.run_id if scope is not None else None,
                "sandbox_id": workspace.sandbox_id if workspace is not None else None,
                "refresh_reason": reason.value,
                "reason_code": "sandbox_profile_refresh_failed",
                "error_message": self._safe_error_message(exc),
            },
        )
        if scope is None or workspace is None:
            return
        existing_profile = await self.get_profile(user_id=user_id, session_id=session_id)
        refresh_error = self._build_refresh_error(
            reason_code="sandbox_profile_refresh_failed",
            stage="refresh",
            message="sandbox capability profile refresh failed",
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        )
        if existing_profile is not None:
            profile = self._copy_profile_with_error(
                existing_profile,
                reason=reason,
                refresh_error=refresh_error,
            )
        else:
            profile = self._build_unknown_profile(
                scope=scope,
                workspace=workspace,
                reason=reason,
                refresh_error=refresh_error,
                probe_status=SandboxCapabilityStatus.UNKNOWN,
            )
        await self._record_profile(profile=profile)

    def _build_workspace_runtime(self, *, user_id: str, session_id: str) -> WorkspaceRuntimeService:
        return WorkspaceRuntimeService(
            user_id=user_id,
            session_id=session_id,
            uow_factory=self._uow_factory,
            artifact_ledger=ArtifactLedgerService(uow_factory=self._uow_factory),
        )

    def _build_refresh_error(
            self,
            *,
            reason_code: str,
            stage: str,
            message: str,
            probe_status: SandboxCapabilityStatus,
    ) -> SandboxCapabilityRefreshError:
        return SandboxCapabilityRefreshError(
            reason_code=reason_code,
            stage=stage,
            message=self._sanitize_message(message),
            occurred_at=datetime.now(),
            retryable=True,
            probe_status=probe_status,
        )

    def _build_prompt_summary(
            self,
            *,
            health_status: SandboxCapabilityStatus,
            cwd: str,
            capabilities: Sequence[SandboxCapabilityItem],
            resource_limits: SandboxResourceLimits,
            generated_at: datetime,
            sandbox_profile_stale: bool,
    ) -> SandboxCapabilityPromptSummary:
        available_runtime: dict[str, str] = {}
        available_tools: list[str] = []
        unavailable_capabilities: list[str] = []
        requires_confirmation: list[str] = []
        for capability in capabilities:
            if capability.kind == SandboxCapabilityKind.PROXY:
                continue
            if capability.kind == SandboxCapabilityKind.NETWORK and capability.name != "network_egress":
                continue
            if capability.status == SandboxCapabilityStatus.AVAILABLE:
                if capability.kind in {
                    SandboxCapabilityKind.PYTHON,
                    SandboxCapabilityKind.NODE,
                    SandboxCapabilityKind.OS,
                }:
                    available_runtime[capability.kind.value] = capability.version or capability.name
                else:
                    available_tools.append(capability.kind.value)
            if capability.status in {
                SandboxCapabilityStatus.UNAVAILABLE,
                SandboxCapabilityStatus.DEGRADED,
                SandboxCapabilityStatus.UNKNOWN,
            }:
                unavailable_capabilities.append(capability.name)
            if capability.requires_confirmation:
                requires_confirmation.append(capability.name)
        summary = SandboxCapabilityPromptSummary(
            health_status=health_status,
            cwd=cwd,
            available_runtime=available_runtime,
            available_tools=sorted(set(available_tools)),
            unavailable_capabilities=sorted(set(unavailable_capabilities)),
            requires_confirmation=sorted(set(requires_confirmation)),
            resource_limits=self._build_prompt_resource_limits(resource_limits),
            generated_at=generated_at,
            sandbox_profile_stale=sandbox_profile_stale,
        )
        return summary

    @staticmethod
    def _build_prompt_resource_limits(resource_limits: SandboxResourceLimits) -> dict[str, object]:
        payload: dict[str, object] = {}
        if resource_limits.writable_dirs:
            payload["writable_dirs"] = resource_limits.writable_dirs
        if resource_limits.max_command_seconds is not None:
            payload["max_command_seconds"] = resource_limits.max_command_seconds
        if resource_limits.max_file_read_bytes is not None:
            payload["max_file_read_bytes"] = resource_limits.max_file_read_bytes
        return payload

    @staticmethod
    def _merge_capabilities_by_kind(
            *,
            existing_capabilities: Sequence[SandboxCapabilityItem],
            refreshed_capabilities: Sequence[SandboxCapabilityItem],
            selected_kinds: set[SandboxCapabilityKind],
    ) -> list[SandboxCapabilityItem]:
        preserved = [
            capability
            for capability in existing_capabilities
            if capability.kind not in selected_kinds
        ]
        refreshed = [
            capability
            for capability in refreshed_capabilities
            if capability.kind in selected_kinds
        ]
        return [*preserved, *refreshed]

    @staticmethod
    def _normalize_section_kinds(section_kinds: list[SandboxCapabilityKind] | None) -> list[str]:
        if section_kinds is None:
            return []
        return sorted({
            kind.value if isinstance(kind, SandboxCapabilityKind) else str(kind)
            for kind in section_kinds
        })

    @staticmethod
    def _is_expired(profile: SandboxCapabilityProfile) -> bool:
        if profile.expires_at is None:
            return True
        return profile.expires_at <= datetime.now(profile.expires_at.tzinfo)

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)

    @staticmethod
    def _sanitize_message(message: str) -> str:
        return sanitize_sandbox_profile_message(message)

    def _safe_error_message(self, exc: Exception) -> str:
        return self._sanitize_message(str(exc.__class__.__name__) or "sandbox profile refresh failed")

    def _log_refresh_started(
            self,
            *,
            scope: AccessScopeResult,
            workspace: Workspace,
            reason: SandboxProfileRefreshReason,
            section_kinds: list[str],
    ) -> None:
        logger.info(
            "sandbox_profile_refresh_started",
            extra={
                "user_id": scope.user_id,
                "session_id": scope.session_id,
                "workspace_id": workspace.id,
                "run_id": scope.run_id,
                "sandbox_id": workspace.sandbox_id,
                "refresh_reason": reason.value,
                "section_kinds": section_kinds,
            },
        )

    @staticmethod
    def _log_refresh_skipped(
            *,
            profile: SandboxCapabilityProfile,
            reason: SandboxProfileRefreshReason,
            reason_code: str,
    ) -> None:
        logger.info(
            "sandbox_profile_refresh_skipped",
            extra={
                "user_id": profile.user_id,
                "session_id": profile.session_id,
                "workspace_id": profile.workspace_id,
                "run_id": profile.run_id,
                "sandbox_id": profile.sandbox_id,
                "refresh_reason": reason.value,
                "health_status": profile.health_status.value,
                "profile_hash_prefix": profile.profile_hash[:16],
                "reason_code": reason_code,
            },
        )

    @staticmethod
    def _log_prompt_summary_built(*, profile: SandboxCapabilityProfile) -> None:
        logger.info(
            "sandbox_profile_prompt_summary_built",
            extra={
                "user_id": profile.user_id,
                "session_id": profile.session_id,
                "workspace_id": profile.workspace_id,
                "run_id": profile.run_id,
                "sandbox_id": profile.sandbox_id,
                "health_status": profile.health_status.value,
                "profile_hash_prefix": profile.profile_hash[:16],
            },
        )

    @staticmethod
    def _log_refresh_finished(
            *,
            profile: SandboxCapabilityProfile,
            section_kinds: list[str],
            elapsed_ms: int,
    ) -> None:
        logger.info(
            "sandbox_profile_refresh_finished",
            extra={
                "user_id": profile.user_id,
                "session_id": profile.session_id,
                "workspace_id": profile.workspace_id,
                "run_id": profile.run_id,
                "sandbox_id": profile.sandbox_id,
                "refresh_reason": profile.refresh_reason.value,
                "section_kinds": section_kinds,
                "health_status": profile.health_status.value,
                "profile_hash_prefix": profile.profile_hash[:16],
                "elapsed_ms": elapsed_ms,
            },
        )

    @staticmethod
    def _log_runtime_tool_snapshot_recorded(*, profile: SandboxCapabilityProfile) -> None:
        logger.info(
            "sandbox_profile_runtime_tool_snapshot_recorded",
            extra={
                "user_id": profile.user_id,
                "session_id": profile.session_id,
                "workspace_id": profile.workspace_id,
                "run_id": profile.run_id,
                "sandbox_id": profile.sandbox_id,
                "health_status": profile.health_status.value,
                "profile_hash_prefix": profile.profile_hash[:16],
            },
        )
