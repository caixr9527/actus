#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Fact 投影上下文构造器。"""

from __future__ import annotations

from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.domain.models.sandbox_fact import SandboxFactProfileRef
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.data_access_contract import DataAccessAction
from app.domain.services.runtime.contracts.sandbox_fact_ports import (
    SandboxFactProjectionContext,
    SandboxFactProjectionContextBuilderPort,
)
from app.domain.services.workspace_runtime import WorkspaceRuntimeService


class SandboxFactProjectionContextBuilder(SandboxFactProjectionContextBuilderPort):
    """为 runner/runtime 集中构造 ToolEvent fact 投影上下文。"""

    def __init__(
            self,
            *,
            access_control_service: RuntimeAccessControlService,
            workspace_runtime_service: WorkspaceRuntimeService,
            user_id: str | None,
            session_id: str,
    ) -> None:
        self._access_control_service = access_control_service
        self._workspace_runtime_service = workspace_runtime_service
        self._user_id = str(user_id or "").strip()
        self._session_id = session_id

    async def build_for_tool_event(
            self,
            *,
            source_event_id: str,
            current_step_id: str | None = None,
    ) -> SandboxFactProjectionContext:
        if not self._user_id:
            raise ValueError("Sandbox fact projection context 需要 user_id")
        scope = await self._access_control_service.assert_session_access(
            user_id=self._user_id,
            session_id=self._session_id,
            action=DataAccessAction.READ,
        )
        resolved_step_id = self._resolve_current_step_id(scope=scope, current_step_id=current_step_id)
        if resolved_step_id != scope.current_step_id:
            scope = scope.model_copy(update={"current_step_id": resolved_step_id})
        return await self._build_context(source_event_id=source_event_id, scope=scope)

    @staticmethod
    def _resolve_current_step_id(
            *,
            scope: AccessScopeResult,
            current_step_id: str | None,
    ) -> str | None:
        scope_step_id = str(scope.current_step_id or "").strip() or None
        requested_step_id = str(current_step_id or "").strip() or None
        if scope_step_id and requested_step_id and scope_step_id != requested_step_id:
            raise ValueError("Sandbox fact current_step_id 与 access scope 不一致")
        return scope_step_id or requested_step_id

    async def build_for_document_input(
            self,
            *,
            source_event_id: str,
            scope: AccessScopeResult,
    ) -> SandboxFactProjectionContext:
        if not self._user_id:
            raise ValueError("Sandbox fact projection context 需要 user_id")
        return await self._build_context(source_event_id=source_event_id, scope=scope)

    async def _build_context(
            self,
            *,
            source_event_id: str,
            scope: AccessScopeResult,
    ) -> SandboxFactProjectionContext:
        workspace = await self._workspace_runtime_service.get_workspace()
        profile_error: Exception | None = None
        try:
            profile = await self._workspace_runtime_service.get_sandbox_capability_profile()
        except Exception as exc:
            profile = None
            profile_error = exc
        sandbox_id = str(getattr(workspace, "sandbox_id", None) or "").strip() or None
        if profile is not None:
            profile_ref = SandboxFactProfileRef(
                profile_id=profile.profile_id,
                profile_hash=profile.profile_hash,
                sandbox_id=profile.sandbox_id,
                generated_at=profile.generated_at,
                status="available",
            )
        elif profile_error is not None:
            profile_ref = SandboxFactProfileRef(status="invalid")
        else:
            profile_ref = SandboxFactProfileRef(status="missing")
        return SandboxFactProjectionContext(
            scope=scope,
            profile_ref=profile_ref,
            sandbox_id=sandbox_id,
            source_event_id=str(source_event_id or "").strip() or None,
            current_step_id=scope.current_step_id,
        )
