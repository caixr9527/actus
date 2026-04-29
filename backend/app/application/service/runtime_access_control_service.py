#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 数据访问控制应用服务。"""

import logging
from datetime import datetime
from typing import Callable

from pydantic import BaseModel

from app.application.errors import NotFoundError, error_keys
from app.application.service.data_retention_policy_service import DataRetentionPolicyService
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.data_access_contract import (
    DataAccessAction,
    DataClassificationResult,
    DataOrigin,
    DataResourceKind,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
    normalize_tenant_id,
)

logger = logging.getLogger(__name__)


class AccessScopeResult(BaseModel):
    """已校验的数据访问范围。"""

    tenant_id: str
    user_id: str
    session_id: str | None = None
    workspace_id: str | None = None
    run_id: str | None = None
    current_step_id: str | None = None


class AccessDecisionResult(BaseModel):
    """数据访问判定结果。"""

    allowed: bool
    resource_kind: DataResourceKind
    action: DataAccessAction
    reason_code: str = ""


class RetentionPolicyResult(BaseModel):
    """保留策略结果。"""

    tenant_id: str
    policy_kind: RetentionPolicyKind
    expires_at: datetime | None = None
    delete_after_days: int | None = None


class DataEgressRecordResult(BaseModel):
    """数据出域元数据记录，供后续 Safety Audit 接管。"""

    user_id: str
    session_id: str | None = None
    workspace_id: str | None = None
    run_id: str | None = None
    destination: str
    resource_kind: DataResourceKind
    privacy_level: PrivacyLevel
    trust_level: DataTrustLevel
    reason_code: str


class RuntimeAccessControlService:
    """Runtime 数据访问控制统一入口。"""

    def __init__(
            self,
            *,
            uow_factory: Callable[[], IUnitOfWork],
            retention_policy_service: DataRetentionPolicyService | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._retention_policy_service = retention_policy_service or DataRetentionPolicyService()

    async def resolve_session_scope(
            self,
            *,
            user_id: str,
            session_id: str,
    ) -> AccessScopeResult:
        tenant_id = normalize_tenant_id(user_id)
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id=session_id, user_id=user_id)
            if session is None:
                self._log_denied(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    resource_kind=DataResourceKind.SESSION,
                    action=DataAccessAction.READ,
                    resource_id=session_id,
                    session_id=session_id,
                    reason_code="session_not_found",
                )
                raise self._session_not_found(session_id)

            workspace = None
            workspace_id = str(session.workspace_id or "").strip()
            if workspace_id:
                workspace = await uow.workspace.get_by_id_for_user(
                    workspace_id=workspace_id,
                    user_id=user_id,
                )
                if workspace is None:
                    self._log_denied(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        resource_kind=DataResourceKind.WORKSPACE,
                        action=DataAccessAction.READ,
                        resource_id=workspace_id,
                        session_id=session.id,
                        workspace_id=workspace_id,
                        reason_code="workspace_not_found",
                    )
                    raise self._session_not_found(session_id)
                if workspace is not None and workspace.session_id != session.id:
                    self._log_denied(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        resource_kind=DataResourceKind.WORKSPACE,
                        action=DataAccessAction.READ,
                        resource_id=workspace_id,
                        session_id=session.id,
                        workspace_id=workspace_id,
                        reason_code="workspace_session_mismatch",
                    )
                    raise self._session_not_found(session_id)

            if not workspace_id:
                workspaces = await uow.workspace.list_by_session_id(session_id=session.id)
                if len(workspaces) > 1:
                    self._log_denied(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        resource_kind=DataResourceKind.WORKSPACE,
                        action=DataAccessAction.READ,
                        resource_id=session.id,
                        session_id=session.id,
                        reason_code="multiple_workspaces_for_session",
                    )
                    raise self._session_not_found(session_id)
                if len(workspaces) == 1:
                    workspace = workspaces[0]
                    workspace_user_id = str(workspace.user_id or "").strip()
                    if not workspace_user_id:
                        self._log_denied(
                            user_id=user_id,
                            tenant_id=tenant_id,
                            resource_kind=DataResourceKind.WORKSPACE,
                            action=DataAccessAction.READ,
                            resource_id=workspace.id,
                            session_id=session.id,
                            workspace_id=workspace.id,
                            reason_code="workspace_owner_missing",
                        )
                        raise self._session_not_found(session_id)
                    if workspace_user_id != user_id:
                        self._log_denied(
                            user_id=user_id,
                            tenant_id=tenant_id,
                            resource_kind=DataResourceKind.WORKSPACE,
                            action=DataAccessAction.READ,
                            resource_id=workspace.id,
                            session_id=session.id,
                            workspace_id=workspace.id,
                            reason_code="workspace_owner_mismatch",
                        )
                        raise self._session_not_found(session_id)
                    workspace_id = workspace.id

            run = None
            run_id = str(getattr(workspace, "current_run_id", None) or session.current_run_id or "").strip()
            if run_id:
                run = await uow.workflow_run.get_by_id_for_user(run_id=run_id, user_id=user_id)
                if run is None or run.session_id != session.id or run.user_id != user_id:
                    self._log_denied(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        resource_kind=DataResourceKind.RUN,
                        action=DataAccessAction.READ,
                        resource_id=run_id,
                        session_id=session.id,
                        workspace_id=workspace_id or None,
                        run_id=run_id,
                        reason_code="run_owner_mismatch",
                    )
                    raise self._session_not_found(session_id)

        return AccessScopeResult(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            workspace_id=workspace_id or None,
            run_id=run.id if run is not None else None,
            current_step_id=run.current_step_id if run is not None else None,
        )

    async def assert_session_access(
            self,
            *,
            user_id: str,
            session_id: str,
            action: DataAccessAction,
    ) -> AccessScopeResult:
        return await self.resolve_session_scope(user_id=user_id, session_id=session_id)

    async def assert_workspace_access(
            self,
            *,
            user_id: str,
            workspace_id: str,
            action: DataAccessAction,
            expected_session_id: str | None = None,
    ) -> AccessScopeResult:
        tenant_id = normalize_tenant_id(user_id)
        async with self._uow_factory() as uow:
            workspace = await uow.workspace.get_by_id_for_user(
                workspace_id=workspace_id,
                user_id=user_id,
            )
            if workspace is None:
                self._log_denied(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    resource_kind=DataResourceKind.WORKSPACE,
                    action=action,
                    resource_id=workspace_id,
                    reason_code="workspace_not_found",
                )
                raise self._session_not_found(expected_session_id or "")

            if expected_session_id is not None and workspace.session_id != expected_session_id:
                self._log_denied(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    resource_kind=DataResourceKind.WORKSPACE,
                    action=action,
                    resource_id=workspace_id,
                    session_id=expected_session_id,
                    workspace_id=workspace_id,
                    reason_code="workspace_expected_session_mismatch",
                )
                raise self._session_not_found(expected_session_id)

        scope = await self.resolve_session_scope(user_id=user_id, session_id=workspace.session_id)
        if scope.workspace_id != workspace_id:
            self._log_denied(
                user_id=user_id,
                tenant_id=tenant_id,
                resource_kind=DataResourceKind.WORKSPACE,
                action=action,
                resource_id=workspace_id,
                session_id=workspace.session_id,
                workspace_id=workspace_id,
                reason_code="workspace_not_in_session_scope",
            )
            raise self._session_not_found(workspace.session_id)
        return scope

    async def assert_run_access(
            self,
            *,
            user_id: str,
            run_id: str,
            action: DataAccessAction,
            expected_session_id: str | None = None,
    ) -> AccessScopeResult:
        tenant_id = normalize_tenant_id(user_id)
        async with self._uow_factory() as uow:
            run = await uow.workflow_run.get_by_id_for_user(run_id=run_id, user_id=user_id)
        if run is None:
            self._log_denied(
                user_id=user_id,
                tenant_id=tenant_id,
                resource_kind=DataResourceKind.RUN,
                action=action,
                resource_id=run_id,
                run_id=run_id,
                reason_code="run_not_found",
            )
            raise self._session_not_found(expected_session_id or "")
        if expected_session_id is not None and run.session_id != expected_session_id:
            self._log_denied(
                user_id=user_id,
                tenant_id=tenant_id,
                resource_kind=DataResourceKind.RUN,
                action=action,
                resource_id=run_id,
                session_id=expected_session_id,
                run_id=run_id,
                reason_code="run_expected_session_mismatch",
            )
            raise self._session_not_found(expected_session_id)
        scope = await self.resolve_session_scope(user_id=user_id, session_id=run.session_id)
        return scope.model_copy(
            update={
                "run_id": run.id,
                "current_step_id": run.current_step_id,
            }
        )

    async def assert_session_replay_access(
            self,
            *,
            user_id: str,
            session_id: str,
    ) -> AccessScopeResult:
        return await self.assert_session_access(
            user_id=user_id,
            session_id=session_id,
            action=DataAccessAction.STREAM,
        )

    async def assert_file_access(
            self,
            *,
            user_id: str,
            file_id: str,
            action: DataAccessAction,
    ) -> AccessScopeResult:
        tenant_id = normalize_tenant_id(user_id)
        async with self._uow_factory() as uow:
            file = await uow.file.get_by_id_and_user_id(file_id=file_id, user_id=user_id)
        if file is None:
            self._log_denied(
                user_id=user_id,
                tenant_id=tenant_id,
                resource_kind=DataResourceKind.FILE,
                action=action,
                resource_id=file_id,
                reason_code="file_not_found",
            )
            raise NotFoundError(
                msg=f"该文件[{file_id}]不存在",
                error_key=error_keys.FILE_NOT_FOUND,
                error_params={"file_id": file_id},
            )
        return AccessScopeResult(tenant_id=tenant_id, user_id=user_id)

    async def assert_artifact_access(
            self,
            *,
            user_id: str,
            session_id: str,
            path: str,
            action: DataAccessAction,
    ) -> AccessScopeResult:
        scope = await self.assert_session_access(
            user_id=user_id,
            session_id=session_id,
            action=action,
        )
        if scope.workspace_id is None:
            self._log_denied(
                user_id=user_id,
                tenant_id=scope.tenant_id,
                resource_kind=DataResourceKind.ARTIFACT,
                action=action,
                resource_id=path,
                session_id=session_id,
                reason_code="workspace_required",
            )
            raise self._session_not_found(session_id)

        async with self._uow_factory() as uow:
            artifacts = await uow.workspace_artifact.list_by_user_workspace_id_and_paths(
                user_id=user_id,
                workspace_id=scope.workspace_id,
                paths=[path],
            )
        if not artifacts:
            self._log_denied(
                user_id=user_id,
                tenant_id=scope.tenant_id,
                resource_kind=DataResourceKind.ARTIFACT,
                action=action,
                resource_id=path,
                session_id=session_id,
                workspace_id=scope.workspace_id,
                run_id=scope.run_id,
                reason_code="artifact_not_found",
            )
            raise self._session_not_found(session_id)
        return scope

    async def assert_sandbox_access(
            self,
            *,
            user_id: str,
            session_id: str,
            resource_kind: DataResourceKind,
            action: DataAccessAction,
    ) -> AccessScopeResult:
        if resource_kind not in {
            DataResourceKind.SANDBOX_FILE,
            DataResourceKind.SANDBOX_SHELL,
            DataResourceKind.SANDBOX_VNC,
        }:
            raise ValueError("resource_kind 必须是 sandbox 资源类型")
        scope = await self.assert_session_access(
            user_id=user_id,
            session_id=session_id,
            action=action,
        )
        if scope.workspace_id is not None:
            return scope
        self._log_denied(
            user_id=user_id,
            tenant_id=scope.tenant_id,
            resource_kind=resource_kind,
            action=action,
            resource_id=session_id,
            session_id=session_id,
            reason_code="workspace_required",
        )
        raise self._session_not_found(session_id)

    def classify_data(
            self,
            *,
            tenant_id: str,
            origin: DataOrigin,
            requested_privacy_level: PrivacyLevel | None = None,
            retention_policy: RetentionPolicyKind | None = None,
    ) -> DataClassificationResult:
        return self._retention_policy_service.classify_data(
            tenant_id=tenant_id,
            origin=origin,
            requested_privacy_level=requested_privacy_level,
            retention_policy=retention_policy,
        )

    @staticmethod
    def _session_not_found(session_id: str) -> NotFoundError:
        return NotFoundError(
            msg="该会话不存在，请核实后重试",
            error_key=error_keys.SESSION_NOT_FOUND,
            error_params={"session_id": session_id},
        )

    @staticmethod
    def _log_denied(
            *,
            user_id: str,
            tenant_id: str,
            resource_kind: DataResourceKind,
            action: DataAccessAction,
            resource_id: str,
            session_id: str | None = None,
            workspace_id: str | None = None,
            run_id: str | None = None,
            reason_code: str,
    ) -> None:
        logger.warning(
            "Runtime数据访问被拒绝",
            extra={
                "event": "data_access_denied",
                "user_id": user_id,
                "tenant_id": tenant_id,
                "resource_kind": resource_kind.value,
                "action": action.value,
                "resource_id": resource_id,
                "session_id": session_id,
                "workspace_id": workspace_id,
                "run_id": run_id,
                "reason_code": reason_code,
            },
        )
