#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 数据保留策略应用服务。"""

from datetime import datetime

from pydantic import BaseModel

from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    RetentionPolicyKind,
)


class RetentionPolicyResult(BaseModel):
    """应用层保留策略结果。"""

    tenant_id: str
    policy_kind: RetentionPolicyKind
    expires_at: datetime | None = None
    delete_after_days: int | None = None


class DataRetentionPolicyService:
    """集中提供 P0-4 默认保留策略，避免业务服务散落硬编码。"""

    def default_policy_for_origin(
            self,
            *,
            tenant_id: str,
            origin: DataOrigin,
            requested_policy: RetentionPolicyKind | None = None,
    ) -> RetentionPolicyResult:
        if requested_policy is not None:
            return RetentionPolicyResult(tenant_id=tenant_id, policy_kind=requested_policy)

        if origin == DataOrigin.LONG_TERM_MEMORY:
            policy_kind = RetentionPolicyKind.USER_MEMORY
        elif origin in {DataOrigin.USER_UPLOAD, DataOrigin.AGENT_GENERATED, DataOrigin.SANDBOX_STATE}:
            policy_kind = RetentionPolicyKind.WORKSPACE_BOUND
        elif origin == DataOrigin.SYSTEM_OPERATIONAL:
            policy_kind = RetentionPolicyKind.EPHEMERAL
        else:
            policy_kind = RetentionPolicyKind.SESSION_BOUND
        return RetentionPolicyResult(tenant_id=tenant_id, policy_kind=policy_kind)
