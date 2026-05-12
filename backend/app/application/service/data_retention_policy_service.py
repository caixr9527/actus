#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 数据保留策略应用服务。"""

from datetime import datetime

from pydantic import BaseModel

from app.domain.services.runtime.contracts.data_access_contract import (
    DataClassificationResult,
    DefaultDataClassificationPolicy,
    DataOrigin,
    PrivacyLevel,
    RetentionPolicyKind,
    default_retention_policy,
    default_privacy_level,
    default_trust_level,
    normalize_tenant_id,
)


class RetentionPolicyResult(BaseModel):
    """应用层保留策略结果。"""

    tenant_id: str
    policy_kind: RetentionPolicyKind
    expires_at: datetime | None = None
    delete_after_days: int | None = None


class DataRetentionPolicyService(DefaultDataClassificationPolicy):
    """集中提供 P0-4 默认保留策略，避免业务服务散落硬编码。"""

    def default_policy_for_origin(
            self,
            *,
            tenant_id: str,
            origin: DataOrigin,
            requested_policy: RetentionPolicyKind | None = None,
    ) -> RetentionPolicyResult:
        normalized_tenant_id = normalize_tenant_id(tenant_id)
        if requested_policy is not None:
            return RetentionPolicyResult(tenant_id=normalized_tenant_id, policy_kind=requested_policy)

        policy_kind = default_retention_policy(origin)
        return RetentionPolicyResult(tenant_id=normalized_tenant_id, policy_kind=policy_kind)

    def classify_data(
            self,
            *,
            tenant_id: str,
            origin: DataOrigin,
            requested_privacy_level: PrivacyLevel | None = None,
            retention_policy: RetentionPolicyKind | None = None,
    ) -> DataClassificationResult:
        policy = self.default_policy_for_origin(
            tenant_id=tenant_id,
            origin=origin,
            requested_policy=retention_policy,
        )
        return DataClassificationResult(
            tenant_id=policy.tenant_id,
            origin=origin,
            trust_level=default_trust_level(origin),
            privacy_level=requested_privacy_level or default_privacy_level(origin),
            retention_policy=policy.policy_kind,
        )
