#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 数据访问治理的纯领域契约。"""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel


class DataResourceKind(str, Enum):
    """受 Runtime 数据访问控制保护的资源类型。"""

    SESSION = "session"
    WORKSPACE = "workspace"
    RUN = "run"
    EVENT = "event"
    CURSOR = "cursor"
    FILE = "file"
    SANDBOX_FILE = "sandbox_file"
    SANDBOX_SHELL = "sandbox_shell"
    SANDBOX_VNC = "sandbox_vnc"
    ARTIFACT = "artifact"
    MEMORY = "memory"
    RAG_CHUNK = "rag_chunk"
    EVIDENCE = "evidence"
    SKILL_CACHE = "skill_cache"
    EXTERNAL_RESULT = "external_result"


class DataAccessAction(str, Enum):
    """Runtime 数据访问动作。"""

    READ = "read"
    LIST = "list"
    STREAM = "stream"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    DOWNLOAD = "download"
    EXECUTE = "execute"
    RECALL = "recall"
    EXPORT = "export"


class DataOrigin(str, Enum):
    """数据来源，用于决定默认信任等级、隐私等级和保留策略。"""

    USER_MESSAGE = "user_message"
    USER_UPLOAD = "user_upload"
    AGENT_GENERATED = "agent_generated"
    SANDBOX_STATE = "sandbox_state"
    EXTERNAL_WEB = "external_web"
    EXTERNAL_TOOL = "external_tool"
    LONG_TERM_MEMORY = "long_term_memory"
    RAG_DERIVED = "rag_derived"
    SYSTEM_OPERATIONAL = "system_operational"


class DataTrustLevel(str, Enum):
    """数据可信等级。"""

    USER_PROVIDED = "user_provided"
    AGENT_GENERATED = "agent_generated"
    EXTERNAL_UNTRUSTED = "external_untrusted"
    SYSTEM_GENERATED = "system_generated"


class PrivacyLevel(str, Enum):
    """数据隐私等级。"""

    PRIVATE = "private"
    SENSITIVE = "sensitive"
    INTERNAL = "internal"
    PUBLIC = "public"


class RetentionPolicyKind(str, Enum):
    """数据保留策略类型。"""

    SESSION_BOUND = "session_bound"
    WORKSPACE_BOUND = "workspace_bound"
    USER_MEMORY = "user_memory"
    EPHEMERAL = "ephemeral"
    LEGAL_HOLD = "legal_hold"


class DataClassificationResult(BaseModel):
    """数据分类结果，供应用层策略服务与运行时写入链路共享。"""

    tenant_id: str
    origin: DataOrigin
    trust_level: DataTrustLevel
    privacy_level: PrivacyLevel
    retention_policy: RetentionPolicyKind


class DataClassificationPolicy(Protocol):
    """运行时数据分类策略端口。"""

    def classify_data(
            self,
            *,
            tenant_id: str,
            origin: DataOrigin,
            requested_privacy_level: PrivacyLevel | None = None,
            retention_policy: RetentionPolicyKind | None = None,
    ) -> DataClassificationResult:
        """返回指定来源的信任等级、隐私等级和保留策略。"""
        ...


class DefaultDataClassificationPolicy:
    """领域默认数据分类策略，供直接构图等无应用服务注入场景使用。"""

    def classify_data(
            self,
            *,
            tenant_id: str,
            origin: DataOrigin,
            requested_privacy_level: PrivacyLevel | None = None,
            retention_policy: RetentionPolicyKind | None = None,
    ) -> DataClassificationResult:
        normalized_tenant_id = normalize_tenant_id(tenant_id)
        return DataClassificationResult(
            tenant_id=normalized_tenant_id,
            origin=origin,
            trust_level=default_trust_level(origin),
            privacy_level=requested_privacy_level or default_privacy_level(origin),
            retention_policy=retention_policy or default_retention_policy(origin),
        )


def normalize_tenant_id(user_id: str) -> str:
    """当前组织模型上线前，租户边界固定为 user_id。"""
    tenant_id = str(user_id or "").strip()
    if not tenant_id:
        raise ValueError("user_id 不能为空")
    return tenant_id


def default_privacy_level(origin: DataOrigin) -> PrivacyLevel:
    """按来源返回默认隐私等级，P0-4 不产生公开数据。"""
    if origin == DataOrigin.LONG_TERM_MEMORY:
        return PrivacyLevel.SENSITIVE
    if origin == DataOrigin.SYSTEM_OPERATIONAL:
        return PrivacyLevel.INTERNAL
    return PrivacyLevel.PRIVATE


def default_trust_level(origin: DataOrigin) -> DataTrustLevel:
    """按来源返回默认可信等级。"""
    if origin in {DataOrigin.USER_MESSAGE, DataOrigin.USER_UPLOAD}:
        return DataTrustLevel.USER_PROVIDED
    if origin == DataOrigin.AGENT_GENERATED:
        return DataTrustLevel.AGENT_GENERATED
    if origin in {DataOrigin.EXTERNAL_WEB, DataOrigin.EXTERNAL_TOOL}:
        return DataTrustLevel.EXTERNAL_UNTRUSTED
    return DataTrustLevel.SYSTEM_GENERATED


def default_retention_policy(origin: DataOrigin) -> RetentionPolicyKind:
    """按来源返回默认保留策略，具体配置入口由应用层服务统一封装。"""
    if origin == DataOrigin.LONG_TERM_MEMORY:
        return RetentionPolicyKind.USER_MEMORY
    if origin in {DataOrigin.USER_UPLOAD, DataOrigin.AGENT_GENERATED, DataOrigin.SANDBOX_STATE}:
        return RetentionPolicyKind.WORKSPACE_BOUND
    if origin == DataOrigin.SYSTEM_OPERATIONAL:
        return RetentionPolicyKind.EPHEMERAL
    return RetentionPolicyKind.SESSION_BOUND


def is_cross_scope_access_allowed(
        *,
        source_user_id: str | None,
        target_user_id: str | None,
) -> bool:
    """P0-4 阶段禁止跨用户访问，缺失任一归属也不视为允许。"""
    source = str(source_user_id or "").strip()
    target = str(target_user_id or "").strip()
    return bool(source and target and source == target)
