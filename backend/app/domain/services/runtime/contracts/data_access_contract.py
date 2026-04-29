#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 数据访问治理的纯领域契约。"""

from enum import Enum


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


def is_cross_scope_access_allowed(
        *,
        source_user_id: str | None,
        target_user_id: str | None,
) -> bool:
    """P0-4 阶段禁止跨用户访问，缺失任一归属也不视为允许。"""
    source = str(source_user_id or "").strip()
    target = str(target_user_id or "").strip()
    return bool(source and target and source == target)
