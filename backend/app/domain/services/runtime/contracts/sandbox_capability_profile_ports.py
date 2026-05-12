#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox capability profile runtime 侧端口契约。"""

from __future__ import annotations

from typing import Protocol

from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    RuntimeToolCapabilitySnapshot,
    SandboxCapabilityProfile,
    SandboxProfileRefreshReason,
)


class SandboxCapabilityProfileRefresherPort(Protocol):
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
        """sandbox 已绑定到权威 workspace/run 后刷新 profile；失败不推进 runtime 状态。"""
        ...


class RuntimeToolSnapshotRecorderPort(Protocol):
    async def record_runtime_tool_snapshot(
            self,
            *,
            user_id: str,
            session_id: str,
            snapshot: RuntimeToolCapabilitySnapshot,
    ) -> SandboxCapabilityProfile:
        """记录当前 run 实际暴露给模型的工具能力快照；scope 由实现方重新解析。"""
        ...
