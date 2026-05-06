#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Fact 仓储协议。"""

from typing import Protocol

from app.domain.models.sandbox_fact import (
    SandboxFactKind,
    SandboxFactRecord,
    SandboxFactScope,
)


class SandboxFactRepository(Protocol):
    """Sandbox Fact Ledger 持久化端口。"""

    async def save_once(self, fact: SandboxFactRecord) -> SandboxFactRecord:
        """按 idempotency_key 幂等保存 fact。"""
        ...

    async def list_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            fact_scope: SandboxFactScope | None = None,
            run_id: str | None = None,
            step_id: str | None = None,
            fact_kinds: list[SandboxFactKind] | None = None,
            limit: int = 100,
    ) -> list[SandboxFactRecord]:
        """按用户、会话和可选 scope 强字段查询 fact。"""
        ...

    async def list_by_source_event(
            self,
            *,
            user_id: str,
            session_id: str,
            source_event_id: str,
    ) -> list[SandboxFactRecord]:
        """按 workflow_run_events.event_id 回链查询 fact。"""
        ...

    async def list_by_ids(
            self,
            *,
            user_id: str,
            session_id: str,
            fact_ids: list[str],
            limit: int = 100,
    ) -> list[SandboxFactRecord]:
        """按用户、会话和 fact id 列表查询，禁止裸 id 全局读取。"""
        ...
