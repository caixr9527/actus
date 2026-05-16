#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safety Audit Ledger 仓储协议。"""

from datetime import datetime
from typing import Protocol

from app.domain.models.safety_audit import SafetyAuditDecision, SafetyAuditRecord, SafetyAuditRiskLevel


class SafetyAuditRepository(Protocol):
    """Safety Audit Ledger 持久化端口。"""

    async def save_once(self, record: SafetyAuditRecord) -> SafetyAuditRecord:
        """按 user_id + session_id + run_id + action_id 幂等保存审计记录。"""
        ...

    async def get_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            audit_id: str,
    ) -> SafetyAuditRecord | None:
        """按强 scope 读取单条记录，禁止裸 audit_id 全局读取。"""
        ...

    async def list_by_run(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            limit: int = 100,
    ) -> list[SafetyAuditRecord]:
        """按用户、会话和 run 查询审计记录。"""
        ...

    async def list_by_step(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            step_id: str,
            limit: int = 100,
    ) -> list[SafetyAuditRecord]:
        """按用户、会话、run 和 step 查询审计记录。"""
        ...

    async def list_by_decision_and_risk(
            self,
            *,
            user_id: str,
            session_id: str,
            decision: SafetyAuditDecision,
            risk_level: SafetyAuditRiskLevel,
            limit: int = 100,
    ) -> list[SafetyAuditRecord]:
        """按 decision 和 risk 查询审计记录。"""
        ...

    async def list_by_tool_event_source(
            self,
            *,
            user_id: str,
            session_id: str,
            tool_event_source_event_id: str,
    ) -> list[SafetyAuditRecord]:
        """按 ToolEvent source event 回链查询审计记录。"""
        ...

    async def list_by_decision_event(
            self,
            *,
            user_id: str,
            session_id: str,
            decision_event_id: str,
    ) -> list[SafetyAuditRecord]:
        """按 SafetyAuditEvent 回链查询审计记录。"""
        ...

    async def list_by_confirmation_event(
            self,
            *,
            user_id: str,
            session_id: str,
            confirmation_event_id: str,
    ) -> list[SafetyAuditRecord]:
        """按用户确认/拒绝 event 回链查询审计记录。"""
        ...

    async def attach_linkage(
            self,
            *,
            user_id: str,
            session_id: str,
            audit_id: str,
            decision_event_id: str | None = None,
            tool_event_source_event_id: str | None = None,
            confirmation_event_id: str | None = None,
            source_event_type: str | None = None,
            source_linked_at: datetime | None = None,
    ) -> SafetyAuditRecord:
        """只补齐 linkage 字段，禁止修改决策 payload。"""
        ...
