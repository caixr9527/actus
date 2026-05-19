#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback Ledger 仓储协议。"""

from typing import Protocol

from datetime import datetime

from app.domain.models.feedback import (
    FeedbackRecord,
    FeedbackResolutionResult,
    FeedbackScopeKind,
    FeedbackTargetType,
)


class FeedbackRepository(Protocol):
    """Feedback Ledger 持久化端口。"""

    async def save_once(self, record: FeedbackRecord) -> FeedbackRecord:
        """按 user/session/scope/dedupe_key 幂等保存反馈记录。"""
        ...

    async def get_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            feedback_scope_kind: FeedbackScopeKind,
            scope_id: str,
            feedback_id: str,
    ) -> FeedbackRecord | None:
        """按强 scope 读取单条反馈，禁止裸 feedback_id 全局读取。"""
        ...

    async def list_by_run(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        """按用户、会话和 run 查询反馈记录。"""
        ...

    async def list_by_step(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            step_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        """按用户、会话、run 和 step 查询反馈记录。"""
        ...

    async def list_active_by_run(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        """run scope 的活跃反馈便捷查询。"""
        ...

    async def list_active_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            feedback_scope_kind: FeedbackScopeKind,
            scope_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        """按用户、会话和 feedback scope 查询活跃反馈。"""
        ...

    async def list_by_scope(
            self,
            *,
            user_id: str,
            session_id: str,
            feedback_scope_kind: FeedbackScopeKind,
            scope_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        """按用户、会话和 feedback scope 查询反馈记录，不限制 status。"""
        ...

    async def list_by_target(
            self,
            *,
            user_id: str,
            session_id: str,
            target_type: FeedbackTargetType,
            target_id: str,
            target_revision_id: str | None = None,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        """按目标强字段查询反馈记录。"""
        ...

    async def list_by_source_event(
            self,
            *,
            user_id: str,
            session_id: str,
            source_event_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        """按 source event 回链查询反馈记录。"""
        ...

    async def update_resolution(
            self,
            *,
            user_id: str,
            session_id: str,
            feedback_scope_kind: FeedbackScopeKind,
            scope_id: str,
            feedback_id: str,
            resolution: FeedbackResolutionResult,
            updated_at: datetime,
    ) -> FeedbackRecord:
        """只更新 lifecycle 强列与 resolution/updated_at，禁止覆盖业务 payload。"""
        ...
