#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence Ledger 仓储协议。"""

from typing import Protocol

from app.domain.models.evidence import EvidenceRecord, EvidenceScope


class EvidenceRepository(Protocol):
    """Evidence Ledger 持久化端口。"""

    async def save_once(self, evidence: EvidenceRecord) -> EvidenceRecord:
        """按 idempotency_key 幂等保存 evidence。"""
        ...

    async def list_by_ids(
            self,
            *,
            user_id: str,
            session_id: str,
            evidence_ids: list[str],
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        """按用户、会话和 evidence id 列表查询，禁止裸 id 全局读取。"""
        ...

    async def list_by_step(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            step_id: str,
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        """按用户、会话、run 和 step 强字段查询 evidence。"""
        ...

    async def list_by_run(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            evidence_scope: EvidenceScope | None = None,
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        """按用户、会话和 run 强字段查询 evidence。"""
        ...

    async def list_by_fact_ids(
            self,
            *,
            user_id: str,
            session_id: str,
            fact_ids: list[str],
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        """按用户、会话和 fact ids 查询 evidence。"""
        ...

    async def list_reusable_by_run(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        """查询同一 run 内可复用 evidence，用于后续 digest/去重。"""
        ...

    async def list_by_action_subject(
            self,
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            action_key: str,
            subject_key: str,
            limit: int = 100,
    ) -> list[EvidenceRecord]:
        """按归一 action/subject 查询，禁止重复执行判断使用。"""
        ...
