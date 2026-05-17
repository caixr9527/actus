#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""带 SafetyAuditEvent 投影的非工具安全审计入口。"""

from __future__ import annotations

import logging

from app.domain.models.safety_audit import (
    NonToolSafetyAuditCommand,
    NonToolSafetyAuditRecorderPort,
    SafetyAuditNonToolActionKind,
    SafetyAuditRecorderPort,
    SafetyAuditRecordCommand,
    SafetyAuditWriteResult,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.safety_audit_contract import SafetyAuditEventProjectorPort

logger = logging.getLogger(__name__)


class ProjectingSafetyAuditRecorder(SafetyAuditRecorderPort, NonToolSafetyAuditRecorderPort):
    """包装账本 recorder，在无 ToolEvent 的安全决策写入后立即投影 runtime event。"""

    def __init__(
            self,
            *,
            recorder: SafetyAuditRecorderPort,
            projector: SafetyAuditEventProjectorPort,
    ) -> None:
        self._recorder = recorder
        self._projector = projector

    async def record_constraint_decision(self, command: SafetyAuditRecordCommand) -> SafetyAuditWriteResult:
        return await self._recorder.record_constraint_decision(command)

    async def record_confirmation_decision(self, command: SafetyAuditRecordCommand) -> SafetyAuditWriteResult:
        return await self._recorder.record_confirmation_decision(command)

    async def record_non_tool_action(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        return await self._record_and_project(command)

    async def attach_tool_event_source(
            self,
            audit_id: str,
            tool_event_source_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        return await self._recorder.attach_tool_event_source(
            audit_id,
            tool_event_source_event_id,
            scope=scope,
        )

    async def attach_decision_event(
            self,
            audit_id: str,
            decision_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        return await self._recorder.attach_decision_event(
            audit_id,
            decision_event_id,
            scope=scope,
        )

    async def attach_confirmation_event(
            self,
            audit_id: str,
            confirmation_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        return await self._recorder.attach_confirmation_event(
            audit_id,
            confirmation_event_id,
            scope=scope,
        )

    async def record_artifact_download_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        self._require_action_kind(command, SafetyAuditNonToolActionKind.ARTIFACT_DOWNLOAD)
        return await self.record_non_tool_action(command)

    async def record_artifact_preview_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        self._require_action_kind(command, SafetyAuditNonToolActionKind.ARTIFACT_PREVIEW)
        return await self.record_non_tool_action(command)

    async def record_document_preflight_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        self._require_action_kind(command, SafetyAuditNonToolActionKind.DOCUMENT_PREFLIGHT)
        return await self.record_non_tool_action(command)

    async def record_external_capability_governance_decision(
            self,
        command: NonToolSafetyAuditCommand,
    ) -> SafetyAuditWriteResult:
        self._require_action_kind(command, SafetyAuditNonToolActionKind.EXTERNAL_CAPABILITY_GOVERNANCE)
        return await self.record_non_tool_action(command)

    async def _record_and_project(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        result = await self._recorder.record_non_tool_action(command)
        try:
            await self._projector.project_single_audit(
                scope=command.scope,
                audit_id=result.audit_id,
            )
        except Exception as exc:
            logger.exception(
                "safety_audit_event_projection_failed",
                extra={
                    "user_id": command.user_id,
                    "session_id": command.session_id,
                    "run_id": command.run_id,
                    "step_id": command.step_id,
                    "audit_id": result.audit_id,
                    "action_kind": command.action_kind.value,
                    "error_type": exc.__class__.__name__,
                    "reason_code": "safety_audit_event_projection_failed",
                },
            )
        return result

    @staticmethod
    def _require_action_kind(
            command: NonToolSafetyAuditCommand,
            expected: SafetyAuditNonToolActionKind,
    ) -> None:
        if command.action_kind != expected:
            raise ValueError("非工具审计 action_kind 与端口不一致")
