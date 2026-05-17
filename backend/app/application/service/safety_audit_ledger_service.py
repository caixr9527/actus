#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safety Audit Ledger 应用服务。"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Literal

from app.domain.models import WorkflowRunEventRecord
from app.domain.models.safety_audit import (
    NonToolSafetyAuditCommand,
    NonToolSafetyAuditRecorderPort,
    SafetyAuditDataClassification,
    SafetyAuditDecision,
    SafetyAuditDecisionCounts,
    SafetyAuditNonToolActionKind,
    SafetyAuditRecorderPort,
    SafetyAuditRecord,
    SafetyAuditRecordCommand,
    SafetyAuditRecordResult,
    SafetyAuditRiskClassificationDigest,
    SafetyAuditRiskClassificationInput,
    SafetyAuditRiskCounts,
    SafetyAuditRiskLevel,
    SafetyAuditSnapshotRecordRef,
    SafetyAuditSnapshotResult,
    SafetyAuditWriteResult,
    SafetyAuditWriteStatus,
    SafetyAuditRewriteMetadataDigest,
    build_args_digest,
    build_hash,
    build_safety_audit_action_id,
    classify_safety_audit_risk,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)

logger = logging.getLogger(__name__)


class SafetyAuditLedgerError(RuntimeError):
    """Safety Audit 应用服务错误。"""


class SafetyAuditScopeError(SafetyAuditLedgerError):
    """缺少 scope 或 command 与 scope 不一致，必须 fail closed。"""


class SafetyAuditLinkedEventError(SafetyAuditLedgerError):
    """linkage event 不存在或类型不匹配。"""


class SafetyAuditLedgerService(SafetyAuditRecorderPort, NonToolSafetyAuditRecorderPort):
    """Safety Audit 写入、回链和只读 snapshot 的应用层入口。"""

    def __init__(self, *, uow_factory: Callable[[], IUnitOfWork]) -> None:
        self._uow_factory = uow_factory

    async def record_constraint_decision(self, command: SafetyAuditRecordCommand) -> SafetyAuditWriteResult:
        return await self._record_command(command)

    async def record_confirmation_decision(self, command: SafetyAuditRecordCommand) -> SafetyAuditWriteResult:
        return await self._record_command(command)

    async def record_non_tool_action(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        record_command = self._non_tool_to_record_command(command)
        return await self._record_command(record_command)

    async def record_artifact_download_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        return await self._record_expected_non_tool_action(
            command=command,
            expected_action_kind=SafetyAuditNonToolActionKind.ARTIFACT_DOWNLOAD,
        )

    async def record_artifact_preview_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        return await self._record_expected_non_tool_action(
            command=command,
            expected_action_kind=SafetyAuditNonToolActionKind.ARTIFACT_PREVIEW,
        )

    async def record_document_preflight_decision(self, command: NonToolSafetyAuditCommand) -> SafetyAuditWriteResult:
        return await self._record_expected_non_tool_action(
            command=command,
            expected_action_kind=SafetyAuditNonToolActionKind.DOCUMENT_PREFLIGHT,
        )

    async def record_external_capability_governance_decision(
            self,
            command: NonToolSafetyAuditCommand,
    ) -> SafetyAuditWriteResult:
        return await self._record_expected_non_tool_action(
            command=command,
            expected_action_kind=SafetyAuditNonToolActionKind.EXTERNAL_CAPABILITY_GOVERNANCE,
        )

    async def attach_tool_event_source(
            self,
            audit_id: str,
            tool_event_source_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        return await self._attach_event_with_scope(
            scope=self._require_attach_scope(scope),
            audit_id=audit_id,
            event_id=tool_event_source_event_id,
            expected_event_type="tool",
            linkage_kind="tool",
        )

    async def attach_decision_event(
            self,
            audit_id: str,
            decision_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        return await self._attach_event_with_scope(
            scope=self._require_attach_scope(scope),
            audit_id=audit_id,
            event_id=decision_event_id,
            expected_event_type="safety_audit",
            linkage_kind="decision",
        )

    async def attach_confirmation_event(
            self,
            audit_id: str,
            confirmation_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        return await self._attach_event_with_scope(
            scope=self._require_attach_scope(scope),
            audit_id=audit_id,
            event_id=confirmation_event_id,
            expected_event_type="message",
            linkage_kind="confirmation",
            require_user_message=True,
        )

    async def list_by_run(
            self,
            *,
            scope: AccessScopeResult,
            run_id: str | None = None,
            limit: int = 100,
    ) -> list[SafetyAuditRecord]:
        self._validate_scope_basics(scope)
        actual_run_id = self._resolve_run_id(scope=scope, run_id=run_id)
        async with self._uow_factory() as uow:
            return await uow.safety_audit.list_by_run(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                run_id=actual_run_id,
                limit=limit,
            )

    async def list_by_step(
            self,
            *,
            scope: AccessScopeResult,
            run_id: str | None = None,
            step_id: str | None = None,
            limit: int = 100,
    ) -> list[SafetyAuditRecord]:
        self._validate_scope_basics(scope)
        actual_run_id = self._resolve_run_id(scope=scope, run_id=run_id)
        actual_step_id = str(step_id or scope.current_step_id or "").strip()
        if not actual_step_id:
            raise SafetyAuditScopeError("Safety Audit step 查询必须包含 step_id")
        async with self._uow_factory() as uow:
            return await uow.safety_audit.list_by_step(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                run_id=actual_run_id,
                step_id=actual_step_id,
                limit=limit,
            )

    async def list_by_tool_event_source(
            self,
            *,
            scope: AccessScopeResult,
            tool_event_source_event_id: str,
    ) -> list[SafetyAuditRecord]:
        self._validate_scope_basics(scope)
        event_id = self._require_event_id(tool_event_source_event_id)
        async with self._uow_factory() as uow:
            return await uow.safety_audit.list_by_tool_event_source(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                tool_event_source_event_id=event_id,
            )

    async def list_by_decision_event(
            self,
            *,
            scope: AccessScopeResult,
            decision_event_id: str,
    ) -> list[SafetyAuditRecord]:
        self._validate_scope_basics(scope)
        event_id = self._require_event_id(decision_event_id)
        async with self._uow_factory() as uow:
            return await uow.safety_audit.list_by_decision_event(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                decision_event_id=event_id,
            )

    async def list_by_confirmation_event(
            self,
            *,
            scope: AccessScopeResult,
            confirmation_event_id: str,
    ) -> list[SafetyAuditRecord]:
        self._validate_scope_basics(scope)
        event_id = self._require_event_id(confirmation_event_id)
        async with self._uow_factory() as uow:
            return await uow.safety_audit.list_by_confirmation_event(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                confirmation_event_id=event_id,
            )

    async def get_snapshot(
            self,
            *,
            scope: AccessScopeResult,
            run_id: str | None = None,
            limit: int = 100,
    ) -> SafetyAuditSnapshotResult:
        records = await self.list_by_run(scope=scope, run_id=run_id, limit=limit)
        actual_run_id = self._resolve_run_id(scope=scope, run_id=run_id)
        latest = [_snapshot_ref(record) for record in records]
        return SafetyAuditSnapshotResult(
            run_id=actual_run_id,
            audit_cursor=latest[0].audit_id if latest else None,
            decision_counts=_decision_counts(records),
            risk_counts=_risk_counts(records),
            blocked_actions=[ref for ref in latest if ref.decision == SafetyAuditDecision.BLOCK],
            rewritten_actions=[ref for ref in latest if ref.decision == SafetyAuditDecision.REWRITE],
            confirmation_decisions=[
                ref for ref in latest
                if ref.decision in {
                    SafetyAuditDecision.REQUIRE_CONFIRMATION,
                    SafetyAuditDecision.CONFIRMATION_APPROVED,
                    SafetyAuditDecision.CONFIRMATION_REJECTED,
                }
            ],
            critical_findings=[ref for ref in latest if ref.risk_level == SafetyAuditRiskLevel.CRITICAL],
            latest_records=latest[:limit],
        )

    async def _record_expected_non_tool_action(
            self,
            *,
            command: NonToolSafetyAuditCommand,
            expected_action_kind: SafetyAuditNonToolActionKind,
    ) -> SafetyAuditWriteResult:
        if command.action_kind != expected_action_kind:
            raise SafetyAuditScopeError("非工具审计 action_kind 与端口不一致")
        return await self.record_non_tool_action(command)

    async def _record_command(self, command: SafetyAuditRecordCommand) -> SafetyAuditWriteResult:
        self._validate_command_scope(command)
        record = self._build_record(command)
        async with self._uow_factory() as uow:
            saved = await uow.safety_audit.save_once(record)
        status = SafetyAuditWriteStatus.CREATED if saved.id == record.id else SafetyAuditWriteStatus.REUSED
        logger.info(
            "safety_audit_record_%s user_id=%s session_id=%s run_id=%s step_id=%s tool_call_id=%s action_id=%s decision=%s reason_code=%s risk_level=%s winning_policy=%s tool_call_fingerprint=%s",
            status.value,
            saved.user_id,
            saved.session_id,
            saved.run_id,
            saved.step_id,
            saved.tool_call_id,
            saved.action_id,
            saved.decision.value,
            saved.reason_code,
            saved.risk_level.value,
            saved.winning_policy,
            saved.tool_call_fingerprint,
        )
        return _write_result(saved, status=status, reason_code=saved.reason_code)

    @staticmethod
    def _require_attach_scope(scope: AccessScopeResult | None) -> AccessScopeResult:
        if scope is None:
            raise SafetyAuditScopeError("attach linkage 缺少 AccessScopeResult")
        return scope

    async def _attach_event_with_scope(
            self,
            *,
            scope: AccessScopeResult,
            audit_id: str,
            event_id: str,
            expected_event_type: str,
            linkage_kind: Literal["tool", "decision", "confirmation"],
            require_user_message: bool = False,
    ) -> SafetyAuditWriteResult:
        self._validate_scope_basics(scope)
        actual_run_id = self._resolve_run_id(scope=scope, run_id=scope.run_id)
        normalized_audit_id = str(audit_id or "").strip()
        normalized_event_id = str(event_id or "").strip()
        if not normalized_audit_id or not normalized_event_id:
            raise SafetyAuditLinkedEventError("audit_id 和 event_id 不能为空")
        async with self._uow_factory() as uow:
            record = await uow.safety_audit.get_by_scope(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                audit_id=normalized_audit_id,
            )
            if record is None or record.run_id != actual_run_id:
                raise SafetyAuditLinkedEventError("linked_event_mismatch")
            event_record = await uow.workflow_run.get_event_record_by_event_id(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                run_id=actual_run_id,
                event_id=normalized_event_id,
            )
            self._validate_linked_event(
                record=record,
                event_record=event_record,
                expected_event_type=expected_event_type,
                linkage_kind=linkage_kind,
                require_user_message=require_user_message,
            )
            update_kwargs = {}
            if record.source_event_type is None and record.source_linked_at is None:
                update_kwargs["source_event_type"] = expected_event_type
                update_kwargs["source_linked_at"] = datetime.now()
            if linkage_kind == "tool":
                update_kwargs["tool_event_source_event_id"] = normalized_event_id
            elif linkage_kind == "decision":
                update_kwargs["decision_event_id"] = normalized_event_id
            else:
                update_kwargs["confirmation_event_id"] = normalized_event_id
            updated = await uow.safety_audit.attach_linkage(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                audit_id=normalized_audit_id,
                **update_kwargs,
            )
            return _write_result(updated, status=SafetyAuditWriteStatus.REUSED, reason_code=updated.reason_code)

    def _non_tool_to_record_command(self, command: NonToolSafetyAuditCommand) -> SafetyAuditRecordCommand:
        normalized_function_name = command.function_name.strip().lower()
        return SafetyAuditRecordCommand(
            scope=command.scope,
            user_id=command.user_id,
            session_id=command.session_id,
            workspace_id=command.workspace_id,
            run_id=command.run_id,
            step_id=command.step_id,
            tool_call_id=command.action_id_hint,
            capability_id=command.capability_id,
            tool_family=command.tool_family,
            function_name=command.function_name,
            normalized_function_name=normalized_function_name,
            requested_args=command.requested_args,
            final_function_name=command.function_name,
            final_normalized_function_name=normalized_function_name,
            final_args=command.final_args,
            decision=command.decision,
            reason_code=command.reason_code,
            policy_trace=[],
            winning_policy="non_tool_action",
            tool_call_fingerprint=build_hash({
                "action_kind": command.action_kind.value,
                "action_id_hint": command.action_id_hint,
                "requested_args": command.requested_args,
            }),
            related_artifact_revisions=command.related_artifact_revisions,
            data_classification=command.data_classification,
            external_capability_governance=command.external_capability,
            risk_input=SafetyAuditRiskClassificationInput(
                decision=command.decision,
                normalized_function_name=normalized_function_name,
                tool_family=command.tool_family,
                capability_id=command.capability_id,
                action_kind=command.action_kind.value,
                reason_code=command.reason_code,
                artifact_delivery_state=command.artifact_delivery_state,
                external_provider=command.external_provider,
                external_capability=command.external_capability,
            ),
            safe_path_roots=command.safe_path_roots,
        )

    def _build_record(self, command: SafetyAuditRecordCommand) -> SafetyAuditRecord:
        risk_input = command.risk_input or SafetyAuditRiskClassificationInput(
            decision=command.decision,
            normalized_function_name=command.normalized_function_name,
            tool_family=command.tool_family,
            capability_id=command.capability_id,
            action_kind=command.normalized_function_name,
            reason_code=command.reason_code,
        )
        risk_result = classify_safety_audit_risk(risk_input)
        action_id = build_safety_audit_action_id(
            run_id=command.run_id,
            step_id=command.step_id,
            tool_call_id=command.tool_call_id,
            tool_call_fingerprint=command.tool_call_fingerprint,
            decision=command.decision,
            reason_code=command.reason_code,
        )
        classification = command.data_classification or _default_classification()
        return SafetyAuditRecord(
            user_id=command.user_id,
            session_id=command.session_id,
            workspace_id=command.workspace_id,
            run_id=command.run_id,
            step_id=command.step_id,
            action_id=action_id,
            tool_call_id=command.tool_call_id,
            capability_id=command.capability_id,
            tool_family=command.tool_family,
            function_name=command.function_name,
            normalized_function_name=command.normalized_function_name,
            requested_args_digest=build_args_digest(command.requested_args, safe_path_roots=command.safe_path_roots),
            final_function_name=command.final_function_name,
            final_normalized_function_name=command.final_normalized_function_name,
            final_args_digest=build_args_digest(command.final_args, safe_path_roots=command.safe_path_roots),
            decision=command.decision,
            reason_code=command.reason_code,
            risk_level=risk_result.risk_level,
            policy_trace=command.policy_trace,
            winning_policy=command.winning_policy,
            tool_call_fingerprint=command.tool_call_fingerprint,
            rewrite_applied=command.rewrite_applied,
            rewrite_reason=command.rewrite_reason,
            rewrite_metadata_digest=SafetyAuditRewriteMetadataDigest.model_validate(
                build_args_digest(command.rewrite_metadata, safe_path_roots=command.safe_path_roots).model_dump(mode="json")
            ),
            confirmation_id=command.confirmation_id,
            related_fact_ids=command.related_fact_ids,
            related_evidence_ids=command.related_evidence_ids,
            related_artifact_revisions=command.related_artifact_revisions,
            profile_hash=command.profile_hash,
            external_capability_governance=command.external_capability_governance,
            origin=classification.origin,
            trust_level=classification.trust_level,
            privacy_level=classification.privacy_level,
            retention_policy=classification.retention_policy,
            classification=classification,
            risk_classification_digest=SafetyAuditRiskClassificationDigest(
                risk_level=risk_result.risk_level,
                matched_rule=risk_result.matched_rule,
            ),
            created_at=datetime.now(),
        )

    @staticmethod
    def _validate_command_scope(command: SafetyAuditRecordCommand) -> None:
        SafetyAuditLedgerService._validate_scope_basics(command.scope)
        scope = command.scope
        mismatches = []
        if scope.user_id != command.user_id:
            mismatches.append("user_id")
        if str(scope.session_id or "") != command.session_id:
            mismatches.append("session_id")
        if str(scope.workspace_id or "") != command.workspace_id:
            mismatches.append("workspace_id")
        if str(scope.run_id or "") != command.run_id:
            mismatches.append("run_id")
        scope_step_id = str(scope.current_step_id or "").strip()
        command_step_id = str(command.step_id or "").strip()
        if scope_step_id != command_step_id:
            mismatches.append("step_id")
        if mismatches:
            raise SafetyAuditScopeError("Safety Audit command scope mismatch: " + ",".join(mismatches))

    @staticmethod
    def _validate_scope_basics(scope: AccessScopeResult | None) -> None:
        if scope is None:
            raise SafetyAuditScopeError("Safety Audit 缺少 AccessScopeResult")
        if not scope.user_id or not scope.session_id or not scope.workspace_id or not scope.run_id:
            raise SafetyAuditScopeError("Safety Audit scope 缺少 user/session/workspace/run")

    @staticmethod
    def _resolve_run_id(*, scope: AccessScopeResult, run_id: str | None) -> str:
        actual_run_id = str(run_id or scope.run_id or "").strip()
        if not actual_run_id:
            raise SafetyAuditScopeError("Safety Audit run_id 不能为空")
        if scope.run_id and actual_run_id != scope.run_id:
            raise SafetyAuditScopeError("Safety Audit run_id 与 scope 不一致")
        return actual_run_id

    @staticmethod
    def _require_event_id(event_id: str | None) -> str:
        normalized_event_id = str(event_id or "").strip()
        if not normalized_event_id:
            raise SafetyAuditScopeError("Safety Audit linkage 查询必须包含 event_id")
        return normalized_event_id

    @staticmethod
    def _validate_linked_event(
            *,
            record: SafetyAuditRecord,
            event_record: WorkflowRunEventRecord | None,
            expected_event_type: str,
            linkage_kind: Literal["tool", "decision", "confirmation"],
            require_user_message: bool = False,
    ) -> None:
        if event_record is None or event_record.event_type != expected_event_type:
            raise SafetyAuditLinkedEventError("linked_event_mismatch")
        if linkage_kind == "tool":
            payload = event_record.event_payload
            event_step_id = str(getattr(payload, "step_id", "") or "").strip()
            event_tool_call_id = str(getattr(payload, "tool_call_id", "") or "").strip()
            record_step_id = str(record.step_id or "").strip()
            record_tool_call_id = str(record.tool_call_id or "").strip()
            if event_step_id != record_step_id or event_tool_call_id != record_tool_call_id:
                raise SafetyAuditLinkedEventError("linked_event_mismatch")
        if require_user_message:
            role = str(getattr(event_record.event_payload, "role", "") or "")
            if role != "user":
                raise SafetyAuditLinkedEventError("linked_event_mismatch")


def _default_classification() -> SafetyAuditDataClassification:
    return SafetyAuditDataClassification(
        origin=DataOrigin.SYSTEM_OPERATIONAL,
        trust_level=DataTrustLevel.SYSTEM_GENERATED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
    )


def _write_result(
        record: SafetyAuditRecord,
        *,
        status: SafetyAuditWriteStatus,
        reason_code: str,
) -> SafetyAuditWriteResult:
    ref = SafetyAuditRecordResult(
        audit_id=record.id,
        action_id=record.action_id,
        decision=record.decision,
        risk_level=record.risk_level,
        reason_code=record.reason_code,
        run_id=record.run_id,
        step_id=record.step_id,
        tool_call_id=record.tool_call_id,
    )
    return SafetyAuditWriteResult(
        audit_id=record.id,
        record=ref,
        status=status,
        reason_code=reason_code,
    )


def _snapshot_ref(record: SafetyAuditRecord) -> SafetyAuditSnapshotRecordRef:
    return SafetyAuditSnapshotRecordRef(
        audit_id=record.id,
        decision=record.decision,
        risk_level=record.risk_level,
        reason_code=record.reason_code,
        step_id=record.step_id,
        tool_call_id=record.tool_call_id,
        function_name=record.function_name,
        created_at=record.created_at,
    )


def _decision_counts(records: list[SafetyAuditRecord]) -> SafetyAuditDecisionCounts:
    counts = SafetyAuditDecisionCounts()
    for record in records:
        setattr(counts, record.decision.value, getattr(counts, record.decision.value) + 1)
    return counts


def _risk_counts(records: list[SafetyAuditRecord]) -> SafetyAuditRiskCounts:
    counts = SafetyAuditRiskCounts()
    for record in records:
        setattr(counts, record.risk_level.value, getattr(counts, record.risk_level.value) + 1)
    return counts
