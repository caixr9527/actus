#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback Ledger PR2 validator。"""

from __future__ import annotations

from typing import Any
from app.application.service.feedback_ledger_common import (
    FeedbackSanitizer,
    FeedbackScopeValidationError,
    FeedbackValidationIssue,
    SOURCE_EVENT_TYPE_BY_KIND,
    TARGET_EVENT_TYPE_BY_TYPE,
)
from app.domain.models import FeedbackInputEvent, WorkflowRunEventRecord
from app.domain.models.feedback import (
    FeedbackCategory,
    FeedbackGapKind,
    FeedbackKind,
    FeedbackReasonCode,
    FeedbackResolutionCommand,
    FeedbackScopeKind,
    FeedbackScopeResult,
    FeedbackSnapshotScopeResult,
    FeedbackSourceKind,
    FeedbackSourceRefResult,
    FeedbackTargetRefResult,
    FeedbackTargetType,
    QualityFeedbackCommand,
    RuntimeFeedbackCommand,
    UserFeedbackCommand,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactRevisionResolveCommand,
    ArtifactRevisionResolveStatus,
)
from app.domain.services.runtime.contracts.feedback_runtime_ports import ArtifactRevisionResolverPort


class FeedbackScopeValidator:
    """统一 scope/source/target owned 校验。"""

    def __init__(
            self,
            *,
            artifact_revision_resolver: ArtifactRevisionResolverPort | None = None,
            sanitizer: FeedbackSanitizer | None = None,
    ) -> None:
        self._artifact_revision_resolver = artifact_revision_resolver
        self._sanitizer = sanitizer or FeedbackSanitizer()

    async def validate_record_command(
            self,
            *,
            uow: IUnitOfWork,
            command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand,
            resolved_scope_kind: FeedbackScopeKind,
    ) -> FeedbackScopeResult:
        access_scope = self._require_scope(command.access_scope)
        source_event = await self._load_source_event(
            workflow_run=uow.workflow_run,
            access_scope=access_scope,
            command=command,
        )
        self._validate_controlled_user_feedback_source(command=command, source_event=source_event)
        effective_source_run_id = command.source_ref.source_run_id or source_event.run_id
        await self._validate_source_owned_refs(
            uow=uow,
            access_scope=access_scope,
            command=command,
            effective_source_run_id=effective_source_run_id,
        )
        await self._validate_target(uow=uow, access_scope=access_scope, command=command)
        return self._build_scope_result(
            access_scope=access_scope,
            requested_feedback_scope_kind=resolved_scope_kind,
            requested_scope_id=command.requested_scope_id,
            source_run_id=effective_source_run_id,
            target_run_id=command.target_ref.target_run_id,
            current_run_id=command.current_run_id_at_record_time or (
                str(access_scope.run_id) if access_scope.run_id else None
            ),
        )

    async def validate_resolution_command(
            self,
            *,
            uow: IUnitOfWork,
            command: FeedbackResolutionCommand,
    ) -> FeedbackScopeResult:
        access_scope = self._require_scope(command.access_scope)
        if command.resolution_source_event_id is not None:
            await self._load_resolution_source_event(
                workflow_run=uow.workflow_run,
                access_scope=access_scope,
                source_event_id=command.resolution_source_event_id,
                feedback_scope_kind=command.requested_feedback_scope_kind,
                requested_scope_id=command.requested_scope_id,
            )
        feedback_scope_kind = command.requested_feedback_scope_kind or (
            FeedbackScopeKind.RUN if access_scope.run_id else FeedbackScopeKind.SESSION
        )
        scope_id = command.requested_scope_id or (
            str(access_scope.run_id) if feedback_scope_kind == FeedbackScopeKind.RUN else str(access_scope.session_id)
        )
        return self._build_scope_result(
            access_scope=access_scope,
            requested_feedback_scope_kind=feedback_scope_kind,
            requested_scope_id=scope_id,
            source_run_id=str(access_scope.run_id) if access_scope.run_id else None,
            target_run_id=str(access_scope.run_id) if access_scope.run_id else None,
            current_run_id=str(access_scope.run_id) if access_scope.run_id else None,
        )

    def build_snapshot_scope(
            self,
            *,
            access_scope: AccessScopeResult,
            feedback_scope_kind: FeedbackScopeKind,
            requested_scope_id: str | None,
    ) -> FeedbackSnapshotScopeResult:
        scope = self._require_scope(access_scope)
        if feedback_scope_kind == FeedbackScopeKind.RUN:
            run_id = str(requested_scope_id or scope.run_id or "").strip()
            if not run_id:
                raise self._issue(FeedbackReasonCode.FEEDBACK_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "run snapshot 缺少 run_id")
            if scope.run_id and str(scope.run_id) != run_id:
                raise self._issue(FeedbackReasonCode.FEEDBACK_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "run snapshot scope 与当前 access scope 不一致")
            return FeedbackSnapshotScopeResult(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                workspace_id=str(scope.workspace_id),
                feedback_scope_kind=FeedbackScopeKind.RUN,
                scope_id=run_id,
                current_run_id_at_snapshot_time=run_id,
            )
        if requested_scope_id is not None and str(requested_scope_id).strip() != str(scope.session_id):
            raise self._issue(FeedbackReasonCode.FEEDBACK_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "session snapshot scope_id 必须等于 session_id")
        return FeedbackSnapshotScopeResult(
            user_id=scope.user_id,
            session_id=str(scope.session_id),
            workspace_id=str(scope.workspace_id),
            feedback_scope_kind=FeedbackScopeKind.SESSION,
            scope_id=str(requested_scope_id or scope.session_id),
            current_run_id_at_snapshot_time=str(scope.run_id) if scope.run_id else None,
        )

    def _require_scope(self, scope: AccessScopeResult | None) -> AccessScopeResult:
        if scope is None or not str(scope.user_id or "").strip():
            raise self._issue(FeedbackReasonCode.FEEDBACK_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "feedback access scope 缺失")
        if not str(scope.session_id or "").strip() or not str(scope.workspace_id or "").strip():
            raise self._issue(FeedbackReasonCode.FEEDBACK_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "feedback access scope 必须包含 session_id 和 workspace_id")
        return scope

    async def _load_source_event(
            self,
            *,
            workflow_run: Any,
            access_scope: AccessScopeResult,
            command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand,
    ) -> WorkflowRunEventRecord:
        event_id = str(command.source_ref.source_event_id or "").strip()
        if not event_id:
            raise self._source_issue_for_command(command, "缺少 source_event_id")
        if command.source_ref.source_run_id:
            record = await workflow_run.get_event_record_by_event_id(
                user_id=access_scope.user_id,
                session_id=str(access_scope.session_id),
                run_id=str(command.source_ref.source_run_id),
                event_id=event_id,
            )
        else:
            record = await workflow_run.get_event_record_by_event_id_in_session(
                user_id=access_scope.user_id,
                session_id=str(access_scope.session_id),
                event_id=event_id,
            )
        if record is None:
            raise self._source_issue_for_command(command, "source event 不存在或不属于当前 session")
        if command.source_ref.source_run_id and record.run_id != command.source_ref.source_run_id:
            raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISMATCH, FeedbackGapKind.RECORD_FAILED, "source event 与 source_run_id 不一致")
        expected_event_type = SOURCE_EVENT_TYPE_BY_KIND.get(command.source_ref.source_kind)
        if expected_event_type and record.event_type != expected_event_type:
            raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISMATCH, FeedbackGapKind.RECORD_FAILED, f"source event_type 必须是 {expected_event_type}")
        return record

    def _validate_controlled_user_feedback_source(
            self,
            *,
            command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand,
            source_event: WorkflowRunEventRecord,
    ) -> None:
        if command.kind != FeedbackKind.USER_FEEDBACK or command.category != FeedbackCategory.CONTINUE_CANCELLED:
            return
        if command.source_ref.source_kind != FeedbackSourceKind.FEEDBACK_INPUT:
            raise self._issue(
                FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISMATCH,
                FeedbackGapKind.RECORD_FAILED,
                "continue_cancelled 必须使用 feedback_input source event",
            )
        if command.target_ref.target_type != FeedbackTargetType.WAIT_EVENT:
            raise self._issue(
                FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH,
                FeedbackGapKind.RECORD_FAILED,
                "continue_cancelled 必须指向 wait_event target",
            )
        if not isinstance(source_event.event_payload, FeedbackInputEvent):
            raise self._issue(
                FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISMATCH,
                FeedbackGapKind.RECORD_FAILED,
                "continue_cancelled source event payload 必须是 feedback_input",
            )
        payload = source_event.event_payload.payload
        if payload.source_action != "continue_cancelled":
            raise self._issue(
                FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISMATCH,
                FeedbackGapKind.RECORD_FAILED,
                "continue_cancelled source_action 不匹配",
            )
        if payload.target_ref != command.target_ref:
            raise self._issue(
                FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH,
                FeedbackGapKind.RECORD_FAILED,
                "continue_cancelled source event target 与 command target 不一致",
            )

    async def _load_resolution_source_event(
            self,
            *,
            workflow_run: Any,
            access_scope: AccessScopeResult,
            source_event_id: str,
            feedback_scope_kind: FeedbackScopeKind | None,
            requested_scope_id: str | None,
    ) -> None:
        record = await workflow_run.get_event_record_by_event_id_in_session(
            user_id=access_scope.user_id,
            session_id=str(access_scope.session_id),
            event_id=source_event_id,
        )
        if record is None:
            raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISSING, FeedbackGapKind.SOURCE_MISSING, "resolution source event 不存在或不属于当前 session")
        if feedback_scope_kind == FeedbackScopeKind.RUN and requested_scope_id and record.run_id != requested_scope_id:
            raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISMATCH, FeedbackGapKind.RECORD_FAILED, "resolution source event 与 run scope 不一致")

    async def _validate_source_owned_refs(
            self,
            *,
            uow: IUnitOfWork,
            access_scope: AccessScopeResult,
            command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand,
            effective_source_run_id: str | None,
    ) -> None:
        source_kind = command.source_ref.source_kind
        if source_kind == FeedbackSourceKind.FINAL_DELIVERY:
            raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "final_delivery resolver 尚未接入")
        if source_kind in {
            FeedbackSourceKind.FEEDBACK_INPUT,
            FeedbackSourceKind.MESSAGE_EVENT,
            FeedbackSourceKind.WAIT_EVENT,
            FeedbackSourceKind.TOOL_EVENT,
            FeedbackSourceKind.SELF_REVIEW,
            FeedbackSourceKind.EVALUATION,
        }:
            return
        if source_kind == FeedbackSourceKind.SANDBOX_FACT:
            fact_ids = self._collect_ids(command.source_ref.source_record_refs, "fact_id")
            if not fact_ids:
                raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "sandbox_fact source 缺少 fact_id 引用")
            facts = await uow.sandbox_fact.list_by_ids(user_id=access_scope.user_id, session_id=str(access_scope.session_id), fact_ids=fact_ids, limit=len(fact_ids))
            if len(facts) != len(set(fact_ids)):
                raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "sandbox_fact source 引用不存在或不属于当前 session")
            self._validate_source_record_run_ids(
                source_kind=source_kind,
                source_run_id=effective_source_run_id,
                records=facts,
            )
            return
        if source_kind in {FeedbackSourceKind.EVIDENCE, FeedbackSourceKind.EVIDENCE_GAP}:
            evidence_ids = self._collect_ids(command.source_ref.source_record_refs, "evidence_id")
            if not evidence_ids:
                raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "evidence source 缺少 evidence_id 引用")
            evidence_list = await uow.evidence.list_by_ids(user_id=access_scope.user_id, session_id=str(access_scope.session_id), evidence_ids=evidence_ids, limit=len(evidence_ids))
            if len(evidence_list) != len(set(evidence_ids)):
                raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "evidence source 引用不存在或不属于当前 session")
            self._validate_source_record_run_ids(
                source_kind=source_kind,
                source_run_id=effective_source_run_id,
                records=evidence_list,
            )
            return
        if source_kind == FeedbackSourceKind.SAFETY_AUDIT:
            audit_ids = self._collect_ids(command.source_ref.source_record_refs, "audit_id")
            if not audit_ids:
                raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "safety_audit source 缺少 audit_id 引用")
            if len(set(audit_ids)) != 1:
                raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "safety_audit source 必须且只能引用一个 audit_id")
            audit = await uow.safety_audit.get_by_scope(user_id=access_scope.user_id, session_id=str(access_scope.session_id), audit_id=audit_ids[0])
            if audit is None:
                raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "safety_audit source 引用不存在或不属于当前 session")
            self._validate_source_record_run_ids(
                source_kind=source_kind,
                source_run_id=effective_source_run_id,
                records=[audit],
            )
            return
        if source_kind == FeedbackSourceKind.ARTIFACT_REVISION:
            await self._validate_artifact_source_ref(
                access_scope=access_scope,
                source_ref=command.source_ref,
                effective_source_run_id=effective_source_run_id,
            )

    async def _validate_artifact_source_ref(
            self,
            *,
            access_scope: AccessScopeResult,
            source_ref: FeedbackSourceRefResult,
            effective_source_run_id: str | None,
    ) -> None:
        if self._artifact_revision_resolver is None:
            raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "artifact revision resolver 未装配")
        artifact_id = self._collect_ids(source_ref.source_record_refs, "artifact_id")
        revision_id = self._collect_ids(source_ref.source_record_refs, "revision_id")
        content_hash = self._collect_ids(source_ref.source_record_refs, "content_hash")
        if not artifact_id or not revision_id or not content_hash:
            raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "artifact source 缺少 artifact/revision/content_hash 引用")
        resolved = await self._artifact_revision_resolver.resolve(
            ArtifactRevisionResolveCommand(
                user_id=access_scope.user_id,
                workspace_id=str(access_scope.workspace_id),
                session_id=str(access_scope.session_id),
                artifact_id=artifact_id[0],
                revision_id=revision_id[0],
                content_hash=content_hash[0],
                run_id=effective_source_run_id,
                source_run_id=effective_source_run_id,
            )
        )
        if resolved.status != ArtifactRevisionResolveStatus.RESOLVED:
            raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, str(resolved.reason_code or "artifact source 解析失败"))

    async def _validate_target(
            self,
            *,
            uow: IUnitOfWork,
            access_scope: AccessScopeResult,
            command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand,
    ) -> None:
        target_ref = command.target_ref
        source_ref = command.source_ref
        if target_ref.target_type == FeedbackTargetType.FINAL_DELIVERY or source_ref.source_kind == FeedbackSourceKind.FINAL_DELIVERY:
            raise self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING, FeedbackGapKind.SOURCE_INCOMPLETE, "final_delivery resolver 尚未接入")
        if target_ref.target_type in {FeedbackTargetType.RUN, FeedbackTargetType.STEP, FeedbackTargetType.TOOL_CALL, FeedbackTargetType.USER_GOAL}:
            await self._validate_target_run_ownership(
                workflow_run=uow.workflow_run,
                access_scope=access_scope,
                target_ref=target_ref,
            )
            return
        if target_ref.target_type in {FeedbackTargetType.MESSAGE_EVENT, FeedbackTargetType.WAIT_EVENT}:
            await self._validate_target_event(workflow_run=uow.workflow_run, access_scope=access_scope, target_ref=target_ref)
            return
        if target_ref.target_type == FeedbackTargetType.ARTIFACT_REVISION:
            await self._validate_artifact_target(access_scope=access_scope, source_ref=source_ref, target_ref=target_ref)
            return
        if target_ref.target_type == FeedbackTargetType.SANDBOX_FACT:
            facts = await uow.sandbox_fact.list_by_ids(user_id=access_scope.user_id, session_id=str(access_scope.session_id), fact_ids=[target_ref.target_id], limit=1)
            if not facts:
                raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_MISSING, FeedbackGapKind.RECORD_FAILED, "sandbox_fact target 不存在或不属于当前 session")
            if target_ref.target_run_id and str(getattr(facts[0], "run_id", "") or "") != target_ref.target_run_id:
                raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "sandbox_fact target_run_id 不一致")
            return
        if target_ref.target_type in {FeedbackTargetType.EVIDENCE, FeedbackTargetType.EVIDENCE_GAP}:
            evidence_list = await uow.evidence.list_by_ids(user_id=access_scope.user_id, session_id=str(access_scope.session_id), evidence_ids=[target_ref.target_id], limit=1)
            if not evidence_list:
                raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_MISSING, FeedbackGapKind.RECORD_FAILED, "evidence target 不存在或不属于当前 session")
            if target_ref.target_run_id and str(getattr(evidence_list[0], "run_id", "") or "") != target_ref.target_run_id:
                raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "evidence target_run_id 不一致")
            return
        if target_ref.target_type == FeedbackTargetType.SAFETY_AUDIT:
            audit = await uow.safety_audit.get_by_scope(user_id=access_scope.user_id, session_id=str(access_scope.session_id), audit_id=target_ref.target_id)
            if audit is None:
                raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_MISSING, FeedbackGapKind.RECORD_FAILED, "safety_audit target 不存在或不属于当前 session")
            if target_ref.target_run_id and str(getattr(audit, "run_id", "") or "") != target_ref.target_run_id:
                raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "safety_audit target_run_id 不一致")

    async def _validate_target_event(self, *, workflow_run: Any, access_scope: AccessScopeResult, target_ref: FeedbackTargetRefResult) -> None:
        record = await workflow_run.get_event_record_by_event_id_in_session(user_id=access_scope.user_id, session_id=str(access_scope.session_id), event_id=target_ref.target_id)
        if record is None:
            raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_MISSING, FeedbackGapKind.RECORD_FAILED, f"{target_ref.target_type.value} target 不存在或不属于当前 session")
        expected_event_type = TARGET_EVENT_TYPE_BY_TYPE[target_ref.target_type]
        if record.event_type != expected_event_type:
            raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, f"{target_ref.target_type.value} target event_type 必须是 {expected_event_type}")
        if target_ref.target_run_id != record.run_id:
            raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, f"{target_ref.target_type.value} target_run_id 与 event.run_id 不一致")

    async def _validate_artifact_target(self, *, access_scope: AccessScopeResult, source_ref: FeedbackSourceRefResult, target_ref: FeedbackTargetRefResult) -> None:
        if self._artifact_revision_resolver is None:
            raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_REVISION_MISSING, FeedbackGapKind.RECORD_FAILED, "artifact revision resolver 未装配")
        resolved = await self._artifact_revision_resolver.resolve(
            ArtifactRevisionResolveCommand(
                user_id=access_scope.user_id,
                workspace_id=str(access_scope.workspace_id),
                session_id=str(access_scope.session_id),
                artifact_id=target_ref.target_id,
                revision_id=str(target_ref.target_revision_id),
                content_hash=str(target_ref.target_content_hash),
                run_id=target_ref.target_run_id,
                source_run_id=source_ref.source_run_id,
            )
        )
        if resolved.status != ArtifactRevisionResolveStatus.RESOLVED:
            reason_code = FeedbackReasonCode.FEEDBACK_TARGET_REVISION_MISSING if resolved.status == ArtifactRevisionResolveStatus.NOT_FOUND else FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH
            raise self._issue(reason_code, FeedbackGapKind.RECORD_FAILED, str(resolved.reason_code or "artifact revision 解析失败"))

    async def _validate_target_run_ownership(
            self,
            *,
            workflow_run: Any,
            access_scope: AccessScopeResult,
            target_ref: FeedbackTargetRefResult,
    ) -> None:
        target_run_id = str(target_ref.target_run_id or "").strip()
        if not target_run_id:
            raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, f"{target_ref.target_type.value} target 缺少 target_run_id")
        run_record = await workflow_run.get_by_id_for_user_session(
            run_id=target_run_id,
            user_id=access_scope.user_id,
            session_id=str(access_scope.session_id),
        )
        if run_record is None:
            raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, f"{target_ref.target_type.value} target_run_id 不属于当前 session")

    def _build_scope_result(
            self,
            *,
            access_scope: AccessScopeResult,
            requested_feedback_scope_kind: FeedbackScopeKind,
            requested_scope_id: str | None,
            source_run_id: str | None,
            target_run_id: str | None,
            current_run_id: str | None,
    ) -> FeedbackScopeResult:
        normalized_source_run_id = str(source_run_id or "").strip() or None
        normalized_target_run_id = str(target_run_id or "").strip() or None
        normalized_current_run_id = str(current_run_id or "").strip() or None
        if requested_feedback_scope_kind == FeedbackScopeKind.RUN:
            run_id = str(requested_scope_id or normalized_current_run_id or access_scope.run_id or "").strip()
            if not run_id:
                raise self._issue(FeedbackReasonCode.FEEDBACK_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "run scope 缺少 run_id")
            if normalized_source_run_id != run_id or normalized_target_run_id != run_id or normalized_current_run_id != run_id:
                raise self._issue(FeedbackReasonCode.FEEDBACK_TARGET_SCOPE_MISMATCH, FeedbackGapKind.RECORD_FAILED, "run scope 要求 scope/source/target/current run 一致")
            return FeedbackScopeResult(
                user_id=access_scope.user_id,
                session_id=str(access_scope.session_id),
                workspace_id=str(access_scope.workspace_id),
                feedback_scope_kind=FeedbackScopeKind.RUN,
                scope_id=run_id,
                run_id=run_id,
                source_run_id=run_id,
                target_run_id=run_id,
                current_run_id_at_record_time=run_id,
            )
        return FeedbackScopeResult(
            user_id=access_scope.user_id,
            session_id=str(access_scope.session_id),
            workspace_id=str(access_scope.workspace_id),
            feedback_scope_kind=FeedbackScopeKind.SESSION,
            scope_id=str(access_scope.session_id),
            run_id=normalized_current_run_id,
            source_run_id=normalized_source_run_id,
            target_run_id=normalized_target_run_id,
            current_run_id_at_record_time=normalized_current_run_id,
        )

    def _source_issue_for_command(
            self,
            command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand,
            diagnostic_summary: str,
    ) -> FeedbackScopeValidationError:
        if command.kind == FeedbackKind.RUNTIME_FEEDBACK and command.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE:
            return self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE, FeedbackGapKind.SOURCE_INCOMPLETE, diagnostic_summary)
        return self._issue(FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISSING, FeedbackGapKind.SOURCE_MISSING, diagnostic_summary)

    def _collect_ids(self, refs: list[dict[str, str | None]], key: str) -> list[str]:
        values: list[str] = []
        for item in refs:
            value = item.get(key)
            normalized = str(value or "").strip()
            if normalized:
                values.append(normalized)
        return values

    def _validate_source_record_run_ids(
            self,
            *,
            source_kind: FeedbackSourceKind,
            source_run_id: str | None,
            records: list[object],
    ) -> None:
        expected_run_id = str(source_run_id or "").strip()
        if not expected_run_id:
            return
        for record in records:
            record_run_id = str(getattr(record, "run_id", "") or "").strip()
            if record_run_id != expected_run_id:
                raise self._issue(
                    FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING,
                    FeedbackGapKind.SOURCE_INCOMPLETE,
                    f"{source_kind.value} source 引用的 run_id 与 source_run_id 不一致",
                )

    def _issue(
            self,
            reason_code: FeedbackReasonCode,
            gap_kind: FeedbackGapKind,
            diagnostic_summary: str,
    ) -> FeedbackScopeValidationError:
        return FeedbackScopeValidationError(
            FeedbackValidationIssue(
                reason_code=reason_code,
                gap_kind=gap_kind,
                diagnostic_summary=self._sanitizer.sanitize_diagnostic_summary(diagnostic_summary),
            )
        )
