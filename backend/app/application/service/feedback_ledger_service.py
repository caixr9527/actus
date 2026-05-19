#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback Ledger PR2 应用服务。"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Callable

from app.application.service.feedback_ledger_common import (
    FeedbackLedgerError,
    FeedbackRequiredRecordError,
    FeedbackRetentionPolicy,
    FeedbackSanitizer,
    FeedbackScopeValidationError,
    FeedbackSeverityPolicy,
)
from app.application.service.feedback_ledger_support import FeedbackScopeValidator
from app.application.service.feedback_snapshot_builder import FeedbackSnapshotBuilder, FeedbackSnapshotPolicy
from app.domain.models.feedback import (
    FeedbackCategory,
    FeedbackGapResult,
    FeedbackKind,
    FeedbackRecord,
    FeedbackRecorderPort,
    FeedbackReasonCode,
    FeedbackResolutionCommand,
    FeedbackResolutionResult,
    FeedbackScopeKind,
    FeedbackScopeResult,
    FeedbackSeverity,
    FeedbackSnapshotProviderPort,
    FeedbackSnapshotResult,
    FeedbackSnapshotScopeResult,
    FeedbackSnapshotStage,
    FeedbackSourceRefResult,
    FeedbackStatus,
    FeedbackTargetType,
    FeedbackWriteResult,
    QualityFeedbackCommand,
    RuntimeFeedbackCommand,
    UserFeedbackCommand,
    build_feedback_record_result,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.feedback_runtime_ports import ArtifactRevisionResolverPort

logger = logging.getLogger(__name__)


class FeedbackLedgerService(FeedbackRecorderPort, FeedbackSnapshotProviderPort):
    """Feedback Ledger PR2 应用层入口。"""

    def __init__(
            self,
            *,
            uow_factory: Callable[[], IUnitOfWork],
            sanitizer: FeedbackSanitizer | None = None,
            severity_policy: FeedbackSeverityPolicy | None = None,
            retention_policy: FeedbackRetentionPolicy | None = None,
            snapshot_policy: FeedbackSnapshotPolicy | None = None,
            artifact_revision_resolver: ArtifactRevisionResolverPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._sanitizer = sanitizer or FeedbackSanitizer()
        self._severity_policy = severity_policy or FeedbackSeverityPolicy()
        self._retention_policy = retention_policy or FeedbackRetentionPolicy()
        self._snapshot_policy = snapshot_policy or FeedbackSnapshotPolicy()
        self._scope_validator = FeedbackScopeValidator(
            artifact_revision_resolver=artifact_revision_resolver,
            sanitizer=self._sanitizer,
        )
        self._snapshot_builder = FeedbackSnapshotBuilder(
            policy=self._snapshot_policy,
            sanitizer=self._sanitizer,
        )

    async def record_user_feedback(self, command: UserFeedbackCommand) -> FeedbackWriteResult:
        return await self._record(command=command, required=True)

    async def record_runtime_feedback(self, command: RuntimeFeedbackCommand) -> FeedbackWriteResult:
        return await self._record(command=command, required=False)

    async def record_quality_feedback(self, command: QualityFeedbackCommand) -> FeedbackWriteResult:
        return await self._record(command=command, required=False)

    async def resolve_feedback(self, command: FeedbackResolutionCommand) -> FeedbackWriteResult:
        now = command.updated_at
        try:
            async with self._uow_factory() as uow:
                scope = await self._scope_validator.validate_resolution_command(uow=uow, command=command)
                record = await uow.feedback.update_resolution(
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    feedback_scope_kind=scope.feedback_scope_kind,
                    scope_id=scope.scope_id,
                    feedback_id=command.feedback_id,
                    resolution=command.resolution,
                    updated_at=command.updated_at,
                )
                logger.info(
                    "feedback_resolution_updated user_id=%s session_id=%s scope_kind=%s scope_id=%s feedback_id=%s status=%s",
                    scope.user_id,
                    scope.session_id,
                    scope.feedback_scope_kind.value,
                    scope.scope_id,
                    record.id,
                    record.status.value,
                )
                return FeedbackWriteResult(
                    success=True,
                    created=False,
                    reused=False,
                    reason_code=record.reason_code,
                    feedback_id=record.id,
                    record_ref=build_feedback_record_result(record),
                    scope=record.scope,
                    source_ref=record.source_ref,
                    target_ref=record.target_ref,
                    resolution=record.resolution,
                    gap=None,
                    created_at=record.created_at,
                )
        except FeedbackScopeValidationError as exc:
            gap = self._build_resolution_gap(issue=exc.issue, command=command, now=now)
            logger.warning("feedback_scope_validation_failed detail=%s", exc)
            return FeedbackWriteResult(
                success=False,
                created=False,
                reused=False,
                reason_code=gap.reason_code,
                feedback_id=None,
                record_ref=None,
                scope=None,
                source_ref=None,
                target_ref=None,
                resolution=None,
                gap=gap,
                created_at=None,
            )

    async def build_snapshot(
            self,
            *,
            access_scope: AccessScopeResult,
            stage: FeedbackSnapshotStage,
            feedback_scope_kind: FeedbackScopeKind,
            requested_scope_id: str | None = None,
            runtime_gaps: list[FeedbackGapResult] | None = None,
    ) -> FeedbackSnapshotResult:
        now = datetime.now()
        async with self._uow_factory() as uow:
            scope = self._scope_validator.build_snapshot_scope(
                access_scope=access_scope,
                feedback_scope_kind=feedback_scope_kind,
                requested_scope_id=requested_scope_id,
            )
            records = await uow.feedback.list_by_scope(
                user_id=scope.user_id,
                session_id=scope.session_id,
                feedback_scope_kind=scope.feedback_scope_kind,
                scope_id=scope.scope_id,
                limit=self._snapshot_policy.candidate_scan_limit,
            )
        snapshot = self._snapshot_builder.build(
            scope=scope,
            stage=stage,
            records=records,
            runtime_gaps=runtime_gaps,
            now=now,
        )
        logger.info(
            "feedback_snapshot_built user_id=%s session_id=%s scope_kind=%s scope_id=%s stage=%s included=%s gaps=%s",
            scope.user_id,
            scope.session_id,
            scope.feedback_scope_kind.value,
            scope.scope_id,
            stage.value,
            len(snapshot.included_feedback_ids),
            len(snapshot.feedback_gaps),
        )
        return snapshot

    async def list_by_run(
            self,
            *,
            access_scope: AccessScopeResult,
            run_id: str | None = None,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        scope = self._scope_validator._require_scope(access_scope)
        actual_run_id = str(run_id or scope.run_id or "").strip()
        if not actual_run_id:
            raise FeedbackLedgerError("feedback run 查询缺少 run_id")
        async with self._uow_factory() as uow:
            return await uow.feedback.list_by_run(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                run_id=actual_run_id,
                limit=limit,
            )

    async def list_by_step(
            self,
            *,
            access_scope: AccessScopeResult,
            run_id: str | None = None,
            step_id: str | None = None,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        scope = self._scope_validator._require_scope(access_scope)
        actual_run_id = str(run_id or scope.run_id or "").strip()
        actual_step_id = str(step_id or scope.current_step_id or "").strip()
        if not actual_run_id or not actual_step_id:
            raise FeedbackLedgerError("feedback step 查询缺少 run_id 或 step_id")
        async with self._uow_factory() as uow:
            return await uow.feedback.list_by_step(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                run_id=actual_run_id,
                step_id=actual_step_id,
                limit=limit,
            )

    async def list_by_target(
            self,
            *,
            access_scope: AccessScopeResult,
            target_type: FeedbackTargetType,
            target_id: str,
            target_revision_id: str | None = None,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        scope = self._scope_validator._require_scope(access_scope)
        async with self._uow_factory() as uow:
            return await uow.feedback.list_by_target(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                target_type=target_type,
                target_id=target_id,
                target_revision_id=target_revision_id,
                limit=limit,
            )

    async def list_by_source_event(
            self,
            *,
            access_scope: AccessScopeResult,
            source_event_id: str,
            limit: int = 100,
    ) -> list[FeedbackRecord]:
        scope = self._scope_validator._require_scope(access_scope)
        async with self._uow_factory() as uow:
            return await uow.feedback.list_by_source_event(
                user_id=scope.user_id,
                session_id=str(scope.session_id),
                source_event_id=source_event_id,
                limit=limit,
            )

    async def _record(
            self,
            *,
            command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand,
            required: bool,
    ) -> FeedbackWriteResult:
        now = datetime.now()
        try:
            retention = self._retention_policy.decide(
                kind=command.kind,
                category=command.category,
                requested_scope_kind=command.requested_feedback_scope_kind,
                current_run_id=command.current_run_id_at_record_time or (
                    str(command.access_scope.run_id) if command.access_scope.run_id else None
                ),
                now=now,
            )
            async with self._uow_factory() as uow:
                scope = await self._scope_validator.validate_record_command(
                    uow=uow,
                    command=command,
                    resolved_scope_kind=retention.resolved_scope_kind,
                )
                normalized_source_ref = command.source_ref.model_copy(
                    update={"source_run_id": scope.source_run_id}
                )
                feedback_summary, prompt_safe_summary = self._sanitizer.summarize(
                    kind=command.kind,
                    summary_hint=self._extract_summary_hint(command),
                    source_summary=normalized_source_ref.source_summary,
                    reason_code=command.reason_code,
                )
                severity = self._severity_policy.classify(
                    command_kind=command.kind,
                    category=command.category,
                    reason_code=command.reason_code,
                    classification=command.classification,
                    target_ref=command.target_ref,
                    source_ref=normalized_source_ref,
                )
                if (
                        command.kind == FeedbackKind.RUNTIME_FEEDBACK
                        and command.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE
                        and severity != FeedbackSeverity.WARNING
                ):
                    raise FeedbackLedgerError("feedback_source_incomplete 弱反馈必须是 warning 级")
                feedback_key = self._build_feedback_key(
                    kind=command.kind,
                    category=command.category,
                    target_ref=command.target_ref,
                    reason_code=command.reason_code,
                )
                dedupe_key = self._build_dedupe_key(
                    scope=scope,
                    feedback_key=feedback_key,
                    source_ref=normalized_source_ref,
                )
                record = FeedbackRecord(
                    id=f"fb_{uuid.uuid4().hex}",
                    scope=scope,
                    source_ref=normalized_source_ref,
                    target_ref=command.target_ref,
                    resolution=FeedbackResolutionResult(status=FeedbackStatus.OPEN),
                    feedback_summary=feedback_summary,
                    prompt_safe_summary=prompt_safe_summary,
                    classification=command.classification,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    workspace_id=scope.workspace_id,
                    run_id=scope.run_id,
                    feedback_scope_kind=scope.feedback_scope_kind,
                    scope_id=scope.scope_id,
                    source_run_id=scope.source_run_id,
                    target_run_id=scope.target_run_id,
                    step_id=command.step_id,
                    kind=command.kind,
                    category=command.category,
                    status=FeedbackStatus.OPEN,
                    severity=severity,
                    source_kind=normalized_source_ref.source_kind,
                    source_event_id=normalized_source_ref.source_event_id,
                    source_record_refs=list(normalized_source_ref.source_record_refs),
                    target_type=command.target_ref.target_type,
                    target_id=command.target_ref.target_id,
                    target_revision_id=command.target_ref.target_revision_id,
                    target_content_hash=command.target_ref.target_content_hash,
                    dedupe_key=dedupe_key,
                    feedback_key=feedback_key,
                    reason_code=command.reason_code,
                    resolution_reason_code=None,
                    resolved_by_ref=None,
                    decay_policy=retention.decay_policy,
                    expires_at=retention.expires_at,
                    ttl_scope=retention.ttl_scope,
                    profile_hash=command.profile_hash,
                    origin=command.origin,
                    trust_level=command.trust_level,
                    privacy_level=command.privacy_level,
                    retention_policy=command.retention_policy,
                    created_at=now,
                    updated_at=now,
                )
                saved = await uow.feedback.save_once(record)
                created = saved.id == record.id
                logger.info(
                    "feedback_record_%s user_id=%s session_id=%s scope_kind=%s scope_id=%s feedback_id=%s kind=%s category=%s severity=%s",
                    "created" if created else "reused",
                    saved.user_id,
                    saved.session_id,
                    saved.feedback_scope_kind.value,
                    saved.scope_id,
                    saved.id,
                    saved.kind.value,
                    saved.category.value,
                    saved.severity.value,
                )
                return FeedbackWriteResult(
                    success=True,
                    created=created,
                    reused=not created,
                    reason_code=saved.reason_code,
                    feedback_id=saved.id,
                    record_ref=build_feedback_record_result(saved),
                    scope=saved.scope,
                    source_ref=saved.source_ref,
                    target_ref=saved.target_ref,
                    resolution=None,
                    gap=None,
                    created_at=saved.created_at,
                )
        except FeedbackScopeValidationError as exc:
            gap = self._build_gap_from_issue(
                issue=exc.issue,
                command=command,
                stage=self._infer_gap_stage(command),
                now=now,
            )
            logger.warning(
                "feedback_record_failed user_id=%s session_id=%s kind=%s category=%s reason_code=%s detail=%s",
                command.access_scope.user_id if command.access_scope else "",
                command.access_scope.session_id if command.access_scope else "",
                command.kind.value,
                command.category.value,
                gap.reason_code.value,
                exc,
            )
            if required:
                raise FeedbackRequiredRecordError(str(exc))
            return FeedbackWriteResult(
                success=False,
                created=False,
                reused=False,
                reason_code=gap.reason_code,
                feedback_id=None,
                record_ref=None,
                scope=None,
                source_ref=None,
                target_ref=None,
                resolution=None,
                gap=gap,
                created_at=None,
            )

    def _infer_gap_stage(
            self,
            command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand,
    ) -> FeedbackSnapshotStage:
        if command.kind == FeedbackKind.QUALITY_FEEDBACK:
            return FeedbackSnapshotStage.FINAL_GATE
        if command.kind == FeedbackKind.RUNTIME_FEEDBACK:
            return FeedbackSnapshotStage.REPLAN
        return FeedbackSnapshotStage.SUMMARY

    def _build_gap_from_issue(
            self,
            *,
            issue,
            command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand,
            stage: FeedbackSnapshotStage,
            now: datetime,
    ) -> FeedbackGapResult:
        scope = self._try_build_gap_scope(command.access_scope, command.requested_feedback_scope_kind)
        return FeedbackGapResult(
            gap_kind=issue.gap_kind,
            reason_code=issue.reason_code,
            source_ref=issue.source_ref,
            target_ref=issue.target_ref,
            stage=stage,
            scope=scope,
            diagnostic_summary=issue.diagnostic_summary,
            created_at=now,
        )

    def _build_resolution_gap(
            self,
            *,
            issue,
            command: FeedbackResolutionCommand,
            now: datetime,
    ) -> FeedbackGapResult:
        scope = self._try_build_gap_scope(command.access_scope, command.requested_feedback_scope_kind)
        return FeedbackGapResult(
            gap_kind=issue.gap_kind,
            reason_code=issue.reason_code,
            source_ref=issue.source_ref,
            target_ref=issue.target_ref,
            stage=FeedbackSnapshotStage.REPLAN,
            scope=scope,
            diagnostic_summary=issue.diagnostic_summary,
            created_at=now,
        )

    def _try_build_gap_scope(
            self,
            access_scope: AccessScopeResult | None,
            requested_scope_kind: FeedbackScopeKind | None,
    ) -> FeedbackSnapshotScopeResult | None:
        if access_scope is None:
            return None
        session_id = str(access_scope.session_id or "").strip()
        workspace_id = str(access_scope.workspace_id or "").strip()
        if not access_scope.user_id or not session_id or not workspace_id:
            return None
        scope_kind = requested_scope_kind or (
            FeedbackScopeKind.RUN if access_scope.run_id else FeedbackScopeKind.SESSION
        )
        if scope_kind == FeedbackScopeKind.RUN:
            run_id = str(access_scope.run_id or "").strip()
            if not run_id:
                return None
            return FeedbackSnapshotScopeResult(
                user_id=access_scope.user_id,
                session_id=session_id,
                workspace_id=workspace_id,
                feedback_scope_kind=FeedbackScopeKind.RUN,
                scope_id=run_id,
                current_run_id_at_snapshot_time=run_id,
            )
        return FeedbackSnapshotScopeResult(
            user_id=access_scope.user_id,
            session_id=session_id,
            workspace_id=workspace_id,
            feedback_scope_kind=FeedbackScopeKind.SESSION,
            scope_id=session_id,
            current_run_id_at_snapshot_time=str(access_scope.run_id) if access_scope.run_id else None,
        )

    def _extract_summary_hint(self, command: UserFeedbackCommand | RuntimeFeedbackCommand | QualityFeedbackCommand) -> str | None:
        if isinstance(command, UserFeedbackCommand):
            return command.intent.summary_hint
        return None

    @staticmethod
    def _build_feedback_key(
            *,
            kind: FeedbackKind,
            category: FeedbackCategory,
            target_ref,
            reason_code,
    ) -> str:
        payload = {
            "kind": kind.value,
            "category": category.value,
            "target_type": target_ref.target_type.value,
            "target_id": target_ref.target_id,
            "target_run_id": target_ref.target_run_id,
            "target_revision_id": target_ref.target_revision_id,
            "target_content_hash": target_ref.target_content_hash,
            "reason_code": reason_code.value,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return f"feedback_key:{digest}"

    @staticmethod
    def _build_dedupe_key(
            *,
            scope: FeedbackScopeResult,
            feedback_key: str,
            source_ref: FeedbackSourceRefResult,
    ) -> str:
        payload = {
            "user_id": scope.user_id,
            "session_id": scope.session_id,
            "feedback_scope_kind": scope.feedback_scope_kind.value,
            "scope_id": scope.scope_id,
            "feedback_key": feedback_key,
            "source_event_id": source_ref.source_event_id,
            "source_record_refs": source_ref.source_record_refs,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return f"feedback_dedupe:{digest}"


__all__ = [
    "FeedbackLedgerError",
    "FeedbackLedgerService",
    "FeedbackRequiredRecordError",
    "FeedbackScopeValidationError",
]
