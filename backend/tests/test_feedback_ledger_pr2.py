from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

from app.application.service.feedback_ledger_service import (
    FeedbackLedgerService,
    FeedbackRequiredRecordError,
    FeedbackScopeValidationError,
)
from app.application.service.feedback_snapshot_builder import FeedbackSnapshotBuilder, FeedbackSnapshotPolicy
from app.domain.models import ArtifactEvent, FeedbackInputEvent, MessageEvent, ToolEvent, WaitEvent, WorkflowRunEventRecord
from app.domain.models.feedback import (
    FeedbackCategory,
    FeedbackClassificationResult,
    FeedbackDataOrigin,
    FeedbackInputEventPayloadResult,
    FeedbackGapKind,
    FeedbackGapResult,
    FeedbackKind,
    FeedbackPromptSafeSummaryResult,
    FeedbackReasonCode,
    FeedbackRecord,
    FeedbackResolutionCommand,
    FeedbackResolutionReasonCode,
    FeedbackResolutionResult,
    FeedbackScopeKind,
    FeedbackScopeResult,
    FeedbackSeverity,
    FeedbackSnapshotScopeResult,
    FeedbackSnapshotStage,
    FeedbackSourceConfidence,
    FeedbackSourceKind,
    FeedbackSourceRefResult,
    FeedbackStatus,
    FeedbackSummaryKind,
    FeedbackSummaryResult,
    FeedbackTargetRefResult,
    FeedbackTargetType,
    QualityFeedbackCommand,
    RuntimeFeedbackCommand,
    UserFeedbackCommand,
    UserFeedbackIntent,
    UserFeedbackIntentKind,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)


def _access_scope(
        *,
        user_id: str = "user-1",
        session_id: str = "session-1",
        workspace_id: str = "workspace-1",
        run_id: str | None = "run-1",
        current_step_id: str | None = "step-1",
) -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id=user_id,
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        run_id=run_id,
        current_step_id=current_step_id,
    )


def _classification(
        *,
        source_confidence: FeedbackSourceConfidence = FeedbackSourceConfidence.STRONG,
        data_origin: FeedbackDataOrigin = FeedbackDataOrigin.USER,
) -> FeedbackClassificationResult:
    return FeedbackClassificationResult(
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.SESSION_BOUND,
        trust_level=DataTrustLevel.USER_PROVIDED if data_origin == FeedbackDataOrigin.USER else DataTrustLevel.SYSTEM_GENERATED,
        source_confidence=source_confidence,
        data_origin=data_origin,
    )


def _feedback_summary(text: str = "用户指出上一轮结果有误，需要修正。") -> FeedbackSummaryResult:
    return FeedbackSummaryResult(
        summary_text=text,
        summary_kind=FeedbackSummaryKind.USER_STATED,
        is_truncated=False,
        truncation_reason=None,
        language="zh-CN",
    )


def _prompt_safe_summary(text: str = "用户指出上一轮结果有误，需要修正。") -> FeedbackPromptSafeSummaryResult:
    return FeedbackPromptSafeSummaryResult(
        summary_text=text,
        is_truncated=False,
        sanitization_applied=False,
        sanitization_reasons=[],
        prompt_visible=True,
    )


def _source_ref(
        *,
        source_kind: FeedbackSourceKind = FeedbackSourceKind.MESSAGE_EVENT,
        source_event_id: str = "evt-source-1",
        source_run_id: str | None = "run-1",
        source_summary: str = "用户给出纠错反馈。",
        source_record_refs: list[dict[str, str | None]] | None = None,
) -> FeedbackSourceRefResult:
    refs = source_record_refs or [{"event_id": source_event_id}]
    return FeedbackSourceRefResult(
        source_kind=source_kind,
        source_event_id=source_event_id,
        source_record_refs=refs,
        source_run_id=source_run_id,
        source_step_id="step-1" if source_run_id else None,
        source_summary=source_summary,
    )


def _target_ref(
        *,
        target_type: FeedbackTargetType = FeedbackTargetType.MESSAGE_EVENT,
        target_id: str = "evt-target-1",
        target_run_id: str | None = "run-1",
        target_revision_id: str | None = None,
        target_content_hash: str | None = None,
) -> FeedbackTargetRefResult:
    return FeedbackTargetRefResult(
        target_type=target_type,
        target_id=target_id,
        target_run_id=target_run_id,
        target_revision_id=target_revision_id,
        target_content_hash=target_content_hash,
    )


def _user_command(
        *,
        access_scope: AccessScopeResult | None = None,
        source_ref: FeedbackSourceRefResult | None = None,
        target_ref: FeedbackTargetRefResult | None = None,
        category: FeedbackCategory = FeedbackCategory.CORRECTION,
        reason_code: FeedbackReasonCode = FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
        summary_hint: str = "上一轮答案和需求不一致",
) -> UserFeedbackCommand:
    scope = access_scope or _access_scope()
    actual_target = target_ref or _target_ref()
    return UserFeedbackCommand(
        access_scope=scope,
        source_ref=source_ref or _source_ref(),
        target_ref=actual_target,
        category=category,
        reason_code=reason_code,
        feedback_summary=_feedback_summary(),
        prompt_safe_summary=_prompt_safe_summary(),
        classification=_classification(),
        requested_feedback_scope_kind=FeedbackScopeKind.RUN if scope.run_id else FeedbackScopeKind.SESSION,
        requested_scope_id=str(scope.run_id) if scope.run_id else str(scope.session_id),
        current_run_id_at_record_time=str(scope.run_id) if scope.run_id else None,
        step_id=scope.current_step_id,
        profile_hash="sha256:" + "a" * 64,
        decay_policy="placeholder",
        ttl_scope="placeholder",
        origin=DataOrigin.USER_MESSAGE,
        trust_level=DataTrustLevel.USER_PROVIDED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.SESSION_BOUND,
        intent=UserFeedbackIntent(
            intent_kind=UserFeedbackIntentKind.CORRECTION if category == FeedbackCategory.CORRECTION else UserFeedbackIntentKind.DISSATISFACTION,
            target_ref=actual_target,
            reason_code=reason_code,
            summary_hint=summary_hint,
        ),
    )


def _runtime_command(
        *,
        access_scope: AccessScopeResult | None = None,
        source_ref: FeedbackSourceRefResult | None = None,
        target_ref: FeedbackTargetRefResult | None = None,
        category: FeedbackCategory = FeedbackCategory.TOOL_FAILURE,
        reason_code: FeedbackReasonCode = FeedbackReasonCode.TOOL_FAILED,
        source_confidence: FeedbackSourceConfidence = FeedbackSourceConfidence.STRONG,
) -> RuntimeFeedbackCommand:
    scope = access_scope or _access_scope()
    classification = _classification(
        source_confidence=source_confidence,
        data_origin=FeedbackDataOrigin.RUNTIME,
    )
    return RuntimeFeedbackCommand(
        access_scope=scope,
        source_ref=source_ref or _source_ref(
            source_kind=FeedbackSourceKind.TOOL_EVENT,
            source_event_id="evt-tool-1",
            source_summary="工具调用失败。",
        ),
        target_ref=target_ref or _target_ref(target_type=FeedbackTargetType.TOOL_CALL, target_id="call-1"),
        category=category,
        reason_code=reason_code,
        feedback_summary=_feedback_summary("运行时诊断反馈。"),
        prompt_safe_summary=_prompt_safe_summary("运行时诊断反馈。"),
        classification=classification,
        requested_feedback_scope_kind=FeedbackScopeKind.RUN if scope.run_id else FeedbackScopeKind.SESSION,
        requested_scope_id=str(scope.run_id) if scope.run_id else str(scope.session_id),
        current_run_id_at_record_time=str(scope.run_id) if scope.run_id else None,
        step_id=scope.current_step_id,
        profile_hash="sha256:" + "b" * 64,
        decay_policy="placeholder",
        ttl_scope="placeholder",
        origin=DataOrigin.SYSTEM_OPERATIONAL,
        trust_level=DataTrustLevel.SYSTEM_GENERATED,
        privacy_level=classification.privacy_level,
        retention_policy=classification.retention_policy,
    )


def _quality_command(
        *,
        access_scope: AccessScopeResult | None = None,
        source_ref: FeedbackSourceRefResult | None = None,
        target_ref: FeedbackTargetRefResult | None = None,
        category: FeedbackCategory = FeedbackCategory.ARTIFACT_UNUSABLE,
        reason_code: FeedbackReasonCode = FeedbackReasonCode.ARTIFACT_UNUSABLE,
) -> QualityFeedbackCommand:
    scope = access_scope or _access_scope()
    classification = _classification(
        source_confidence=FeedbackSourceConfidence.STRONG,
        data_origin=FeedbackDataOrigin.SYSTEM_QUALITY,
    )
    return QualityFeedbackCommand(
        access_scope=scope,
        source_ref=source_ref or _source_ref(
            source_kind=FeedbackSourceKind.SAFETY_AUDIT,
            source_event_id="evt-audit-1",
            source_summary="受控质量裁决。",
            source_record_refs=[{"audit_id": "audit-1"}],
        ),
        target_ref=target_ref or _target_ref(
            target_type=FeedbackTargetType.ARTIFACT_REVISION,
            target_id="artifact-1",
            target_run_id="run-1",
            target_revision_id="rev-1",
            target_content_hash="sha256:artifact",
        ),
        category=category,
        reason_code=reason_code,
        feedback_summary=_feedback_summary("质量裁决反馈。"),
        prompt_safe_summary=_prompt_safe_summary("质量裁决反馈。"),
        classification=classification,
        requested_feedback_scope_kind=FeedbackScopeKind.SESSION if category in {
            FeedbackCategory.ARTIFACT_UNUSABLE,
            FeedbackCategory.FINAL_ANSWER_MISMATCH,
        } else FeedbackScopeKind.RUN,
        requested_scope_id=str(scope.session_id) if category in {
            FeedbackCategory.ARTIFACT_UNUSABLE,
            FeedbackCategory.FINAL_ANSWER_MISMATCH,
        } else str(scope.run_id),
        current_run_id_at_record_time=str(scope.run_id) if scope.run_id else None,
        step_id=scope.current_step_id,
        profile_hash=None,
        decay_policy="placeholder",
        ttl_scope="placeholder",
        origin=DataOrigin.SYSTEM_OPERATIONAL,
        trust_level=DataTrustLevel.SYSTEM_GENERATED,
        privacy_level=classification.privacy_level,
        retention_policy=classification.retention_policy,
    )


def _record(
        *,
        feedback_id: str,
        scope_kind: FeedbackScopeKind = FeedbackScopeKind.RUN,
        scope_id: str = "run-1",
        run_id: str | None = "run-1",
        source_run_id: str | None = "run-1",
        target_run_id: str | None = "run-1",
        kind: FeedbackKind = FeedbackKind.USER_FEEDBACK,
        category: FeedbackCategory = FeedbackCategory.CORRECTION,
        reason_code: FeedbackReasonCode = FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
        severity: FeedbackSeverity = FeedbackSeverity.ERROR,
        status: FeedbackStatus = FeedbackStatus.OPEN,
        source_kind: FeedbackSourceKind = FeedbackSourceKind.MESSAGE_EVENT,
        source_event_id: str = "evt-source-1",
        target_type: FeedbackTargetType = FeedbackTargetType.MESSAGE_EVENT,
        target_id: str = "evt-target-1",
        target_revision_id: str | None = None,
        target_content_hash: str | None = None,
        decay_policy: str = "session_persistent",
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        resolution_reason_code: FeedbackResolutionReasonCode | None = None,
) -> FeedbackRecord:
    created = created_at or datetime(2026, 5, 19, 10, 0, 0)
    updated = updated_at or created
    classification = _classification(
        data_origin=FeedbackDataOrigin.USER if kind == FeedbackKind.USER_FEEDBACK else (
            FeedbackDataOrigin.RUNTIME if kind == FeedbackKind.RUNTIME_FEEDBACK else FeedbackDataOrigin.SYSTEM_QUALITY
        )
    )
    origin = DataOrigin.USER_MESSAGE if kind == FeedbackKind.USER_FEEDBACK else DataOrigin.SYSTEM_OPERATIONAL
    trust_level = classification.trust_level
    privacy_level = classification.privacy_level
    retention_policy = classification.retention_policy
    scope = FeedbackScopeResult(
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        feedback_scope_kind=scope_kind,
        scope_id=scope_id,
        run_id=run_id,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        current_run_id_at_record_time=run_id,
    )
    resolution = FeedbackResolutionResult(status=FeedbackStatus.OPEN)
    if status != FeedbackStatus.OPEN:
        resolution = FeedbackResolutionResult(
            status=status,
            resolution_reason_code=resolution_reason_code or FeedbackResolutionReasonCode.RESOLVED_BY_REPLAN,
            resolved_by_ref={"feedback_id": "fb-resolver"},
            resolved_at=updated,
            resolution_summary="后续流程已处理。",
        )
    return FeedbackRecord(
        id=feedback_id,
        scope=scope,
        source_ref=_source_ref(
            source_kind=source_kind,
            source_event_id=source_event_id,
            source_run_id=source_run_id,
            source_summary=f"{feedback_id} summary",
            source_record_refs=[{"event_id": source_event_id}],
        ),
        target_ref=_target_ref(
            target_type=target_type,
            target_id=target_id,
            target_run_id=target_run_id,
            target_revision_id=target_revision_id,
            target_content_hash=target_content_hash,
        ),
        resolution=resolution,
        feedback_summary=_feedback_summary(feedback_id),
        prompt_safe_summary=_prompt_safe_summary(feedback_id),
        classification=classification,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id=run_id,
        feedback_scope_kind=scope_kind,
        scope_id=scope_id,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        step_id="step-1",
        kind=kind,
        category=category,
        status=status,
        severity=severity,
        source_kind=source_kind,
        source_event_id=source_event_id,
        source_record_refs=[{"event_id": source_event_id}],
        target_type=target_type,
        target_id=target_id,
        target_revision_id=target_revision_id,
        target_content_hash=target_content_hash,
        dedupe_key=f"dedupe:{feedback_id}",
        feedback_key=f"key:{feedback_id}",
        reason_code=reason_code,
        resolution_reason_code=resolution.resolution_reason_code,
        resolved_by_ref=resolution.resolved_by_ref,
        decay_policy=decay_policy,
        expires_at=None,
        ttl_scope="session" if "session" in decay_policy else "run",
        profile_hash=None,
        origin=origin,
        trust_level=trust_level,
        privacy_level=privacy_level,
        retention_policy=retention_policy,
        created_at=created,
        updated_at=updated,
    )


class _FeedbackRepo:
    def __init__(self, records: list[FeedbackRecord] | None = None) -> None:
        self.records = list(records or [])
        self.saved: list[FeedbackRecord] = []
        self.updated: list[dict] = []

    async def save_once(self, record: FeedbackRecord) -> FeedbackRecord:
        self.saved.append(record)
        for existing in self.records:
            if (
                    existing.user_id == record.user_id
                    and existing.session_id == record.session_id
                    and existing.feedback_scope_kind == record.feedback_scope_kind
                    and existing.scope_id == record.scope_id
                    and existing.dedupe_key == record.dedupe_key
            ):
                return existing
        self.records.append(record)
        return record

    async def list_by_scope(self, *, user_id: str, session_id: str, feedback_scope_kind: FeedbackScopeKind, scope_id: str, limit: int = 100):
        return [
            record for record in self.records
            if record.user_id == user_id
            and record.session_id == session_id
            and record.feedback_scope_kind == feedback_scope_kind
            and record.scope_id == scope_id
        ][:limit]

    async def list_by_run(self, *, user_id: str, session_id: str, run_id: str, limit: int = 100):
        return [record for record in self.records if record.user_id == user_id and record.session_id == session_id and record.run_id == run_id][:limit]

    async def list_by_step(self, *, user_id: str, session_id: str, run_id: str, step_id: str, limit: int = 100):
        return [
            record for record in self.records
            if record.user_id == user_id and record.session_id == session_id and record.run_id == run_id and record.step_id == step_id
        ][:limit]

    async def list_by_target(self, *, user_id: str, session_id: str, target_type: FeedbackTargetType, target_id: str, target_revision_id: str | None = None, limit: int = 100):
        return [
            record for record in self.records
            if record.user_id == user_id
            and record.session_id == session_id
            and record.target_type == target_type
            and record.target_id == target_id
            and record.target_revision_id == target_revision_id
        ][:limit]

    async def list_by_source_event(self, *, user_id: str, session_id: str, source_event_id: str, limit: int = 100):
        return [
            record for record in self.records
            if record.user_id == user_id and record.session_id == session_id and record.source_event_id == source_event_id
        ][:limit]

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
        self.updated.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "feedback_scope_kind": feedback_scope_kind,
                "scope_id": scope_id,
                "feedback_id": feedback_id,
                "resolution": resolution,
                "updated_at": updated_at,
            }
        )
        for index, record in enumerate(self.records):
            if (
                    record.user_id == user_id
                    and record.session_id == session_id
                    and record.feedback_scope_kind == feedback_scope_kind
                    and record.scope_id == scope_id
                    and record.id == feedback_id
            ):
                updated = record.model_copy(
                    update={
                        "resolution": resolution,
                        "status": resolution.status,
                        "resolution_reason_code": resolution.resolution_reason_code,
                        "resolved_by_ref": resolution.resolved_by_ref,
                        "updated_at": updated_at,
                    }
                )
                self.records[index] = updated
                return updated
        raise ValueError("not found")


class _WorkflowRunRepo:
    def __init__(self, records: dict[str, WorkflowRunEventRecord] | None = None) -> None:
        self.records = dict(records or {})
        self.by_run_calls: list[dict] = []
        self.by_session_calls: list[dict] = []
        self.run_scope_calls: list[dict] = []

    async def get_event_record_by_event_id(self, **kwargs):
        self.by_run_calls.append(kwargs)
        record = self.records.get(kwargs["event_id"])
        if record is None:
            return None
        if record.user_id != kwargs["user_id"] or record.session_id != kwargs["session_id"] or record.run_id != kwargs["run_id"]:
            return None
        return record

    async def get_event_record_by_event_id_in_session(self, **kwargs):
        self.by_session_calls.append(kwargs)
        record = self.records.get(kwargs["event_id"])
        if record is None:
            return None
        if record.user_id != kwargs["user_id"] or record.session_id != kwargs["session_id"]:
            return None
        return record

    async def get_by_id_for_user_session(self, *, run_id: str, user_id: str, session_id: str):
        self.run_scope_calls.append({"run_id": run_id, "user_id": user_id, "session_id": session_id})
        if not run_id.startswith("run-"):
            return None
        if run_id == "run-cross-session":
            return None
        if run_id == "run-missing":
            return None
        return SimpleNamespace(id=run_id, user_id=user_id, session_id=session_id)


class _SandboxFactRepo:
    def __init__(self, records: dict[str, object] | None = None) -> None:
        self.records = dict(records or {})

    async def list_by_ids(self, *, user_id: str, session_id: str, fact_ids: list[str], limit: int = 100):
        result = []
        for fact_id in fact_ids[:limit]:
            fact = self.records.get(fact_id)
            if fact is not None and fact.user_id == user_id and fact.session_id == session_id:
                result.append(fact)
        return result


class _EvidenceRepo:
    def __init__(self, records: dict[str, object] | None = None) -> None:
        self.records = dict(records or {})

    async def list_by_ids(self, *, user_id: str, session_id: str, evidence_ids: list[str], limit: int = 100):
        result = []
        for evidence_id in evidence_ids[:limit]:
            evidence = self.records.get(evidence_id)
            if evidence is not None and evidence.user_id == user_id and evidence.session_id == session_id:
                result.append(evidence)
        return result


class _SafetyAuditRepo:
    def __init__(self, records: dict[str, object] | None = None) -> None:
        self.records = dict(records or {})

    async def get_by_scope(self, *, user_id: str, session_id: str, audit_id: str):
        audit = self.records.get(audit_id)
        if audit is None or audit.user_id != user_id or audit.session_id != session_id:
            return None
        return audit


class _ArtifactResolver:
    def __init__(self, *, status: str = "resolved", revision_run_id: str = "run-1") -> None:
        self.status = status
        self.revision_run_id = revision_run_id
        self.calls: list[object] = []

    async def resolve(self, command):
        self.calls.append(command)
        if self.status != "resolved":
            return SimpleNamespace(status=self.status, reason_code="artifact_revision_not_found", revision=None)
        revision = SimpleNamespace(
            artifact_id=command.artifact_id,
            revision_id=command.revision_id,
            content_hash=command.content_hash,
            run_id=self.revision_run_id,
            source_run_id=command.source_run_id,
        )
        return SimpleNamespace(status="resolved", reason_code=None, revision=revision)


class _UoW:
    def __init__(
            self,
            *,
            feedback_repo: _FeedbackRepo | None = None,
            workflow_run_repo: _WorkflowRunRepo | None = None,
            sandbox_fact_repo: _SandboxFactRepo | None = None,
            evidence_repo: _EvidenceRepo | None = None,
            safety_audit_repo: _SafetyAuditRepo | None = None,
    ) -> None:
        self.feedback = feedback_repo or _FeedbackRepo()
        self.workflow_run = workflow_run_repo or _WorkflowRunRepo()
        self.sandbox_fact = sandbox_fact_repo or _SandboxFactRepo()
        self.evidence = evidence_repo or _EvidenceRepo()
        self.safety_audit = safety_audit_repo or _SafetyAuditRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self):
        return None

    async def rollback(self):
        return None


def _event_record(
        *,
        event_id: str,
        run_id: str = "run-1",
        session_id: str = "session-1",
        user_id: str = "user-1",
        event_type: str = "message",
):
    if event_type == "message":
        payload = MessageEvent(id=event_id, role="user", message="ok")
    elif event_type == "tool":
        payload = ToolEvent(
            id=event_id,
            step_id="step-1",
            tool_call_id="call-1",
            tool_name="tool",
            function_name="read_file",
            function_args={"path": "/tmp/a.txt"},
            function_result=None,
            status="called",
        )
    elif event_type == "wait":
        payload = WaitEvent(id=event_id, interrupt_id="interrupt-1", payload={"type": "confirm"})
    elif event_type == "feedback_input":
        target_ref = _target_ref(target_type=FeedbackTargetType.MESSAGE_EVENT, target_id="evt-target-1")
        payload = FeedbackInputEvent(
            id=event_id,
            payload=FeedbackInputEventPayloadResult(
                source_action="final_satisfaction",
                intent_kind=UserFeedbackIntentKind.DISSATISFACTION,
                target_ref=target_ref,
                reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
                sanitized_summary="用户反馈上一轮答案不满意",
                input_hash=f"feedback_input:{event_id}",
                runtime_metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "workspace_id": "workspace-1",
                    "source_run_id": run_id,
                    "target_run_id": target_ref.target_run_id,
                },
            ),
        )
    elif event_type == "artifact":
        payload = ArtifactEvent(
            id=event_id,
            payload={
                "artifact_refs": [
                    {
                        "artifact_id": "artifact-1",
                        "path": "/workspace/report.md",
                        "artifact_type": "file",
                        "delivery_state": "candidate",
                        "current_revision_id": "rev-1",
                        "latest_content_hash": "sha256:" + "a" * 64,
                    }
                ],
                "revision_refs": [
                    {
                        "artifact_id": "artifact-1",
                        "revision_id": "rev-1",
                        "content_hash": "sha256:" + "a" * 64,
                        "path": "/workspace/report.md",
                        "artifact_type": "file",
                        "delivery_state": "candidate",
                        "source_event_id": event_id,
                    }
                ],
                "counts": {"revision_count": 1},
                "summary": "artifact event",
                "source_event_ids": [event_id],
                "runtime_metadata": {},
            },
        )
    elif event_type == "sandbox_fact":
        payload = MessageEvent(id=event_id, role="assistant", message="sandbox fact projected")
    elif event_type == "safety_audit":
        payload = MessageEvent(id=event_id, role="assistant", message="audit")
    else:
        payload = MessageEvent(id=event_id, role="assistant", message="other")
    return WorkflowRunEventRecord(
        run_id=run_id,
        session_id=session_id,
        user_id=user_id,
        event_id=event_id,
        event_type=event_type,
        event_payload=payload,
    )


def _service(
        *,
        uow: _UoW,
        artifact_resolver: _ArtifactResolver | None = None,
) -> FeedbackLedgerService:
    return FeedbackLedgerService(
        uow_factory=lambda: uow,
        artifact_revision_resolver=artifact_resolver,
    )


def test_record_user_feedback_should_write_correction_record() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {
                "evt-source-1": _event_record(event_id="evt-source-1", event_type="message"),
                "evt-target-1": _event_record(event_id="evt-target-1", event_type="message"),
            }
        )
        feedback_repo = _FeedbackRepo()
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo))

        result = await service.record_user_feedback(_user_command())

        assert result.success is True
        assert result.created is True
        assert result.record_ref is not None
        assert result.record_ref.scope.feedback_scope_kind == FeedbackScopeKind.SESSION
        assert result.record_ref.severity == FeedbackSeverity.ERROR
        assert result.record_ref.reason_code == FeedbackReasonCode.USER_CORRECTED_REQUIREMENT
        assert feedback_repo.saved[0].decay_policy == "session_persistent"
        assert feedback_repo.saved[0].feedback_scope_kind == FeedbackScopeKind.SESSION

    asyncio.run(_run())


def test_record_runtime_feedback_should_return_source_missing_gap_when_event_missing() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=_WorkflowRunRepo()))

        result = await service.record_runtime_feedback(_runtime_command())

        assert result.success is False
        assert result.gap is not None
        assert result.gap.gap_kind == FeedbackGapKind.SOURCE_MISSING
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISSING
        assert result.gap.source_ref is None
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_runtime_feedback_should_allow_weak_incomplete_tool_feedback_with_valid_tool_event() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo({"evt-tool-1": _event_record(event_id="evt-tool-1", event_type="tool")})
        feedback_repo = _FeedbackRepo()
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo))

        result = await service.record_runtime_feedback(
            _runtime_command(
                target_ref=_target_ref(target_type=FeedbackTargetType.TOOL_CALL, target_id="call-1"),
                reason_code=FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE,
                source_confidence=FeedbackSourceConfidence.WEAK,
            )
        )

        assert result.success is True
        assert result.record_ref is not None
        assert result.record_ref.severity == FeedbackSeverity.WARNING
        assert feedback_repo.saved[0].reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE

    asyncio.run(_run())


def test_record_runtime_feedback_should_return_source_incomplete_gap_without_valid_source_event() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo()
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=_WorkflowRunRepo()))

        result = await service.record_runtime_feedback(
            _runtime_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.TOOL_EVENT,
                    source_event_id="evt-missing-tool",
                    source_summary="仅有日志摘要。",
                    source_record_refs=[{"log_line": "tool failed"}],
                ),
                reason_code=FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE,
                source_confidence=FeedbackSourceConfidence.WEAK,
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert result.gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_INCOMPLETE
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_quality_feedback_should_require_owned_audit_and_artifact_revision() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo({"evt-audit-1": _event_record(event_id="evt-audit-1", event_type="safety_audit")})
        safety_repo = _SafetyAuditRepo({"audit-1": SimpleNamespace(id="audit-1", user_id="user-1", session_id="session-1", run_id="run-1")})
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                safety_audit_repo=safety_repo,
            ),
            artifact_resolver=_ArtifactResolver(),
        )

        result = await service.record_quality_feedback(_quality_command())

        assert result.success is True
        assert result.record_ref is not None
        assert result.record_ref.scope.feedback_scope_kind == FeedbackScopeKind.SESSION
        assert result.record_ref.category == FeedbackCategory.ARTIFACT_UNUSABLE

    asyncio.run(_run())


def test_record_user_feedback_should_fail_closed_for_required_missing_source() -> None:
    async def _run() -> None:
        service = _service(uow=_UoW(workflow_run_repo=_WorkflowRunRepo()))

        try:
            await service.record_user_feedback(_user_command())
        except FeedbackRequiredRecordError:
            return
        raise AssertionError("expected FeedbackRequiredRecordError")

    asyncio.run(_run())


def test_record_should_reject_cross_session_target() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {
                "evt-source-1": _event_record(event_id="evt-source-1", event_type="message"),
                "evt-target-1": _event_record(event_id="evt-target-1", session_id="session-2", event_type="message"),
            }
        )
        service = _service(uow=_UoW(workflow_run_repo=workflow_repo))

        try:
            await service.record_user_feedback(_user_command())
        except FeedbackRequiredRecordError:
            return
        raise AssertionError("expected FeedbackRequiredRecordError")

    asyncio.run(_run())


def test_resolve_feedback_should_validate_resolution_source_event_ownership() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {
                "evt-resolve-1": _event_record(event_id="evt-resolve-1", session_id="session-2", event_type="message"),
            }
        )
        feedback_repo = _FeedbackRepo(records=[_record(feedback_id="fb-1")])
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo))

        command = FeedbackResolutionCommand(
            access_scope=_access_scope(),
            feedback_id="fb-1",
            requested_feedback_scope_kind=FeedbackScopeKind.RUN,
            requested_scope_id="run-1",
            resolution=FeedbackResolutionResult(
                status=FeedbackStatus.RESOLVED,
                resolution_reason_code=FeedbackResolutionReasonCode.RESOLVED_BY_REPLAN,
                resolved_by_ref={"event_id": "evt-resolve-1"},
                resolved_at=datetime(2026, 5, 19, 11, 0, 0),
                resolution_summary="已重规划处理。",
            ),
            updated_at=datetime(2026, 5, 19, 11, 0, 0),
            resolution_source_event_id="evt-resolve-1",
        )

        result = await service.resolve_feedback(command)

        assert result.success is False
        assert result.gap is not None
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISSING
        assert feedback_repo.updated == []

    asyncio.run(_run())


def test_build_snapshot_should_merge_runtime_gaps_and_keep_top_level_current_run() -> None:
    async def _run() -> None:
        records = [
            _record(
                feedback_id="fb-old",
            scope_kind=FeedbackScopeKind.SESSION,
            scope_id="session-1",
            run_id="run-1",
            source_run_id="run-prev",
            target_run_id="run-prev",
            kind=FeedbackKind.RUNTIME_FEEDBACK,
            category=FeedbackCategory.TOOL_FAILURE,
            reason_code=FeedbackReasonCode.TOOL_FAILED,
            severity=FeedbackSeverity.WARNING,
            source_kind=FeedbackSourceKind.TOOL_EVENT,
            source_event_id="evt-tool-old",
            target_type=FeedbackTargetType.TOOL_CALL,
            target_id="call-old",
            decay_policy="run_window",
        ),
        _record(
            feedback_id="fb-current",
            scope_kind=FeedbackScopeKind.SESSION,
            scope_id="session-1",
            run_id="run-1",
            source_run_id="run-1",
            target_run_id="run-1",
            kind=FeedbackKind.USER_FEEDBACK,
            category=FeedbackCategory.CORRECTION,
            reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
            severity=FeedbackSeverity.ERROR,
            source_kind=FeedbackSourceKind.MESSAGE_EVENT,
            source_event_id="evt-msg-current",
            target_type=FeedbackTargetType.MESSAGE_EVENT,
            target_id="evt-target-current",
            decay_policy="session_persistent",
        ),
    ]
        feedback_repo = _FeedbackRepo(records=records)
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=_WorkflowRunRepo()))

        runtime_gap = FeedbackGapResult(
            gap_kind=FeedbackGapKind.PROJECTION_MISSING,
            reason_code=FeedbackReasonCode.FEEDBACK_PROJECTION_GAP,
            source_ref=None,
            target_ref=None,
            stage=FeedbackSnapshotStage.REPLAN,
            scope=FeedbackSnapshotScopeResult(
                user_id="user-1",
                session_id="session-1",
                workspace_id="workspace-1",
                feedback_scope_kind=FeedbackScopeKind.SESSION,
                scope_id="session-1",
                current_run_id_at_snapshot_time="run-1",
            ),
            diagnostic_summary="feedback 投影存在暂时缺口。",
            created_at=datetime(2026, 5, 19, 12, 0, 0),
        )

        snapshot = await service.build_snapshot(
            access_scope=_access_scope(),
            stage=FeedbackSnapshotStage.REPLAN,
            feedback_scope_kind=FeedbackScopeKind.SESSION,
            runtime_gaps=[runtime_gap],
        )

        assert snapshot.scope.current_run_id_at_snapshot_time == "run-1"
        assert snapshot.source_run_id == "run-1"
        assert snapshot.feedback_gaps[0].reason_code == FeedbackReasonCode.FEEDBACK_PROJECTION_GAP
        assert "fb-current" in snapshot.included_feedback_ids
        assert "fb-old" not in snapshot.included_feedback_ids
        assert any(ref.feedback_id == "fb-old" and ref.excluded_by.value == "ttl" for ref in snapshot.excluded_feedback_refs)

    asyncio.run(_run())


def test_build_session_snapshot_should_include_continue_cancelled_targeting_old_run_wait_event() -> None:
    async def _run() -> None:
        record = _record(
            feedback_id="fb-continue-cancelled",
            scope_kind=FeedbackScopeKind.SESSION,
            scope_id="session-1",
            run_id="run-2",
            source_run_id="run-2",
            target_run_id="run-1",
            category=FeedbackCategory.CONTINUE_CANCELLED,
            reason_code=FeedbackReasonCode.USER_CONTINUED_CANCELLED,
            severity=FeedbackSeverity.INFO,
            source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
            source_event_id="evt-feedback-input-continue",
            target_type=FeedbackTargetType.WAIT_EVENT,
            target_id="evt-wait-old",
            decay_policy="session_window",
        )
        feedback_repo = _FeedbackRepo(records=[record])
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=_WorkflowRunRepo()))

        snapshot = await service.build_snapshot(
            access_scope=_access_scope(run_id="run-2"),
            stage=FeedbackSnapshotStage.FUTURE_REVIEW,
            feedback_scope_kind=FeedbackScopeKind.SESSION,
        )

        assert snapshot.scope.current_run_id_at_snapshot_time == "run-2"
        assert snapshot.source_run_id == "run-2"
        assert snapshot.included_feedback_ids == ["fb-continue-cancelled"]
        assert snapshot.open_feedback_items[0].source_run_id == "run-2"
        assert snapshot.open_feedback_items[0].target_run_id == "run-1"
        assert snapshot.open_feedback_items[0].target_ref.target_id == "evt-wait-old"

    asyncio.run(_run())


def test_record_continue_cancelled_run_scope_should_be_rejected_for_cross_run_target() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {
                "evt-feedback-input-continue": _event_record(
                    event_id="evt-feedback-input-continue",
                    run_id="run-2",
                    event_type="feedback_input",
                ),
                "evt-wait-old": _event_record(
                    event_id="evt-wait-old",
                    run_id="run-1",
                    event_type="wait",
                ),
            }
        )
        service = _service(uow=_UoW(feedback_repo=_FeedbackRepo(), workflow_run_repo=workflow_repo))
        command = _user_command(
            access_scope=_access_scope(run_id="run-2"),
            source_ref=_source_ref(
                source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
                source_event_id="evt-feedback-input-continue",
                source_run_id="run-2",
                source_record_refs=[
                    {"event_id": "evt-feedback-input-continue", "run_id": "run-2"},
                    {"event_id": "evt-wait-old", "run_id": "run-1"},
                ],
            ),
            target_ref=_target_ref(
                target_type=FeedbackTargetType.WAIT_EVENT,
                target_id="evt-wait-old",
                target_run_id="run-1",
            ),
            category=FeedbackCategory.CONTINUE_CANCELLED,
            reason_code=FeedbackReasonCode.USER_CONTINUED_CANCELLED,
            summary_hint="用户继续执行已取消任务",
        ).model_copy(
            update={
                "requested_feedback_scope_kind": FeedbackScopeKind.RUN,
                "requested_scope_id": "run-2",
                "current_run_id_at_record_time": "run-2",
                "step_id": None,
                "intent": UserFeedbackIntent(
                    intent_kind=UserFeedbackIntentKind.CONTINUE_CANCELLED,
                    target_ref=_target_ref(
                        target_type=FeedbackTargetType.WAIT_EVENT,
                        target_id="evt-wait-old",
                        target_run_id="run-1",
                    ),
                    reason_code=FeedbackReasonCode.USER_CONTINUED_CANCELLED,
                    summary_hint="用户继续执行已取消任务",
                ),
            }
        )

        try:
            await service.record_user_feedback(command)
        except FeedbackRequiredRecordError:
            return
        raise AssertionError("expected FeedbackRequiredRecordError")

    asyncio.run(_run())


def test_record_continue_cancelled_should_reject_forged_message_source() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {
                "evt-source-1": _event_record(event_id="evt-source-1", event_type="message"),
                "evt-wait-old": _event_record(event_id="evt-wait-old", event_type="wait"),
            }
        )
        feedback_repo = _FeedbackRepo()
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo))
        target_ref = _target_ref(
            target_type=FeedbackTargetType.WAIT_EVENT,
            target_id="evt-wait-old",
            target_run_id="run-1",
        )
        command = _user_command(
            source_ref=_source_ref(
                source_kind=FeedbackSourceKind.MESSAGE_EVENT,
                source_event_id="evt-source-1",
                source_run_id="run-1",
            ),
            target_ref=target_ref,
            category=FeedbackCategory.CONTINUE_CANCELLED,
            reason_code=FeedbackReasonCode.USER_CONTINUED_CANCELLED,
            summary_hint="用户继续执行已取消任务",
        ).model_copy(
            update={
                "requested_feedback_scope_kind": FeedbackScopeKind.SESSION,
                "requested_scope_id": "session-1",
                "intent": UserFeedbackIntent(
                    intent_kind=UserFeedbackIntentKind.CONTINUE_CANCELLED,
                    target_ref=target_ref,
                    reason_code=FeedbackReasonCode.USER_CONTINUED_CANCELLED,
                    summary_hint="用户继续执行已取消任务",
                ),
            }
        )

        try:
            await service.record_user_feedback(command)
        except FeedbackRequiredRecordError:
            assert feedback_repo.saved == []
            return
        raise AssertionError("expected FeedbackRequiredRecordError")

    asyncio.run(_run())


def test_snapshot_builder_should_dedupe_by_open_severity_and_created_at() -> None:
    now = datetime(2026, 5, 19, 12, 0, 0)
    scope = FeedbackSnapshotScopeResult(
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        feedback_scope_kind=FeedbackScopeKind.RUN,
        scope_id="run-1",
        current_run_id_at_snapshot_time="run-1",
    )
    records = [
        _record(
            feedback_id="fb-resolved",
            severity=FeedbackSeverity.ERROR,
            status=FeedbackStatus.RESOLVED,
            resolution_reason_code=FeedbackResolutionReasonCode.RESOLVED_BY_REPLAN,
            created_at=now - timedelta(minutes=20),
            updated_at=now - timedelta(minutes=5),
            kind=FeedbackKind.RUNTIME_FEEDBACK,
            category=FeedbackCategory.TOOL_FAILURE,
            reason_code=FeedbackReasonCode.TOOL_FAILED,
            source_kind=FeedbackSourceKind.TOOL_EVENT,
            source_event_id="evt-tool-1",
            target_type=FeedbackTargetType.TOOL_CALL,
            target_id="call-1",
            decay_policy="run_window",
        ),
        _record(
            feedback_id="fb-open-warning",
            severity=FeedbackSeverity.WARNING,
            created_at=now - timedelta(minutes=10),
            kind=FeedbackKind.RUNTIME_FEEDBACK,
            category=FeedbackCategory.TOOL_FAILURE,
            reason_code=FeedbackReasonCode.TOOL_FAILED,
            source_kind=FeedbackSourceKind.TOOL_EVENT,
            source_event_id="evt-tool-2",
            target_type=FeedbackTargetType.TOOL_CALL,
            target_id="call-1",
            decay_policy="run_window",
        ),
        _record(
            feedback_id="fb-open-error",
            severity=FeedbackSeverity.ERROR,
            created_at=now - timedelta(minutes=1),
            kind=FeedbackKind.RUNTIME_FEEDBACK,
            category=FeedbackCategory.TOOL_FAILURE,
            reason_code=FeedbackReasonCode.TOOL_FAILED,
            source_kind=FeedbackSourceKind.TOOL_EVENT,
            source_event_id="evt-tool-3",
            target_type=FeedbackTargetType.TOOL_CALL,
            target_id="call-1",
            decay_policy="run_window",
        ),
    ]
    builder = FeedbackSnapshotBuilder(policy=FeedbackSnapshotPolicy())

    snapshot = builder.build(
        scope=scope,
        stage=FeedbackSnapshotStage.REPLAN,
        records=records,
        runtime_gaps=[],
        now=now,
    )

    assert snapshot.included_feedback_ids == ["fb-open-error"]
    assert any(ref.feedback_id == "fb-open-warning" and ref.excluded_by.value == "dedupe" for ref in snapshot.excluded_feedback_refs)
    assert any(ref.feedback_id == "fb-resolved" and ref.excluded_by.value == "dedupe" for ref in snapshot.excluded_feedback_refs)


def test_build_snapshot_should_default_scope_ids_when_requested_scope_missing() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo(records=[])
        service = _service(uow=_UoW(feedback_repo=feedback_repo))

        run_snapshot = await service.build_snapshot(
            access_scope=_access_scope(run_id="run-9"),
            stage=FeedbackSnapshotStage.EXECUTE,
            feedback_scope_kind=FeedbackScopeKind.RUN,
        )
        session_snapshot = await service.build_snapshot(
            access_scope=_access_scope(run_id="run-9"),
            stage=FeedbackSnapshotStage.FUTURE_REVIEW,
            feedback_scope_kind=FeedbackScopeKind.SESSION,
        )

        assert run_snapshot.scope.scope_id == "run-9"
        assert run_snapshot.scope.current_run_id_at_snapshot_time == "run-9"
        assert session_snapshot.scope.scope_id == "session-1"
        assert session_snapshot.source_run_id == "run-9"

    asyncio.run(_run())


def test_build_snapshot_should_reject_mismatched_session_scope_id() -> None:
    async def _run() -> None:
        feedback_repo = _FeedbackRepo(records=[])
        service = _service(uow=_UoW(feedback_repo=feedback_repo))

        try:
            await service.build_snapshot(
                access_scope=_access_scope(run_id="run-9"),
                stage=FeedbackSnapshotStage.FUTURE_REVIEW,
                feedback_scope_kind=FeedbackScopeKind.SESSION,
                requested_scope_id="session-other",
            )
        except FeedbackScopeValidationError as exc:
            assert exc.issue.reason_code == FeedbackReasonCode.FEEDBACK_SCOPE_MISMATCH
            return
        raise AssertionError("expected FeedbackScopeValidationError")

    asyncio.run(_run())


def test_record_should_allow_same_session_cross_run_target() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo({"evt-source-1": _event_record(event_id="evt-source-1", event_type="message")})
        feedback_repo = _FeedbackRepo()
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo))

        result = await service.record_user_feedback(
            _user_command(
                target_ref=_target_ref(target_type=FeedbackTargetType.RUN, target_id="run-2", target_run_id="run-2")
            )
        )

        assert result.success is True
        assert result.record_ref is not None
        assert result.record_ref.target_ref.target_run_id == "run-2"

    asyncio.run(_run())


def test_record_should_reject_cross_session_run_target() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo({"evt-source-1": _event_record(event_id="evt-source-1", event_type="message")})
        feedback_repo = _FeedbackRepo()
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo))

        try:
            await service.record_user_feedback(
                _user_command(
                    target_ref=_target_ref(target_type=FeedbackTargetType.RUN, target_id="run-cross-session", target_run_id="run-cross-session")
                )
            )
        except FeedbackRequiredRecordError:
            assert feedback_repo.saved == []
            return
        raise AssertionError("expected FeedbackRequiredRecordError")

    asyncio.run(_run())


def test_record_should_reject_missing_run_target() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo({"evt-source-1": _event_record(event_id="evt-source-1", event_type="message")})
        feedback_repo = _FeedbackRepo()
        service = _service(uow=_UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo))

        try:
            # PR2 仅校验 STEP/TOOL_CALL target_run_id 归属，不在此处校验 tool_call 实体是否存在。
            await service.record_user_feedback(
                _user_command(
                    target_ref=_target_ref(target_type=FeedbackTargetType.TOOL_CALL, target_id="call-1", target_run_id="run-missing")
                )
            )
        except FeedbackRequiredRecordError:
            assert feedback_repo.saved == []
            return
        raise AssertionError("expected FeedbackRequiredRecordError")

    asyncio.run(_run())


def test_record_should_return_gap_for_final_delivery_target() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo({"evt-audit-1": _event_record(event_id="evt-audit-1", event_type="safety_audit")})
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                safety_audit_repo=_SafetyAuditRepo(
                    records={"audit-1": SimpleNamespace(id="audit-1", user_id="user-1", session_id="session-1", run_id="run-1")}
                ),
            )
        )

        result = await service.record_quality_feedback(
            _quality_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.SAFETY_AUDIT,
                    source_event_id="evt-audit-1",
                    source_summary="受控质量裁决。",
                    source_record_refs=[{"audit_id": "audit-1"}],
                ),
                target_ref=_target_ref(target_type=FeedbackTargetType.FINAL_DELIVERY, target_id="final-1", target_run_id="run-1"),
                category=FeedbackCategory.FINAL_GATE_BLOCKED,
                reason_code=FeedbackReasonCode.FINAL_GATE_BLOCKED,
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert result.gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_fail_closed_when_sandbox_fact_source_missing() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo({"evt-source-1": _event_record(event_id="evt-source-1", event_type="sandbox_fact")})
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                sandbox_fact_repo=_SandboxFactRepo(records={}),
            )
        )

        result = await service.record_runtime_feedback(
            _runtime_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.SANDBOX_FACT,
                    source_event_id="evt-source-1",
                    source_summary="fact source",
                    source_record_refs=[{"fact_id": "fact-missing"}],
                )
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_fail_closed_when_sandbox_fact_source_run_mismatch() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {"evt-source-2": _event_record(event_id="evt-source-2", run_id="run-2", event_type="sandbox_fact")}
        )
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                sandbox_fact_repo=_SandboxFactRepo(
                    records={
                        "fact-1": SimpleNamespace(id="fact-1", user_id="user-1", session_id="session-1", run_id="run-1")
                    }
                ),
            )
        )

        result = await service.record_runtime_feedback(
            _runtime_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.SANDBOX_FACT,
                    source_event_id="evt-source-2",
                    source_run_id="run-2",
                    source_summary="fact source",
                    source_record_refs=[{"fact_id": "fact-1"}],
                )
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert result.gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_fail_closed_when_sandbox_fact_source_run_inferred_from_event_mismatch() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {"evt-source-inferred-2": _event_record(event_id="evt-source-inferred-2", run_id="run-2", event_type="sandbox_fact")}
        )
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                sandbox_fact_repo=_SandboxFactRepo(
                    records={
                        "fact-inferred-1": SimpleNamespace(id="fact-inferred-1", user_id="user-1", session_id="session-1", run_id="run-1")
                    }
                ),
            )
        )

        result = await service.record_runtime_feedback(
            _runtime_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.SANDBOX_FACT,
                    source_event_id="evt-source-inferred-2",
                    source_run_id=None,
                    source_summary="fact source inferred run",
                    source_record_refs=[{"fact_id": "fact-inferred-1"}],
                )
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert result.gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_fail_closed_when_evidence_target_missing() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo({"evt-tool-1": _event_record(event_id="evt-tool-1", event_type="tool")})
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                evidence_repo=_EvidenceRepo(records={}),
            )
        )

        result = await service.record_runtime_feedback(
            _runtime_command(
                target_ref=_target_ref(target_type=FeedbackTargetType.EVIDENCE, target_id="ev-missing", target_run_id="run-1")
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_fail_closed_when_evidence_source_run_mismatch() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {"evt-evidence-2": _event_record(event_id="evt-evidence-2", run_id="run-2", event_type="evidence")}
        )
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                evidence_repo=_EvidenceRepo(
                    records={
                        "ev-1": SimpleNamespace(id="ev-1", user_id="user-1", session_id="session-1", run_id="run-1")
                    }
                ),
            )
        )

        result = await service.record_runtime_feedback(
            _runtime_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.EVIDENCE_GAP,
                    source_event_id="evt-evidence-2",
                    source_run_id="run-2",
                    source_summary="evidence gap source",
                    source_record_refs=[{"evidence_id": "ev-1"}],
                )
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert result.gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_fail_closed_when_evidence_source_run_inferred_from_event_mismatch() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {"evt-evidence-inferred-2": _event_record(event_id="evt-evidence-inferred-2", run_id="run-2", event_type="evidence")}
        )
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                evidence_repo=_EvidenceRepo(
                    records={
                        "ev-inferred-1": SimpleNamespace(id="ev-inferred-1", user_id="user-1", session_id="session-1", run_id="run-1")
                    }
                ),
            )
        )

        result = await service.record_runtime_feedback(
            _runtime_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.EVIDENCE_GAP,
                    source_event_id="evt-evidence-inferred-2",
                    source_run_id=None,
                    source_summary="evidence gap source inferred run",
                    source_record_refs=[{"evidence_id": "ev-inferred-1"}],
                )
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert result.gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_fail_closed_when_safety_audit_target_missing() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo({"evt-tool-1": _event_record(event_id="evt-tool-1", event_type="tool")})
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                safety_audit_repo=_SafetyAuditRepo(records={}),
            )
        )

        result = await service.record_runtime_feedback(
            _runtime_command(
                target_ref=_target_ref(target_type=FeedbackTargetType.SAFETY_AUDIT, target_id="audit-missing", target_run_id="run-1")
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_fail_closed_when_safety_audit_source_run_mismatch() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {"evt-audit-2": _event_record(event_id="evt-audit-2", run_id="run-2", event_type="safety_audit")}
        )
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                safety_audit_repo=_SafetyAuditRepo(
                    records={
                        "audit-2": SimpleNamespace(id="audit-2", user_id="user-1", session_id="session-1", run_id="run-1")
                    }
                ),
            ),
            artifact_resolver=_ArtifactResolver(),
        )

        result = await service.record_quality_feedback(
            _quality_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.SAFETY_AUDIT,
                    source_event_id="evt-audit-2",
                    source_run_id="run-2",
                    source_summary="audit source",
                    source_record_refs=[{"audit_id": "audit-2"}],
                )
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert result.gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_fail_closed_when_safety_audit_source_run_inferred_from_event_mismatch() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {"evt-audit-inferred-2": _event_record(event_id="evt-audit-inferred-2", run_id="run-2", event_type="safety_audit")}
        )
        feedback_repo = _FeedbackRepo()
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
                safety_audit_repo=_SafetyAuditRepo(
                    records={
                        "audit-inferred-1": SimpleNamespace(id="audit-inferred-1", user_id="user-1", session_id="session-1", run_id="run-1")
                    }
                ),
            ),
            artifact_resolver=_ArtifactResolver(),
        )

        result = await service.record_quality_feedback(
            _quality_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.SAFETY_AUDIT,
                    source_event_id="evt-audit-inferred-2",
                    source_run_id=None,
                    source_summary="audit source inferred run",
                    source_record_refs=[{"audit_id": "audit-inferred-1"}],
                )
            )
        )

        assert result.success is False
        assert result.gap is not None
        assert result.gap.gap_kind == FeedbackGapKind.SOURCE_INCOMPLETE
        assert result.gap.reason_code == FeedbackReasonCode.FEEDBACK_SOURCE_RECORD_MISSING
        assert feedback_repo.saved == []

    asyncio.run(_run())


def test_record_should_pass_effective_source_run_id_to_artifact_source_resolver() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo(
            {"evt-artifact-source-2": _event_record(event_id="evt-artifact-source-2", run_id="run-2", event_type="artifact")}
        )
        feedback_repo = _FeedbackRepo()
        artifact_resolver = _ArtifactResolver(revision_run_id="run-2")
        service = _service(
            uow=_UoW(
                feedback_repo=feedback_repo,
                workflow_run_repo=workflow_repo,
            ),
            artifact_resolver=artifact_resolver,
        )

        result = await service.record_quality_feedback(
            _quality_command(
                source_ref=_source_ref(
                    source_kind=FeedbackSourceKind.ARTIFACT_REVISION,
                    source_event_id="evt-artifact-source-2",
                    source_run_id=None,
                    source_summary="artifact source inferred run",
                    source_record_refs=[
                        {
                            "artifact_id": "artifact-1",
                            "revision_id": "rev-1",
                            "content_hash": "sha256:artifact",
                        }
                    ],
                )
            )
        )

        assert result.success is True
        assert artifact_resolver.calls
        assert artifact_resolver.calls[0].run_id == "run-2"
        assert artifact_resolver.calls[0].source_run_id == "run-2"

    asyncio.run(_run())


def test_snapshot_builder_should_enforce_limits_and_gap_truncation() -> None:
    now = datetime(2026, 5, 19, 12, 0, 0)
    scope = FeedbackSnapshotScopeResult(
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        feedback_scope_kind=FeedbackScopeKind.RUN,
        scope_id="run-1",
        current_run_id_at_snapshot_time="run-1",
    )
    records: list[FeedbackRecord] = []
    for index in range(220):
        status = FeedbackStatus.OPEN if index < 70 else FeedbackStatus.RESOLVED
        records.append(
            _record(
                feedback_id=f"fb-{index:03d}",
                kind=FeedbackKind.USER_FEEDBACK if index % 3 == 0 else (
                    FeedbackKind.RUNTIME_FEEDBACK if index % 3 == 1 else FeedbackKind.QUALITY_FEEDBACK
                ),
                category=FeedbackCategory.CORRECTION if index % 3 == 0 else (
                    FeedbackCategory.TOOL_FAILURE if index % 3 == 1 else FeedbackCategory.FINAL_GATE_BLOCKED
                ),
                reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT if index % 3 == 0 else (
                    FeedbackReasonCode.TOOL_FAILED if index % 3 == 1 else FeedbackReasonCode.FINAL_GATE_BLOCKED
                ),
                severity=FeedbackSeverity.ERROR if index % 2 == 0 else FeedbackSeverity.WARNING,
                status=status,
                resolution_reason_code=FeedbackResolutionReasonCode.RESOLVED_BY_REPLAN if status != FeedbackStatus.OPEN else None,
                source_kind=FeedbackSourceKind.MESSAGE_EVENT if index % 3 == 0 else FeedbackSourceKind.TOOL_EVENT,
                source_event_id=f"evt-{index:03d}",
                target_type=FeedbackTargetType.MESSAGE_EVENT if index % 3 == 0 else FeedbackTargetType.TOOL_CALL,
                target_id=f"target-{index:03d}",
                created_at=now - timedelta(minutes=index),
                updated_at=now - timedelta(minutes=index),
                decay_policy="session_persistent" if index % 3 == 0 else "run_persistent",
            )
        )
    runtime_gaps = [
        FeedbackGapResult(
            gap_kind=FeedbackGapKind.PROJECTION_MISSING,
            reason_code=FeedbackReasonCode.FEEDBACK_PROJECTION_GAP,
            source_ref=None,
            target_ref=None,
            stage=FeedbackSnapshotStage.FUTURE_REVIEW,
            scope=scope,
            diagnostic_summary=f"gap-{idx}",
            created_at=now,
        )
        for idx in range(25)
    ]
    builder = FeedbackSnapshotBuilder(policy=FeedbackSnapshotPolicy())

    snapshot = builder.build(
        scope=scope,
        stage=FeedbackSnapshotStage.FUTURE_REVIEW,
        records=records,
        runtime_gaps=runtime_gaps,
        now=now,
    )

    assert len(snapshot.cursor.source_record_ids) == 200
    assert len(snapshot.active_user_feedback) <= 20
    assert len(snapshot.active_runtime_feedback) <= 20
    assert len(snapshot.active_quality_feedback) <= 20
    assert len(snapshot.open_feedback_items) == 50
    assert len(snapshot.resolved_feedback_items) == 20
    assert len(snapshot.excluded_feedback_refs) <= 100
    assert len(snapshot.feedback_gaps) == 20
    assert snapshot.feedback_gaps[-1].reason_code == FeedbackReasonCode.FEEDBACK_PROJECTION_GAP


def test_snapshot_builder_should_reject_invalid_stage_string() -> None:
    scope = FeedbackSnapshotScopeResult(
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        feedback_scope_kind=FeedbackScopeKind.RUN,
        scope_id="run-1",
        current_run_id_at_snapshot_time="run-1",
    )
    builder = FeedbackSnapshotBuilder(policy=FeedbackSnapshotPolicy())

    try:
        builder.build(scope=scope, stage="execute", records=[], runtime_gaps=[], now=datetime(2026, 5, 19, 12, 0, 0))  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError")
