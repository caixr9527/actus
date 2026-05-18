import asyncio
import inspect
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import MetaData, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.domain.models.feedback import (
    FeedbackCategory,
    FeedbackClassificationResult,
    FeedbackDataOrigin,
    FeedbackGapKind,
    FeedbackGapResult,
    FeedbackKind,
    FeedbackPromptSafeSummaryResult,
    FeedbackReasonCode,
    FeedbackRecord,
    FeedbackRecordCommand,
    FeedbackRecordResult,
    FeedbackResolutionCommand,
    FeedbackResolutionReasonCode,
    FeedbackResolutionResult,
    FeedbackScopeKind,
    FeedbackScopeResult,
    FeedbackSeverity,
    FeedbackSnapshotCursorResult,
    FeedbackSnapshotItemResult,
    FeedbackSnapshotResult,
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
    FeedbackWriteResult,
    UserFeedbackCommand,
    UserFeedbackIntent,
    UserFeedbackIntentKind,
    build_feedback_record_result,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.infrastructure.models.feedback import FeedbackRecordModel
from app.infrastructure.repositories.db_feedback_repository import DBFeedbackRepository


async def _run_feedback_postgres_integration(
        assertion: Callable[[AsyncSession], Awaitable[None] | None],
) -> None:
    from core.config import get_settings

    database_uri = get_settings().sqlalchemy_database_uri
    schema_name = f"feedback_pr1_{uuid4().hex}"
    engine = create_async_engine(database_uri, pool_pre_ping=True)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    try:
        async with engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema_name}"'))
            scoped_table = FeedbackRecordModel.__table__.to_metadata(MetaData(), schema=schema_name)
            await conn.run_sync(lambda sync_conn: scoped_table.create(sync_conn))

        async with session_factory() as session:
            await session.execute(text(f'SET search_path TO "{schema_name}"'))
            maybe_coro = assertion(session)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
        await engine.dispose()


def _source_ref(
        *,
        source_kind: FeedbackSourceKind = FeedbackSourceKind.MESSAGE_EVENT,
        source_event_id: str = "evt-source-1",
        source_run_id: str | None = "run-1",
        source_step_id: str | None = "step-1",
) -> FeedbackSourceRefResult:
    return FeedbackSourceRefResult(
        source_kind=source_kind,
        source_event_id=source_event_id,
        source_record_refs=[{"event_id": source_event_id}],
        source_run_id=source_run_id,
        source_step_id=source_step_id,
        source_summary="已脱敏来源摘要",
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


def _scope(
        *,
        feedback_scope_kind: FeedbackScopeKind = FeedbackScopeKind.RUN,
        scope_id: str = "run-1",
        run_id: str | None = "run-1",
        source_run_id: str | None = "run-1",
        target_run_id: str | None = "run-1",
        current_run_id_at_record_time: str | None = "run-1",
) -> FeedbackScopeResult:
    return FeedbackScopeResult(
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        feedback_scope_kind=feedback_scope_kind,
        scope_id=scope_id,
        run_id=run_id,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        current_run_id_at_record_time=current_run_id_at_record_time,
    )


def _snapshot_scope(
        *,
        feedback_scope_kind: FeedbackScopeKind = FeedbackScopeKind.RUN,
        scope_id: str = "run-1",
        current_run_id_at_snapshot_time: str | None = "run-1",
) -> FeedbackSnapshotScopeResult:
    return FeedbackSnapshotScopeResult(
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        feedback_scope_kind=feedback_scope_kind,
        scope_id=scope_id,
        current_run_id_at_snapshot_time=current_run_id_at_snapshot_time,
    )


def _feedback_summary() -> FeedbackSummaryResult:
    return FeedbackSummaryResult(
        summary_text="用户指出上一轮结果有误，需要修正。",
        summary_kind=FeedbackSummaryKind.USER_STATED,
        is_truncated=False,
        truncation_reason=None,
        language="zh-CN",
    )


def _prompt_safe_summary() -> FeedbackPromptSafeSummaryResult:
    return FeedbackPromptSafeSummaryResult(
        summary_text="用户指出上一轮结果有误，需要修正。",
        is_truncated=False,
        sanitization_applied=False,
        sanitization_reasons=[],
        prompt_visible=True,
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


def _resolution(
        *,
        status: FeedbackStatus = FeedbackStatus.OPEN,
) -> FeedbackResolutionResult:
    if status == FeedbackStatus.OPEN:
        return FeedbackResolutionResult(status=status)
    return FeedbackResolutionResult(
        status=status,
        resolution_reason_code=FeedbackResolutionReasonCode.RESOLVED_BY_REPLAN,
        resolved_by_ref={"feedback_id": "fb-resolver-1"},
        resolved_at=datetime(2026, 5, 18, 11, 0, 0),
        superseded_by_feedback_id="fb-new-1" if status == FeedbackStatus.SUPERSEDED else None,
        resolution_summary="后续流程已处理该反馈。",
    )


def _record(
        *,
        feedback_id: str = "fb-1",
        feedback_scope_kind: FeedbackScopeKind = FeedbackScopeKind.RUN,
        scope_id: str = "run-1",
        run_id: str | None = "run-1",
        source_run_id: str | None = "run-1",
        target_run_id: str | None = "run-1",
        target_type: FeedbackTargetType = FeedbackTargetType.MESSAGE_EVENT,
        target_id: str = "evt-target-1",
        target_revision_id: str | None = None,
        target_content_hash: str | None = None,
        kind: FeedbackKind = FeedbackKind.USER_FEEDBACK,
        category: FeedbackCategory = FeedbackCategory.CORRECTION,
        status: FeedbackStatus = FeedbackStatus.OPEN,
        severity: FeedbackSeverity = FeedbackSeverity.ERROR,
        source_kind: FeedbackSourceKind = FeedbackSourceKind.MESSAGE_EVENT,
        source_event_id: str = "evt-source-1",
        reason_code: FeedbackReasonCode = FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
        dedupe_key: str = "dedupe-1",
        feedback_key: str = "key-1",
        step_id: str | None = "step-1",
        profile_hash: str | None = "sha256:" + "a" * 64,
        classification: FeedbackClassificationResult | None = None,
        resolution: FeedbackResolutionResult | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
) -> FeedbackRecord:
    classification = classification or _classification()
    resolution = resolution or _resolution(status=status)
    created_at = created_at or datetime(2026, 5, 18, 10, 0, 0)
    updated_at = updated_at or datetime(2026, 5, 18, 10, 0, 0)
    source_ref = _source_ref(
        source_kind=source_kind,
        source_event_id=source_event_id,
        source_run_id=source_run_id,
        source_step_id=step_id,
    )
    target_ref = _target_ref(
        target_type=target_type,
        target_id=target_id,
        target_run_id=target_run_id,
        target_revision_id=target_revision_id,
        target_content_hash=target_content_hash,
    )
    scope = _scope(
        feedback_scope_kind=feedback_scope_kind,
        scope_id=scope_id,
        run_id=run_id,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        current_run_id_at_record_time=run_id,
    )
    return FeedbackRecord(
        id=feedback_id,
        scope=scope,
        source_ref=source_ref,
        target_ref=target_ref,
        resolution=resolution,
        feedback_summary=_feedback_summary(),
        prompt_safe_summary=_prompt_safe_summary(),
        classification=classification,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id=run_id,
        feedback_scope_kind=feedback_scope_kind,
        scope_id=scope_id,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        step_id=step_id,
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
        dedupe_key=dedupe_key,
        feedback_key=feedback_key,
        reason_code=reason_code,
        resolution_reason_code=resolution.resolution_reason_code,
        resolved_by_ref=resolution.resolved_by_ref,
        decay_policy="session_window",
        expires_at=None,
        ttl_scope="session",
        profile_hash=profile_hash,
        origin=DataOrigin.USER_MESSAGE if kind == FeedbackKind.USER_FEEDBACK else DataOrigin.SYSTEM_OPERATIONAL,
        trust_level=classification.trust_level,
        privacy_level=classification.privacy_level,
        retention_policy=classification.retention_policy,
        created_at=created_at,
        updated_at=updated_at,
    )


def _access_scope() -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        current_step_id="step-1",
    )


def test_feedback_record_and_related_contracts_should_be_strict() -> None:
    payload = _record().model_dump(mode="json")
    payload["extra"] = "forbidden"
    with pytest.raises(ValidationError):
        FeedbackRecord.model_validate(payload)

    command_payload = UserFeedbackCommand(
        access_scope=_access_scope(),
        source_ref=_source_ref(),
        target_ref=_target_ref(),
        category=FeedbackCategory.CORRECTION,
        reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
        feedback_summary=_feedback_summary(),
        prompt_safe_summary=_prompt_safe_summary(),
        classification=_classification(),
        intent=UserFeedbackIntent(
            intent_kind=UserFeedbackIntentKind.CORRECTION,
            target_ref=_target_ref(),
            reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
            summary_hint="用户纠错",
        ),
        requested_feedback_scope_kind=FeedbackScopeKind.RUN,
        requested_scope_id="run-1",
        current_run_id_at_record_time="run-1",
        step_id="step-1",
        profile_hash="sha256:" + "b" * 64,
        decay_policy="run_window",
        ttl_scope="run",
        origin=DataOrigin.USER_MESSAGE,
        trust_level=DataTrustLevel.USER_PROVIDED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.SESSION_BOUND,
    ).model_dump(mode="json")
    command_payload["scope"] = _scope().model_dump(mode="json")
    with pytest.raises(ValidationError):
        FeedbackRecordCommand.model_validate(command_payload)


def test_feedback_command_kind_and_category_matrix_should_reject_spoofed_inputs() -> None:
    base_kwargs = {
        "access_scope": _access_scope(),
        "source_ref": _source_ref(),
        "target_ref": _target_ref(),
        "reason_code": FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
        "feedback_summary": _feedback_summary(),
        "prompt_safe_summary": _prompt_safe_summary(),
        "classification": _classification(),
        "requested_feedback_scope_kind": FeedbackScopeKind.RUN,
        "requested_scope_id": "run-1",
        "current_run_id_at_record_time": "run-1",
        "step_id": "step-1",
        "profile_hash": "sha256:" + "b" * 64,
        "decay_policy": "run_window",
        "ttl_scope": "run",
        "origin": DataOrigin.USER_MESSAGE,
        "trust_level": DataTrustLevel.USER_PROVIDED,
        "privacy_level": PrivacyLevel.PRIVATE,
        "retention_policy": RetentionPolicyKind.SESSION_BOUND,
    }

    with pytest.raises(ValidationError):
        UserFeedbackCommand(
            **base_kwargs,
            kind=FeedbackKind.QUALITY_FEEDBACK,
            category=FeedbackCategory.CORRECTION,
            intent=UserFeedbackIntent(
                intent_kind=UserFeedbackIntentKind.CORRECTION,
                target_ref=_target_ref(),
                reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
            ),
        )

    with pytest.raises(ValidationError):
        UserFeedbackCommand(
            **base_kwargs,
            category=FeedbackCategory.ARTIFACT_UNUSABLE,
            intent=UserFeedbackIntent(
                intent_kind=UserFeedbackIntentKind.DISSATISFACTION,
                target_ref=_target_ref(),
                reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
            ),
        )

    with pytest.raises(ValidationError):
        FeedbackRecordCommand.model_validate(
            {
                **base_kwargs,
                "kind": FeedbackKind.RUNTIME_FEEDBACK.value,
                "category": FeedbackCategory.ARTIFACT_UNUSABLE.value,
                "reason_code": FeedbackReasonCode.ARTIFACT_UNUSABLE.value,
                "feedback_summary": _feedback_summary().model_dump(mode="json"),
                "prompt_safe_summary": _prompt_safe_summary().model_dump(mode="json"),
                "classification": _classification(data_origin=FeedbackDataOrigin.RUNTIME).model_dump(mode="json"),
            }
        )

    with pytest.raises(ValidationError):
        FeedbackRecordCommand.model_validate(
            {
                **base_kwargs,
                "kind": FeedbackKind.USER_FEEDBACK.value,
                "category": FeedbackCategory.ARTIFACT_UNUSABLE.value,
                "reason_code": FeedbackReasonCode.USER_REPORTED_DISSATISFACTION.value,
                "feedback_summary": _feedback_summary().model_dump(mode="json"),
                "prompt_safe_summary": _prompt_safe_summary().model_dump(mode="json"),
                "classification": _classification().model_dump(mode="json"),
            }
        )


def test_feedback_record_command_should_reject_mismatched_classification_governance_fields() -> None:
    command_payload = {
        "access_scope": _access_scope().model_dump(mode="json"),
        "source_ref": _source_ref().model_dump(mode="json"),
        "target_ref": _target_ref().model_dump(mode="json"),
        "kind": FeedbackKind.USER_FEEDBACK.value,
        "category": FeedbackCategory.CORRECTION.value,
        "reason_code": FeedbackReasonCode.USER_CORRECTED_REQUIREMENT.value,
        "feedback_summary": _feedback_summary().model_dump(mode="json"),
        "prompt_safe_summary": _prompt_safe_summary().model_dump(mode="json"),
        "classification": _classification().model_dump(mode="json"),
        "requested_feedback_scope_kind": FeedbackScopeKind.RUN.value,
        "requested_scope_id": "run-1",
        "current_run_id_at_record_time": "run-1",
        "step_id": "step-1",
        "profile_hash": "sha256:" + "b" * 64,
        "decay_policy": "run_window",
        "ttl_scope": "run",
        "origin": DataOrigin.USER_MESSAGE.value,
        "trust_level": DataTrustLevel.USER_PROVIDED.value,
        "privacy_level": PrivacyLevel.PRIVATE.value,
        "retention_policy": RetentionPolicyKind.SESSION_BOUND.value,
    }

    mismatch_privacy = dict(command_payload)
    mismatch_privacy["privacy_level"] = PrivacyLevel.INTERNAL.value
    with pytest.raises(ValidationError):
        FeedbackRecordCommand.model_validate(mismatch_privacy)

    mismatch_retention = dict(command_payload)
    mismatch_retention["retention_policy"] = RetentionPolicyKind.EPHEMERAL.value
    with pytest.raises(ValidationError):
        FeedbackRecordCommand.model_validate(mismatch_retention)

    mismatch_trust = dict(command_payload)
    mismatch_trust["trust_level"] = DataTrustLevel.SYSTEM_GENERATED.value
    with pytest.raises(ValidationError):
        FeedbackRecordCommand.model_validate(mismatch_trust)


def test_feedback_target_ref_validation_matrix_should_reject_invalid_shapes() -> None:
    with pytest.raises(ValidationError):
        FeedbackTargetRefResult(
            target_type=FeedbackTargetType.ARTIFACT_REVISION,
            target_id="artifact-1",
            target_run_id="run-1",
        )

    with pytest.raises(ValidationError):
        FeedbackTargetRefResult(
            target_type=FeedbackTargetType.MESSAGE_EVENT,
            target_id="evt-message-1",
        )

    with pytest.raises(ValidationError):
        FeedbackTargetRefResult(
            target_type=FeedbackTargetType.TOOL_CALL,
            target_id="tool-1",
        )

    with pytest.raises(ValidationError):
        FeedbackTargetRefResult.model_validate(
            {
                "target_type": FeedbackTargetType.ARTIFACT_REVISION.value,
                "target_id": "artifact-1",
                "target_run_id": "run-1",
                "target_revision_id": "rev-1",
                "target_content_hash": "sha256:artifact",
                "artifact_id": "artifact-1",
            }
        )


def test_feedback_source_kind_must_use_fixed_enum() -> None:
    with pytest.raises(ValidationError):
        FeedbackSourceRefResult.model_validate(
            {
                "source_kind": "free_text_source",
                "source_event_id": "evt-1",
                "source_record_refs": [],
                "source_run_id": "run-1",
                "source_step_id": "step-1",
                "source_summary": "摘要",
            }
        )


def test_feedback_resolution_command_should_reject_caller_supplied_scope_source_target() -> None:
    with pytest.raises(ValidationError):
        FeedbackResolutionCommand.model_validate(
            {
                "access_scope": _access_scope().model_dump(mode="json"),
                "feedback_id": "fb-1",
                "scope": _scope().model_dump(mode="json"),
                "resolution": _resolution(status=FeedbackStatus.RESOLVED).model_dump(mode="json"),
                "updated_at": datetime(2026, 5, 18, 12, 0, 0).isoformat(),
            }
        )

    with pytest.raises(ValidationError):
        FeedbackResolutionCommand.model_validate(
            {
                "access_scope": _access_scope().model_dump(mode="json"),
                "feedback_id": "fb-1",
                "source_ref": _source_ref().model_dump(mode="json"),
                "resolution": _resolution(status=FeedbackStatus.RESOLVED).model_dump(mode="json"),
                "updated_at": datetime(2026, 5, 18, 12, 0, 0).isoformat(),
            }
        )

    with pytest.raises(ValidationError):
        FeedbackResolutionCommand.model_validate(
            {
                "access_scope": _access_scope().model_dump(mode="json"),
                "feedback_id": "fb-1",
                "target_ref": _target_ref().model_dump(mode="json"),
                "resolution": _resolution(status=FeedbackStatus.RESOLVED).model_dump(mode="json"),
                "updated_at": datetime(2026, 5, 18, 12, 0, 0).isoformat(),
            }
        )


def test_runtime_signal_should_be_rejected_for_record_command_result_and_snapshot_item() -> None:
    with pytest.raises(ValidationError):
        _record(source_kind=FeedbackSourceKind.RUNTIME_SIGNAL)

    with pytest.raises(ValidationError):
        FeedbackRecordCommand.model_validate(
            {
                "access_scope": _access_scope().model_dump(mode="json"),
                "source_ref": _source_ref(source_kind=FeedbackSourceKind.RUNTIME_SIGNAL).model_dump(mode="json"),
                "target_ref": _target_ref().model_dump(mode="json"),
                "kind": FeedbackKind.RUNTIME_FEEDBACK.value,
                "category": FeedbackCategory.NO_PROGRESS.value,
                "reason_code": FeedbackReasonCode.EXECUTION_NO_PROGRESS.value,
                "feedback_summary": _feedback_summary().model_dump(mode="json"),
                "prompt_safe_summary": _prompt_safe_summary().model_dump(mode="json"),
                "classification": _classification(
                    data_origin=FeedbackDataOrigin.RUNTIME,
                    source_confidence=FeedbackSourceConfidence.WEAK,
                ).model_dump(mode="json"),
                "requested_feedback_scope_kind": FeedbackScopeKind.RUN.value,
                "requested_scope_id": "run-1",
                "current_run_id_at_record_time": "run-1",
                "step_id": "step-1",
                "profile_hash": "sha256:" + "b" * 64,
                "decay_policy": "run_window",
                "ttl_scope": "run",
                "origin": DataOrigin.SYSTEM_OPERATIONAL.value,
                "trust_level": DataTrustLevel.SYSTEM_GENERATED.value,
                "privacy_level": PrivacyLevel.INTERNAL.value,
                "retention_policy": RetentionPolicyKind.SESSION_BOUND.value,
            }
        )

    with pytest.raises(ValidationError):
        FeedbackSnapshotItemResult(
            feedback_id="fb-1",
            kind=FeedbackKind.RUNTIME_FEEDBACK,
            category=FeedbackCategory.NO_PROGRESS,
            status=FeedbackStatus.OPEN,
            severity=FeedbackSeverity.WARNING,
            target_ref=_target_ref(target_type=FeedbackTargetType.RUN, target_id="run-1", target_run_id="run-1"),
            source_kind=FeedbackSourceKind.RUNTIME_SIGNAL,
            source_event_id="evt-source-1",
            source_run_id="run-1",
            target_run_id="run-1",
            prompt_safe_summary="当前步骤缺少进展。",
            reason_code=FeedbackReasonCode.EXECUTION_NO_PROGRESS,
            resolution_reason_code=None,
            created_at=datetime(2026, 5, 18, 10, 0, 0),
        )

    gap = FeedbackGapResult(
        gap_kind=FeedbackGapKind.PROJECTION_MISSING,
        reason_code=FeedbackReasonCode.FEEDBACK_PROJECTION_GAP,
        source_ref=FeedbackSourceRefResult(
            source_kind=FeedbackSourceKind.RUNTIME_SIGNAL,
            source_event_id="evt-runtime-gap-1",
            source_record_refs=[{"diagnostic_id": "diag-1"}],
            source_run_id="run-1",
            source_step_id="step-1",
            source_summary="当前上下文诊断到 feedback 投影缺口。",
        ),
        target_ref=_target_ref(),
        stage=FeedbackSnapshotStage.REPLAN,
        scope=_snapshot_scope(),
        diagnostic_summary="投影链路缺失，仅作为诊断项暴露。",
        created_at=datetime(2026, 5, 18, 10, 6, 0),
    )
    assert gap.source_ref is not None
    assert gap.source_ref.source_kind == FeedbackSourceKind.RUNTIME_SIGNAL


def test_feedback_write_result_status_matrix_should_be_serializable() -> None:
    record = _record()
    record_result = build_feedback_record_result(record)

    created = FeedbackWriteResult(
        success=True,
        created=True,
        reused=False,
        reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
        feedback_id=record.id,
        record_ref=record_result,
        scope=record.scope,
        source_ref=record.source_ref,
        target_ref=record.target_ref,
        resolution=None,
        gap=None,
        created_at=record.created_at,
    )
    assert created.model_dump(mode="json")["created"] is True

    reused = FeedbackWriteResult(
        success=True,
        created=False,
        reused=True,
        reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
        feedback_id=record.id,
        record_ref=record_result,
        scope=record.scope,
        source_ref=record.source_ref,
        target_ref=record.target_ref,
        resolution=None,
        gap=None,
        created_at=record.created_at,
    )
    assert reused.model_dump(mode="json")["reused"] is True

    failed_gap = FeedbackGapResult(
        gap_kind=FeedbackGapKind.SOURCE_MISSING,
        reason_code=FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISSING,
        source_ref=None,
        target_ref=None,
        stage=FeedbackSnapshotStage.REPLAN,
        scope=None,
        diagnostic_summary="缺少 source event，不能落库。",
        created_at=datetime(2026, 5, 18, 10, 5, 0),
    )
    failed = FeedbackWriteResult(
        success=False,
        created=False,
        reused=False,
        reason_code=FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISSING,
        feedback_id=None,
        record_ref=None,
        scope=None,
        source_ref=None,
        target_ref=None,
        resolution=None,
        gap=failed_gap,
        created_at=None,
    )
    assert failed.model_dump(mode="json")["gap"]["gap_kind"] == FeedbackGapKind.SOURCE_MISSING.value

    resolved_record = _record(
        status=FeedbackStatus.RESOLVED,
        resolution=_resolution(status=FeedbackStatus.RESOLVED),
        updated_at=datetime(2026, 5, 18, 10, 30, 0),
    )
    resolved_result = build_feedback_record_result(resolved_record)
    resolved = FeedbackWriteResult(
        success=True,
        created=False,
        reused=False,
        reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
        feedback_id=resolved_record.id,
        record_ref=resolved_result,
        scope=resolved_record.scope,
        source_ref=resolved_record.source_ref,
        target_ref=resolved_record.target_ref,
        resolution=resolved_record.resolution,
        gap=None,
        created_at=resolved_record.created_at,
    )
    assert resolved.model_dump(mode="json")["resolution"]["status"] == FeedbackStatus.RESOLVED.value

    with pytest.raises(ValidationError):
        FeedbackWriteResult(
            success=True,
            created=False,
            reused=False,
            reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
            feedback_id=record.id,
            record_ref=record_result,
            scope=record.scope,
            source_ref=record.source_ref,
            target_ref=record.target_ref,
            resolution=None,
            gap=None,
            created_at=record.created_at,
        )

    with pytest.raises(ValidationError):
        FeedbackWriteResult(
            success=True,
            created=True,
            reused=False,
            reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
            feedback_id=resolved_record.id,
            record_ref=resolved_result,
            scope=resolved_record.scope,
            source_ref=resolved_record.source_ref,
            target_ref=resolved_record.target_ref,
            resolution=resolved_record.resolution,
            gap=None,
            created_at=resolved_record.created_at,
        )

    with pytest.raises(ValidationError):
        FeedbackWriteResult(
            success=True,
            created=False,
            reused=True,
            reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
            feedback_id=resolved_record.id,
            record_ref=resolved_result,
            scope=resolved_record.scope,
            source_ref=resolved_record.source_ref,
            target_ref=resolved_record.target_ref,
            resolution=resolved_record.resolution,
            gap=None,
            created_at=resolved_record.created_at,
        )


def test_feedback_gap_source_missing_should_reject_source_ref() -> None:
    with pytest.raises(ValidationError):
        FeedbackGapResult(
            gap_kind=FeedbackGapKind.SOURCE_MISSING,
            reason_code=FeedbackReasonCode.FEEDBACK_SOURCE_EVENT_MISSING,
            source_ref=_source_ref(),
            target_ref=None,
            stage=FeedbackSnapshotStage.REPLAN,
            scope=None,
            diagnostic_summary="非法 source ref",
            created_at=datetime(2026, 5, 18, 10, 0, 0),
        )


def test_feedback_record_model_should_round_trip_result_and_json_fields() -> None:
    record = _record(
        feedback_scope_kind=FeedbackScopeKind.SESSION,
        scope_id="session-1",
        run_id=None,
        source_run_id="run-prev",
        target_run_id="run-prev",
        step_id=None,
        target_type=FeedbackTargetType.ARTIFACT_REVISION,
        target_id="artifact-1",
        target_revision_id="rev-1",
        target_content_hash="sha256:artifact",
        source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
        source_event_id="evt-feedback-input-1",
        kind=FeedbackKind.USER_FEEDBACK,
        category=FeedbackCategory.DISSATISFACTION,
        reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
    )
    result = build_feedback_record_result(record)
    model = FeedbackRecordModel.from_domain(record)
    round_trip_result = model.to_result()

    assert result.model_dump(mode="json") == round_trip_result.model_dump(mode="json")
    assert model.source_ref["source_event_id"] == "evt-feedback-input-1"
    assert model.target_ref["target_revision_id"] == "rev-1"
    assert model.classification["source_confidence"] == FeedbackSourceConfidence.STRONG.value
    assert model.source_event_id == "evt-feedback-input-1"
    assert model.target_revision_id == "rev-1"
    assert model.target_content_hash == "sha256:artifact"


def test_feedback_record_model_round_trip_should_preserve_origin_strong_column() -> None:
    record = _record(
        kind=FeedbackKind.RUNTIME_FEEDBACK,
        category=FeedbackCategory.TOOL_FAILURE,
        reason_code=FeedbackReasonCode.TOOL_FAILED,
        source_kind=FeedbackSourceKind.TOOL_EVENT,
        source_event_id="evt-tool-1",
        classification=_classification(
            data_origin=FeedbackDataOrigin.RUNTIME,
            source_confidence=FeedbackSourceConfidence.STRONG,
        ),
    ).model_copy(update={"origin": DataOrigin.EXTERNAL_TOOL})

    round_trip = FeedbackRecordModel.from_domain(record).to_domain()

    assert round_trip.origin == DataOrigin.EXTERNAL_TOOL


def test_feedback_record_model_to_domain_should_fail_closed_for_raw_default_json_payloads() -> None:
    model = FeedbackRecordModel(
        id="fb-raw-default-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        feedback_scope_kind=FeedbackScopeKind.RUN.value,
        scope_id="run-1",
        source_run_id="run-1",
        target_run_id="run-1",
        step_id="step-1",
        kind=FeedbackKind.USER_FEEDBACK.value,
        category=FeedbackCategory.CORRECTION.value,
        status=FeedbackStatus.OPEN.value,
        severity=FeedbackSeverity.ERROR.value,
        source_kind=FeedbackSourceKind.MESSAGE_EVENT.value,
        source_event_id="evt-source-1",
        target_type=FeedbackTargetType.MESSAGE_EVENT.value,
        target_id="evt-target-1",
        target_revision_id=None,
        target_content_hash=None,
        feedback_key="key-1",
        dedupe_key="dedupe-1",
        reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT.value,
        resolution_reason_code=None,
        decay_policy="run_window",
        ttl_scope="run",
        expires_at=None,
        origin=DataOrigin.USER_MESSAGE.value,
        trust_level=DataTrustLevel.USER_PROVIDED.value,
        privacy_level=PrivacyLevel.PRIVATE.value,
        retention_policy=RetentionPolicyKind.SESSION_BOUND.value,
        profile_hash="sha256:" + "c" * 64,
        source_record_refs=[],
        source_ref={},
        target_ref={},
        feedback_summary={},
        prompt_safe_summary={},
        resolution={},
        classification={},
        created_at=datetime(2026, 5, 18, 10, 0, 0),
        updated_at=datetime(2026, 5, 18, 10, 0, 0),
    )

    with pytest.raises(ValidationError):
        model.to_domain()


def test_feedback_orm_model_should_define_required_columns_and_json_fields() -> None:
    columns = set(FeedbackRecordModel.__table__.columns.keys())
    for column in {
        "feedback_scope_kind",
        "scope_id",
        "source_run_id",
        "target_run_id",
        "target_revision_id",
        "target_content_hash",
        "feedback_key",
        "dedupe_key",
        "resolution_reason_code",
        "decay_policy",
        "ttl_scope",
        "updated_at",
    }:
        assert column in columns

    assert FeedbackRecordModel.__tablename__ == "feedback_records"
    assert isinstance(FeedbackRecordModel.__table__.c.source_ref.type, postgresql.JSONB)
    assert isinstance(FeedbackRecordModel.__table__.c.feedback_summary.type, postgresql.JSONB)
    assert FeedbackRecordModel.__table__.c.source_event_id.nullable is False
    assert FeedbackRecordModel.__table__.c.run_id.nullable is True


def test_feedback_repository_should_save_once_by_scope_and_dedupe_key() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: "fb-1"))
    )
    repository = DBFeedbackRepository(db_session=db_session)

    saved = asyncio.run(repository.save_once(_record()))

    assert saved.id == "fb-1"
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_feedback_records_user_session_scope_dedupe DO NOTHING" in compiled_sql
    assert "feedback_records" in compiled_sql


def test_feedback_repository_duplicate_should_return_existing_record() -> None:
    existing = _record(feedback_id="fb-existing")
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    existing_result = SimpleNamespace(scalar_one_or_none=lambda: FeedbackRecordModel.from_domain(existing))
    db_session = SimpleNamespace(execute=AsyncMock(side_effect=[execute_result, existing_result]))
    repository = DBFeedbackRepository(db_session=db_session)

    saved = asyncio.run(repository.save_once(_record()))

    assert saved.id == "fb-existing"
    lookup_statement = db_session.execute.call_args_list[1].args[0]
    compiled_sql = str(lookup_statement.compile(dialect=postgresql.dialect()))
    assert "feedback_records.user_id" in compiled_sql
    assert "feedback_records.session_id" in compiled_sql
    assert "feedback_records.feedback_scope_kind" in compiled_sql
    assert "feedback_records.scope_id" in compiled_sql
    assert "feedback_records.dedupe_key" in compiled_sql


def test_feedback_repository_queries_should_require_user_session_and_scope_filters() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])))
    )
    repository = DBFeedbackRepository(db_session=db_session)

    result = asyncio.run(
        repository.list_active_by_scope(
            user_id="user-1",
            session_id="session-1",
            feedback_scope_kind=FeedbackScopeKind.SESSION,
            scope_id="session-1",
        )
    )

    assert result == []
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "feedback_records.feedback_scope_kind" in compiled_sql
    assert "feedback_records.scope_id" in compiled_sql
    assert "feedback_records.status IN" in compiled_sql
    assert "feedback_records.run_id =" not in compiled_sql

    with pytest.raises(ValueError):
        asyncio.run(
            repository.list_by_run(user_id="", session_id="session-1", run_id="run-1")
        )


def test_feedback_repository_list_active_by_run_should_delegate_to_run_scope_query() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])))
    )
    repository = DBFeedbackRepository(db_session=db_session)

    result = asyncio.run(
        repository.list_active_by_run(
            user_id="user-1",
            session_id="session-1",
            run_id="run-1",
        )
    )

    assert result == []
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "feedback_records.feedback_scope_kind" in compiled_sql
    assert "feedback_records.scope_id" in compiled_sql
    compiled_params = statement.compile(dialect=postgresql.dialect()).params
    assert compiled_params["scope_id_1"] == "run-1"


def test_feedback_repository_update_resolution_should_only_update_lifecycle_fields() -> None:
    updated = _record(
        status=FeedbackStatus.RESOLVED,
        resolution=_resolution(status=FeedbackStatus.RESOLVED),
        updated_at=datetime(2026, 5, 18, 12, 0, 0),
    )
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(rowcount=1),
                SimpleNamespace(scalar_one_or_none=lambda: FeedbackRecordModel.from_domain(updated)),
            ]
        )
    )
    repository = DBFeedbackRepository(db_session=db_session)

    saved = asyncio.run(
        repository.update_resolution(
            user_id="user-1",
            session_id="session-1",
            feedback_scope_kind=FeedbackScopeKind.RUN,
            scope_id="run-1",
            feedback_id="fb-1",
            resolution=updated.resolution,
            updated_at=updated.updated_at,
        )
    )

    assert saved.status == FeedbackStatus.RESOLVED
    update_statement = db_session.execute.call_args_list[0].args[0]
    compiled_sql = str(update_statement.compile(dialect=postgresql.dialect()))
    assert "status=" in compiled_sql or "status =" in compiled_sql
    assert "resolution_reason_code" in compiled_sql
    assert "resolution" in compiled_sql
    assert "severity" not in compiled_sql
    assert "feedback_records.reason_code" not in compiled_sql


def test_feedback_repository_update_resolution_signature_should_not_accept_split_lifecycle_inputs() -> None:
    parameters = inspect.signature(DBFeedbackRepository.update_resolution).parameters

    assert "status" not in parameters
    assert "resolution_reason_code" not in parameters


def test_feedback_repository_postgres_constraints_and_session_scope_should_hold_integration() -> None:
    async def _assertions(session: AsyncSession) -> None:
        repository = DBFeedbackRepository(db_session=session)
        session_record = _record(
            feedback_id="fb-int-1",
            feedback_scope_kind=FeedbackScopeKind.SESSION,
            scope_id="session-1",
            run_id=None,
            source_run_id="run-prev-1",
            target_run_id="run-prev-1",
            step_id=None,
            source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
            source_event_id="evt-feedback-input-1",
            target_type=FeedbackTargetType.ARTIFACT_REVISION,
            target_id="artifact-1",
            target_revision_id="rev-1",
            target_content_hash="sha256:artifact",
            category=FeedbackCategory.DISSATISFACTION,
            reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
            dedupe_key="dedupe-int-1",
            feedback_key="key-int-1",
        )
        duplicate_record = _record(
            feedback_id="fb-int-2",
            feedback_scope_kind=FeedbackScopeKind.SESSION,
            scope_id="session-1",
            run_id=None,
            source_run_id="run-prev-1",
            target_run_id="run-prev-1",
            step_id=None,
            source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
            source_event_id="evt-feedback-input-1",
            target_type=FeedbackTargetType.ARTIFACT_REVISION,
            target_id="artifact-1",
            target_revision_id="rev-1",
            target_content_hash="sha256:artifact",
            category=FeedbackCategory.DISSATISFACTION,
            reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
            dedupe_key="dedupe-int-1",
            feedback_key="key-int-2",
        )

        first_saved = await repository.save_once(session_record)
        duplicate_saved = await repository.save_once(duplicate_record)
        assert first_saved.id == "fb-int-1"
        assert duplicate_saved.id == "fb-int-1"

        rows = await session.execute(select(FeedbackRecordModel.id))
        assert rows.scalars().all() == ["fb-int-1"]

        active_records = await repository.list_active_by_scope(
            user_id="user-1",
            session_id="session-1",
            feedback_scope_kind=FeedbackScopeKind.SESSION,
            scope_id="session-1",
        )
        assert [record.id for record in active_records] == ["fb-int-1"]
        assert active_records[0].run_id is None

        invalid_model = FeedbackRecordModel.from_domain(
            _record(
                feedback_id="fb-int-invalid",
                feedback_scope_kind=FeedbackScopeKind.SESSION,
                scope_id="session-1",
                run_id=None,
                source_run_id="run-prev-1",
                target_run_id="run-prev-1",
                step_id=None,
                source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
                source_event_id="evt-feedback-input-invalid",
                target_type=FeedbackTargetType.ARTIFACT_REVISION,
                target_id="artifact-2",
                target_revision_id="rev-2",
                target_content_hash="sha256:artifact-2",
                category=FeedbackCategory.DISSATISFACTION,
                reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
                dedupe_key="dedupe-int-invalid",
                feedback_key="key-int-invalid",
            )
        )
        invalid_model.source_event_id = None  # type: ignore[assignment]

        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                session.add(invalid_model)
                await session.flush()

    asyncio.run(_run_feedback_postgres_integration(_assertions))


def test_feedback_snapshot_contract_should_use_complete_cursor_and_string_prompt_summary() -> None:
    item = FeedbackSnapshotItemResult(
        feedback_id="fb-1",
        kind=FeedbackKind.USER_FEEDBACK,
        category=FeedbackCategory.CORRECTION,
        status=FeedbackStatus.OPEN,
        severity=FeedbackSeverity.ERROR,
        target_ref=_target_ref(),
        source_kind=FeedbackSourceKind.MESSAGE_EVENT,
        source_event_id="evt-source-1",
        source_run_id="run-1",
        target_run_id="run-1",
        prompt_safe_summary="用户指出上一轮结果有误，需要修正。",
        reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
        resolution_reason_code=None,
        created_at=datetime(2026, 5, 18, 10, 0, 0),
    )
    snapshot = FeedbackSnapshotResult(
        scope=_snapshot_scope(),
        snapshot_id="snapshot-1",
        source_run_id="run-1",
        stage=FeedbackSnapshotStage.REPLAN,
        active_user_feedback=[item],
        active_runtime_feedback=[],
        active_quality_feedback=[],
        open_feedback_items=[item],
        resolved_feedback_items=[],
        do_not_repeat_feedback=[],
        user_constraints=[],
        replan_hints=[],
        review_hints=[],
        final_gate_hints=[],
        feedback_gaps=[],
        included_feedback_ids=["fb-1"],
        excluded_feedback_refs=[],
        cursor=FeedbackSnapshotCursorResult(
            latest_feedback_id="fb-1",
            source_record_ids=["fb-1"],
        ),
        created_at=datetime(2026, 5, 18, 10, 5, 0),
    )

    assert snapshot.model_dump(mode="json")["cursor"]["latest_feedback_id"] == "fb-1"
    with pytest.raises(ValidationError):
        FeedbackSnapshotCursorResult.model_validate(
            {"latest_feedback_id": "fb-1", "source_record_ids": ["fb-1"], "extra": "forbidden"}
        )

    with pytest.raises(ValidationError):
        FeedbackSnapshotScopeResult(
            user_id="user-1",
            session_id="session-1",
            workspace_id="workspace-1",
            feedback_scope_kind=FeedbackScopeKind.RUN,
            scope_id="run-1",
            current_run_id_at_snapshot_time=None,
        )


def test_feedback_artifact_target_should_require_revision_and_hash() -> None:
    with pytest.raises(ValidationError):
        _record(
            target_type=FeedbackTargetType.ARTIFACT_REVISION,
            target_id="artifact-1",
            target_revision_id=None,
            target_content_hash="sha256:artifact",
        )


def test_feedback_source_event_id_missing_should_fail_closed_in_contracts() -> None:
    with pytest.raises(ValidationError):
        _record(source_event_id="")

    with pytest.raises(ValidationError):
        FeedbackSourceRefResult(
            source_kind=FeedbackSourceKind.MESSAGE_EVENT,
            source_event_id="",
            source_record_refs=[],
            source_run_id="run-1",
            source_step_id="step-1",
            source_summary="摘要",
        )


def test_feedback_migration_should_define_required_indexes_and_not_null_source_event() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "9d1e2f3a4b5c_create_feedback_records.py"
    )
    migration_text = migration_path.read_text(encoding="utf-8")

    assert "feedback_records" in migration_text
    assert "uq_feedback_records_user_session_scope_dedupe" in migration_text
    assert "ix_feedback_user_session_run_created" in migration_text
    assert "ix_feedback_user_session_scope_status" in migration_text
    assert 'sa.Column("source_event_id", sa.String(length=255), nullable=False)' in migration_text
    assert 'sa.Column("run_id", sa.String(length=255), nullable=True)' in migration_text
