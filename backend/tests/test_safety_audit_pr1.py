import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from app.domain.models.safety_audit import (
    EXISTING_CONSTRAINT_REASON_CODES,
    SafetyAuditDecision,
    SafetyAuditRiskClassificationDigest,
    SafetyAuditRewriteMetadataDigest,
    SafetyAuditPolicyTraceEntry,
    SafetyAuditRecord,
    SafetyAuditRiskClassificationInput,
    SafetyAuditRiskLevel,
    build_args_digest,
    build_safety_audit_action_id,
    classify_safety_audit_risk,
)
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.infrastructure.models.safety_audit import SafetyAuditRecordModel
from app.infrastructure.repositories.db_safety_audit_repository import (
    DBSafetyAuditRepository,
    SafetyAuditLinkageConflictError,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine import reason_codes


def _record(
        *,
        audit_id: str = "audit-1",
        user_id: str = "user-1",
        session_id: str = "session-1",
        workspace_id: str = "workspace-1",
        run_id: str = "run-1",
        step_id: str | None = "step-1",
        tool_call_id: str | None = "call-1",
        decision: SafetyAuditDecision = SafetyAuditDecision.ALLOW,
        reason_code: str = "allow",
        risk_level: SafetyAuditRiskLevel = SafetyAuditRiskLevel.LOW,
        function_name: str = "read_file",
        normalized_function_name: str = "read_file",
        final_function_name: str = "read_file",
        final_normalized_function_name: str = "read_file",
        capability_id: str = "filesystem.read",
        tool_family: str = "filesystem",
        fingerprint: str = "sha256:fingerprint",
) -> SafetyAuditRecord:
    action_id = build_safety_audit_action_id(
        run_id=run_id,
        step_id=step_id,
        tool_call_id=tool_call_id,
        tool_call_fingerprint=fingerprint,
        decision=decision,
        reason_code=reason_code,
    )
    requested_digest = build_args_digest({"path": "/workspace/a.txt", "token": "token=secret-value-12345"})
    final_digest = build_args_digest({"path": "/workspace/a.txt"})
    return SafetyAuditRecord(
        id=audit_id,
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        run_id=run_id,
        step_id=step_id,
        action_id=action_id,
        tool_call_id=tool_call_id,
        capability_id=capability_id,
        tool_family=tool_family,
        function_name=function_name,
        normalized_function_name=normalized_function_name,
        requested_args_digest=requested_digest,
        final_function_name=final_function_name,
        final_normalized_function_name=final_normalized_function_name,
        final_args_digest=final_digest,
        decision=decision,
        reason_code=reason_code,
        risk_level=risk_level,
        policy_trace=[
            SafetyAuditPolicyTraceEntry(
                policy_name="task_mode_policy",
                action=decision.value,
                reason_code=reason_code,
            )
        ],
        winning_policy="task_mode_policy",
        tool_call_fingerprint=fingerprint,
        rewrite_applied=decision == SafetyAuditDecision.REWRITE,
        rewrite_reason=reason_code if decision == SafetyAuditDecision.REWRITE else None,
        rewrite_metadata_digest=SafetyAuditRewriteMetadataDigest.model_validate(
            build_args_digest({"rewrite_reason": reason_code}).model_dump(mode="json")
        ),
        profile_hash="sha256:" + "a" * 64,
        origin=DataOrigin.SYSTEM_OPERATIONAL,
        trust_level=DataTrustLevel.SYSTEM_GENERATED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        classification={
            "origin": DataOrigin.SYSTEM_OPERATIONAL.value,
            "trust_level": DataTrustLevel.SYSTEM_GENERATED.value,
            "privacy_level": PrivacyLevel.PRIVATE.value,
            "retention_policy": RetentionPolicyKind.WORKSPACE_BOUND.value,
            "has_sensitive_refs": False,
            "data_categories": [],
        },
        risk_classification_digest=SafetyAuditRiskClassificationDigest(
            risk_level=risk_level,
            matched_rule="low_reason_code",
        ),
        created_at=datetime(2026, 5, 16, 10, 0, 0),
    )


def test_safety_audit_record_schema_is_strict_and_action_id_is_required() -> None:
    record = _record()
    payload = record.model_dump(mode="json")
    payload["extra"] = "forbidden"
    with pytest.raises(ValidationError):
        SafetyAuditRecord.model_validate(payload)

    payload = record.model_dump(mode="json")
    payload["action_id"] = "wrong"
    with pytest.raises(ValidationError, match="action_id"):
        SafetyAuditRecord.model_validate(payload)


def test_args_digest_redacts_secret_long_text_url_query_and_unverified_path() -> None:
    digest = build_args_digest(
        {
            "api_key": "api_key=sk-secret-value-123456",
            "content": "hello" * 100,
            "url": "https://example.com/search?q=token=abc&safe=1",
            "path": "/Users/alice/private/report.txt",
        }
    )
    serialized = str(digest)
    assert "sk-secret-value" not in serialized
    assert "token=abc" not in serialized
    assert "/Users/alice/private/report.txt" not in serialized

    assert digest.fields["api_key"].has_secret is True
    assert digest.fields["content"].truncated is True
    assert digest.fields["url"].url.query_stripped is True
    path_digest = digest.fields["path"].path
    assert path_digest.basename == "report.txt"
    assert path_digest.scope_status == "unverified"
    assert path_digest.relative_path is None


def test_args_digest_allows_relative_path_only_after_safe_root_validation() -> None:
    digest = build_args_digest(
        {"path": "/workspace/project/report.txt"},
        safe_path_roots=["/workspace"],
    )
    path_digest = digest.fields["path"].path
    assert path_digest.scope_status == "verified"
    assert path_digest.relative_path == "project/report.txt"


def test_args_digest_preserves_relative_scope_hint_without_raw_path() -> None:
    digest = build_args_digest({"path": "reports/final.md"})

    path_digest = digest.fields["path"].path
    assert path_digest.is_absolute is False
    assert path_digest.scope_hint == "relative"
    assert path_digest.relative_path is None


@pytest.mark.parametrize(
    "field_name",
    ["requested_args_digest", "final_args_digest", "rewrite_metadata_digest", "classification"],
)
def test_record_rejects_raw_payload_fields(field_name: str) -> None:
    payload = _record().model_dump(mode="json")
    payload[field_name] = {
        "raw_args": {"token": "token=secret-value-123456"},
        "stdout": "secret output",
        "content": "raw file body",
        "path": "/Users/alice/private/report.txt",
        "matched_rule": "low_reason_code",
    }

    with pytest.raises(ValidationError):
        SafetyAuditRecord.model_validate(payload)


def test_risk_classification_digest_is_separate_from_data_classification() -> None:
    record = _record()
    model = SafetyAuditRecordModel.from_domain(record)

    assert "matched_rule" not in model.classification
    assert model.risk_classification_digest == {
        "risk_level": SafetyAuditRiskLevel.LOW.value,
        "matched_rule": "low_reason_code",
    }

    payload = record.model_dump(mode="json")
    payload["classification"]["matched_rule"] = "low_reason_code"
    with pytest.raises(ValidationError):
        SafetyAuditRecord.model_validate(payload)


def test_rewrite_metadata_digest_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SafetyAuditRewriteMetadataDigest.model_validate({"raw_args": {"path": "/tmp/a.txt"}})


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.BLOCK,
                normalized_function_name="read_file",
                tool_family="filesystem",
                capability_id="filesystem.read",
                action_kind="read_file",
                reason_code="constraint_engine_error",
            ),
            SafetyAuditRiskLevel.CRITICAL,
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="run_shell",
                tool_family="shell",
                capability_id="shell.execute",
                action_kind="execute",
                reason_code="allow",
            ),
            SafetyAuditRiskLevel.HIGH,
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.REWRITE,
                normalized_function_name="search",
                tool_family="research",
                capability_id="research.search",
                action_kind="search",
                reason_code="research_search_to_fetch_rewrite",
            ),
            SafetyAuditRiskLevel.MEDIUM,
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="message",
                tool_family="message",
                capability_id="message.tool",
                action_kind="message",
                reason_code="allow",
            ),
            SafetyAuditRiskLevel.LOW,
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                reason_code="new_unknown_reason",
            ),
            SafetyAuditRiskLevel.HIGH,
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="search",
                tool_family="research",
                capability_id="research.search",
                action_kind=None,
                reason_code="allow",
            ),
            SafetyAuditRiskLevel.HIGH,
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="message",
                tool_family="message",
                capability_id="unknown",
                action_kind="message",
                reason_code="allow",
            ),
            SafetyAuditRiskLevel.HIGH,
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="unknown",
                tool_family="unknown",
                capability_id="message.tool",
                action_kind="message",
                reason_code="allow",
            ),
            SafetyAuditRiskLevel.HIGH,
        ),
    ],
)
def test_risk_classifier_matrix(input_data: SafetyAuditRiskClassificationInput, expected: SafetyAuditRiskLevel) -> None:
    assert classify_safety_audit_risk(input_data).risk_level == expected


def test_risk_classifier_covers_current_constraint_reason_codes() -> None:
    actual_reason_codes = {
        value
        for name, value in vars(reason_codes).items()
        if name.startswith("REASON_") and isinstance(value, str)
    }
    assert actual_reason_codes.issubset(EXISTING_CONSTRAINT_REASON_CODES)

    missing: list[str] = []
    for reason_code in EXISTING_CONSTRAINT_REASON_CODES:
        result = classify_safety_audit_risk(
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.BLOCK if reason_code != "allow" else SafetyAuditDecision.ALLOW,
                normalized_function_name="message",
                tool_family="message",
                capability_id="message.tool",
                action_kind="message",
                reason_code=reason_code,
            )
        )
        if result.matched_rule == "unknown_default":
            missing.append(reason_code)
    assert missing == []


def test_orm_model_contains_required_strong_columns() -> None:
    columns = set(SafetyAuditRecordModel.__table__.columns.keys())
    for column in {
        "capability_id",
        "tool_family",
        "rewrite_applied",
        "rewrite_reason",
        "confirmation_id",
        "origin",
        "trust_level",
        "privacy_level",
        "retention_policy",
    }:
        assert column in columns

    assert SafetyAuditRecordModel.__tablename__ == "safety_audit_records"
    assert isinstance(SafetyAuditRecordModel.__table__.c.requested_args_digest.type, postgresql.JSONB)
    assert isinstance(SafetyAuditRecordModel.__table__.c.risk_classification_digest.type, postgresql.JSONB)


def test_repository_save_once_should_use_action_scope_idempotency() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: "audit-1"))
    )
    repository = DBSafetyAuditRepository(db_session=db_session)

    saved = asyncio.run(repository.save_once(_record()))

    assert saved.id == "audit-1"
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_safety_audit_records_user_session_run_action DO NOTHING" in compiled_sql
    assert "safety_audit_records" in compiled_sql


def test_repository_duplicate_should_return_existing_record_by_action_scope() -> None:
    existing = _record(audit_id="existing-audit")
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    existing_result = SimpleNamespace(scalar_one_or_none=lambda: SafetyAuditRecordModel.from_domain(existing))
    db_session = SimpleNamespace(execute=AsyncMock(side_effect=[execute_result, existing_result]))
    repository = DBSafetyAuditRepository(db_session=db_session)

    saved = asyncio.run(repository.save_once(_record()))

    assert saved.id == "existing-audit"
    lookup_statement = db_session.execute.call_args_list[1].args[0]
    compiled_sql = str(lookup_statement.compile(dialect=postgresql.dialect()))
    assert "safety_audit_records.user_id" in compiled_sql
    assert "safety_audit_records.session_id" in compiled_sql
    assert "safety_audit_records.run_id" in compiled_sql
    assert "safety_audit_records.action_id" in compiled_sql


def test_repository_queries_should_require_user_session_scope_and_strong_filters() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])))
    )
    repository = DBSafetyAuditRepository(db_session=db_session)

    result = asyncio.run(
        repository.list_by_step(
            user_id="user-1",
            session_id="session-1",
            run_id="run-1",
            step_id="step-1",
        )
    )

    assert result == []
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "safety_audit_records.user_id" in compiled_sql
    assert "safety_audit_records.session_id" in compiled_sql
    assert "safety_audit_records.run_id" in compiled_sql
    assert "safety_audit_records.step_id" in compiled_sql

    with pytest.raises(ValueError):
        asyncio.run(repository.list_by_run(user_id="", session_id="session-1", run_id="run-1"))


def test_repository_attach_linkage_only_updates_linkage_fields() -> None:
    record = _record()
    linked = _record().model_copy(
        update={
            "decision_event_id": "event-decision-1",
            "tool_event_source_event_id": "event-tool-1",
            "source_event_type": "tool_event",
            "source_linked_at": datetime(2026, 5, 16, 11, 0, 0),
        }
    )
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(scalar_one_or_none=lambda: SafetyAuditRecordModel.from_domain(record)),
                SimpleNamespace(rowcount=1),
                SimpleNamespace(scalar_one_or_none=lambda: SafetyAuditRecordModel.from_domain(linked)),
            ]
        )
    )
    repository = DBSafetyAuditRepository(db_session=db_session)

    updated = asyncio.run(
        repository.attach_linkage(
            user_id="user-1",
            session_id="session-1",
            audit_id=record.id,
            decision_event_id="event-decision-1",
            tool_event_source_event_id="event-tool-1",
            source_event_type="tool_event",
            source_linked_at=datetime(2026, 5, 16, 11, 0, 0),
        )
    )

    assert updated.decision_event_id == "event-decision-1"
    assert updated.tool_event_source_event_id == "event-tool-1"
    assert updated.decision == record.decision
    assert updated.risk_level == record.risk_level
    assert updated.requested_args_digest == record.requested_args_digest
    update_statement = db_session.execute.call_args_list[1].args[0]
    compiled_sql = str(update_statement.compile(dialect=postgresql.dialect()))
    assert "decision_event_id" in compiled_sql
    assert "IS NULL" in compiled_sql
    assert "risk_level" not in compiled_sql
    assert "requested_args_digest" not in compiled_sql


def test_repository_attach_linkage_rejects_conflicting_rewrite() -> None:
    record = _record().model_copy(update={"decision_event_id": "event-decision-1"})
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: SafetyAuditRecordModel.from_domain(record)))
    )
    repository = DBSafetyAuditRepository(db_session=db_session)

    with pytest.raises(SafetyAuditLinkageConflictError):
        asyncio.run(
            repository.attach_linkage(
                user_id="user-1",
                session_id="session-1",
                audit_id=record.id,
                decision_event_id="event-decision-2",
            )
        )


def test_repository_attach_linkage_rejects_concurrent_conditional_update_failure() -> None:
    record = _record()
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(scalar_one_or_none=lambda: SafetyAuditRecordModel.from_domain(record)),
                SimpleNamespace(rowcount=0),
            ]
        )
    )
    repository = DBSafetyAuditRepository(db_session=db_session)

    with pytest.raises(SafetyAuditLinkageConflictError):
        asyncio.run(
            repository.attach_linkage(
                user_id="user-1",
                session_id="session-1",
                audit_id=record.id,
                decision_event_id="event-decision-1",
            )
        )


def test_safety_audit_migration_should_define_table_indexes_and_unique_key() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "8c9d0e1f2a3b_create_safety_audit_records.py"
    )
    migration_text = migration_path.read_text(encoding="utf-8")

    assert "safety_audit_records" in migration_text
    assert "uq_safety_audit_records_user_session_run_action" in migration_text
    assert "ix_safety_audit_user_session_run_created" in migration_text
    assert "ix_safety_audit_user_session_run_step" in migration_text
    assert "ix_safety_audit_user_session_decision_risk" in migration_text
    assert "risk_classification_digest" in migration_text
