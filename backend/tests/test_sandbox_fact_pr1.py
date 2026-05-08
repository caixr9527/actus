import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from app.domain.models.sandbox_fact import (
    BrowserActionPayload,
    BrowserSnapshotPayload,
    CommandExecutionPayload,
    CorrectionPayload,
    DocumentContextPayload,
    FetchedPagePayload,
    FileListPayload,
    FileMutationPayload,
    FileReadPayload,
    FileSearchPayload,
    HumanInteractionPayload,
    ProfileReferencePayload,
    SandboxFactKind,
    SandboxFactProfileRef,
    SandboxFactRecord,
    SandboxFactScope,
    SandboxFactSourceRef,
    SandboxFactSourceType,
    SandboxFactSubjectRef,
    SearchResultPayload,
    ShellOutputPayload,
    ToolFailurePayload,
    SupersededPayload,
    build_sandbox_fact_idempotency_key,
    build_sandbox_fact_payload_hash,
    classify_sandbox_fact_data,
    validate_sandbox_fact_payload,
)
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.infrastructure.models.sandbox_fact import SandboxFactModel
from app.infrastructure.repositories.db_sandbox_fact_repository import (
    DBFactSupersededTargetError,
    DBSandboxFactRepository,
)


def _fact(
        *,
        fact_id: str = "fact-1",
        user_id: str = "user-1",
        session_id: str = "session-1",
        workspace_id: str = "workspace-1",
        fact_scope: SandboxFactScope = SandboxFactScope.STEP,
        run_id: str | None = "run-1",
        step_id: str | None = "step-1",
        fact_kind: SandboxFactKind = SandboxFactKind.COMMAND_EXECUTION,
        source_event_id: str | None = "event-1",
        tool_call_id: str | None = "tool-call-1",
        subject_key: str = "command:abc",
        payload: dict | None = None,
        supersedes_fact_id: str | None = None,
) -> SandboxFactRecord:
    raw_payload = payload or _payload_for_kind(fact_kind, supersedes_fact_id=supersedes_fact_id)
    actual_payload = validate_sandbox_fact_payload(
        fact_kind=fact_kind,
        payload=raw_payload,
    ).model_dump(mode="json")
    payload_hash = build_sandbox_fact_payload_hash(actual_payload)
    source_ref = SandboxFactSourceRef(
        source_type=SandboxFactSourceType.TOOL_EVENT,
        source_event_id=source_event_id,
        source_event_status="available" if source_event_id else "missing",
        tool_event_id="tool-event-1",
        tool_call_id=tool_call_id,
        function_name="run_shell",
    )
    subject_ref = SandboxFactSubjectRef(
        subject_type="command",
        subject_key=subject_key,
    )
    idempotency_key = build_sandbox_fact_idempotency_key(
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        fact_scope=fact_scope,
        run_id=run_id,
        step_id=step_id,
        fact_kind=fact_kind,
        source_event_id=source_ref.source_event_id,
        tool_call_id=source_ref.tool_call_id,
        subject_key=subject_ref.subject_key,
        payload_hash=payload_hash,
    )
    classification = classify_sandbox_fact_data(
        fact_kind=fact_kind,
        source_type=source_ref.source_type,
    )
    return SandboxFactRecord(
        id=fact_id,
        user_id=user_id,
        session_id=session_id,
        workspace_id=workspace_id,
        fact_scope=fact_scope,
        run_id=run_id,
        step_id=step_id,
        sandbox_id="sandbox-1",
        fact_kind=fact_kind,
        source_ref=source_ref,
        subject_ref=subject_ref,
        profile_ref=SandboxFactProfileRef(
            profile_id="profile-1",
            profile_hash="sha256:" + "a" * 64,
            sandbox_id="sandbox-1",
            generated_at=datetime(2026, 5, 6, 9, 0, 0),
            status="available",
        ),
        supersedes_fact_id=supersedes_fact_id,
        summary="执行了命令",
        payload=actual_payload,
        payload_hash=payload_hash,
        idempotency_key=idempotency_key,
        origin=classification.origin,
        trust_level=classification.trust_level,
        privacy_level=classification.privacy_level,
        retention_policy=classification.retention_policy,
        created_at=datetime(2026, 5, 6, 10, 0, 0),
    )


def _payload_for_kind(
        fact_kind: SandboxFactKind,
        *,
        supersedes_fact_id: str | None = None,
) -> dict:
    base_file_mutation = {
        "path": "/workspace/a.txt",
        "operation": "write",
        "mutation_intent_hash": "sha256:intent",
        "exists": True,
        "before_content_sha256": None,
        "after_content_sha256": "sha256:after",
        "content_sha256_kind": "read_content_sha256",
        "size_after": 12,
        "changed": True,
    }
    payloads: dict[SandboxFactKind, dict] = {
        SandboxFactKind.COMMAND_EXECUTION: {
            "command_fingerprint": "sha256:abc",
            "cwd": "/workspace",
            "exit_code": 0,
            "duration_ms": 10,
            "stdout_excerpt": "",
            "stderr_excerpt": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "changed_paths": [],
            "timeout": False,
        },
        SandboxFactKind.SHELL_OUTPUT: {
            "session_ref": "shell-1",
            "output_excerpt": "ok",
            "output_truncated": False,
            "console_record_count": 1,
            "process_status": "completed",
        },
        SandboxFactKind.FILE_READ: {
            "path": "/workspace/a.txt",
            "exists": True,
            "size": 12,
            "content_sha256": "sha256:file",
            "content_sha256_kind": "read_content_sha256",
            "mime_type": "text/plain",
            "line_range": {"start": 1, "end": 1},
            "excerpt": "hello",
            "is_truncated": False,
        },
        SandboxFactKind.FILE_WRITE: base_file_mutation,
        SandboxFactKind.FILE_DELETE: {**base_file_mutation, "operation": "delete", "exists": False, "after_content_sha256": None, "size_after": None},
        SandboxFactKind.FILE_SNAPSHOT: {**base_file_mutation, "operation": "snapshot"},
        SandboxFactKind.FILE_LIST: {
            "dir_path": "/workspace",
            "entry_count": 1,
            "entries": [{"name": "a.txt", "type": "file", "size": 12}],
            "is_truncated": False,
        },
        SandboxFactKind.FILE_SEARCH: {
            "path": "/workspace",
            "regex_hash": "sha256:regex",
            "match_count": 1,
            "matches": [{"path": "/workspace/a.txt", "line_number": 1, "excerpt": "hello"}],
            "is_truncated": False,
        },
        SandboxFactKind.BROWSER_SNAPSHOT: {
            "url_hash": "sha256:url",
            "url_origin": "https://example.com",
            "title": "Example",
            "screenshot_artifact_id": "artifact-1",
            "screenshot_artifact_path": "/artifacts/s.png",
            "structured_summary": "page",
            "actionable_element_count": 1,
            "degrade_reason": None,
        },
        SandboxFactKind.BROWSER_ACTION: {
            "action": "click",
            "target_summary": "button",
            "url_hash_before": "sha256:before",
            "url_hash_after": "sha256:after",
            "success": True,
            "degrade_reason": None,
        },
        SandboxFactKind.SEARCH_RESULT: {
            "query_hash": "sha256:query",
            "query_excerpt": "query",
            "result_count": 1,
            "top_results": [{"title": "A", "origin": "https://example.com", "url_hash": "sha256:url", "snippet_excerpt": "snippet"}],
            "is_truncated": False,
        },
        SandboxFactKind.FETCHED_PAGE: {
            "fetched_url_hash": "sha256:url",
            "final_url_origin": "https://example.com",
            "status_code": 200,
            "content_type": "text/html",
            "title": "Example",
            "excerpt": "page",
            "is_truncated": False,
        },
        SandboxFactKind.DOCUMENT_CONTEXT: {
            "file_id": "file-1",
            "filename_extension": ".pdf",
            "mime_type": "application/pdf",
            "parse_status": "success",
            "reason_code": None,
            "full_file_sha256": None,
            "read_content_sha256": "sha256:read",
            "is_truncated": False,
            "excerpt_char_count": 10,
        },
        SandboxFactKind.PROFILE_REFERENCE: {
            "profile_id": "profile-1",
            "profile_hash": "sha256:" + "a" * 64,
            "sandbox_id": "sandbox-1",
            "generated_at": "2026-05-06T09:00:00",
            "status": "available",
        },
        SandboxFactKind.TOOL_FAILURE: {
            "function_name": "run_shell",
            "reason_code": "timeout",
            "message_excerpt": "timeout",
            "retry_count": 1,
            "timeout": True,
            "diagnostic_type": "tool_error",
        },
        SandboxFactKind.HUMAN_INTERACTION: {
            "interaction_type": "confirmation",
            "message_excerpt": "approved",
            "confirmed": True,
        },
        SandboxFactKind.CORRECTION: {
            "corrected_fact_ids": ["fact-1"],
            "reason_code": "wrong_value",
            "message_excerpt": "修正",
        },
        SandboxFactKind.SUPERSEDED: {
            "supersedes_fact_id": supersedes_fact_id or "fact-1",
            "reason_code": "replaced",
            "message_excerpt": "废弃",
        },
    }
    return payloads[fact_kind]


def test_sandbox_fact_contract_should_reject_extra_fields() -> None:
    payload = _fact().model_dump(mode="json")
    payload["legacy_event_ref"] = "legacy-event"

    with pytest.raises(ValidationError):
        SandboxFactRecord.model_validate(payload)


def test_sandbox_fact_scope_should_validate_required_run_and_step_fields() -> None:
    with pytest.raises(ValidationError):
        _fact(fact_scope=SandboxFactScope.STEP, run_id=None, step_id="step-1")

    with pytest.raises(ValidationError):
        _fact(fact_scope=SandboxFactScope.RUN, run_id="run-1", step_id="step-1")

    workspace_fact = _fact(fact_scope=SandboxFactScope.WORKSPACE, run_id=None, step_id=None)
    assert workspace_fact.fact_scope == SandboxFactScope.WORKSPACE


def test_sandbox_fact_should_use_source_event_id_only() -> None:
    source_ref_fields = SandboxFactSourceRef.model_fields
    record_fields = SandboxFactRecord.model_fields

    assert "source_event_id" in source_ref_fields
    assert "event_ref_id" not in source_ref_fields
    assert "event_ref_id" not in record_fields


def test_sandbox_fact_payload_schema_should_reject_extra_and_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        CommandExecutionPayload.model_validate(
            {
                "command_fingerprint": "sha256:abc",
                "cwd": "/workspace",
                "exit_code": 0,
                "duration_ms": 10,
                "stdout_excerpt": "",
                "stderr_excerpt": "",
                "stdout_truncated": False,
                "stderr_truncated": False,
                "changed_paths": [],
                "timeout": False,
                "raw_stdout": "forbidden",
            }
        )

    with pytest.raises(ValidationError):
        SearchResultPayload.model_validate(
            {
                "query_hash": "sha256:abc",
                "query_excerpt": "query",
                "result_count": 0,
                "is_truncated": False,
            }
        )

    DocumentContextPayload.model_validate(
        {
            "file_id": "file-1",
            "filename_extension": ".pdf",
            "mime_type": "application/pdf",
            "parse_status": "success",
            "reason_code": None,
            "full_file_sha256": None,
            "read_content_sha256": "sha256:def",
            "is_truncated": False,
            "excerpt_char_count": 100,
        }
    )
    ToolFailurePayload.model_validate(
        {
            "function_name": "search",
            "reason_code": "timeout",
            "message_excerpt": "timeout",
            "retry_count": 1,
            "timeout": True,
            "diagnostic_type": "tool_error",
        }
    )


def test_sandbox_fact_record_should_validate_payload_by_fact_kind() -> None:
    with pytest.raises(ValidationError):
        _fact(payload={"command_fingerprint": "sha256:abc", "exit_code": 0})


def test_sandbox_fact_record_should_reject_mismatched_payload_hash() -> None:
    payload = _fact().model_dump(mode="json")
    payload["payload_hash"] = "sha256:" + "0" * 64

    with pytest.raises(ValidationError, match="payload_hash"):
        SandboxFactRecord.model_validate(payload)


def test_sandbox_fact_record_should_reject_mismatched_idempotency_key() -> None:
    payload = _fact().model_dump(mode="json")
    payload["idempotency_key"] = "sha256:" + "1" * 64

    with pytest.raises(ValidationError, match="idempotency_key"):
        SandboxFactRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("fact_kind", "payload_schema"),
    [
        (SandboxFactKind.COMMAND_EXECUTION, CommandExecutionPayload),
        (SandboxFactKind.SHELL_OUTPUT, ShellOutputPayload),
        (SandboxFactKind.FILE_READ, FileReadPayload),
        (SandboxFactKind.FILE_WRITE, FileMutationPayload),
        (SandboxFactKind.FILE_DELETE, FileMutationPayload),
        (SandboxFactKind.FILE_LIST, FileListPayload),
        (SandboxFactKind.FILE_SEARCH, FileSearchPayload),
        (SandboxFactKind.FILE_SNAPSHOT, FileMutationPayload),
        (SandboxFactKind.BROWSER_SNAPSHOT, BrowserSnapshotPayload),
        (SandboxFactKind.BROWSER_ACTION, BrowserActionPayload),
        (SandboxFactKind.SEARCH_RESULT, SearchResultPayload),
        (SandboxFactKind.FETCHED_PAGE, FetchedPagePayload),
        (SandboxFactKind.DOCUMENT_CONTEXT, DocumentContextPayload),
        (SandboxFactKind.PROFILE_REFERENCE, ProfileReferencePayload),
        (SandboxFactKind.TOOL_FAILURE, ToolFailurePayload),
        (SandboxFactKind.HUMAN_INTERACTION, HumanInteractionPayload),
        (SandboxFactKind.CORRECTION, CorrectionPayload),
        (SandboxFactKind.SUPERSEDED, SupersededPayload),
    ],
)
def test_sandbox_fact_record_should_accept_each_kind_payload_schema(
        fact_kind: SandboxFactKind,
        payload_schema,
) -> None:
    supersedes_fact_id = "fact-original" if fact_kind == SandboxFactKind.SUPERSEDED else None
    fact = _fact(
        fact_kind=fact_kind,
        subject_key=f"{fact_kind.value}:subject",
        payload=_payload_for_kind(fact_kind, supersedes_fact_id=supersedes_fact_id),
        supersedes_fact_id=supersedes_fact_id,
    )

    assert payload_schema.model_validate(fact.payload)


def test_sandbox_fact_data_classification_should_match_representative_kinds() -> None:
    document = classify_sandbox_fact_data(
        fact_kind=SandboxFactKind.DOCUMENT_CONTEXT,
        source_type=SandboxFactSourceType.DOCUMENT_INPUT,
    )
    assert document.origin == DataOrigin.USER_UPLOAD
    assert document.trust_level == DataTrustLevel.USER_PROVIDED

    search = classify_sandbox_fact_data(
        fact_kind=SandboxFactKind.SEARCH_RESULT,
        source_type=SandboxFactSourceType.SANDBOX_API,
    )
    assert search.origin == DataOrigin.EXTERNAL_WEB
    assert search.trust_level == DataTrustLevel.EXTERNAL_UNTRUSTED

    profile = classify_sandbox_fact_data(
        fact_kind=SandboxFactKind.PROFILE_REFERENCE,
        source_type=SandboxFactSourceType.PROFILE,
    )
    assert profile.privacy_level == PrivacyLevel.INTERNAL
    assert profile.retention_policy == RetentionPolicyKind.WORKSPACE_BOUND


def test_sandbox_fact_model_should_round_trip_ref_snapshots() -> None:
    fact = _fact()
    restored = SandboxFactModel.from_domain(fact).to_domain()

    assert restored.source_ref == fact.source_ref
    assert restored.subject_ref == fact.subject_ref
    assert restored.profile_ref == fact.profile_ref
    assert restored.payload == fact.payload


def test_sandbox_fact_correction_should_append_without_mutating_original() -> None:
    original = _fact()
    original_payload = dict(original.payload)
    correction = _fact(
        fact_id="fact-correction",
        fact_kind=SandboxFactKind.CORRECTION,
        subject_key="correction:1",
        payload={"corrected_fact_ids": [original.id], "message_excerpt": "修正", "reason_code": "wrong_value"},
    ).model_copy(update={"related_fact_ids": [original.id]})

    assert original.payload == original_payload
    assert correction.id != original.id
    assert correction.related_fact_ids == [original.id]


def test_sandbox_fact_superseded_should_append_and_reference_original() -> None:
    original = _fact()
    original_payload = dict(original.payload)
    superseded = _fact(
        fact_id="fact-super",
        fact_kind=SandboxFactKind.SUPERSEDED,
        subject_key="superseded:1",
        payload={"supersedes_fact_id": original.id, "message_excerpt": "废弃", "reason_code": "replaced"},
        supersedes_fact_id=original.id,
    )

    assert superseded.id != original.id
    assert superseded.supersedes_fact_id == original.id
    assert original.payload == original_payload


def test_sandbox_fact_repository_should_save_once_by_idempotency_key() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: "fact-1"))
    )
    repository = DBSandboxFactRepository(db_session=db_session)

    saved = asyncio.run(repository.save_once(_fact()))

    assert saved.id == "fact-1"
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_sandbox_facts_idempotency_key DO NOTHING" in compiled_sql
    assert "sandbox_facts" in compiled_sql


def test_sandbox_fact_repository_duplicate_should_return_existing_record() -> None:
    existing = _fact(fact_id="existing-fact")
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    existing_result = SimpleNamespace(scalar_one_or_none=lambda: SandboxFactModel.from_domain(existing))
    db_session = SimpleNamespace(execute=AsyncMock(side_effect=[execute_result, existing_result]))
    repository = DBSandboxFactRepository(db_session=db_session)

    saved = asyncio.run(repository.save_once(_fact()))

    assert saved.id == "existing-fact"


def test_sandbox_fact_repository_duplicate_lookup_should_filter_user_session() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    filtered_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    db_session = SimpleNamespace(execute=AsyncMock(side_effect=[execute_result, filtered_result]))
    repository = DBSandboxFactRepository(db_session=db_session)

    with pytest.raises(RuntimeError):
        asyncio.run(repository.save_once(_fact()))

    lookup_statement = db_session.execute.call_args_list[1].args[0]
    compiled_sql = str(lookup_statement.compile(dialect=postgresql.dialect()))
    assert "sandbox_facts.user_id" in compiled_sql
    assert "sandbox_facts.session_id" in compiled_sql
    assert "sandbox_facts.idempotency_key" in compiled_sql


def test_sandbox_fact_repository_should_validate_superseded_target_scope() -> None:
    execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    db_session = SimpleNamespace(execute=AsyncMock(return_value=execute_result))
    repository = DBSandboxFactRepository(db_session=db_session)
    fact = _fact(
        fact_id="fact-super",
        fact_kind=SandboxFactKind.SUPERSEDED,
        payload={"supersedes_fact_id": "missing", "message_excerpt": "废弃", "reason_code": "replaced"},
        subject_key="superseded:missing",
        supersedes_fact_id="missing",
    )

    with pytest.raises(DBFactSupersededTargetError):
        asyncio.run(repository.save_once(fact))


def test_sandbox_fact_query_should_require_user_session_scope() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])))
    )
    repository = DBSandboxFactRepository(db_session=db_session)

    asyncio.run(
        repository.list_by_scope(
            user_id="user-1",
            session_id="session-1",
            fact_scope=SandboxFactScope.STEP,
            run_id="run-1",
            step_id="step-1",
            fact_kinds=[SandboxFactKind.COMMAND_EXECUTION],
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "sandbox_facts.user_id" in compiled_sql
    assert "sandbox_facts.session_id" in compiled_sql
    assert "sandbox_facts.run_id" in compiled_sql
    assert "sandbox_facts.step_id" in compiled_sql


def test_sandbox_fact_query_should_reject_empty_user_session_scope() -> None:
    db_session = SimpleNamespace(execute=AsyncMock())
    repository = DBSandboxFactRepository(db_session=db_session)

    with pytest.raises(ValueError):
        asyncio.run(repository.list_by_scope(user_id="", session_id="session-1"))

    with pytest.raises(ValueError):
        asyncio.run(
            repository.list_by_source_event(
                user_id="user-1",
                session_id="",
                source_event_id="event-1",
            )
        )

    db_session.execute.assert_not_awaited()


def test_sandbox_fact_list_by_ids_should_require_user_session_scope() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])))
    )
    repository = DBSandboxFactRepository(db_session=db_session)

    result = asyncio.run(
        repository.list_by_ids(
            user_id="user-1",
            session_id="session-1",
            fact_ids=["fact-1", ""],
        )
    )

    assert result == []
    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "sandbox_facts.user_id" in compiled_sql
    assert "sandbox_facts.session_id" in compiled_sql
    assert "sandbox_facts.id" in compiled_sql


def test_sandbox_fact_repository_list_by_ids_should_restore_ref_snapshots_from_model() -> None:
    fact = _fact()
    model = SandboxFactModel.from_domain(fact)
    db_session = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: [model])
            )
        )
    )
    repository = DBSandboxFactRepository(db_session=db_session)

    result = asyncio.run(
        repository.list_by_ids(
            user_id=fact.user_id,
            session_id=fact.session_id,
            fact_ids=[fact.id],
        )
    )

    assert len(result) == 1
    restored = result[0]
    assert isinstance(restored, SandboxFactRecord)
    assert restored.source_ref == fact.source_ref
    assert restored.subject_ref == fact.subject_ref
    assert restored.profile_ref == fact.profile_ref
    assert restored.payload == fact.payload


def test_sandbox_fact_migration_should_define_table_indexes_and_unique_key() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "7f8e9a0b1c2d_create_sandbox_facts.py"
    )
    content = migration_path.read_text(encoding="utf-8")

    assert '"sandbox_facts"' in content
    assert "uq_sandbox_facts_idempotency_key" in content
    assert "ix_sandbox_facts_user_session_created" in content
    assert "ix_sandbox_facts_user_run_step" in content
    assert "ix_sandbox_facts_source_event" in content
