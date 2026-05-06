import asyncio
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.application.service.sandbox_fact_ledger_service import (
    CommandExecutionFactInput,
    DocumentContextFactInput,
    FetchedPageFactInput,
    HumanInteractionFactInput,
    ProfileReferenceFactInput,
    SandboxFactLedgerService,
    SandboxFactProfileMismatchError,
    SandboxFactScopeError,
    SearchResultFactInput,
    SearchResultItemInput,
    ToolFailureFactInput,
    normalize_fact_input,
)
from app.domain.models.sandbox_fact import (
    SandboxFactKind,
    SandboxFactProfileRef,
    SandboxFactRecord,
    SandboxFactScope,
    SandboxFactSourceType,
)
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.domain.services.runtime.contracts.sandbox_fact_ports import (
    SandboxFactProjectionContext,
)


class _SandboxFactRepo:
    def __init__(self) -> None:
        self.saved: list[SandboxFactRecord] = []
        self.list_calls: list[dict] = []

    async def save_once(self, fact: SandboxFactRecord) -> SandboxFactRecord:
        self.saved.append(fact)
        return fact

    async def list_by_scope(self, **kwargs):
        self.list_calls.append(kwargs)
        return []


class _UoW:
    def __init__(self, repo: _SandboxFactRepo) -> None:
        self.sandbox_fact = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self):
        return None

    async def rollback(self):
        return None


def _context(
        *,
        user_id: str = "user-1",
        session_id: str | None = "session-1",
        workspace_id: str | None = "workspace-1",
        run_id: str | None = "run-1",
        current_step_id: str | None = "step-1",
        sandbox_id: str | None = "sandbox-1",
        profile_ref: SandboxFactProfileRef | None = None,
) -> SandboxFactProjectionContext:
    return SandboxFactProjectionContext(
        scope=AccessScopeResult(
            tenant_id=user_id,
            user_id=user_id,
            session_id=session_id,
            workspace_id=workspace_id,
            run_id=run_id,
            current_step_id=current_step_id,
        ),
        profile_ref=profile_ref or SandboxFactProfileRef(
            profile_id="profile-1",
            profile_hash="sha256:" + "a" * 64,
            sandbox_id=sandbox_id,
            generated_at=datetime(2026, 5, 6, 9, 0, 0),
            status="available",
        ),
        sandbox_id=sandbox_id,
        source_event_id="event-1",
        current_step_id=current_step_id,
    )


def _service(repo: _SandboxFactRepo) -> SandboxFactLedgerService:
    return SandboxFactLedgerService(uow_factory=lambda: _UoW(repo))


def test_normalizer_should_redact_secret_and_truncate_command_output() -> None:
    long_stdout = '"api_key": "abcdefghijklmnop" ' + "x" * 5000
    payload, subject_ref, _ = normalize_fact_input(
        CommandExecutionFactInput(
            command='echo {"password":"secret12345"}',
            cwd="/workspace",
            exit_code=0,
            duration_ms=12,
            stdout=long_stdout,
            stderr='{"password":"secret12345"}',
        )
    )

    assert "[REDACTED]" in payload["stdout_excerpt"]
    assert "[REDACTED]" in payload["stderr_excerpt"]
    assert "abcdefghijklmnop" not in str(payload)
    assert "secret12345" not in str(payload)
    assert len(payload["stdout_excerpt"]) == 4000
    assert payload["stdout_truncated"] is True
    assert subject_ref.subject_type == "command"


def test_record_fact_should_write_classified_sanitized_command_fact() -> None:
    repo = _SandboxFactRepo()
    service = _service(repo)

    fact = asyncio.run(
        service.record_fact(
            context=_context(),
            fact_input=CommandExecutionFactInput(
                source_type=SandboxFactSourceType.SANDBOX_API,
                summary="token=abcdefghijklmnop",
                command="python run.py --api_key=abcdefghijklmnop",
                cwd="/workspace",
                exit_code=0,
                duration_ms=20,
                stdout="ok",
                stderr="",
                changed_paths=["/workspace/out.txt"],
            ),
        )
    )

    assert repo.saved == [fact]
    assert fact.user_id == "user-1"
    assert fact.session_id == "session-1"
    assert fact.workspace_id == "workspace-1"
    assert fact.run_id == "run-1"
    assert fact.step_id == "step-1"
    assert fact.source_ref.source_event_id == "event-1"
    assert fact.origin == DataOrigin.SANDBOX_STATE
    assert fact.trust_level == DataTrustLevel.SYSTEM_GENERATED
    assert "[REDACTED]" in str(fact.payload)
    assert "[REDACTED]" in fact.summary
    assert "abcdefghijklmnop" not in fact.summary


def test_fact_input_should_forbid_source_event_id_override() -> None:
    with pytest.raises(ValidationError):
        CommandExecutionFactInput.model_validate(
            {
                "source_event_id": "forged-event",
                "command": "echo ok",
                "cwd": "/workspace",
                "exit_code": 0,
                "duration_ms": 1,
            }
        )


def test_record_fact_should_use_context_source_event_id_only() -> None:
    repo = _SandboxFactRepo()
    service = _service(repo)

    fact = asyncio.run(
        service.record_fact(
            context=_context(),
            fact_input=CommandExecutionFactInput(
                command="echo ok",
                cwd="/workspace",
                exit_code=0,
                duration_ms=1,
            ),
        )
    )

    assert fact.source_ref.source_event_id == "event-1"


def test_record_fact_should_fail_closed_for_missing_scope_user_session_workspace() -> None:
    repo = _SandboxFactRepo()
    service = _service(repo)

    with pytest.raises(SandboxFactScopeError):
        asyncio.run(
            service.record_fact(
                context=_context(session_id=None),
                fact_input=CommandExecutionFactInput(
                    command="echo ok",
                    cwd="/workspace",
                    exit_code=0,
                    duration_ms=1,
                ),
            )
        )

    assert repo.saved == []


def test_record_fact_should_fail_when_step_scope_uses_non_current_step() -> None:
    repo = _SandboxFactRepo()
    service = _service(repo)

    with pytest.raises(SandboxFactScopeError):
        asyncio.run(
            service.record_fact(
                context=_context(current_step_id="step-1"),
                fact_input=CommandExecutionFactInput(
                    step_id="step-2",
                    command="echo ok",
                    cwd="/workspace",
                    exit_code=0,
                    duration_ms=1,
                ),
            )
        )


def test_record_fact_should_reject_step_id_for_run_and_workspace_scope() -> None:
    repo = _SandboxFactRepo()
    service = _service(repo)

    with pytest.raises(SandboxFactScopeError):
        asyncio.run(
            service.record_fact(
                context=_context(),
                fact_input=CommandExecutionFactInput(
                    fact_scope=SandboxFactScope.RUN,
                    step_id="step-1",
                    command="echo ok",
                    cwd="/workspace",
                    exit_code=0,
                    duration_ms=1,
                ),
            )
        )

    with pytest.raises(SandboxFactScopeError):
        asyncio.run(
            service.record_fact(
                context=_context(),
                fact_input=CommandExecutionFactInput(
                    fact_scope=SandboxFactScope.WORKSPACE,
                    step_id="step-1",
                    command="echo ok",
                    cwd="/workspace",
                    exit_code=0,
                    duration_ms=1,
                ),
            )
        )


def test_record_fact_should_fail_closed_for_profile_sandbox_mismatch() -> None:
    repo = _SandboxFactRepo()
    service = _service(repo)

    with pytest.raises(SandboxFactProfileMismatchError):
        asyncio.run(
            service.record_fact(
                context=_context(
                    sandbox_id="sandbox-1",
                    profile_ref=SandboxFactProfileRef(
                        profile_id="profile-1",
                        profile_hash="sha256:" + "a" * 64,
                        sandbox_id="sandbox-2",
                        status="available",
                    ),
                ),
                fact_input=CommandExecutionFactInput(
                    command="echo ok",
                    cwd="/workspace",
                    exit_code=0,
                    duration_ms=1,
                ),
            )
        )


def test_record_fact_should_preserve_missing_profile_ref() -> None:
    repo = _SandboxFactRepo()
    service = _service(repo)

    fact = asyncio.run(
        service.record_fact(
            context=_context(
                sandbox_id=None,
                profile_ref=SandboxFactProfileRef(status="missing"),
            ),
            fact_input=ProfileReferenceFactInput(fact_scope=SandboxFactScope.RUN),
        )
    )

    assert fact.fact_kind == SandboxFactKind.PROFILE_REFERENCE
    assert fact.step_id is None
    assert fact.profile_ref.status == "missing"
    assert fact.payload["status"] == "missing"
    assert "profile_hash" in fact.payload["missing_fields"]
    assert fact.privacy_level == PrivacyLevel.INTERNAL
    assert fact.retention_policy == RetentionPolicyKind.WORKSPACE_BOUND


def test_record_fact_should_write_document_search_fetch_tool_failure_and_user_classification() -> None:
    cases = [
        (
            DocumentContextFactInput(
                fact_scope=SandboxFactScope.RUN,
                file_id="file-1",
                filename_extension=".pdf",
                mime_type="application/pdf",
                parse_status="failed",
                reason_code="parse_failed",
                is_truncated=True,
                excerpt_char_count=0,
            ),
            DataOrigin.USER_UPLOAD,
            DataTrustLevel.USER_PROVIDED,
        ),
        (
            SearchResultFactInput(
                query="api_key=abcdefghijklmnop langgraph",
                result_count=1,
                top_results=[
                    SearchResultItemInput(
                        title="Title",
                        url="https://example.com/a?token=secret",
                        snippet="cookie=abcdefghijklmnop snippet",
                    )
                ],
            ),
            DataOrigin.EXTERNAL_WEB,
            DataTrustLevel.EXTERNAL_UNTRUSTED,
        ),
        (
            FetchedPageFactInput(
                fetched_url="https://example.com/page?token=secret",
                final_url="https://example.com/page",
                status_code=200,
                content_type="text/html",
                title="Page",
                content="password=secret12345 body",
            ),
            DataOrigin.EXTERNAL_WEB,
            DataTrustLevel.EXTERNAL_UNTRUSTED,
        ),
        (
            ToolFailureFactInput(
                function_name="search",
                reason_code="timeout",
                message="Bearer abcdefghijklmnop failed",
                retry_count=1,
                timeout=True,
                diagnostic_type="tool_error",
            ),
            DataOrigin.SYSTEM_OPERATIONAL,
            DataTrustLevel.SYSTEM_GENERATED,
        ),
        (
            HumanInteractionFactInput(
                interaction_type="confirmation",
                message="用户确认执行",
                confirmed=True,
            ),
            DataOrigin.USER_MESSAGE,
            DataTrustLevel.USER_PROVIDED,
        ),
    ]
    for fact_input, expected_origin, expected_trust in cases:
        repo = _SandboxFactRepo()
        service = _service(repo)

        fact = asyncio.run(service.record_fact(context=_context(), fact_input=fact_input))

        assert fact.origin == expected_origin
        assert fact.trust_level == expected_trust
        assert fact.payload_hash.startswith("sha256:")
        assert fact.idempotency_key.startswith("sha256:")
        assert repo.saved[0] == fact


def test_record_fact_should_reject_invalid_domain_input_required_fields() -> None:
    with pytest.raises(ValidationError):
        SearchResultFactInput.model_validate({"result_count": 1})


def test_list_facts_should_pass_user_session_scope_to_repository() -> None:
    repo = _SandboxFactRepo()
    service = _service(repo)

    result = asyncio.run(
        service.list_facts(
            context=_context(),
            fact_scope=SandboxFactScope.STEP,
            run_id="run-1",
            step_id="step-1",
            fact_kinds=[SandboxFactKind.COMMAND_EXECUTION],
            limit=50,
        )
    )

    assert result == []
    assert repo.list_calls == [
        {
            "user_id": "user-1",
            "session_id": "session-1",
            "fact_scope": SandboxFactScope.STEP,
            "run_id": "run-1",
            "step_id": "step-1",
            "fact_kinds": [SandboxFactKind.COMMAND_EXECUTION],
            "limit": 50,
        }
    ]


def test_sandbox_fact_ledger_service_should_not_implement_tool_event_recorder() -> None:
    service = _service(_SandboxFactRepo())

    assert not hasattr(service, "record_from_tool_event")
