import asyncio
from datetime import datetime

from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.application.service.sandbox_fact_event_projector import SandboxFactEventProjector
from app.domain.models import SandboxFactEvent
from app.domain.models.sandbox_fact import (
    SandboxFactKind,
    SandboxFactProfileRef,
    SandboxFactRecord,
    SandboxFactScope,
    SandboxFactSourceRef,
    SandboxFactSourceType,
    SandboxFactSubjectRef,
    build_sandbox_fact_idempotency_key,
    build_sandbox_fact_payload_hash,
    classify_sandbox_fact_data,
    validate_sandbox_fact_payload,
)
from app.domain.services.runtime.contracts.sandbox_fact_ports import SandboxFactProjectionContext


class _SandboxFactRepo:
    def __init__(self, facts: list[SandboxFactRecord]) -> None:
        self.facts = facts
        self.list_by_ids_calls: list[dict] = []

    async def list_by_ids(self, **kwargs) -> list[SandboxFactRecord]:
        self.list_by_ids_calls.append(kwargs)
        fact_ids = set(kwargs["fact_ids"])
        return [
            fact
            for fact in self.facts
            if fact.id in fact_ids
            and fact.user_id == kwargs["user_id"]
            and fact.session_id == kwargs["session_id"]
        ][:kwargs["limit"]]


class _UoW:
    def __init__(self, *, sandbox_fact: _SandboxFactRepo) -> None:
        self.sandbox_fact = sandbox_fact

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _context() -> SandboxFactProjectionContext:
    return SandboxFactProjectionContext(
        scope=AccessScopeResult(
            tenant_id="user-1",
            user_id="user-1",
            session_id="session-1",
            workspace_id="workspace-1",
            run_id="run-1",
            current_step_id="step-1",
        ),
        profile_ref=SandboxFactProfileRef(
            profile_id="profile-1",
            profile_hash="sha256:" + "a" * 64,
            sandbox_id="sandbox-1",
            generated_at=datetime(2026, 5, 6, 9, 0, 0),
            status="available",
        ),
        sandbox_id="sandbox-1",
        source_event_id="tool-event-1",
        current_step_id="step-1",
    )


def _payload_for_kind(fact_kind: SandboxFactKind) -> dict:
    if fact_kind == SandboxFactKind.TOOL_FAILURE:
        return {
            "function_name": "exec_command",
            "reason_code": "tool_failed",
            "message_excerpt": "tool failed",
            "retry_count": 0,
            "diagnostic_type": "tool_execution",
            "timeout": False,
            "missing_fields": None,
        }
    if fact_kind == SandboxFactKind.SHELL_OUTPUT:
        return {
            "session_ref": "shell-session-1",
            "output_excerpt": "pytest output",
            "output_truncated": False,
            "console_record_count": 1,
            "process_status": "completed",
            "exit_code": 0,
            "duration_ms": 1,
        }
    return {
        "command_fingerprint": "sha256:" + "b" * 64,
        "cwd": "/workspace",
        "exit_code": 0,
        "duration_ms": 1,
        "stdout_excerpt": "",
        "stderr_excerpt": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "changed_paths": [],
        "timeout": False,
    }


def _fact(
        *,
        fact_id: str,
        user_id: str = "user-1",
        session_id: str = "session-1",
        fact_kind: SandboxFactKind = SandboxFactKind.COMMAND_EXECUTION,
        summary: str = "exec_command tool fact",
        source_event_id: str | None = "tool-event-1",
) -> SandboxFactRecord:
    payload = validate_sandbox_fact_payload(
        fact_kind=fact_kind,
        payload=_payload_for_kind(fact_kind),
    ).model_dump(mode="json")
    source_ref = SandboxFactSourceRef(
        source_type=SandboxFactSourceType.SANDBOX_API,
        source_event_id=source_event_id,
        source_event_status="available" if source_event_id else "missing",
        tool_event_id=source_event_id,
        tool_call_id="call-1",
        function_name="exec_command",
    )
    subject_ref = SandboxFactSubjectRef(subject_type="command", subject_key="command:abc")
    payload_hash = build_sandbox_fact_payload_hash(payload)
    idempotency_key = build_sandbox_fact_idempotency_key(
        user_id=user_id,
        session_id=session_id,
        workspace_id="workspace-1",
        fact_scope=SandboxFactScope.STEP,
        run_id="run-1",
        step_id="step-1",
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
        workspace_id="workspace-1",
        fact_scope=SandboxFactScope.STEP,
        run_id="run-1",
        step_id="step-1",
        sandbox_id="sandbox-1",
        fact_kind=fact_kind,
        source_ref=source_ref,
        subject_ref=subject_ref,
        profile_ref=_context().profile_ref,
        summary=summary,
        payload=payload,
        payload_hash=payload_hash,
        idempotency_key=idempotency_key,
        origin=classification.origin,
        trust_level=classification.trust_level,
        privacy_level=classification.privacy_level,
        retention_policy=classification.retention_policy,
    )


def test_sandbox_fact_event_projector_should_write_one_event_with_scoped_refs() -> None:
    fact = _fact(fact_id="fact-1")
    sandbox_fact_repo = _SandboxFactRepo([fact])
    projector = SandboxFactEventProjector(
        uow_factory=lambda: _UoW(
            sandbox_fact=sandbox_fact_repo,
        )
    )

    event = asyncio.run(projector.project_tool_event_facts(context=_context(), facts=[fact]))

    assert isinstance(event, SandboxFactEvent)
    assert event.source_event_id == "tool-event-1"
    assert event.step_id == "step-1"
    assert event.fact_refs[0].fact_id == "fact-1"
    assert event.fact_refs[0].fact_kind == SandboxFactKind.COMMAND_EXECUTION
    assert sandbox_fact_repo.list_by_ids_calls == [
        {
            "user_id": "user-1",
            "session_id": "session-1",
            "fact_ids": ["fact-1"],
            "limit": 1,
        }
    ]


def test_sandbox_fact_event_should_reference_multiple_fact_kinds_from_one_tool_event() -> None:
    command_fact = _fact(
        fact_id="fact-command",
        fact_kind=SandboxFactKind.COMMAND_EXECUTION,
        summary="command fact",
    )
    output_fact = _fact(
        fact_id="fact-output",
        fact_kind=SandboxFactKind.SHELL_OUTPUT,
        summary="shell output fact",
    )
    sandbox_fact_repo = _SandboxFactRepo([output_fact, command_fact])
    projector = SandboxFactEventProjector(
        uow_factory=lambda: _UoW(
            sandbox_fact=sandbox_fact_repo,
        )
    )

    event = asyncio.run(
        projector.project_tool_event_facts(
            context=_context(),
            facts=[command_fact, output_fact],
        )
    )

    assert isinstance(event, SandboxFactEvent)
    assert event.source_event_id == "tool-event-1"
    assert [ref.fact_id for ref in event.fact_refs] == ["fact-command", "fact-output"]
    assert [ref.fact_kind for ref in event.fact_refs] == [
        SandboxFactKind.COMMAND_EXECUTION,
        SandboxFactKind.SHELL_OUTPUT,
    ]
    assert "command fact" in event.summary
    assert "shell output fact" in event.summary


def test_sandbox_fact_event_should_include_tool_failure_fact() -> None:
    failure_fact = _fact(
        fact_id="fact-failure",
        fact_kind=SandboxFactKind.TOOL_FAILURE,
        summary="tool failed",
    )
    sandbox_fact_repo = _SandboxFactRepo([failure_fact])
    projector = SandboxFactEventProjector(
        uow_factory=lambda: _UoW(
            sandbox_fact=sandbox_fact_repo,
        )
    )

    event = asyncio.run(projector.project_tool_event_facts(context=_context(), facts=[failure_fact]))

    assert isinstance(event, SandboxFactEvent)
    assert event.fact_refs[0].fact_kind == SandboxFactKind.TOOL_FAILURE
    assert event.summary == "tool failed"


def test_sandbox_fact_event_projector_should_fail_closed_for_cross_scope_ref() -> None:
    fact = _fact(fact_id="fact-1")
    cross_scope_fact = _fact(fact_id="fact-1", user_id="user-2")
    sandbox_fact_repo = _SandboxFactRepo([cross_scope_fact])
    projector = SandboxFactEventProjector(
        uow_factory=lambda: _UoW(
            sandbox_fact=sandbox_fact_repo,
        )
    )

    event = asyncio.run(projector.project_tool_event_facts(context=_context(), facts=[fact]))

    assert event is None


def test_sandbox_fact_event_should_reject_missing_source_event_id() -> None:
    fact = _fact(fact_id="fact-1")
    sandbox_fact_repo = _SandboxFactRepo([fact])
    context = _context().model_copy(update={"source_event_id": None})
    projector = SandboxFactEventProjector(
        uow_factory=lambda: _UoW(
            sandbox_fact=sandbox_fact_repo,
        )
    )

    event = asyncio.run(projector.project_tool_event_facts(context=context, facts=[fact]))

    assert event is None


def test_sandbox_fact_event_should_reject_fact_with_different_source_event_id() -> None:
    fact = _fact(
        fact_id="fact-1",
        source_event_id="other-tool-event",
    )
    sandbox_fact_repo = _SandboxFactRepo([fact])
    projector = SandboxFactEventProjector(
        uow_factory=lambda: _UoW(
            sandbox_fact=sandbox_fact_repo,
        )
    )

    event = asyncio.run(projector.project_tool_event_facts(context=_context(), facts=[fact]))

    assert event is None
