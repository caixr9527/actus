import asyncio
import json
import logging

import pytest
from pydantic import ValidationError

from app.application.service.evidence_digest_projector import EvidenceDigestProjector
from app.application.service.evidence_fact_assembler import EvidenceFactAssembler
from app.application.service.evidence_fact_assembler import (
    BrowserEvidenceStrategy,
    CommandEvidenceStrategy,
    DocumentEvidenceStrategy,
    FileEvidenceStrategy,
    HumanConfirmationEvidenceStrategy,
    PageEvidenceStrategy,
    SearchEvidenceStrategy,
    ToolFailureEvidenceStrategy,
)
from app.application.service.evidence_ledger_service import EvidenceLedgerService
from app.application.service.evidence_result_handle_resolver import EvidenceResultHandleResolver
from app.application.service.evidence_runtime_context_provider import EvidenceRuntimeContextProvider
from app.domain.models import ExecutionStatus, Plan, Session, Step, StepEvent, StepOutcome, ToolResult, WorkflowRun, Workspace
from app.domain.models.evidence import (
    EvidenceBackedFactProjection,
    EvidenceDuplicateDecision,
    EvidenceKind,
    EvidenceQualityStatus,
    EvidenceReadStrategy,
    EvidenceResolvedResult,
    EvidenceResolvedStatus,
    EvidenceResultRef,
    EvidenceResultRefType,
    EvidenceReusePolicy,
    EvidenceReuseSnapshot,
    EvidenceScope,
    EvidenceSourceRef,
    EvidenceSourceType,
    EvidenceStalenessPolicy,
    EvidenceSubjectRef,
    EvidenceSupportLevel,
    validate_evidence_payload,
    build_evidence_result_handle,
)
from app.domain.models.sandbox_fact import (
    SandboxFactKind,
    SandboxFactRecord,
    SandboxFactScope,
    SandboxFactSourceRef,
    SandboxFactSourceType,
    SandboxFactSubjectRef,
    build_sandbox_fact_idempotency_key,
    build_sandbox_fact_payload_hash,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.evidence_key_normalizer import (
    build_file_mutation_intent_hash,
    build_evidence_action_subject_key_from_fact,
    build_evidence_action_subject_key_from_tool_call,
    hash_query,
    hash_url,
)
from app.domain.services.tools import BaseTool
from app.domain.services.tools.base import tool
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintInput
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.engine import ConstraintEngine
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import (
    REASON_EVIDENCE_REUSE_SNAPSHOT_MISSING,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.evidence_reuse_policy import (
    evaluate_evidence_reuse_policy,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.graph import build_planner_react_langgraph_graph
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_context import ExecutionContext
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import ExecutionState
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.tools import execute_step_with_prompt
from app.infrastructure.runtime.langgraph.graphs.planner_react.policy_engine.engine import ToolPolicyEngine
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.execute_nodes import (
    _resolve_pending_evidence_reuse,
    execute_step_node,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.execute_helpers import prepare_execute_step_input
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.prompt_context_helpers import (
    _append_prompt_context_to_prompt,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.replan import ReplanMergeEngine
from app.domain.services.runtime.contracts.langgraph_settings import (
    REPLAN_META_VALIDATION_ALLOW_PATTERN,
    REPLAN_META_VALIDATION_DENY_PATTERN,
    REPLAN_META_VALIDATION_STEP_PATTERN,
)
from app.domain.services.agent_task_runner import AgentTaskRunner


class _EvidenceRepo:
    def __init__(self, records=None) -> None:
        self.records = list(records or [])
        self.saved = []

    async def save_once(self, evidence):
        self.saved.append(evidence)
        self.records.append(evidence)
        return evidence

    async def list_by_run(self, **kwargs):
        return list(self.records)

    async def list_by_step(self, *, step_id, **kwargs):
        return [record for record in self.records if record.step_id == step_id]


class _FactRepo:
    def __init__(self, facts=None) -> None:
        self.facts = list(facts or [])
        self.raise_on_list_by_scope = False

    async def list_by_scope(self, **kwargs):
        if self.raise_on_list_by_scope:
            raise RuntimeError("fact repo unavailable")
        return [
            fact for fact in self.facts
            if fact.run_id == kwargs.get("run_id") and fact.step_id == kwargs.get("step_id")
        ]

    async def list_by_ids(self, *, fact_ids, **kwargs):
        return [fact for fact in self.facts if fact.id in set(fact_ids)]


class _WorkflowRunRepo:
    async def get_event_record_by_event_id(self, **kwargs):
        return object()


class _ArtifactRepo:
    async def get_by_user_workspace_id_and_id(self, **kwargs):
        return None


class _UoW:
    def __init__(self, *, evidence=None, facts=None) -> None:
        self.evidence = evidence or _EvidenceRepo()
        self.sandbox_fact = _FactRepo(facts or [])
        self.workflow_run = _WorkflowRunRepo()
        self.workspace_artifact = _ArtifactRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FailingFactUoW(_UoW):
    def __init__(self, *, evidence) -> None:
        super().__init__(evidence=evidence)
        self.sandbox_fact.raise_on_list_by_scope = True


def _scope(current_step_id: str = "step-2") -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        current_step_id=current_step_id,
    )


def _ledger_service(*, uow_factory) -> EvidenceLedgerService:
    return EvidenceLedgerService(uow_factory=uow_factory, assembler=EvidenceFactAssembler())


def _snapshot_from_fact(kind: SandboxFactKind, *, payload=None) -> EvidenceReuseSnapshot:
    fact = _fact(kind, payload=payload)
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=[fact], evidence=evidence_repo)
    service = _ledger_service(uow_factory=uow_factory)
    asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    context = asyncio.run(
        EvidenceDigestProjector(uow_factory=uow_factory).build_context(
            stage="execute",
            scope=_scope("step-2"),
            completed_step_ids=["step-1"],
            step=Step(id="step-2"),
            task_mode="general",
        )
    )
    assert context.evidence_reuse_snapshot is not None
    return context.evidence_reuse_snapshot


def _fact(kind: SandboxFactKind, *, fact_id: str = "fact-1", step_id: str = "step-1", payload=None) -> SandboxFactRecord:
    payload = payload or _payload_for_fact(kind)
    payload_hash = build_sandbox_fact_payload_hash(payload)
    source_ref = SandboxFactSourceRef(
        source_type=SandboxFactSourceType.TOOL_EVENT,
        source_event_id="event-1",
        source_event_status="available",
        tool_call_id="tool-call-1",
        function_name=_function_for_fact(kind),
    )
    subject_ref = SandboxFactSubjectRef(subject_type=_subject_type_for_fact(kind), subject_key="subject:1")
    idempotency_key = build_sandbox_fact_idempotency_key(
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        fact_scope=SandboxFactScope.STEP,
        run_id="run-1",
        step_id=step_id,
        fact_kind=kind,
        source_event_id=source_ref.source_event_id,
        tool_call_id=source_ref.tool_call_id,
        subject_key=subject_ref.subject_key,
        payload_hash=payload_hash,
    )
    return SandboxFactRecord(
        id=fact_id,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        fact_scope=SandboxFactScope.STEP,
        run_id="run-1",
        step_id=step_id,
        fact_kind=kind,
        source_ref=source_ref,
        subject_ref=subject_ref,
        summary="safe summary",
        payload=payload,
        payload_hash=payload_hash,
        idempotency_key=idempotency_key,
    )


@pytest.mark.parametrize(
    ("kind", "expected_kind"),
    [
        (SandboxFactKind.COMMAND_EXECUTION, EvidenceKind.ACTION_EVIDENCE),
        (SandboxFactKind.SHELL_OUTPUT, EvidenceKind.ACTION_EVIDENCE),
        (SandboxFactKind.DOCUMENT_CONTEXT, EvidenceKind.DOCUMENT_EVIDENCE),
        (SandboxFactKind.SEARCH_RESULT, EvidenceKind.SEARCH_EVIDENCE),
        (SandboxFactKind.FETCHED_PAGE, EvidenceKind.PAGE_EVIDENCE),
        (SandboxFactKind.FILE_READ, EvidenceKind.FILE_EVIDENCE),
        (SandboxFactKind.BROWSER_SNAPSHOT, EvidenceKind.BROWSER_EVIDENCE),
        (SandboxFactKind.TOOL_FAILURE, EvidenceKind.TOOL_FAILURE_EVIDENCE),
        (SandboxFactKind.HUMAN_INTERACTION, EvidenceKind.HUMAN_CONFIRMATION_EVIDENCE),
    ],
)
def test_assembler_should_cover_pr3_minimum_fact_mappings(kind, expected_kind) -> None:
    result = EvidenceFactAssembler().assemble_step(step=Step(id="step-1"), facts=[_fact(kind)])

    all_inputs = [*result.evidence_inputs, *result.gap_inputs]
    assert any(item.evidence_kind == expected_kind for item in all_inputs)


def test_assembler_should_register_documented_evidence_strategies() -> None:
    assembler = EvidenceFactAssembler()

    assert assembler._strategies_by_kind[SandboxFactKind.COMMAND_EXECUTION].__class__ is CommandEvidenceStrategy
    assert assembler._strategies_by_kind[SandboxFactKind.SHELL_OUTPUT].__class__ is CommandEvidenceStrategy
    assert assembler._strategies_by_kind[SandboxFactKind.FILE_READ].__class__ is FileEvidenceStrategy
    assert assembler._strategies_by_kind[SandboxFactKind.SEARCH_RESULT].__class__ is SearchEvidenceStrategy
    assert assembler._strategies_by_kind[SandboxFactKind.FETCHED_PAGE].__class__ is PageEvidenceStrategy
    assert assembler._strategies_by_kind[SandboxFactKind.BROWSER_SNAPSHOT].__class__ is BrowserEvidenceStrategy
    assert assembler._strategies_by_kind[SandboxFactKind.DOCUMENT_CONTEXT].__class__ is DocumentEvidenceStrategy
    assert assembler._strategies_by_kind[SandboxFactKind.TOOL_FAILURE].__class__ is ToolFailureEvidenceStrategy
    assert assembler._strategies_by_kind[SandboxFactKind.HUMAN_INTERACTION].__class__ is HumanConfirmationEvidenceStrategy


def test_document_evidence_should_treat_parsed_as_strong_reusable() -> None:
    result = EvidenceFactAssembler().assemble_step(
        step=Step(id="step-1"),
        facts=[_fact(SandboxFactKind.DOCUMENT_CONTEXT)],
    )

    evidence_input = result.evidence_inputs[0]
    assert evidence_input.quality_status == EvidenceQualityStatus.VALID
    assert evidence_input.support_level == EvidenceSupportLevel.STRONG
    assert evidence_input.reuse_policy == EvidenceReusePolicy.REUSE_ALLOWED
    assert evidence_input.result_refs[0].read_strategy == EvidenceReadStrategy.READ_DOCUMENT_SOURCE


def test_document_evidence_should_not_accept_success_parse_status_as_success() -> None:
    fact = _fact(
        SandboxFactKind.DOCUMENT_CONTEXT,
        payload={**_payload_for_fact(SandboxFactKind.DOCUMENT_CONTEXT), "parse_status": "success"},
    )

    result = EvidenceFactAssembler().assemble_step(step=Step(id="step-1"), facts=[fact])
    evidence_input = result.evidence_inputs[0]

    assert evidence_input.quality_status == EvidenceQualityStatus.PARTIAL
    assert evidence_input.support_level == EvidenceSupportLevel.PARTIAL
    assert evidence_input.reuse_policy != EvidenceReusePolicy.REUSE_ALLOWED
    assert evidence_input.result_refs[0].read_strategy == EvidenceReadStrategy.VERIFY_BEFORE_USE
    with pytest.raises(ValidationError):
        validate_evidence_payload(
            evidence_kind=EvidenceKind.DOCUMENT_EVIDENCE,
            payload=evidence_input.payload,
        )


def test_digest_should_build_handles_only_from_persisted_result_refs() -> None:
    fact = _fact(SandboxFactKind.FILE_READ)
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(facts=[fact], evidence=evidence_repo))
    saved = asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    projector = EvidenceDigestProjector(uow_factory=lambda: _UoW(evidence=evidence_repo))

    digest = asyncio.run(projector.build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
    ))

    assert saved
    assert digest is not None
    assert digest.result_handles
    assert digest.do_not_repeat
    assert digest.do_not_repeat[0].result_handle_id == digest.result_handles[0].result_handle_id
    assert digest.evidence_backed_facts
    assert digest.evidence_backed_facts[0].text == "safe summary"
    assert digest.evidence_backed_facts[0].evidence_ids
    assert digest.evidence_backed_facts[0].fact_ids == ["fact-1"]


def test_digest_should_not_truncate_long_evidence_backed_projection_text() -> None:
    long_summary = "前置条件：" + "需要保留完整限定条件。" * 30
    fact = _fact(SandboxFactKind.FILE_READ)
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=[fact], evidence=evidence_repo)
    projector = EvidenceDigestProjector(uow_factory=uow_factory)
    service = EvidenceLedgerService(
        uow_factory=uow_factory,
        assembler=EvidenceFactAssembler(),
        step_projection=projector,
    )
    saved = asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    evidence_repo.records[0].summary = long_summary

    digest = asyncio.run(projector.build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
    ))
    projections = asyncio.run(service.build_step_evidence_backed_facts(
        scope=_scope("step-1"),
        step=Step(id="step-1"),
    ))

    assert saved
    assert digest is not None
    projection = digest.evidence_backed_facts[0]
    assert projection.text == "该 step 已形成可审计 evidence，摘要过长未注入 prompt；请通过 evidence refs/result handle 读取。"
    assert len(projection.text) <= 300
    assert projection.text != long_summary[:300]
    assert projection.evidence_ids
    assert projection.fact_ids == ["fact-1"]
    assert projection.source_event_ids == ["event-1"]
    assert projections[0].text == projection.text


def test_step_outcome_should_keep_evidence_backed_facts_and_derive_facts_text() -> None:
    projection = EvidenceBackedFactProjection(
        text="已读取 /workspace/a.txt",
        evidence_ids=["evidence-1"],
        fact_ids=["fact-1"],
        artifact_ids=[],
        source_event_ids=["event-1"],
        user_confirmation_event_ids=[],
    )
    step = Step(
        id="step-1",
        outcome=StepOutcome(
            done=True,
            summary="完成",
            evidence_backed_facts=[projection],
            facts_learned=[projection.text],
        ),
    )

    assert step.outcome is not None
    assert step.outcome.facts_learned == ["已读取 /workspace/a.txt"]
    assert step.model_dump(mode="json")["outcome"]["evidence_backed_facts"][0]["fact_ids"] == ["fact-1"]


def test_digest_prompt_summary_should_truncate_by_complete_lines_without_trimming_structured_context() -> None:
    action_count = 20
    gap_count = 10
    action_step_ids = [f"action-step-{index}" for index in range(action_count)]
    gap_step_ids = [f"gap-step-{index}" for index in range(gap_count)]
    facts = [
        _fact(
            SandboxFactKind.FILE_READ,
            fact_id=f"fact-{index}",
            step_id=step_id,
            payload={
                **_payload_for_fact(SandboxFactKind.FILE_READ),
                "path": f"/workspace/{'very-long-segment-' * 5}{index}.txt",
            },
        )
        for index, step_id in enumerate(action_step_ids)
    ]
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=facts, evidence=evidence_repo)
    service = _ledger_service(uow_factory=uow_factory)
    for step_id in action_step_ids:
        asyncio.run(service.reconcile_step_evidence(scope=_scope(step_id), step=Step(id=step_id)))
    asyncio.run(service.reconcile_previous_steps_evidence(
        scope=_scope("step-final"),
        completed_step_ids=gap_step_ids,
    ))
    projector = EvidenceDigestProjector(uow_factory=uow_factory)

    digest = asyncio.run(projector.build_digest(
        scope=_scope("step-final"),
        current_step_id="step-final",
        completed_step_ids=[*action_step_ids, *gap_step_ids],
    ))

    assert digest is not None
    assert len(digest.completed_actions) == action_count
    assert len(digest.do_not_repeat) == action_count
    assert len(digest.result_handles) == action_count
    assert len(digest.evidence_gaps) == gap_count
    assert len(digest.summary_for_prompt) <= 1200
    assert "... omitted_completed_actions=" in digest.summary_for_prompt
    assert "... omitted_do_not_repeat=" in digest.summary_for_prompt
    assert "... omitted_evidence_gaps=" in digest.summary_for_prompt
    assert "... full_structured_context_available=true" in digest.summary_for_prompt
    for line in digest.summary_for_prompt.splitlines():
        assert line.startswith((
            "completed_actions:",
            "do_not_repeat:",
            "evidence_gaps:",
            "- ",
            "... ",
        ))


def test_runtime_context_service_should_consume_domain_port() -> None:
    provider = _Provider()
    service = RuntimeContextService(evidence_context_provider=provider)
    step1 = Step(id="step-1", status=ExecutionStatus.COMPLETED)
    step2 = Step(id="step-2")
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "plan": Plan(steps=[step1, step2]),
        "step_states": [{"step_id": "step-1", "status": "completed"}],
    }

    packet = asyncio.run(service.build_packet_async(stage="execute", state=state, step=step2, task_mode="general"))

    assert "evidence_context" in packet
    assert packet["evidence_context"]["evidence_reuse_snapshot"]["do_not_repeat"][0]["result_handle_id"]
    assert packet["evidence_prompt_digest"] == "digest"
    assert "evidence_context" not in packet["prompt_visible_fields"]
    assert provider.completed_step_ids == ["step-1"]


def test_prompt_should_only_include_evidence_digest_not_structured_context() -> None:
    packet = {
        "prompt_visible_fields": ["current_step", "evidence_context", "evidence_prompt_digest", "audit_refs"],
        "current_step": {"id": "step-2"},
        "evidence_prompt_digest": "safe digest",
        "evidence_context": _runtime_context().model_dump(mode="json"),
        "audit_refs": {"run_id": "run-1", "thread_id": "thread-1", "current_step_id": "step-2"},
    }

    prompt = _append_prompt_context_to_prompt("base", packet)

    assert "safe digest" in prompt
    assert "prompt_visible_fields" not in prompt
    assert "audit_refs" not in prompt
    assert "thread-1" not in prompt
    assert "evidence_reuse_snapshot" not in prompt
    assert "result_handle_index" not in prompt
    assert "result_handle_id" not in prompt


def test_prepare_execute_input_should_keep_structured_snapshot_when_prompt_digest_missing() -> None:
    runtime_context = _runtime_context()
    provider = _Provider(runtime_context=runtime_context)
    service = RuntimeContextService(evidence_context_provider=provider)
    step1 = Step(id="step-1", status=ExecutionStatus.COMPLETED)
    step2 = Step(id="step-2")
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "plan": Plan(steps=[step1, step2]),
        "step_states": [{"step_id": "step-1", "status": "completed"}],
        "graph_metadata": {},
    }
    runtime_context.prompt_digest = ""

    prepared = asyncio.run(prepare_execute_step_input(
        state=state,
        step=step2,
        llm=object(),
        runtime_context_service=service,
        task_mode="general",
        user_message="read file",
    ))

    assert prepared.runtime_evidence_context is not None
    assert prepared.runtime_evidence_context.evidence_reuse_snapshot is not None
    assert prepared.result_handle_index


def test_execute_should_fail_closed_when_completed_step_exists_but_evidence_context_missing() -> None:
    prepared = asyncio.run(prepare_execute_step_input(
        state={
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "plan": Plan(steps=[
                Step(id="step-1", status=ExecutionStatus.COMPLETED),
                Step(id="step-2"),
            ]),
            "step_states": [{"step_id": "step-1", "status": "completed"}],
            "graph_metadata": {},
        },
        step=Step(id="step-2"),
        llm=object(),
        runtime_context_service=RuntimeContextService(evidence_context_provider=None),
        task_mode="general",
        user_message="读取文件",
    ))
    tool = _CountingRuntimeTool({"read_file"})

    payload, tool_events = asyncio.run(execute_step_with_prompt(
        llm=_FakeToolCallLLM("read_file", {"path": "/workspace/a.txt"}),
        step=Step(id="step-2", description="读取文件"),
        runtime_tools=[tool],
        max_tool_iterations=1,
        task_mode="general",
        user_content=prepared.user_content,
        user_message=prepared.user_message,
        runtime_evidence_context=prepared.runtime_evidence_context,
        has_previous_completed_steps=prepared.has_previous_completed_steps,
    ))

    assert payload["success"] is False
    assert payload["loop_break_reason"] == REASON_EVIDENCE_REUSE_SNAPSHOT_MISSING
    assert tool.invocations == []
    assert any(event.status.value == "called" for event in tool_events)


def test_evidence_reuse_policy_should_return_pending_resolution_from_snapshot_handle() -> None:
    snapshot = _runtime_context().evidence_reuse_snapshot
    decision = evaluate_evidence_reuse_policy(
        ConstraintInput(
            step=Step(id="step-2"),
            task_mode="general",
            function_name="read_file",
            normalized_function_name="read_file",
            function_args={"path": "/workspace/a.txt"},
            matched_tool=None,
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            external_signals_snapshot={
                "evidence_reuse_snapshot": snapshot,
                "has_previous_completed_steps": True,
            },
            runtime_tools=[],
        )
    )

    assert decision is not None
    assert decision.reason_code == "evidence_reuse_pending_resolution"
    assert decision.tool_result_payload.success is False
    assert decision.tool_result_payload.data["result_handle_id"] == snapshot.result_handles[0].result_handle_id
    assert decision.tool_result_payload.data["result_handle"]["result_handle_id"] == snapshot.result_handles[0].result_handle_id


def test_evidence_reuse_policy_should_fail_for_non_strict_snapshot_dict() -> None:
    with pytest.raises(Exception):
        evaluate_evidence_reuse_policy(
            ConstraintInput(
                step=Step(id="step-2"),
                task_mode="general",
                function_name="read_file",
                normalized_function_name="read_file",
                function_args={"path": "/workspace/a.txt"},
                matched_tool=None,
                iteration_blocked_function_names=set(),
                execution_context=_execution_context(),
                execution_state=ExecutionState(),
                external_signals_snapshot={
                    "evidence_reuse_snapshot": {"run_id": "run-1"},
                    "has_previous_completed_steps": True,
                },
                runtime_tools=[],
            )
        )


@pytest.mark.parametrize(
    ("kind", "tool_name", "tool_args"),
    [
        (SandboxFactKind.SEARCH_RESULT, "search_web", {"query_hash": "q1"}),
        (SandboxFactKind.FETCHED_PAGE, "fetch_page", {"url_hash": "u1"}),
        (SandboxFactKind.FILE_READ, "read_file", {"path": "/workspace/a.txt"}),
        (SandboxFactKind.FILE_WRITE, "write_file", {"path": "/workspace/a.txt", "content": "new content"}),
        (SandboxFactKind.FILE_LIST, "find_files", {"dir_path": "/workspace/a.txt"}),
        (SandboxFactKind.FILE_SEARCH, "search_in_file", {"path": "/workspace/a.txt", "regex_hash": "sha256:regex"}),
        (SandboxFactKind.BROWSER_SNAPSHOT, "browser_view", {"url_hash": "u1"}),
        (SandboxFactKind.BROWSER_ACTION, "browser_click", {"target_summary": "Login", "url_hash_after": "u2"}),
        (SandboxFactKind.DOCUMENT_CONTEXT, "read_document", {"file_id": "file-1", "read_content_sha256": "sha256:read"}),
        (SandboxFactKind.HUMAN_INTERACTION, "message_ask_user", {"source_event_id": "event-1"}),
        (SandboxFactKind.COMMAND_EXECUTION, "shell_execute", {"command_fingerprint": "sha256:command"}),
        (
            SandboxFactKind.SHELL_OUTPUT,
            "read_shell_output",
            {
                "session_ref": "shell-1",
                "process_status": "completed",
                "exit_code": 0,
                "output_excerpt": "done",
            },
        ),
    ],
)
def test_tool_call_key_should_match_fact_key_for_real_runtime_tools(kind, tool_name, tool_args) -> None:
    fact = _fact(kind)

    fact_key = build_evidence_action_subject_key_from_fact(fact)
    tool_key = build_evidence_action_subject_key_from_tool_call(tool_name, tool_args)

    assert tool_key.normalization_status == "normalized"
    assert tool_key.action_key == fact_key.action_key
    assert tool_key.subject_key == fact_key.subject_key


@pytest.mark.parametrize(
    ("tool_name", "tool_args"),
    [
        ("write_file", {"path": "/workspace/a.txt", "content": "new content"}),
        ("replace_in_file", {"path": "/workspace/a.txt", "old_str": "a", "new_str": "b"}),
    ],
)
def test_write_tool_call_key_should_skip_without_pre_execute_hash(tool_name, tool_args) -> None:
    result = build_evidence_action_subject_key_from_tool_call(tool_name, tool_args)

    assert result.normalization_status == "normalized"
    assert result.action_key.startswith("file_write:/workspace/a.txt:sha256:")


def test_evidence_reuse_policy_should_allow_marked_verification_search() -> None:
    query = "query"
    query_hash = hash_query(query)
    snapshot = _snapshot_from_fact(
        SandboxFactKind.SEARCH_RESULT,
        payload={
            **_payload_for_fact(SandboxFactKind.SEARCH_RESULT),
            "query_hash": query_hash,
            "query_excerpt": query,
        },
    )

    allowed = evaluate_evidence_reuse_policy(
        ConstraintInput(
            step=Step(id="step-2"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={
                "query": query,
                "query_hash": query_hash,
                "verification_reason_code": "external_evidence_may_change",
            },
            matched_tool=None,
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            external_signals_snapshot={
                "evidence_reuse_snapshot": snapshot,
                "has_previous_completed_steps": True,
            },
            runtime_tools=[],
        )
    )
    assert allowed is None


def test_evidence_reuse_policy_should_rewrite_bare_verification_search() -> None:
    query = "query"
    query_hash = hash_query(query)
    snapshot = _snapshot_from_fact(
        SandboxFactKind.SEARCH_RESULT,
        payload={
            **_payload_for_fact(SandboxFactKind.SEARCH_RESULT),
            "query_hash": query_hash,
            "query_excerpt": query,
        },
    )

    decision = evaluate_evidence_reuse_policy(
        ConstraintInput(
            step=Step(id="step-2"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": query},
            matched_tool=None,
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            external_signals_snapshot={
                "evidence_reuse_snapshot": snapshot,
                "has_previous_completed_steps": True,
            },
            runtime_tools=[],
        )
    )

    assert decision is not None
    assert decision.action == "rewrite"
    assert decision.rewrite_target["function_args"]["query"] == query
    assert decision.rewrite_target["function_args"]["query_hash"] == query_hash
    assert decision.rewrite_target["function_args"]["verification_reason_code"] == "external_evidence_may_change"


@pytest.mark.parametrize(
    "function_args",
    [
        {"query_hash": "q1", "verification_reason_code": "external_evidence_may_change"},
        {"query": "query", "query_hash": "q1"},
        {"query": "query", "query_hash": "q1", "verification_reason_code": "external_evidence_may_change"},
        {"query": "same meaning", "query_hash": "q1", "verification_reason_code": "external_evidence_may_change"},
    ],
)
def test_evidence_reuse_policy_should_block_invalid_verification_search(function_args) -> None:
    snapshot = _snapshot_from_fact(SandboxFactKind.SEARCH_RESULT)

    decision = evaluate_evidence_reuse_policy(
        ConstraintInput(
            step=Step(id="step-2"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args=function_args,
            matched_tool=None,
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            external_signals_snapshot={
                "evidence_reuse_snapshot": snapshot,
                "has_previous_completed_steps": True,
            },
            runtime_tools=[],
        )
    )

    assert decision is not None
    assert decision.reason_code == "evidence_reuse_requires_verification"
    assert decision.tool_result_payload.data["verification_gap"]["reason_code"] == "verification_action_missing"


def test_evidence_reuse_policy_should_allow_marked_verification_fetch_page() -> None:
    url = "https://example.com/a"
    url_hash = hash_url(url)
    snapshot = _snapshot_from_fact(
        SandboxFactKind.FETCHED_PAGE,
        payload={
            **_payload_for_fact(SandboxFactKind.FETCHED_PAGE),
            "fetched_url_hash": url_hash,
            "final_url_origin": url,
        },
    )

    allowed = evaluate_evidence_reuse_policy(
        ConstraintInput(
            step=Step(id="step-2"),
            task_mode="research",
            function_name="fetch_page",
            normalized_function_name="fetch_page",
            function_args={
                "url": url,
                "url_hash": url_hash,
                "verification_reason_code": "external_evidence_may_change",
            },
            matched_tool=None,
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            external_signals_snapshot={
                "evidence_reuse_snapshot": snapshot,
                "has_previous_completed_steps": True,
            },
            runtime_tools=[],
        )
    )

    assert allowed is None


def test_evidence_reuse_policy_should_rewrite_bare_verification_fetch_page() -> None:
    url = "https://example.com/a"
    url_hash = hash_url(url)
    snapshot = _snapshot_from_fact(
        SandboxFactKind.FETCHED_PAGE,
        payload={
            **_payload_for_fact(SandboxFactKind.FETCHED_PAGE),
            "fetched_url_hash": url_hash,
            "final_url_origin": url,
        },
    )

    decision = evaluate_evidence_reuse_policy(
        ConstraintInput(
            step=Step(id="step-2"),
            task_mode="research",
            function_name="fetch_page",
            normalized_function_name="fetch_page",
            function_args={"url": url},
            matched_tool=None,
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            external_signals_snapshot={
                "evidence_reuse_snapshot": snapshot,
                "has_previous_completed_steps": True,
            },
            runtime_tools=[],
        )
    )

    assert decision is not None
    assert decision.action == "rewrite"
    assert decision.rewrite_target["function_args"]["url"] == url
    assert decision.rewrite_target["function_args"]["url_hash"] == url_hash
    assert decision.rewrite_target["function_args"]["verification_reason_code"] == "external_evidence_may_change"


@pytest.mark.parametrize(
    "function_args",
    [
        {"url_hash": "u1", "verification_reason_code": "external_evidence_may_change"},
        {"url": "https://example.com/a", "url_hash": "u1"},
        {"url": "https://example.com/a", "url_hash": "u1", "verification_reason_code": "external_evidence_may_change"},
        {"url": "https://example.com/other", "url_hash": "u1", "verification_reason_code": "external_evidence_may_change"},
    ],
)
def test_evidence_reuse_policy_should_block_invalid_verification_fetch_page(function_args) -> None:
    snapshot = _snapshot_from_fact(SandboxFactKind.FETCHED_PAGE)

    decision = evaluate_evidence_reuse_policy(
        ConstraintInput(
            step=Step(id="step-2"),
            task_mode="research",
            function_name="fetch_page",
            normalized_function_name="fetch_page",
            function_args=function_args,
            matched_tool=None,
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            external_signals_snapshot={
                "evidence_reuse_snapshot": snapshot,
                "has_previous_completed_steps": True,
            },
            runtime_tools=[],
        )
    )

    assert decision is not None
    assert decision.reason_code == "evidence_reuse_requires_verification"
    assert decision.tool_result_payload.data["verification_gap"]["reason_code"] == "verification_action_missing"


@pytest.mark.parametrize(
    ("kind", "function_name", "function_args", "expected_executable_args", "audit_metadata_keys"),
    [
        (
            SandboxFactKind.SEARCH_RESULT,
            "search_web",
            {
                "query": "query",
                "query_hash": hash_query("query"),
                "verification_reason_code": "external_evidence_may_change",
            },
            {"query": "query"},
            {"query_hash", "verification_reason_code"},
        ),
        (
            SandboxFactKind.FETCHED_PAGE,
            "fetch_page",
            {
                "url": "https://example.com/a",
                "url_hash": hash_url("https://example.com/a"),
                "verification_reason_code": "external_evidence_may_change",
            },
            {"url": "https://example.com/a"},
            {"url_hash", "verification_reason_code"},
        ),
    ],
)
def test_tool_policy_engine_should_execute_verification_with_executable_args(
        kind,
        function_name,
        function_args,
        expected_executable_args,
        audit_metadata_keys,
) -> None:
    """verification metadata 只供 policy 审计，executor 只能收到真实工具参数。"""
    payload = _payload_for_fact(kind)
    if kind == SandboxFactKind.SEARCH_RESULT:
        payload = {**payload, "query_hash": function_args["query_hash"], "query_excerpt": function_args["query"]}
    else:
        payload = {**payload, "fetched_url_hash": function_args["url_hash"], "final_url_origin": function_args["url"]}
    snapshot = _snapshot_from_fact(kind, payload=payload)
    tool = _SignatureFilteringTool({function_name})

    result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="验证外部 evidence"),
            task_mode="research",
            function_name=function_name,
            normalized_function_name=function_name,
            function_args=function_args,
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=snapshot,
            has_previous_completed_steps=True,
        )
    )

    assert result.tool_result.success is True
    assert tool.invocations == [(function_name, expected_executable_args)]
    assert set(function_args) >= set(audit_metadata_keys)
    assert not (set(tool.invocations[0][1]) & set(audit_metadata_keys))


def test_tool_policy_engine_should_autofill_verification_metadata_for_bare_search() -> None:
    query = "query"
    query_hash = hash_query(query)
    snapshot = _snapshot_from_fact(
        SandboxFactKind.SEARCH_RESULT,
        payload={
            **_payload_for_fact(SandboxFactKind.SEARCH_RESULT),
            "query_hash": query_hash,
            "query_excerpt": query,
        },
    )
    tool = _SignatureFilteringTool({"search_web"})

    result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="验证外部 evidence"),
            task_mode="research",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": query},
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=snapshot,
            has_previous_completed_steps=True,
        )
    )

    assert result.tool_result.success is True
    assert result.rewrite_reason == "evidence_reuse_requires_verification"
    assert tool.invocations == [("search_web", {"query": query})]


def test_tool_policy_engine_should_allow_system_rewritten_verification_search_in_general_mode() -> None:
    query = "query"
    query_hash = hash_query(query)
    snapshot = _snapshot_from_fact(
        SandboxFactKind.SEARCH_RESULT,
        payload={
            **_payload_for_fact(SandboxFactKind.SEARCH_RESULT),
            "query_hash": query_hash,
            "query_excerpt": query,
        },
    )
    tool = _SignatureFilteringTool({"search_web"})

    result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="验证外部 evidence"),
            task_mode="general",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": query},
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names={"search_web"},
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=snapshot,
            has_previous_completed_steps=True,
        )
    )

    assert result.tool_result.success is True
    assert result.rewrite_reason == "evidence_reuse_requires_verification"
    assert tool.invocations == [("search_web", {"query": query})]


def test_tool_policy_engine_should_not_allow_forged_verification_search_in_general_mode() -> None:
    query = "query"
    snapshot = _snapshot_from_fact(SandboxFactKind.SEARCH_RESULT)
    tool = _SignatureFilteringTool({"search_web"})

    result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="验证外部 evidence"),
            task_mode="general",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={
                "query": query,
                "query_hash": "wrong-hash",
                "verification_reason_code": "external_evidence_may_change",
            },
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names={"search_web"},
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=snapshot,
            has_previous_completed_steps=True,
        )
    )

    assert result.tool_result.success is False
    assert result.tool_result.data["verification_gap"]["reason_code"] == "verification_action_missing"
    assert tool.invocations == []


def test_tool_policy_engine_should_autofill_verification_metadata_for_bare_fetch_page() -> None:
    url = "https://example.com/a"
    url_hash = hash_url(url)
    snapshot = _snapshot_from_fact(
        SandboxFactKind.FETCHED_PAGE,
        payload={
            **_payload_for_fact(SandboxFactKind.FETCHED_PAGE),
            "fetched_url_hash": url_hash,
            "final_url_origin": url,
        },
    )
    tool = _SignatureFilteringTool({"fetch_page"})

    result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="验证外部 evidence"),
            task_mode="research",
            function_name="fetch_page",
            normalized_function_name="fetch_page",
            function_args={"url": url},
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=snapshot,
            has_previous_completed_steps=True,
        )
    )

    assert result.tool_result.success is True
    assert result.rewrite_reason == "evidence_reuse_requires_verification"
    assert tool.invocations == [("fetch_page", {"url": url})]


def test_tool_policy_engine_should_allow_system_rewritten_verification_fetch_page_in_general_mode() -> None:
    url = "https://example.com/a"
    url_hash = hash_url(url)
    snapshot = _snapshot_from_fact(
        SandboxFactKind.FETCHED_PAGE,
        payload={
            **_payload_for_fact(SandboxFactKind.FETCHED_PAGE),
            "fetched_url_hash": url_hash,
            "final_url_origin": url,
        },
    )
    tool = _SignatureFilteringTool({"fetch_page"})

    result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="验证外部 evidence"),
            task_mode="general",
            function_name="fetch_page",
            normalized_function_name="fetch_page",
            function_args={"url": url},
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names={"fetch_page"},
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=snapshot,
            has_previous_completed_steps=True,
        )
    )

    assert result.tool_result.success is True
    assert result.rewrite_reason == "evidence_reuse_requires_verification"
    assert tool.invocations == [("fetch_page", {"url": url})]


def test_tool_policy_engine_should_not_allow_forged_verification_fetch_page_in_general_mode() -> None:
    snapshot = _snapshot_from_fact(SandboxFactKind.FETCHED_PAGE)
    tool = _SignatureFilteringTool({"fetch_page"})

    result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="验证外部 evidence"),
            task_mode="general",
            function_name="fetch_page",
            normalized_function_name="fetch_page",
            function_args={
                "url": "https://example.com/a",
                "url_hash": "wrong-hash",
                "verification_reason_code": "external_evidence_may_change",
            },
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names={"fetch_page"},
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=snapshot,
            has_previous_completed_steps=True,
        )
    )

    assert result.tool_result.success is False
    assert result.tool_result.data["verification_gap"]["reason_code"] == "verification_action_missing"
    assert tool.invocations == []


def test_constraint_engine_should_apply_evidence_reuse_before_task_mode_block_for_repeated_search() -> None:
    query = "query"
    query_hash = hash_query(query)
    key_result = build_evidence_action_subject_key_from_tool_call("search_web", {"query": query})
    result_ref = EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.FACT_REF,
        ref_id="fact-1",
        source_step_id="step-1",
        source_evidence_id="evidence-1",
        source_fact_id="fact-1",
        source_event_id="event-1",
        subject_key=f"query:{query_hash}",
        payload_hash="sha256:payload",
        quality_status=EvidenceQualityStatus.VALID,
        support_level=EvidenceSupportLevel.STRONG,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
        summary="safe summary",
    )
    handle = build_evidence_result_handle(result_ref)
    from app.domain.models.evidence import EvidenceDoNotRepeatResult

    snapshot = EvidenceReuseSnapshot(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        cursor="cursor-1",
        do_not_repeat=[
            EvidenceDoNotRepeatResult(
                action_key=key_result.action_key,
                subject_key=key_result.subject_key,
                reason_code="evidence_reuse_pending_resolution",
                source_step_id="step-1",
                evidence_ids=["evidence-1"],
                reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
                staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
                support_level=EvidenceSupportLevel.STRONG,
                quality_status=EvidenceQualityStatus.VALID,
                result_status="successful",
                duplicate_decision=EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION,
                reuse_result_ref=result_ref,
                result_handle_id=handle.result_handle_id,
                reuse_summary="safe summary",
            )
        ],
        result_handles=[handle],
    )
    tool = _SignatureFilteringTool({"search_web"})
    result = ConstraintEngine(logger=logging.getLogger(__name__)).evaluate_guard(
        constraint_input=ConstraintInput(
            step=Step(id="step-2"),
            task_mode="general",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": query},
            matched_tool=tool,
            iteration_blocked_function_names={"search_web"},
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            external_signals_snapshot={
                "evidence_reuse_snapshot": snapshot,
                "has_previous_completed_steps": True,
            },
            runtime_tools=[tool],
        )
    )

    assert result.winning_policy == "evidence_reuse_policy"
    assert result.constraint_decision.reason_code == "evidence_reuse_pending_resolution"
    assert result.constraint_decision.loop_break_reason == "virtual_success_pending_resolution"
    assert all(
        trace.policy_name != "task_mode_policy"
        for trace in result.policy_trace
        if trace.action == "block"
    )


def test_reconcile_previous_steps_should_persist_gap_before_digest() -> None:
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(evidence=evidence_repo))

    saved = asyncio.run(service.reconcile_previous_steps_evidence(
        scope=_scope("step-2"),
        completed_step_ids=["step-1"],
    ))
    projector = EvidenceDigestProjector(uow_factory=lambda: _UoW(evidence=evidence_repo))
    digest = asyncio.run(projector.build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
    ))

    assert saved
    assert evidence_repo.saved[0].evidence_kind == EvidenceKind.EVIDENCE_GAP
    assert evidence_repo.saved[0].payload["reason_code"] == "previous_step_evidence_missing"
    assert digest is not None
    assert digest.evidence_gaps[0].reason_code == "previous_step_evidence_missing"


def test_runtime_provider_should_reconcile_previous_steps_before_projecting_digest() -> None:
    evidence_repo = _EvidenceRepo()
    ledger_service = _ledger_service(uow_factory=lambda: _UoW(evidence=evidence_repo))
    projector = EvidenceDigestProjector(uow_factory=lambda: _UoW(evidence=evidence_repo))
    provider = EvidenceRuntimeContextProvider(ledger_service=ledger_service, projector=projector)
    context_service = RuntimeContextService(evidence_context_provider=provider)
    step1 = Step(id="step-1", status=ExecutionStatus.COMPLETED)
    step2 = Step(id="step-2")

    packet = asyncio.run(context_service.build_packet_async(
        stage="execute",
        state={
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "plan": Plan(steps=[step1, step2]),
            "step_states": [{"step_id": "step-1", "status": "completed"}],
        },
        step=step2,
        task_mode="general",
    ))

    assert evidence_repo.saved
    assert evidence_repo.saved[0].payload["reason_code"] == "previous_step_evidence_missing"
    assert packet["evidence_context"]["evidence_gaps"][0]["reason_code"] == "previous_step_evidence_missing"


def test_two_step_fact_to_evidence_reuse_closed_loop_should_resolve_without_executor() -> None:
    fact = _fact(SandboxFactKind.FILE_READ, step_id="step-1")
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=[fact], evidence=evidence_repo)
    ledger_service = _ledger_service(uow_factory=uow_factory)

    asyncio.run(ledger_service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))

    provider = EvidenceRuntimeContextProvider(
        ledger_service=ledger_service,
        projector=EvidenceDigestProjector(uow_factory=uow_factory),
    )
    context_service = RuntimeContextService(evidence_context_provider=provider)
    prepared = asyncio.run(prepare_execute_step_input(
        state={
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "plan": Plan(steps=[
                Step(id="step-1", status=ExecutionStatus.COMPLETED),
                Step(id="step-2"),
            ]),
            "step_states": [{"step_id": "step-1", "status": "completed"}],
            "graph_metadata": {},
        },
        step=Step(id="step-2"),
        llm=object(),
        runtime_context_service=context_service,
        task_mode="general",
        user_message="读取文件",
    ))
    assert prepared.runtime_evidence_context is not None
    assert prepared.runtime_evidence_context.evidence_reuse_snapshot is not None

    tool = _CountingRuntimeTool({"read_file"})
    policy_result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="读取文件"),
            task_mode="general",
            function_name="read_file",
            normalized_function_name="read_file",
            function_args={"path": "/workspace/a.txt"},
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=prepared.runtime_evidence_context.evidence_reuse_snapshot,
            has_previous_completed_steps=prepared.has_previous_completed_steps,
        )
    )

    assert policy_result.loop_break_reason == "virtual_success_pending_resolution"
    assert tool.invocations == []

    resolved = asyncio.run(_resolve_pending_evidence_reuse(
        llm_message={
            "success": False,
            "loop_break_reason": policy_result.loop_break_reason,
            "data": policy_result.tool_result.data,
        },
        prepared_input=prepared,
        state={
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
        },
        evidence_result_handle_resolver=EvidenceResultHandleResolver(uow_factory=uow_factory),
    ))

    assert resolved["success"] is True
    assert resolved["loop_break_reason"] == "evidence_reuse_allowed"
    assert resolved["data"]["result_handle_resolved"] is True


def test_execute_step_node_should_resolve_evidence_reuse_without_calling_executor() -> None:
    fact = _fact(SandboxFactKind.FILE_READ, step_id="step-1")
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=[fact], evidence=evidence_repo)
    ledger_service = _ledger_service(uow_factory=uow_factory)
    asyncio.run(ledger_service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    provider = EvidenceRuntimeContextProvider(
        ledger_service=ledger_service,
        projector=EvidenceDigestProjector(uow_factory=uow_factory),
    )
    context_service = RuntimeContextService(evidence_context_provider=provider)
    step1 = Step(id="step-1", status=ExecutionStatus.COMPLETED)
    step2 = Step(id="step-2", description="读取文件")
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "user_message": "读取文件",
        "plan": Plan(steps=[step1, step2]),
        "step_states": [{"step_id": "step-1", "status": "completed"}],
        "graph_metadata": {},
    }
    tool = _CountingRuntimeTool({"read_file"})
    resolver = _SpyEvidenceResultHandleResolver(
        EvidenceResultHandleResolver(uow_factory=uow_factory)
    )

    next_state = asyncio.run(execute_step_node(
        state,
        _FakeToolCallLLM("read_file", {"path": "/workspace/a.txt"}),
        context_service,
        evidence_result_handle_resolver=resolver,
        runtime_tools=[tool],
        max_tool_iterations=1,
    ))

    assert resolver.called is True
    assert tool.invocations == []
    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert next_state["last_executed_step"].outcome.done is True
    assert next_state["last_executed_step"].outcome.summary == "safe summary"
    called_events = [
        event for event in next_state["emitted_events"]
        if getattr(event, "type", "") == "tool" and event.status.value == "called"
    ]
    assert len(called_events) == 1
    assert called_events[0].function_name == "read_file"
    assert called_events[0].function_result.data["result_handle_id"] == resolver.result_handle_id


@pytest.mark.parametrize(
    ("function_name", "function_args", "payload"),
    [
        (
            "write_file",
            {"path": "/workspace/a.txt", "content": "new content"},
            {
                "path": "/workspace/a.txt",
                "operation": "write",
                "mutation_intent_hash": build_file_mutation_intent_hash(
                    path="/workspace/a.txt",
                    operation="write",
                    content="new content",
                    old_str="",
                    new_str="",
                    append=False,
                    leading_newline=False,
                    trailing_newline=False,
                ),
                "exists": True,
                "before_content_sha256": None,
                "after_content_sha256": "sha256:file-v2",
                "content_sha256_kind": "read_content_sha256",
                "size_after": 11,
                "changed": True,
                "missing_fields": None,
                "reason_code": None,
            },
        ),
        (
            "replace_in_file",
            {"path": "/workspace/a.txt", "old_str": "old", "new_str": "new"},
            {
                "path": "/workspace/a.txt",
                "operation": "write",
                "mutation_intent_hash": build_file_mutation_intent_hash(
                    path="/workspace/a.txt",
                    operation="write",
                    content="",
                    old_str="old",
                    new_str="new",
                    append=False,
                    leading_newline=False,
                    trailing_newline=False,
                ),
                "exists": True,
                "before_content_sha256": "sha256:file-v1",
                "after_content_sha256": "sha256:file-v2",
                "content_sha256_kind": "read_content_sha256",
                "size_after": 11,
                "changed": True,
                "missing_fields": None,
                "reason_code": None,
            },
        ),
    ],
)
def test_two_step_file_mutation_duplicate_should_not_call_executor(function_name, function_args, payload) -> None:
    fact = _fact(SandboxFactKind.FILE_WRITE, step_id="step-1", payload=payload)
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=[fact], evidence=evidence_repo)
    ledger_service = _ledger_service(uow_factory=uow_factory)
    asyncio.run(ledger_service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    runtime_context = asyncio.run(
        EvidenceDigestProjector(uow_factory=uow_factory).build_context(
            stage="execute",
            scope=_scope("step-2"),
            completed_step_ids=["step-1"],
            step=Step(id="step-2"),
            task_mode="general",
        )
    )
    tool = _CountingRuntimeTool({function_name})

    result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="重复写文件"),
            task_mode="general",
            function_name=function_name,
            normalized_function_name=function_name,
            function_args=function_args,
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=runtime_context.evidence_reuse_snapshot,
            has_previous_completed_steps=True,
        )
    )

    assert result.loop_break_reason == "virtual_success_pending_resolution"
    assert tool.invocations == []


def test_file_mutation_duplicate_should_normalize_fact_and_tool_paths() -> None:
    intent_hash = build_file_mutation_intent_hash(
        path="/workspace/dir/../a.txt",
        operation="write",
        content="new content",
        old_str="",
        new_str="",
        append=False,
        leading_newline=False,
        trailing_newline=False,
    )
    fact = _fact(
        SandboxFactKind.FILE_WRITE,
        step_id="step-1",
        payload={
            "path": "/workspace/dir/../a.txt",
            "operation": "write",
            "mutation_intent_hash": intent_hash,
            "exists": True,
            "before_content_sha256": None,
            "after_content_sha256": "sha256:file-v2",
            "content_sha256_kind": "read_content_sha256",
            "size_after": 11,
            "changed": True,
            "missing_fields": None,
            "reason_code": None,
        },
    )
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=[fact], evidence=evidence_repo)
    ledger_service = _ledger_service(uow_factory=uow_factory)
    asyncio.run(ledger_service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    runtime_context = asyncio.run(
        EvidenceDigestProjector(uow_factory=uow_factory).build_context(
            stage="execute",
            scope=_scope("step-2"),
            completed_step_ids=["step-1"],
            step=Step(id="step-2"),
            task_mode="general",
        )
    )
    tool = _CountingRuntimeTool({"write_file"})

    result = asyncio.run(
        ToolPolicyEngine(logger=logging.getLogger(__name__)).evaluate_tool_call(
            step=Step(id="step-2", description="重复写文件"),
            task_mode="general",
            function_name="write_file",
            normalized_function_name="write_file",
            function_args={"path": "/workspace/a.txt", "content": "new content"},
            matched_tool=tool,
            runtime_tools=[tool],
            browser_route_state_key="",
            iteration_blocked_function_names=set(),
            execution_context=_execution_context(),
            execution_state=ExecutionState(),
            started_at=0.0,
            evidence_reuse_snapshot=runtime_context.evidence_reuse_snapshot,
            has_previous_completed_steps=True,
        )
    )

    assert runtime_context.evidence_reuse_snapshot.do_not_repeat[0].action_key == (
        f"file_write:/workspace/a.txt:{intent_hash}"
    )
    assert result.loop_break_reason == "virtual_success_pending_resolution"
    assert tool.invocations == []


def test_execute_node_should_resolve_pending_result_before_virtual_success() -> None:
    runtime_context = _runtime_context()
    handle = runtime_context.result_handles[0]
    prepared = type(
        "Prepared",
        (),
        {
            "result_handle_index": runtime_context.result_handle_index,
            "runtime_evidence_context": runtime_context,
        },
    )()
    resolver = _Resolver(handle)
    pending_message = {
        "success": False,
        "loop_break_reason": "virtual_success_pending_resolution",
        "data": {"result_handle_id": handle.result_handle_id, "reuse_summary": "safe summary"},
    }

    resolved = asyncio.run(_resolve_pending_evidence_reuse(
        llm_message=pending_message,
        prepared_input=prepared,
        state={
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
        },
        evidence_result_handle_resolver=resolver,
    ))

    assert resolver.called is True
    assert resolved["success"] is True
    assert resolved["loop_break_reason"] == "evidence_reuse_allowed"
    assert resolved["data"]["result_handle_resolved"] is True


def test_agent_task_runner_should_reconcile_evidence_before_completed_step_event_persisted() -> None:
    step = Step(id="step-1")
    event = StepEvent(step=step, status="completed")
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _RunnerUoW()
    runner._evidence_step_reconciler = _StepReconciler()

    asyncio.run(runner._reconcile_evidence_before_step_completed(event))

    assert runner._evidence_step_reconciler.calls
    call = runner._evidence_step_reconciler.calls[0]
    assert call["step"].id == "step-1"
    assert call["scope"].current_step_id == "step-1"
    assert call["scope"].run_id == "run-1"


def test_agent_task_runner_should_overwrite_step_outcome_with_evidence_backed_projection() -> None:
    step = Step(
        id="step-1",
        outcome=StepOutcome(done=True, summary="完成", facts_learned=["model only fact"]),
    )
    event = StepEvent(step=step, status="completed")
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _RunnerUoW()
    runner._evidence_step_reconciler = _ProjectionStepReconciler()

    asyncio.run(runner._reconcile_evidence_before_step_completed(event))

    assert event.step.outcome is not None
    assert event.step.outcome.facts_learned == ["evidence backed fact"]
    assert event.step.outcome.evidence_backed_facts[0].evidence_ids == ["evidence-1"]


@pytest.mark.parametrize("failure_mode", ["fact_repo", "assembler", "record_evidence"])
def test_reconcile_step_evidence_should_write_gap_when_reconcile_fails(failure_mode) -> None:
    fact = _fact(SandboxFactKind.FILE_READ)
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(facts=[fact], evidence=evidence_repo))
    if failure_mode == "fact_repo":
        service = _ledger_service(uow_factory=lambda: _FailingFactUoW(evidence=evidence_repo))
    elif failure_mode == "assembler":
        service._assembler = _FailingAssembler()
    else:
        service.record_evidence = _record_evidence_that_fails_once_then_saves(service, evidence_repo)  # type: ignore[method-assign]

    saved = asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))

    assert saved
    assert evidence_repo.saved[-1].evidence_kind == EvidenceKind.EVIDENCE_GAP
    assert evidence_repo.saved[-1].payload["reason_code"] == "evidence_reconcile_failed"


def test_agent_task_runner_should_continue_completed_step_when_reconcile_writes_gap() -> None:
    step = Step(id="step-1")
    event = StepEvent(step=step, status="completed")
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _RunnerUoW()
    runner._evidence_step_reconciler = _GapWritingReconciler()

    asyncio.run(runner._reconcile_evidence_before_step_completed(event))

    assert runner._evidence_step_reconciler.called is True


@pytest.mark.parametrize("failure_mode", ["scope", "raise", "none"])
def test_agent_task_runner_should_not_block_completed_event_when_reconcile_fails(failure_mode) -> None:
    step = Step(id="step-1")
    event = StepEvent(step=step, status="completed")
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._user_id = "user-1"
    runner._uow_factory = (
        (lambda: _RunnerUoW(session_repo=_MissingRunnerSessionRepo()))
        if failure_mode == "scope"
        else (lambda: _RunnerUoW())
    )
    reconciler = {
        "scope": _StepReconciler(),
        "raise": _ThrowingStepReconciler(),
        "none": _NoneReturningStepReconciler(),
    }[failure_mode]
    runner._evidence_step_reconciler = reconciler

    asyncio.run(runner._reconcile_evidence_before_step_completed(event))

    if failure_mode == "raise":
        assert reconciler.gap_called is True


def test_graph_execute_closure_should_capture_evidence_result_handle_resolver() -> None:
    execute_closure_code = next(
        const for const in build_planner_react_langgraph_graph.__code__.co_consts
        if hasattr(const, "co_name") and const.co_name == "_execute_step_with_llm"
    )

    assert "evidence_result_handle_resolver" in execute_closure_code.co_freevars


def _runtime_context():
    handle = build_evidence_result_handle(_result_ref())
    do_not_repeat = _do_not_repeat(handle.result_handle_id)
    snapshot = EvidenceReuseSnapshot(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        cursor="cursor-1",
        do_not_repeat=[do_not_repeat],
        result_handles=[handle],
    )
    from app.domain.models.evidence import RuntimeEvidenceContextResult

    return RuntimeEvidenceContextResult(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        has_previous_completed_steps=True,
        prompt_digest="digest",
        evidence_reuse_snapshot=snapshot,
        result_handles=[handle],
        result_handle_index={handle.result_handle_id: handle},
        cursor="cursor-1",
    )


def _result_ref() -> EvidenceResultRef:
    return EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.FACT_REF,
        ref_id="fact-1",
        source_step_id="step-1",
        source_evidence_id="evidence-1",
        source_fact_id="fact-1",
        source_event_id="event-1",
        subject_key="file:/workspace/a.txt",
        payload_hash="sha256:payload",
        quality_status=EvidenceQualityStatus.VALID,
        support_level=EvidenceSupportLevel.STRONG,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
        summary="safe summary",
    )


def _do_not_repeat(result_handle_id: str):
    from app.domain.models.evidence import EvidenceDoNotRepeatResult

    return EvidenceDoNotRepeatResult(
        action_key="file_read:/workspace/a.txt",
        subject_key="file:/workspace/a.txt",
        reason_code="evidence_reuse_pending_resolution",
        source_step_id="step-1",
        evidence_ids=["evidence-1"],
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        support_level=EvidenceSupportLevel.STRONG,
        quality_status=EvidenceQualityStatus.VALID,
        result_status="successful",
        duplicate_decision=EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION,
        reuse_result_ref=_result_ref(),
        result_handle_id=result_handle_id,
        reuse_summary="safe summary",
    )


class _Provider:
    def __init__(self, runtime_context=None) -> None:
        self.completed_step_ids = []
        self._runtime_context = runtime_context

    async def build_context(self, *, completed_step_ids, **kwargs):
        self.completed_step_ids = completed_step_ids
        return self._runtime_context or _runtime_context()


class _Resolver:
    def __init__(self, handle) -> None:
        self.handle = handle
        self.called = False

    async def resolve(self, *, scope, handle):
        self.called = True
        assert handle.result_handle_id == self.handle.result_handle_id
        return EvidenceResolvedResult(
            status=EvidenceResolvedStatus.RESOLVED,
            result_ref_type=handle.result_ref_type,
            source_evidence_id=handle.source_evidence_id,
            source_fact_id=handle.source_fact_id,
            source_event_id=handle.source_event_id,
            subject_key=handle.subject_key,
            read_strategy=handle.read_strategy,
            summary=handle.summary,
            resolved_payload={"summary": "safe"},
            payload_hash=handle.payload_hash,
        )


class _SpyEvidenceResultHandleResolver:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.called = False
        self.result_handle_id = ""

    async def resolve(self, *, scope, handle):
        self.called = True
        self.result_handle_id = handle.result_handle_id
        return await self._delegate.resolve(scope=scope, handle=handle)


class _FakeToolCallLLM:
    def __init__(self, function_name: str, function_args: dict) -> None:
        self.function_name = function_name
        self.function_args = dict(function_args)
        self.calls = 0

    async def invoke(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": self.function_name,
                            "arguments": json.dumps(self.function_args, ensure_ascii=False),
                        },
                    }
                ]
            }
        return {"content": json.dumps({"success": True, "summary": "done"}, ensure_ascii=False)}


class _CountingRuntimeTool(BaseTool):
    name = "counting-tool"

    def __init__(self, function_names: set[str]) -> None:
        super().__init__()
        self.function_names = set(function_names)
        self.invocations = []

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": function_name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for function_name in sorted(self.function_names)
        ]

    def has_tool(self, function_name: str) -> bool:
        return str(function_name or "").strip().lower() in self.function_names

    async def invoke(self, function_name: str, **kwargs):
        self.invocations.append((function_name, dict(kwargs)))
        return ToolResult(success=True, data={"invoked": function_name, **dict(kwargs)})


class _SignatureFilteringTool(BaseTool):
    name = "signature-filtering-tool"

    def __init__(self, function_names: set[str]) -> None:
        super().__init__()
        self.function_names = set(function_names)
        self.invocations = []

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": function_name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for function_name in sorted(self.function_names)
        ]

    def has_tool(self, function_name: str) -> bool:
        return str(function_name or "").strip().lower() in self.function_names

    @tool(name="search_web", description="search", parameters={}, required=["query"])
    async def search_web(self, query: str):
        self.invocations.append(("search_web", {"query": query}))
        return ToolResult(success=True, data={"query": query, "results": []})

    @tool(name="fetch_page", description="fetch", parameters={}, required=["url"])
    async def fetch_page(self, url: str):
        self.invocations.append(("fetch_page", {"url": url}))
        return ToolResult(
            success=True,
            data={
                "url": url,
                "final_url": url,
                "title": "Useful page",
                "content": "useful content " * 20,
            },
        )


class _StepReconciler:
    def __init__(self) -> None:
        self.calls = []

    async def reconcile_step_evidence(self, *, scope, step):
        self.calls.append({"scope": scope, "step": step})
        return []


class _ThrowingStepReconciler:
    def __init__(self) -> None:
        self.gap_called = False

    async def reconcile_step_evidence(self, *, scope, step):
        raise RuntimeError("reconciler failed")

    async def record_reconcile_failed_gap(self, *, scope, step):
        self.gap_called = True
        return object()


class _NoneReturningStepReconciler:
    async def reconcile_step_evidence(self, *, scope, step):
        return None


class _ProjectionStepReconciler:
    async def reconcile_step_evidence(self, *, scope, step):
        return [object()]

    async def build_step_evidence_backed_facts(self, *, scope, step):
        return [
            EvidenceBackedFactProjection(
                text="evidence backed fact",
                evidence_ids=["evidence-1"],
                fact_ids=["fact-1"],
                artifact_ids=[],
                source_event_ids=["event-1"],
                user_confirmation_event_ids=[],
            )
        ]


class _GapWritingReconciler:
    def __init__(self) -> None:
        self.called = False

    async def reconcile_step_evidence(self, *, scope, step):
        self.called = True
        return [object()]


class _FailingAssembler:
    def assemble_step(self, *, step, facts):
        raise RuntimeError("assembler failed")


def _record_evidence_that_fails_once_then_saves(service, evidence_repo):
    calls = {"count": 0}
    original_record_evidence = service.record_evidence

    async def _record_evidence(*, scope, evidence_input):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("record failed")
        return await original_record_evidence(scope=scope, evidence_input=evidence_input)

    return _record_evidence


class _RunnerSessionRepo:
    async def get_by_id(self, *, session_id, user_id=None):
        return Session(id=session_id, user_id=user_id, workspace_id="workspace-1", current_run_id="run-1")


class _MissingRunnerSessionRepo:
    async def get_by_id(self, *, session_id, user_id=None):
        return None


class _RunnerWorkspaceRepo:
    async def get_by_id_for_user(self, *, workspace_id, user_id):
        return Workspace(id=workspace_id, user_id=user_id, session_id="session-1", current_run_id="run-1")

    async def get_by_session_id_for_user(self, *, session_id, user_id):
        return Workspace(id="workspace-1", user_id=user_id, session_id=session_id, current_run_id="run-1")


class _RunnerWorkflowRunRepo:
    async def get_by_id_for_user(self, *, run_id, user_id):
        return WorkflowRun(id=run_id, user_id=user_id, session_id="session-1", current_step_id="step-1")


class _RunnerUoW:
    def __init__(self, *, session_repo=None) -> None:
        self.session = session_repo or _RunnerSessionRepo()
        self.workspace = _RunnerWorkspaceRepo()
        self.workflow_run = _RunnerWorkflowRunRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _execution_context() -> ExecutionContext:
    return ExecutionContext(
        normalized_user_content=[],
        available_tools=[],
        available_function_names=set(),
        browser_route_enabled=False,
        blocked_function_names=set(),
        read_only_file_blocked_function_names=set(),
        research_file_context_blocked_function_names=set(),
        file_processing_shell_blocked_function_names=set(),
        artifact_policy_blocked_function_names=set(),
        requested_max_tool_iterations=1,
        effective_max_tool_iterations=1,
        allow_ask_user=False,
        research_route_enabled=False,
        research_has_explicit_url=False,
    )


def _payload_for_fact(kind: SandboxFactKind) -> dict:
    if kind == SandboxFactKind.DOCUMENT_CONTEXT:
        return {
            "file_id": "file-1",
            "filename_extension": ".pdf",
            "mime_type": "application/pdf",
            "parse_status": "parsed",
            "reason_code": None,
            "full_file_sha256": "sha256:full",
            "read_content_sha256": "sha256:read",
            "is_truncated": False,
            "excerpt_char_count": 10,
            "missing_fields": None,
        }
    if kind == SandboxFactKind.SEARCH_RESULT:
        return {"query_hash": "q1", "query_excerpt": "query", "result_count": 1, "top_results": [], "is_truncated": False, "missing_fields": None, "reason_code": None}
    if kind == SandboxFactKind.FETCHED_PAGE:
        return {"fetched_url_hash": "u1", "final_url_origin": "https://example.com", "status_code": 200, "content_type": "text/html", "title": "title", "excerpt": "body", "is_truncated": False, "missing_fields": None, "reason_code": None}
    if kind == SandboxFactKind.BROWSER_SNAPSHOT:
        return {"url_hash": "u1", "url_origin": "https://example.com", "title": "title", "screenshot_artifact_id": None, "screenshot_artifact_path": None, "structured_summary": "summary", "actionable_element_count": 1, "degrade_reason": None, "missing_fields": None, "reason_code": None}
    if kind == SandboxFactKind.BROWSER_ACTION:
        return {"action": "browser_click", "target_summary": "Login", "url_hash_before": "u1", "url_hash_after": "u2", "success": True, "degrade_reason": None, "missing_fields": None, "reason_code": None}
    if kind == SandboxFactKind.TOOL_FAILURE:
        return {"function_name": "read_file", "reason_code": "failed", "message_excerpt": "failed", "retry_count": 1, "timeout": False, "diagnostic_type": "tool", "missing_fields": None}
    if kind == SandboxFactKind.HUMAN_INTERACTION:
        return {"interaction_type": "confirm", "message_excerpt": "confirmed", "confirmed": True, "reason_code": None, "missing_fields": None}
    if kind == SandboxFactKind.COMMAND_EXECUTION:
        return {
            "command_fingerprint": "sha256:command",
            "cwd": "/workspace",
            "exit_code": 0,
            "duration_ms": 10,
            "stdout_excerpt": "done",
            "stderr_excerpt": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "changed_paths": [],
            "timeout": False,
            "command_excerpt": "echo done",
            "missing_fields": None,
            "reason_code": None,
        }
    if kind == SandboxFactKind.SHELL_OUTPUT:
        return {
            "session_ref": "shell-1",
            "output_excerpt": "done",
            "output_truncated": False,
            "console_record_count": 1,
            "process_status": "completed",
            "exit_code": 0,
            "duration_ms": 10,
            "missing_fields": None,
            "reason_code": None,
        }
    if kind == SandboxFactKind.FILE_WRITE:
        return {
            "path": "/workspace/a.txt",
            "operation": "write",
            "mutation_intent_hash": build_file_mutation_intent_hash(
                path="/workspace/a.txt",
                operation="write",
                content="new content",
                old_str="",
                new_str="",
                append=False,
                leading_newline=False,
                trailing_newline=False,
            ),
            "exists": True,
            "before_content_sha256": None,
            "after_content_sha256": "sha256:file",
            "content_sha256_kind": "read_content_sha256",
            "size_after": 10,
            "changed": True,
            "missing_fields": None,
            "reason_code": None,
        }
    if kind == SandboxFactKind.FILE_LIST:
        return {
            "dir_path": "/workspace/a.txt",
            "entry_count": 1,
            "entries": [{"name": "a.txt", "type": "file", "size": 10, "mtime": None}],
            "is_truncated": False,
            "missing_fields": None,
            "reason_code": None,
        }
    if kind == SandboxFactKind.FILE_SEARCH:
        return {
            "path": "/workspace/a.txt",
            "regex_hash": "sha256:regex",
            "match_count": 1,
            "matches": [{"path": "/workspace/a.txt", "line_number": 1, "excerpt": "safe"}],
            "is_truncated": False,
            "regex_excerpt": "safe",
            "missing_fields": None,
            "reason_code": None,
        }
    return {"path": "/workspace/a.txt", "exists": True, "size": 10, "content_sha256": "sha256:file", "content_sha256_kind": "read_content_sha256", "mime_type": "text/plain", "line_range": None, "excerpt": "safe", "is_truncated": False, "mtime": None, "missing_fields": None, "reason_code": None}


def _function_for_fact(kind: SandboxFactKind) -> str:
    return {
        SandboxFactKind.COMMAND_EXECUTION: "run_shell_command",
        SandboxFactKind.SHELL_OUTPUT: "read_shell_output",
        SandboxFactKind.SEARCH_RESULT: "search_web",
        SandboxFactKind.FETCHED_PAGE: "fetch_page",
        SandboxFactKind.DOCUMENT_CONTEXT: "read_document",
        SandboxFactKind.BROWSER_SNAPSHOT: "browser_snapshot",
        SandboxFactKind.TOOL_FAILURE: "read_file",
        SandboxFactKind.HUMAN_INTERACTION: "message_ask_user",
    }.get(kind, "read_file")


def _subject_type_for_fact(kind: SandboxFactKind) -> str:
    return {
        SandboxFactKind.COMMAND_EXECUTION: "command",
        SandboxFactKind.SHELL_OUTPUT: "command",
        SandboxFactKind.SEARCH_RESULT: "search",
        SandboxFactKind.FETCHED_PAGE: "page",
        SandboxFactKind.DOCUMENT_CONTEXT: "document",
        SandboxFactKind.BROWSER_SNAPSHOT: "browser",
        SandboxFactKind.TOOL_FAILURE: "file",
        SandboxFactKind.HUMAN_INTERACTION: "interaction",
    }.get(kind, "file")
