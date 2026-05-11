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
from app.application.service.sandbox_fact_ledger_service import SandboxFactLedgerService
from app.domain.models import (
    ExecutionStatus,
    FetchedPage,
    Plan,
    Session,
    Step,
    StepEvent,
    StepOutcome,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
    WorkflowRun,
    Workspace,
)
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
from app.domain.services.runtime.contracts.sandbox_fact_contract import SandboxFactProfileRef
from app.domain.services.runtime.contracts.sandbox_fact_ports import (
    SandboxFactProjectionContext,
    ToolEventFactProjectionResult,
)
from app.domain.services.runtime.langgraph_state import GraphStateContractMapper
from app.domain.services.runtime.normalizers import normalize_step_outcome_payload
from app.domain.services.tools import BaseTool
from app.domain.services.tools.base import tool
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.projectors.sandbox_fact_tool_event_projector import (
    SandboxFactToolEventProjector,
)
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
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.execute_nodes import execute_step_node
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.evidence_completion_gate import (
    reconcile_step_evidence_before_state_return,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.execute_helpers import (
    persist_current_step_tool_events_before_evidence_gate,
    prepare_execute_step_input,
)
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

    async def list_by_ids(self, *, evidence_ids, **kwargs):
        return [record for record in self.records if record.id in set(evidence_ids)]


class _FactRepo:
    def __init__(self, facts=None) -> None:
        self.facts = facts if facts is not None else []
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

    async def save_once(self, fact):
        for existing in self.facts:
            if existing.idempotency_key == fact.idempotency_key:
                return existing
        self.facts.append(fact)
        return fact


class _WorkflowRunRepo:
    async def get_event_record_by_event_id(self, **kwargs):
        return object()

    async def add_event_record_if_absent(self, **kwargs):
        return None


class _ArtifactRepo:
    async def get_by_user_workspace_id_and_id(self, **kwargs):
        return None


class _UoW:
    def __init__(self, *, evidence=None, facts=None) -> None:
        self.evidence = evidence or _EvidenceRepo()
        self.sandbox_fact = _FactRepo(facts if facts is not None else [])
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


class _InMemoryToolEventPersistence:
    def __init__(self, *, uow_factory) -> None:
        self._projector = SandboxFactToolEventProjector(
            ledger_service=SandboxFactLedgerService(uow_factory=uow_factory),
        )
        self.persisted_events = []

    async def persist_tool_event_and_record_facts(
            self,
            *,
            event: ToolEvent,
            run_id: str,
            session_id: str,
            current_step_id: str,
    ) -> ToolEventFactProjectionResult:
        source_event_id = str(event.id or f"event-{len(self.persisted_events) + 1}")
        event.id = source_event_id
        if not event.step_id:
            event.step_id = current_step_id
        context = SandboxFactProjectionContext(
            scope=_scope(current_step_id),
            profile_ref=SandboxFactProfileRef(status="missing"),
            source_event_id=source_event_id,
            current_step_id=current_step_id,
        )
        facts = await self._projector.record_from_tool_event(context=context, event=event)
        event.runtime_fact_projection = {
            "source_event_id": source_event_id,
            "fact_count": len(facts),
            "sandbox_fact_event_persisted": False,
            "event_inserted": True,
        }
        self.persisted_events.append(event)
        return ToolEventFactProjectionResult(
            source_event_id=source_event_id,
            fact_count=len(facts),
            sandbox_fact_event_persisted=False,
            event_inserted=True,
        )


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


def _verification_required_search_snapshot(*, query: str = "query") -> EvidenceReuseSnapshot:
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
        staleness_policy=EvidenceStalenessPolicy.EXTERNAL_MAY_CHANGE,
        read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
        reason_code="external_evidence_may_change",
        allowed_verification_actions=["search_web"],
        summary="safe summary",
    )
    handle = build_evidence_result_handle(result_ref)
    from app.domain.models.evidence import EvidenceDoNotRepeatResult

    return EvidenceReuseSnapshot(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        cursor="cursor-1",
        do_not_repeat=[
            EvidenceDoNotRepeatResult(
                action_key=key_result.action_key,
                subject_key=key_result.subject_key,
                reason_code="evidence_reuse_requires_verification",
                source_step_id="step-1",
                evidence_ids=["evidence-1"],
                reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
                staleness_policy=EvidenceStalenessPolicy.EXTERNAL_MAY_CHANGE,
                support_level=EvidenceSupportLevel.STRONG,
                quality_status=EvidenceQualityStatus.VALID,
                result_status="successful",
                duplicate_decision=EvidenceDuplicateDecision.REQUIRE_VERIFICATION,
                reuse_result_ref=result_ref,
                result_handle_id=handle.result_handle_id,
                reuse_summary="safe summary",
            )
        ],
        result_handles=[handle],
    )


def _verification_required_page_snapshot(*, url: str = "https://example.com/a") -> EvidenceReuseSnapshot:
    url_hash = hash_url(url)
    key_result = build_evidence_action_subject_key_from_tool_call("fetch_page", {"url": url})
    result_ref = EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.FACT_REF,
        ref_id="fact-1",
        source_step_id="step-1",
        source_evidence_id="evidence-1",
        source_fact_id="fact-1",
        source_event_id="event-1",
        subject_key=f"page:{url_hash}",
        payload_hash="sha256:payload",
        quality_status=EvidenceQualityStatus.VALID,
        support_level=EvidenceSupportLevel.STRONG,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.EXTERNAL_MAY_CHANGE,
        read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
        reason_code="external_evidence_may_change",
        allowed_verification_actions=["fetch_page"],
        summary="safe summary",
    )
    handle = build_evidence_result_handle(result_ref)
    from app.domain.models.evidence import EvidenceDoNotRepeatResult

    return EvidenceReuseSnapshot(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        cursor="cursor-1",
        do_not_repeat=[
            EvidenceDoNotRepeatResult(
                action_key=key_result.action_key,
                subject_key=key_result.subject_key,
                reason_code="evidence_reuse_requires_verification",
                source_step_id="step-1",
                evidence_ids=["evidence-1"],
                reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
                staleness_policy=EvidenceStalenessPolicy.EXTERNAL_MAY_CHANGE,
                support_level=EvidenceSupportLevel.STRONG,
                quality_status=EvidenceQualityStatus.VALID,
                result_status="successful",
                duplicate_decision=EvidenceDuplicateDecision.REQUIRE_VERIFICATION,
                reuse_result_ref=result_ref,
                result_handle_id=handle.result_handle_id,
                reuse_summary="safe summary",
            )
        ],
        result_handles=[handle],
    )


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


def test_page_evidence_should_be_run_scoped_reusable_when_valid() -> None:
    result = EvidenceFactAssembler().assemble_step(
        step=Step(id="step-1"),
        facts=[_fact(SandboxFactKind.FETCHED_PAGE)],
    )

    evidence_input = result.evidence_inputs[0]
    assert evidence_input.evidence_kind == EvidenceKind.PAGE_EVIDENCE
    assert evidence_input.quality_status == EvidenceQualityStatus.VALID
    assert evidence_input.support_level == EvidenceSupportLevel.STRONG
    assert evidence_input.reusable is True
    assert evidence_input.reuse_policy == EvidenceReusePolicy.REUSE_ALLOWED
    assert evidence_input.staleness_policy == EvidenceStalenessPolicy.RUN_SCOPED
    assert evidence_input.result_refs[0].allowed_verification_actions == []
    assert evidence_input.result_refs[0].reason_code is None


def test_page_evidence_should_require_verification_when_partial() -> None:
    result = EvidenceFactAssembler().assemble_step(
        step=Step(id="step-1"),
        facts=[
            _fact(
                SandboxFactKind.FETCHED_PAGE,
                payload={
                    **_payload_for_fact(SandboxFactKind.FETCHED_PAGE),
                    "excerpt": "",
                    "is_truncated": True,
                    "reason_code": "page_content_truncated",
                },
            )
        ],
    )

    evidence_input = result.evidence_inputs[0]
    assert evidence_input.quality_status == EvidenceQualityStatus.PARTIAL
    assert evidence_input.support_level == EvidenceSupportLevel.PARTIAL
    assert evidence_input.reusable is False
    assert evidence_input.reuse_policy == EvidenceReusePolicy.VERIFY_BEFORE_REUSE
    assert evidence_input.staleness_policy == EvidenceStalenessPolicy.EXTERNAL_MAY_CHANGE
    assert evidence_input.result_refs[0].allowed_verification_actions == ["fetch_page"]


def test_file_evidence_should_accept_read_content_sha256_for_successful_file_read() -> None:
    fact = _fact(
        SandboxFactKind.FILE_READ,
        payload={
            **_payload_for_fact(SandboxFactKind.FILE_READ),
            "content_sha256": None,
            "read_content_sha256": "sha256:read",
            "content_sha256_kind": "read_content_sha256",
            "reason_code": None,
            "missing_fields": None,
        },
    )

    result = EvidenceFactAssembler().assemble_step(step=Step(id="step-1"), facts=[fact])
    evidence_input = result.evidence_inputs[0]

    assert evidence_input.evidence_kind == EvidenceKind.FILE_EVIDENCE
    assert evidence_input.quality_status == EvidenceQualityStatus.VALID
    assert evidence_input.support_level == EvidenceSupportLevel.STRONG
    assert evidence_input.reusable is True
    assert evidence_input.reuse_policy == EvidenceReusePolicy.REUSE_ALLOWED
    assert evidence_input.result_refs[0].content_hash == "sha256:read"
    assert evidence_input.payload["content_sha256"] == "sha256:read"
    assert evidence_input.result_refs[0].read_strategy == EvidenceReadStrategy.READ_FACT_PAYLOAD


def test_file_evidence_should_not_be_strong_or_reusable_without_content_hash() -> None:
    fact = _fact(
        SandboxFactKind.FILE_READ,
        payload={
            **_payload_for_fact(SandboxFactKind.FILE_READ),
            "content_sha256": None,
            "read_content_sha256": None,
            "content_sha256_kind": "unknown",
            "reason_code": None,
            "missing_fields": None,
        },
    )

    result = EvidenceFactAssembler().assemble_step(step=Step(id="step-1"), facts=[fact])
    evidence_input = result.evidence_inputs[0]

    assert evidence_input.quality_status == EvidenceQualityStatus.PARTIAL
    assert evidence_input.support_level == EvidenceSupportLevel.PARTIAL
    assert evidence_input.reusable is False
    assert evidence_input.reuse_policy != EvidenceReusePolicy.REUSE_ALLOWED
    assert evidence_input.result_refs[0].content_hash is None


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


def test_digest_should_include_valid_page_evidence_in_do_not_repeat() -> None:
    url = "https://example.com/a"
    url_hash = hash_url(url)
    fact = _fact(
        SandboxFactKind.FETCHED_PAGE,
        payload={
            **_payload_for_fact(SandboxFactKind.FETCHED_PAGE),
            "fetched_url_hash": url_hash,
            "final_url_origin": url,
        },
    )
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(facts=[fact], evidence=evidence_repo))
    asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    projector = EvidenceDigestProjector(uow_factory=lambda: _UoW(evidence=evidence_repo))

    digest = asyncio.run(projector.build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
    ))

    assert digest is not None
    assert digest.result_handles
    assert digest.do_not_repeat
    assert digest.do_not_repeat[0].action_key == f"fetch:{url_hash}"
    assert digest.do_not_repeat[0].subject_key == f"page:{url_hash}"
    assert digest.do_not_repeat[0].duplicate_decision == (
        EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION
    )
    assert digest.do_not_repeat[0].result_handle_id == digest.result_handles[0].result_handle_id


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
    snapshot = _verification_required_search_snapshot(query=query)

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
    snapshot = _verification_required_search_snapshot(query=query)

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
    assert decision.metadata["rewrite_type"] == "evidence_verification_audit_metadata"
    assert decision.metadata["source_step_id"] == "step-1"
    assert decision.metadata["result_handle_id"] == snapshot.do_not_repeat[0].result_handle_id


@pytest.mark.parametrize(
    "function_args",
    [
        {"query_hash": "q1", "verification_reason_code": "external_evidence_may_change"},
        {"query": "query", "query_hash": "q1", "verification_reason_code": "external_evidence_may_change"},
        {"query": "same meaning", "query_hash": "q1", "verification_reason_code": "external_evidence_may_change"},
    ],
)
def test_evidence_reuse_policy_should_block_invalid_verification_search(function_args) -> None:
    snapshot = _verification_required_search_snapshot()

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
    snapshot = _verification_required_page_snapshot(url=url)

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
    snapshot = _verification_required_page_snapshot(url=url)

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
    assert decision.metadata["rewrite_type"] == "evidence_verification_audit_metadata"
    assert decision.metadata["source_step_id"] == "step-1"
    assert decision.metadata["result_handle_id"] == snapshot.do_not_repeat[0].result_handle_id


@pytest.mark.parametrize(
    "function_args",
    [
        {"url_hash": "u1", "verification_reason_code": "external_evidence_may_change"},
        {"url": "https://example.com/a", "url_hash": "u1", "verification_reason_code": "external_evidence_may_change"},
        {"url": "https://example.com/other", "url_hash": "u1", "verification_reason_code": "external_evidence_may_change"},
    ],
)
def test_evidence_reuse_policy_should_block_invalid_verification_fetch_page(function_args) -> None:
    snapshot = _verification_required_page_snapshot()

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


def test_evidence_reuse_policy_should_block_empty_snapshot_with_previous_steps() -> None:
    snapshot = EvidenceReuseSnapshot(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        cursor="cursor-empty",
        do_not_repeat=[],
        result_handles=[],
    )

    decision = evaluate_evidence_reuse_policy(
        ConstraintInput(
            step=Step(id="step-2"),
            task_mode="general",
            function_name="search_web",
            normalized_function_name="search_web",
            function_args={"query": "query"},
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
    assert decision.action == "block"
    assert decision.reason_code == "evidence_reuse_snapshot_missing"
    assert decision.tool_result_payload.data["duplicate_decision"] == "snapshot_empty_with_previous_completed_step"


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
        snapshot = _verification_required_search_snapshot(query=function_args["query"])
    else:
        snapshot = _verification_required_page_snapshot(url=function_args["url"])
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
    snapshot = _verification_required_search_snapshot(query=query)
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
    snapshot = _verification_required_search_snapshot(query=query)
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
    snapshot = _verification_required_search_snapshot(query=query)
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
    snapshot = _verification_required_page_snapshot(url=url)
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
    snapshot = _verification_required_page_snapshot(url=url)
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
    snapshot = _verification_required_page_snapshot()
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
        evidence_result_handle_resolver=EvidenceResultHandleResolver(uow_factory=uow_factory),
        evidence_resolution_state={
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "current_step_id": "step-2",
        },
    ))

    assert payload["success"] is True
    assert payload["loop_break_reason"] == "evidence_reuse_allowed"
    assert payload["data"]["result_handle_resolved"] is True
    assert tool.invocations == []
    called_events = [
        event for event in tool_events
        if event.status == ToolEventStatus.CALLED and event.function_name == "read_file"
    ]
    assert len(called_events) == 1
    assert called_events[0].function_result.success is True


def test_execute_step_node_should_resolve_evidence_reuse_without_calling_executor(caplog) -> None:
    fact = _fact(SandboxFactKind.FILE_READ, step_id="step-1")
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=[fact], evidence=evidence_repo)
    ledger_service = _ledger_service(uow_factory=uow_factory)
    asyncio.run(ledger_service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    digest = asyncio.run(EvidenceDigestProjector(uow_factory=uow_factory).build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
        stage="execute",
    ))
    assert digest is not None
    assert digest.do_not_repeat
    assert digest.do_not_repeat[0].duplicate_decision == (
        EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION
    )
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

    with caplog.at_level(logging.INFO):
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
    log_text = caplog.text
    expected_log_events = [
        "reuse_existing_evidence_pending_resolution",
        "evidence_result_handle_resolution_started",
        "evidence_result_handle_resolved",
        "evidence_reuse_virtual_tool_result_returned",
    ]
    positions = [log_text.index(event_name) for event_name in expected_log_events]
    assert positions == sorted(positions)
    assert "开始执行真实工具调用" not in log_text


def test_search_evidence_reuse_main_chain_should_resolve_without_second_executor_call(caplog) -> None:
    query = "漳州南靖土楼从厦门出发的交通方式"
    facts: list[SandboxFactRecord] = []
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=facts, evidence=evidence_repo)
    ledger_service = _ledger_service(uow_factory=uow_factory)
    context_service = RuntimeContextService(
        evidence_context_provider=EvidenceRuntimeContextProvider(
            ledger_service=ledger_service,
            projector=EvidenceDigestProjector(uow_factory=uow_factory),
        )
    )
    persistence = _InMemoryToolEventPersistence(uow_factory=uow_factory)
    tool = _SearchMainChainTool()

    step1 = Step(id="step-1", description="搜索漳州南靖土楼从厦门出发的交通方式")
    first_message, first_tool_events = asyncio.run(execute_step_with_prompt(
        llm=_FakeToolCallLLM("search_web", {"query": query}),
        step=step1,
        runtime_tools=[tool],
        max_tool_iterations=1,
        task_mode="research",
        user_message=query,
    ))

    assert tool.invocations == [("search_web", {"query": query})]
    called_events = [
        event for event in first_tool_events
        if event.status == ToolEventStatus.CALLED and event.function_name == "search_web"
    ]
    assert len(called_events) == 1
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is True

    step1.status = ExecutionStatus.COMPLETED
    step1.outcome = StepOutcome(done=True, summary=str(first_message.get("summary") or "search completed"))
    first_state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
    }
    asyncio.run(persist_current_step_tool_events_before_evidence_gate(
        state=first_state,
        step=step1,
        tool_events=first_tool_events,
        runtime_tool_event_persistence=persistence,
    ))

    assert len(persistence.persisted_events) == 1
    assert len(facts) == 1
    assert facts[0].fact_kind == SandboxFactKind.SEARCH_RESULT
    assert facts[0].payload["query_hash"] == hash_query(query)
    assert facts[0].payload["result_count"] > 0

    asyncio.run(reconcile_step_evidence_before_state_return(
        state=first_state,
        step=step1,
        reconciler=ledger_service,
    ))

    search_evidence = [
        record for record in evidence_repo.records
        if record.evidence_kind == EvidenceKind.SEARCH_EVIDENCE
    ]
    assert len(search_evidence) == 1
    assert search_evidence[0].support_level == EvidenceSupportLevel.STRONG
    assert search_evidence[0].quality_status == EvidenceQualityStatus.VALID

    digest = asyncio.run(EvidenceDigestProjector(uow_factory=uow_factory).build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
        stage="execute",
    ))
    assert digest is not None
    assert digest.result_handles
    assert digest.do_not_repeat
    assert digest.do_not_repeat[0].duplicate_decision == (
        EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION
    )

    state_base = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "user_message": "下一步再次确认同一个问题，不要换查询主题。",
    }
    step2 = Step(id="step-2", description="再次确认同一个交通方式问题")
    state = {
        **state_base,
        "plan": Plan(steps=[step1, step2]),
        "step_states": [{"step_id": "step-1", "status": "completed"}],
        "graph_metadata": {},
    }
    resolver = _SpyEvidenceResultHandleResolver(EvidenceResultHandleResolver(uow_factory=uow_factory))

    with caplog.at_level(logging.INFO):
        next_state = asyncio.run(execute_step_node(
            state,
            _FakeToolCallLLM("search_web", {"query": query}),
            context_service,
            evidence_result_handle_resolver=resolver,
            runtime_tools=[tool],
            max_tool_iterations=1,
        ))

    assert resolver.called is True
    assert tool.invocations == [("search_web", {"query": query})]

    prepared = asyncio.run(prepare_execute_step_input(
        state={
            **state_base,
            "plan": Plan(steps=[step1, Step(id="step-2", description="再次确认同一个交通方式问题")]),
            "step_states": [{"step_id": "step-1", "status": "completed"}],
            "graph_metadata": {},
        },
        step=Step(id="step-2", description="再次确认同一个交通方式问题"),
        llm=object(),
        runtime_context_service=context_service,
        task_mode="general",
        user_message=state_base["user_message"],
    ))
    live_events = []

    async def capture_live_event(event):
        live_events.append(event)

    direct_resolver = _SpyEvidenceResultHandleResolver(EvidenceResultHandleResolver(uow_factory=uow_factory))
    direct_payload, direct_tool_events = asyncio.run(execute_step_with_prompt(
        llm=_FakeToolCallLLM("search_web", {"query": query}),
        step=Step(id="step-2", description="再次确认同一个交通方式问题"),
        runtime_tools=[tool],
        max_tool_iterations=1,
        task_mode="general",
        on_tool_event=capture_live_event,
        user_content=prepared.user_content,
        user_message=prepared.user_message,
        runtime_evidence_context=prepared.runtime_evidence_context,
        has_previous_completed_steps=prepared.has_previous_completed_steps,
        evidence_result_handle_resolver=direct_resolver,
        evidence_resolution_state={
            **state_base,
            "current_step_id": "step-2",
        },
    ))

    assert direct_resolver.called is True
    assert direct_payload["success"] is True
    assert direct_payload["loop_break_reason"] == "evidence_reuse_allowed"
    assert tool.invocations == [("search_web", {"query": query})]
    live_called_events = [
        event for event in live_events
        if getattr(event, "type", "") == "tool"
           and event.status == ToolEventStatus.CALLED
           and event.function_name == "search_web"
    ]
    assert len(live_called_events) == 1
    assert live_called_events[0].function_result.success is True
    assert live_called_events[0].function_result.data["result_handle_resolved"] is True
    assert not [
        event for event in live_events
        if getattr(event, "type", "") == "tool"
           and event.status == ToolEventStatus.CALLED
           and event.function_name == "search_web"
           and event.function_result is not None
           and event.function_result.success is False
           and event.function_result.data.get("duplicate_decision") == "reuse_existing_evidence_pending_resolution"
    ]
    assert direct_tool_events == live_events

    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert next_state["last_executed_step"].outcome.done is True
    assert "技术问题" not in next_state["last_executed_step"].outcome.summary
    second_called_events = [
        event for event in next_state["emitted_events"]
        if getattr(event, "type", "") == "tool"
           and event.status == ToolEventStatus.CALLED
           and event.function_name == "search_web"
    ]
    assert len(second_called_events) == 1
    assert second_called_events[0].function_result.success is True
    assert second_called_events[0].function_result.data["result_handle_resolved"] is True

    log_text = caplog.text
    assert "reuse_existing_evidence_pending_resolution" in log_text
    assert "evidence_result_handle_resolved" in log_text
    assert "evidence_reuse_virtual_tool_result_returned" in log_text
    assert "evidence_reuse_snapshot_missing" not in log_text
    assert "开始执行真实工具调用" not in log_text


def test_fetch_page_reuse_main_chain_should_resolve_without_second_executor_call(caplog) -> None:
    url = "http://zhangzhou.bendibao.com/tour/2025125/17804.shtm"
    url_hash = hash_url(url)
    facts: list[SandboxFactRecord] = []
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=facts, evidence=evidence_repo)
    ledger_service = _ledger_service(uow_factory=uow_factory)
    context_service = RuntimeContextService(
        evidence_context_provider=EvidenceRuntimeContextProvider(
            ledger_service=ledger_service,
            projector=EvidenceDigestProjector(uow_factory=uow_factory),
        )
    )
    persistence = _InMemoryToolEventPersistence(uow_factory=uow_factory)
    tool = _FetchPageMainChainTool()

    step1 = Step(id="step-1", description="读取指定 URL 页面内容", task_mode_hint="web_reading")
    first_message, first_tool_events = asyncio.run(execute_step_with_prompt(
        llm=_FakeToolCallLLM("fetch_page", {"url": url}),
        step=step1,
        runtime_tools=[tool],
        max_tool_iterations=1,
        task_mode="web_reading",
        user_message=f"读取这个页面：{url}",
    ))

    assert tool.invocations == [("fetch_page", {"url": url})]
    first_called_events = [
        event for event in first_tool_events
        if event.status == ToolEventStatus.CALLED and event.function_name == "fetch_page"
    ]
    assert len(first_called_events) == 1
    assert first_called_events[0].function_result.success is True

    step1.status = ExecutionStatus.COMPLETED
    step1.outcome = StepOutcome(done=True, summary=str(first_message.get("summary") or "page completed"))
    first_state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
    }
    asyncio.run(persist_current_step_tool_events_before_evidence_gate(
        state=first_state,
        step=step1,
        tool_events=first_tool_events,
        runtime_tool_event_persistence=persistence,
    ))

    assert len(persistence.persisted_events) == 1
    assert len(facts) == 1
    assert facts[0].fact_kind == SandboxFactKind.FETCHED_PAGE
    assert facts[0].payload["fetched_url_hash"] == url_hash

    asyncio.run(reconcile_step_evidence_before_state_return(
        state=first_state,
        step=step1,
        reconciler=ledger_service,
    ))

    page_evidence = [
        record for record in evidence_repo.records
        if record.evidence_kind == EvidenceKind.PAGE_EVIDENCE
    ]
    assert len(page_evidence) == 1
    assert page_evidence[0].action_key == f"fetch:{url_hash}"
    assert page_evidence[0].subject_key == f"page:{url_hash}"
    assert page_evidence[0].reuse_policy == EvidenceReusePolicy.REUSE_ALLOWED
    assert page_evidence[0].staleness_policy == EvidenceStalenessPolicy.RUN_SCOPED
    assert page_evidence[0].support_level == EvidenceSupportLevel.STRONG
    assert page_evidence[0].quality_status == EvidenceQualityStatus.VALID

    digest = asyncio.run(EvidenceDigestProjector(uow_factory=uow_factory).build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
        stage="execute",
    ))
    assert digest is not None
    assert digest.source_step_ids == ["step-1"]
    assert digest.result_handles
    assert digest.do_not_repeat
    assert digest.do_not_repeat[0].duplicate_decision == (
        EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION
    )

    step2 = Step(id="step-2", description="再次读取同一个 URL 确认内容", task_mode_hint="web_reading")
    state_base = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "user_message": f"再次读取同一个 URL 确认内容，不要更换 URL：{url}",
    }
    state = {
        **state_base,
        "plan": Plan(steps=[step1, step2]),
        "step_states": [{"step_id": "step-1", "status": "completed"}],
        "graph_metadata": {},
    }
    resolver = _SpyEvidenceResultHandleResolver(EvidenceResultHandleResolver(uow_factory=uow_factory))

    with caplog.at_level(logging.INFO):
        next_state = asyncio.run(execute_step_node(
            state,
            _FakeToolCallLLM("fetch_page", {"url": url}),
            context_service,
            evidence_result_handle_resolver=resolver,
            runtime_tools=[tool],
            max_tool_iterations=1,
        ))

    assert resolver.called is True
    assert tool.invocations == [("fetch_page", {"url": url})]
    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert next_state["last_executed_step"].outcome.done is True
    second_called_events = [
        event for event in next_state["emitted_events"]
        if getattr(event, "type", "") == "tool"
           and event.status == ToolEventStatus.CALLED
           and event.function_name == "fetch_page"
    ]
    assert len(second_called_events) == 1
    assert second_called_events[0].function_result.success is True
    assert second_called_events[0].function_result.data["result_handle_resolved"] is True

    prepared = asyncio.run(prepare_execute_step_input(
        state={
            **state_base,
            "plan": Plan(steps=[step1, Step(id="step-2", description="再次读取同一个 URL 确认内容", task_mode_hint="web_reading")]),
            "step_states": [{"step_id": "step-1", "status": "completed"}],
            "graph_metadata": {},
        },
        step=Step(id="step-2", description="再次读取同一个 URL 确认内容", task_mode_hint="web_reading"),
        llm=object(),
        runtime_context_service=context_service,
        task_mode="web_reading",
        user_message=state_base["user_message"],
    ))
    live_events = []

    async def capture_live_event(event):
        live_events.append(event)

    direct_resolver = _SpyEvidenceResultHandleResolver(EvidenceResultHandleResolver(uow_factory=uow_factory))
    direct_payload, direct_tool_events = asyncio.run(execute_step_with_prompt(
        llm=_FakeToolCallLLM("fetch_page", {"url": url}),
        step=Step(id="step-2", description="再次读取同一个 URL 确认内容", task_mode_hint="web_reading"),
        runtime_tools=[tool],
        max_tool_iterations=1,
        task_mode="web_reading",
        on_tool_event=capture_live_event,
        user_content=prepared.user_content,
        user_message=prepared.user_message,
        runtime_evidence_context=prepared.runtime_evidence_context,
        has_previous_completed_steps=prepared.has_previous_completed_steps,
        evidence_result_handle_resolver=direct_resolver,
        evidence_resolution_state={
            **state_base,
            "current_step_id": "step-2",
        },
    ))

    assert direct_resolver.called is True
    assert direct_payload["success"] is True
    assert direct_payload["loop_break_reason"] == "evidence_reuse_allowed"
    assert tool.invocations == [("fetch_page", {"url": url})]
    live_called_events = [
        event for event in live_events
        if getattr(event, "type", "") == "tool"
           and event.status == ToolEventStatus.CALLED
           and event.function_name == "fetch_page"
    ]
    assert len(live_called_events) == 1
    assert live_called_events[0].function_result.success is True
    assert live_called_events[0].function_result.data["result_handle_resolved"] is True
    assert not [
        event for event in live_events
        if getattr(event, "type", "") == "tool"
           and event.status == ToolEventStatus.CALLED
           and event.function_name == "fetch_page"
           and event.function_result is not None
           and event.function_result.success is False
           and event.function_result.data.get("duplicate_decision") == "reuse_existing_evidence_pending_resolution"
    ]
    assert direct_tool_events == live_events

    log_text = caplog.text
    assert "reuse_existing_evidence_pending_resolution" in log_text
    assert "evidence_result_handle_resolution_started" in log_text
    assert "evidence_result_handle_resolved" in log_text
    assert "evidence_reuse_virtual_tool_result_returned" in log_text
    assert "evidence_reuse_snapshot_missing" not in log_text
    assert "browser_extract_main_content" not in log_text
    assert "browser_view" not in log_text
    assert "开始执行真实工具调用" not in log_text


def test_web_reading_fetch_page_success_should_stop_before_no_tool_completion() -> None:
    url = "http://zhangzhou.bendibao.com/tour/2025125/17804.shtm"
    tool = _FetchPageMainChainTool()
    llm = _FakeToolCallLLM("fetch_page", {"url": url})

    payload, tool_events = asyncio.run(execute_step_with_prompt(
        llm=llm,
        step=Step(id="step-1", description="读取指定 URL 页面内容", task_mode_hint="web_reading"),
        runtime_tools=[tool],
        max_tool_iterations=3,
        task_mode="web_reading",
        user_message=f"读取这个页面：{url}",
    ))

    assert llm.calls == 1
    assert tool.invocations == [("fetch_page", {"url": url})]
    assert payload["success"] is False
    assert payload["loop_break_reason"] == "page_evidence_pending_gate"
    assert payload["data"]["reason_code"] == "page_evidence_pending_gate"
    assert len([
        event for event in tool_events
        if event.status == ToolEventStatus.CALLED and event.function_name == "fetch_page"
    ]) == 1


def test_web_reading_fetch_page_model_result_should_stop_before_no_tool_completion() -> None:
    url = "http://zhangzhou.bendibao.com/tour/2025125/17804.shtm"
    tool = _FetchPageModelMainChainTool()
    llm = _FakeToolCallLLM("fetch_page", {"url": url})

    payload, tool_events = asyncio.run(execute_step_with_prompt(
        llm=llm,
        step=Step(id="step-1", description="读取指定 URL 页面内容", task_mode_hint="web_reading"),
        runtime_tools=[tool],
        max_tool_iterations=3,
        task_mode="web_reading",
        user_message=f"读取这个页面：{url}",
    ))

    assert llm.calls == 1
    assert tool.invocations == [("fetch_page", {"url": url})]
    assert payload["loop_break_reason"] == "page_evidence_pending_gate"
    called_events = [
        event for event in tool_events
        if event.status == ToolEventStatus.CALLED and event.function_name == "fetch_page"
    ]
    assert len(called_events) == 1
    assert isinstance(called_events[0].function_result.data, FetchedPage)
    fact_inputs = SandboxFactToolEventProjector(
        ledger_service=SandboxFactLedgerService(uow_factory=lambda: _UoW(facts=[], evidence=_EvidenceRepo())),
    )._build_fact_inputs(context=SandboxFactProjectionContext(
        scope=_scope("step-1"),
        profile_ref=SandboxFactProfileRef(status="missing"),
        source_event_id="event-1",
        current_step_id="step-1",
    ), event=called_events[0])
    assert len(fact_inputs) == 1
    assert fact_inputs[0].fact_kind == SandboxFactKind.FETCHED_PAGE


def test_fetch_page_gate_should_complete_failed_web_reading_then_reuse_next_step(caplog) -> None:
    url = "http://zhangzhou.bendibao.com/tour/2025125/17804.shtm"
    facts: list[SandboxFactRecord] = []
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=facts, evidence=evidence_repo)
    ledger_service = _ledger_service(uow_factory=uow_factory)
    context_service = RuntimeContextService(
        evidence_context_provider=EvidenceRuntimeContextProvider(
            ledger_service=ledger_service,
            projector=EvidenceDigestProjector(uow_factory=uow_factory),
        )
    )
    tool = _FetchPageMainChainTool()
    persistence = _InMemoryToolEventPersistence(uow_factory=uow_factory)
    step1 = Step(id="step-1", description="读取指定 URL 页面内容", task_mode_hint="web_reading")
    step2 = Step(id="step-2", description="再次读取同一个 URL 确认内容", task_mode_hint="web_reading")
    base_state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "user_message": f"读取这个页面，之后再次读取同一个 URL 确认内容：{url}",
        "plan": Plan(steps=[step1, step2]),
        "step_states": [],
        "graph_metadata": {},
    }

    with caplog.at_level(logging.INFO):
        first_state = asyncio.run(execute_step_node(
            base_state,
            first_llm := _FakeToolCallLLM("fetch_page", {"url": url}),
            context_service,
            evidence_result_handle_resolver=EvidenceResultHandleResolver(uow_factory=uow_factory),
            runtime_tools=[tool],
            max_tool_iterations=1,
            evidence_step_reconciler=ledger_service,
            runtime_tool_event_persistence=persistence,
        ))

    assert first_llm.calls == 1
    assert tool.invocations == [("fetch_page", {"url": url})]
    assert first_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert first_state["last_executed_step"].outcome.done is True
    assert first_state["last_executed_step"].outcome.blockers == []
    first_log_text = caplog.text
    assert "page_evidence_pending_gate" in first_log_text
    assert "未调用工具直接完成当前轮次" not in first_log_text
    assert first_log_text.index("tool_event_fact_projection_completed_before_evidence_gate") < first_log_text.index(
        "evidence_step_completion_gate_reconciled"
    )
    assert any(record.evidence_kind == EvidenceKind.PAGE_EVIDENCE for record in evidence_repo.records)
    assert any(fact.fact_kind == SandboxFactKind.FETCHED_PAGE for fact in facts)
    digest_after_step1 = asyncio.run(EvidenceDigestProjector(uow_factory=uow_factory).build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
        stage="execute",
    ))
    assert digest_after_step1 is not None

    next_plan = first_state["plan"]
    state_for_second = {
        **first_state,
        "plan": next_plan,
        "step_states": [{"step_id": "step-1", "status": "completed"}],
    }
    resolver = _SpyEvidenceResultHandleResolver(EvidenceResultHandleResolver(uow_factory=uow_factory))

    with caplog.at_level(logging.INFO):
        second_state = asyncio.run(execute_step_node(
            state_for_second,
            _FakeToolCallLLM("fetch_page", {"url": url}),
            context_service,
            evidence_result_handle_resolver=resolver,
            runtime_tools=[tool],
            max_tool_iterations=1,
            evidence_step_reconciler=ledger_service,
            runtime_tool_event_persistence=persistence,
        ))

    assert resolver.called is True
    assert tool.invocations == [("fetch_page", {"url": url})]
    assert second_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert second_state["last_executed_step"].outcome.done is True
    assert [fact.fact_kind for fact in facts] == [SandboxFactKind.FETCHED_PAGE]
    assert [record.evidence_kind for record in evidence_repo.records].count(EvidenceKind.PAGE_EVIDENCE) == 1
    assert persistence.persisted_events[-1].step_id == "step-2"
    assert persistence.persisted_events[-1].runtime_fact_projection["fact_count"] == 0
    second_called_events = [
        event for event in second_state["emitted_events"]
        if getattr(event, "type", "") == "tool"
           and event.status == ToolEventStatus.CALLED
           and event.function_name == "fetch_page"
    ]
    assert second_called_events
    assert second_called_events[-1].function_result.success is True
    assert second_called_events[-1].function_result.data["result_handle_resolved"] is True
    assert second_called_events[-1].runtime_fact_projection["fact_count"] == 0

    final_digest = asyncio.run(EvidenceDigestProjector(uow_factory=uow_factory).build_digest(
        scope=_scope("step-3"),
        current_step_id="step-3",
        completed_step_ids=["step-1", "step-2"],
        stage="execute",
    ))
    assert final_digest is not None
    assert [
        item for item in final_digest.do_not_repeat
        if item.subject_key == f"page:{hash_url(url)}"
    ] == [
        digest_item for digest_item in digest_after_step1.do_not_repeat
        if digest_item.subject_key == f"page:{hash_url(url)}"
    ]
    assert [
        handle for handle in final_digest.result_handles
        if handle.subject_key == f"page:{hash_url(url)}"
    ] == [
        handle for handle in digest_after_step1.result_handles
        if handle.subject_key == f"page:{hash_url(url)}"
    ]

    log_text = caplog.text
    assert "reuse_existing_evidence_pending_resolution" in log_text
    assert "evidence_reuse_virtual_tool_result_returned" in log_text
    assert "evidence_reuse_snapshot_missing" not in log_text
    assert '事件="研究链路状态已写回执行态"' not in log_text
    assert "consecutive_failure_count=1" not in log_text


def test_fetch_page_gate_should_not_complete_partial_web_reading() -> None:
    url = "http://zhangzhou.bendibao.com/tour/2025125/17804.shtm"
    facts: list[SandboxFactRecord] = []
    evidence_repo = _EvidenceRepo()
    uow_factory = lambda: _UoW(facts=facts, evidence=evidence_repo)
    ledger_service = _ledger_service(uow_factory=uow_factory)
    context_service = RuntimeContextService(
        evidence_context_provider=EvidenceRuntimeContextProvider(
            ledger_service=ledger_service,
            projector=EvidenceDigestProjector(uow_factory=uow_factory),
        )
    )
    tool = _FetchPageMainChainTool(content="页面只返回了截断片段", is_truncated=True)
    persistence = _InMemoryToolEventPersistence(uow_factory=uow_factory)
    step1 = Step(id="step-1", description="读取指定 URL 页面内容", task_mode_hint="web_reading")
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "user_message": f"读取这个页面：{url}",
        "plan": Plan(steps=[step1]),
        "step_states": [],
        "graph_metadata": {},
    }

    next_state = asyncio.run(execute_step_node(
        state,
        _FakeToolCallLLM("fetch_page", {"url": url}),
        context_service,
        evidence_result_handle_resolver=EvidenceResultHandleResolver(uow_factory=uow_factory),
        runtime_tools=[tool],
        max_tool_iterations=1,
        evidence_step_reconciler=ledger_service,
        runtime_tool_event_persistence=persistence,
    ))

    assert tool.invocations == [("fetch_page", {"url": url})]
    assert next_state["last_executed_step"].status == ExecutionStatus.FAILED
    assert next_state["last_executed_step"].outcome.done is False
    page_records = [record for record in evidence_repo.records if record.evidence_kind == EvidenceKind.PAGE_EVIDENCE]
    assert not [
        record for record in page_records
        if record.support_level == EvidenceSupportLevel.STRONG
           and record.quality_status == EvidenceQualityStatus.VALID
    ]


def test_execute_step_node_should_run_controlled_page_verification_before_task_mode_policy(caplog) -> None:
    url = "https://example.com/a"
    url_hash = hash_url(url)
    fact = _fact(
        SandboxFactKind.FETCHED_PAGE,
        step_id="step-1",
        payload={
            **_payload_for_fact(SandboxFactKind.FETCHED_PAGE),
            "fetched_url_hash": url_hash,
            "final_url_origin": url,
        },
    )
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
    step2 = Step(id="step-2", description="再次读取页面", task_mode_hint="web_reading")
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "user_message": "再次读取页面",
        "plan": Plan(steps=[step1, step2]),
        "step_states": [{"step_id": "step-1", "status": "completed"}],
        "graph_metadata": {},
    }
    tool = _CountingRuntimeTool({"fetch_page"})

    with caplog.at_level(logging.INFO):
        next_state = asyncio.run(execute_step_node(
            state,
            _FakeToolCallLLM("fetch_page", {"url": url}),
            context_service,
            evidence_result_handle_resolver=EvidenceResultHandleResolver(uow_factory=uow_factory),
            runtime_tools=[tool],
            max_tool_iterations=1,
        ))

    assert tool.invocations == []
    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert next_state["last_executed_step"].outcome.done is True
    evidence_kinds = [record.evidence_kind for record in evidence_repo.records]
    assert evidence_kinds.count(EvidenceKind.PAGE_EVIDENCE) == 1
    assert EvidenceKind.TOOL_FAILURE_EVIDENCE not in evidence_kinds
    log_text = caplog.text
    expected_log_events = [
        "runtime_evidence_context_built",
        "evidence_reuse_snapshot_attached_to_guard",
        "reuse_existing_evidence_pending_resolution",
        "evidence_result_handle_resolved",
        "evidence_reuse_virtual_tool_result_returned",
    ]
    positions = [log_text.index(event_name) for event_name in expected_log_events]
    assert positions == sorted(positions)
    assert "开始执行真实工具调用" not in log_text
    assert "task_mode_tool_blocked" not in log_text


def test_execute_step_node_should_resolve_run_scoped_page_reuse_without_task_mode_policy(caplog) -> None:
    url = "https://example.com/a"
    url_hash = hash_url(url)
    result_ref = EvidenceResultRef(
        result_ref_type=EvidenceResultRefType.FACT_REF,
        ref_id="fact-1",
        source_step_id="step-1",
        source_evidence_id="evidence-1",
        source_fact_id="fact-1",
        source_event_id="event-1",
        subject_key=f"page:{url_hash}",
        payload_hash="sha256:payload",
        quality_status=EvidenceQualityStatus.VALID,
        support_level=EvidenceSupportLevel.STRONG,
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        read_strategy=EvidenceReadStrategy.READ_FACT_PAYLOAD,
        summary="safe summary",
    )
    runtime_context = _runtime_context_for_ref(
        result_ref=result_ref,
        action_key=f"fetch:{url_hash}",
        subject_key=f"page:{url_hash}",
    )
    context_service = RuntimeContextService(evidence_context_provider=_Provider(runtime_context))
    step1 = Step(id="step-1", status=ExecutionStatus.COMPLETED)
    step2 = Step(id="step-2", description="再次读取页面", task_mode_hint="web_reading")
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "user_message": "再次读取页面",
        "plan": Plan(steps=[step1, step2]),
        "step_states": [{"step_id": "step-1", "status": "completed"}],
        "graph_metadata": {},
    }
    tool = _CountingRuntimeTool({"fetch_page"})
    resolver = _Resolver(runtime_context.result_handles[0])

    with caplog.at_level(logging.INFO):
        next_state = asyncio.run(execute_step_node(
            state,
            _FakeToolCallLLM("fetch_page", {"url": url}),
            context_service,
            evidence_result_handle_resolver=resolver,
            runtime_tools=[tool],
            max_tool_iterations=1,
        ))

    assert resolver.called is True
    assert tool.invocations == []
    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    log_text = caplog.text
    expected_log_events = [
        "runtime_evidence_context_built",
        "evidence_reuse_snapshot_attached_to_guard",
        "reuse_existing_evidence_pending_resolution",
        "evidence_result_handle_resolution_started",
        "evidence_result_handle_resolved",
        "evidence_reuse_virtual_tool_result_returned",
    ]
    positions = [log_text.index(event_name) for event_name in expected_log_events]
    assert positions == sorted(positions)
    assert "task_mode_tool_blocked" not in log_text
    assert "开始执行真实工具调用" not in log_text


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
    resolver = _Resolver(handle)
    tool = _CountingRuntimeTool({"read_file"})

    payload, tool_events = asyncio.run(execute_step_with_prompt(
        llm=_FakeToolCallLLM("read_file", {"path": "/workspace/a.txt"}),
        step=Step(id="step-2", description="读取文件"),
        runtime_tools=[tool],
        max_tool_iterations=1,
        task_mode="general",
        runtime_evidence_context=runtime_context,
        has_previous_completed_steps=True,
        evidence_result_handle_resolver=resolver,
        evidence_resolution_state={
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "current_step_id": "step-2",
        },
    ))

    assert resolver.called is True
    assert tool.invocations == []
    assert payload["success"] is True
    assert payload["loop_break_reason"] == "evidence_reuse_allowed"
    assert payload["data"]["result_handle_resolved"] is True
    called_events = [
        event for event in tool_events
        if event.status == ToolEventStatus.CALLED and event.function_name == "read_file"
    ]
    assert len(called_events) == 1
    assert called_events[0].function_result.success is True
    assert called_events[0].function_result.data["result_handle_id"] == handle.result_handle_id


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


def test_graph_completion_gate_should_reconcile_before_next_step_context() -> None:
    reconciler = _ProjectionStepReconciler()
    step = Step(id="step-1", outcome=StepOutcome(done=True, summary="done"))
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
    }

    asyncio.run(reconcile_step_evidence_before_state_return(
        state=state,
        step=step,
        reconciler=reconciler,
    ))

    assert reconciler.reconcile_calls[0]["scope"].current_step_id == "step-1"
    assert step.outcome is not None
    assert step.outcome.facts_learned == ["evidence backed fact"]
    assert step.outcome.evidence_reconcile_metadata["graph_completion_gate"] is True


def test_execute_helper_should_persist_tool_event_facts_before_graph_gate() -> None:
    persistence = _RuntimeToolEventPersistenceSpy()
    event = ToolEvent(
        id="tool-event-1",
        step_id="step-1",
        tool_call_id="call-1",
        tool_name="search",
        function_name="search_web",
        function_args={"query": "漳州南靖土楼从厦门出发的交通方式"},
        function_result=ToolResult(success=True, data={"results": []}),
        status=ToolEventStatus.CALLED,
    )

    asyncio.run(persist_current_step_tool_events_before_evidence_gate(
        state={
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
        },
        step=Step(id="step-1"),
        tool_events=[event],
        runtime_tool_event_persistence=persistence,
    ))

    assert len(persistence.calls) == 1
    call = persistence.calls[0]
    assert call["run_id"] == "run-1"
    assert call["session_id"] == "session-1"
    assert call["current_step_id"] == "step-1"
    assert call["event"] is event
    assert event.runtime_fact_projection["graph_main_chain"] is True


@pytest.mark.parametrize(
    ("state_overrides", "step"),
    [
        ({"run_id": ""}, Step(id="step-1")),
        ({"session_id": ""}, Step(id="step-1")),
        ({}, Step(id="")),
    ],
)
def test_execute_helper_should_fail_closed_when_tool_event_scope_missing(state_overrides, step) -> None:
    persistence = _RuntimeToolEventPersistenceSpy()
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        **state_overrides,
    }

    with pytest.raises(ValueError, match="ToolEvent fact 投影缺少 run_id/session_id/current_step_id"):
        asyncio.run(persist_current_step_tool_events_before_evidence_gate(
            state=state,
            step=step,
            tool_events=[
                ToolEvent(
                    id="tool-event-1",
                    step_id=str(step.id or "") or None,
                    tool_call_id="call-1",
                    tool_name="search",
                    function_name="search_web",
                    function_args={"query": "q"},
                    function_result=ToolResult(success=True, data={}),
                    status=ToolEventStatus.CALLED,
                )
            ],
            runtime_tool_event_persistence=persistence,
        ))

    assert persistence.calls == []


def test_execute_helper_should_fail_closed_when_tool_event_step_id_mismatch() -> None:
    persistence = _RuntimeToolEventPersistenceSpy()

    with pytest.raises(ValueError, match="ToolEvent step_id 与当前 step 不一致"):
        asyncio.run(persist_current_step_tool_events_before_evidence_gate(
            state={
                "user_id": "user-1",
                "session_id": "session-1",
                "workspace_id": "workspace-1",
                "run_id": "run-1",
            },
            step=Step(id="step-1"),
            tool_events=[
                ToolEvent(
                    id="tool-event-1",
                    step_id="other-step",
                    tool_call_id="call-1",
                    tool_name="search",
                    function_name="search_web",
                    function_args={"query": "q"},
                    function_result=ToolResult(success=True, data={}),
                    status=ToolEventStatus.CALLED,
                )
            ],
            runtime_tool_event_persistence=persistence,
        ))

    assert persistence.calls == []


def test_execute_step_node_should_return_graph_gate_marker_in_state_and_completed_event() -> None:
    reconciler = _ProjectionStepReconciler()
    plan = Plan(
        steps=[
            Step(id="step-1", title="完成步骤", description="完成步骤"),
            Step(id="step-2", title="下一步", description="下一步"),
        ]
    )
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "user_message": "完成第一步",
        "plan": plan,
        "step_states": [],
        "graph_metadata": {},
        "working_memory": {},
        "emitted_events": [],
    }

    next_state = asyncio.run(execute_step_node(
        state,
        _FinalJsonLLM({"success": True, "summary": "done"}),
        RuntimeContextService(),
        runtime_tools=[],
        evidence_step_reconciler=reconciler,
    ))

    completed_step = next_state["plan"].steps[0]
    last_step = next_state["last_executed_step"]
    assert completed_step.outcome.evidence_reconcile_metadata["graph_completion_gate"] is True
    assert last_step.outcome.evidence_reconcile_metadata["graph_completion_gate"] is True
    assert AgentTaskRunner._graph_evidence_already_reconciled(last_step) is True
    assert next_state["step_states"][0]["outcome"]["evidence_reconcile_metadata"] == {
        "graph_completion_gate": True,
    }


def test_normalize_step_outcome_payload_should_preserve_graph_gate_marker() -> None:
    outcome = StepOutcome(
        done=True,
        summary="done",
        evidence_reconcile_metadata={"graph_completion_gate": True},
    )

    normalized = normalize_step_outcome_payload(outcome)
    restored = StepOutcome.model_validate(normalized)

    assert normalized["evidence_reconcile_metadata"] == {"graph_completion_gate": True}
    assert restored.evidence_reconcile_metadata == {"graph_completion_gate": True}
    assert GraphStateContractMapper._normalize_step_outcome_state(outcome)[
        "evidence_reconcile_metadata"
    ] == {"graph_completion_gate": True}


def test_normalize_step_outcome_payload_should_drop_invalid_graph_gate_marker() -> None:
    normalized = normalize_step_outcome_payload(
        {
            "done": True,
            "summary": "done",
            "evidence_reconcile_metadata": "bad",
        }
    )

    assert normalized["evidence_reconcile_metadata"] == {}


def test_agent_task_runner_should_skip_reconcile_when_graph_gate_already_reconciled() -> None:
    step = Step(
        id="step-1",
        outcome=StepOutcome(
            done=True,
            summary="done",
            evidence_reconcile_metadata={"graph_completion_gate": True},
        ),
    )
    event = StepEvent(step=step, status="completed")
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _RunnerUoW()
    runner._evidence_step_reconciler = _StepReconciler()

    asyncio.run(runner._reconcile_evidence_before_step_completed(event))

    assert runner._evidence_step_reconciler.calls == []


def test_agent_task_runner_should_skip_tool_fact_projection_when_graph_already_projected() -> None:
    event = ToolEvent(
        id="tool-event-1",
        step_id="step-1",
        tool_call_id="call-1",
        tool_name="search",
        function_name="search_web",
        function_args={"query": "q"},
        function_result=ToolResult(success=True, data={"results": []}),
        status=ToolEventStatus.CALLED,
        runtime_fact_projection={"graph_main_chain": True},
    )
    runner = object.__new__(AgentTaskRunner)
    runner._sandbox_fact_recorder = object()
    runner._sandbox_fact_context_builder = object()
    runner._sandbox_fact_event_projector = None

    asyncio.run(runner._record_sandbox_facts_for_tool_event(
        task=None,
        event=event,
        source_event_id="tool-event-1",
    ))

    assert AgentTaskRunner._graph_tool_event_fact_projected(event) is True


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
    return _runtime_context_for_ref(
        result_ref=_result_ref(),
        action_key="file_read:/workspace/a.txt",
        subject_key="file:/workspace/a.txt",
    )


def _runtime_context_for_ref(
        *,
        result_ref: EvidenceResultRef,
        action_key: str,
        subject_key: str,
):
    handle = build_evidence_result_handle(result_ref)
    do_not_repeat = _do_not_repeat(
        handle.result_handle_id,
        result_ref=result_ref,
        action_key=action_key,
        subject_key=subject_key,
    )
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


def _do_not_repeat(
        result_handle_id: str,
        *,
        result_ref: EvidenceResultRef | None = None,
        action_key: str = "file_read:/workspace/a.txt",
        subject_key: str = "file:/workspace/a.txt",
):
    from app.domain.models.evidence import EvidenceDoNotRepeatResult
    reuse_result_ref = result_ref or _result_ref()

    return EvidenceDoNotRepeatResult(
        action_key=action_key,
        subject_key=subject_key,
        reason_code="evidence_reuse_pending_resolution",
        source_step_id="step-1",
        evidence_ids=["evidence-1"],
        reuse_policy=EvidenceReusePolicy.REUSE_ALLOWED,
        staleness_policy=EvidenceStalenessPolicy.RUN_SCOPED,
        support_level=EvidenceSupportLevel.STRONG,
        quality_status=EvidenceQualityStatus.VALID,
        result_status="successful",
        duplicate_decision=EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION,
        reuse_result_ref=reuse_result_ref,
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


class _FinalJsonLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = dict(payload)
        self.calls = 0

    async def invoke(self, **kwargs):
        self.calls += 1
        return {"content": json.dumps(self.payload, ensure_ascii=False)}


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


class _SearchMainChainTool(BaseTool):
    name = "search-main-chain-tool"

    def __init__(self) -> None:
        super().__init__()
        self.invocations = []

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

    def has_tool(self, function_name: str) -> bool:
        return str(function_name or "").strip().lower() == "search_web"

    @tool(name="search_web", description="search", parameters={}, required=["query"])
    async def search_web(self, query: str):
        self.invocations.append(("search_web", {"query": query}))
        return ToolResult(
            success=True,
            data={
                "query": query,
                "total_results": 0,
                "results": [
                    {
                        "title": "厦门到南靖土楼交通",
                        "url": "https://example.com/xiamen-to-nanjing-tulou",
                        "snippet": "可从厦门乘动车或大巴到南靖，再转乘景区交通前往土楼。",
                    },
                    {
                        "title": "无 URL 结果不进入 top_results",
                        "snippet": "该条只用于确认 result_count 不被 top_results 数量误伤。",
                    },
                ],
            },
        )


class _FetchPageMainChainTool(BaseTool):
    name = "fetch-page-main-chain-tool"

    def __init__(self, *, content: str | None = None, is_truncated: bool = False) -> None:
        super().__init__()
        self.invocations = []
        self.content = (
            content
            if content is not None
            else "从厦门出发可先乘坐动车或大巴到南靖，再换乘土楼景区交通。"
                 "页面还介绍了班次、换乘方式、景区路线和适合一日游的行程安排。" * 4
        )
        self.is_truncated = is_truncated

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "fetch_page",
                    "description": "fetch",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        ]

    def has_tool(self, function_name: str) -> bool:
        return str(function_name or "").strip().lower() == "fetch_page"

    @tool(name="fetch_page", description="fetch", parameters={}, required=["url"])
    async def fetch_page(self, url: str):
        self.invocations.append(("fetch_page", {"url": url}))
        return ToolResult(
            success=True,
            data={
                "url": url,
                "final_url": url,
                "title": "漳州南靖土楼交通指南",
                "content": self.content,
                "is_truncated": self.is_truncated,
                "status_code": 200,
                "content_type": "text/html",
            },
        )


class _FetchPageModelMainChainTool(_FetchPageMainChainTool):
    name = "fetch-page-model-main-chain-tool"

    @tool(name="fetch_page", description="fetch", parameters={}, required=["url"])
    async def fetch_page(self, url: str):
        self.invocations.append(("fetch_page", {"url": url}))
        return ToolResult(
            success=True,
            data=FetchedPage(
                url=url,
                final_url=url,
                title="漳州南靖土楼交通指南",
                content=self.content,
                status_code=200,
                content_type="text/html",
                truncated=self.is_truncated,
            ),
        )


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
    def __init__(self) -> None:
        self.reconcile_calls = []

    async def reconcile_step_evidence(self, *, scope, step):
        self.reconcile_calls.append({"scope": scope, "step": step})
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


class _RuntimeToolEventPersistenceSpy:
    def __init__(self) -> None:
        self.calls = []

    async def persist_tool_event_and_record_facts(
            self,
            *,
            event,
            run_id,
            session_id,
            current_step_id,
    ):
        self.calls.append(
            {
                "event": event,
                "run_id": run_id,
                "session_id": session_id,
                "current_step_id": current_step_id,
            }
        )
        event.runtime_fact_projection = {
            "graph_main_chain": True,
            "source_event_id": event.id,
            "fact_count": 1,
        }

        return type(
            "ToolEventFactProjectionResult",
            (),
            {
                "source_event_id": event.id,
                "fact_count": 1,
                "sandbox_fact_event_persisted": True,
            },
        )()


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
