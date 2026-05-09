import asyncio
import inspect
import logging

from app.application.service.evidence_digest_projector import EvidenceDigestProjector
from app.application.service.evidence_fact_assembler import EvidenceFactAssembler
from app.application.service.evidence_ledger_service import (
    EvidenceLedgerService,
    build_step_evidence_event,
)
from app.application.service.runtime_observation_service import RuntimeObservationService
from app.interfaces.schemas.event import EventMapper
from app.domain.models import (
    EvidenceBackedFactProjection,
    EvidenceEvent,
    EvidenceQualityStatus,
    EvidenceSupportLevel,
    FileToolContent,
    SessionStatus,
    Session,
    Step,
    StepEvent,
    StepOutcome,
    ToolEvent,
    ToolEventStatus,
    Workspace,
    WorkflowRun,
    WorkflowRunEventRecord,
    WorkflowRunStatus,
)
from app.domain.models.evidence import EvidenceKind
from app.domain.models.sandbox_fact import SandboxFactKind
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.infrastructure.models.workflow_run_event import WorkflowRunEventModel
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.contracts import ConstraintInput
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.policies.evidence_reuse_policy import (
    evaluate_evidence_reuse_policy,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import ExecutionState
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.delivery_helpers import (
    _build_reused_step_outcome,
    _merge_step_outcome_into_working_memory,
)

from .test_evidence_ledger_pr3 import (
    _EvidenceRepo,
    _RunnerSessionRepo,
    _RunnerWorkspaceRepo,
    _RunnerWorkflowRunRepo,
    _UoW,
    _execution_context,
    _fact,
    _ledger_service,
    _scope,
)


def test_build_step_evidence_event_should_aggregate_once_without_raw_payload() -> None:
    sandbox_fact = _fact(SandboxFactKind.FILE_READ)
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(facts=[sandbox_fact], evidence=evidence_repo))
    saved = asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))

    event = build_step_evidence_event(step_id="step-1", run_id="run-1", records=saved)

    assert event is not None
    assert event.type == "evidence"
    assert event.step_id == "step-1"
    assert len(event.evidence_refs) == len(saved)
    assert event.source_event_ids == ["event-1"]
    assert event.quality_status_counts[EvidenceQualityStatus.VALID.value] == 1
    assert event.support_level_counts[EvidenceSupportLevel.STRONG.value] == 1
    assert event.gap_count == 0
    payload = event.model_dump(mode="json")
    assert "payload" not in str(payload)
    assert "raw_stdout" not in str(payload)
    assert "full_text" not in str(payload)
    assert "source_event_id" not in payload


def test_build_step_evidence_event_should_emit_single_event_for_multiple_records() -> None:
    facts = [
        _fact(SandboxFactKind.FILE_READ, fact_id="fact-1", step_id="step-1"),
        _fact(SandboxFactKind.FILE_SEARCH, fact_id="fact-2", step_id="step-1"),
    ]
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(facts=facts, evidence=evidence_repo))
    saved = asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))

    event = build_step_evidence_event(step_id="step-1", run_id="run-1", records=saved)

    assert event is not None
    assert event.type == "evidence"
    assert len(saved) == 2
    assert len(event.evidence_refs) == 2
    assert event.summary == "step evidence 对账完成：2 条记录。"


def test_build_step_evidence_event_should_sanitize_sensitive_event_summaries() -> None:
    sandbox_fact = _fact(SandboxFactKind.FILE_READ)
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(facts=[sandbox_fact], evidence=evidence_repo))
    saved = asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    records = [
        saved[0].model_copy(update={"id": "evidence-url", "summary": "来源 https://secret.example.com/a?token=abc"}),
        saved[0].model_copy(update={"id": "evidence-token", "summary": "token=abcdefghi password=hunter2"}),
        saved[0].model_copy(update={"id": "evidence-raw", "summary": "raw stdout: complete shell output"}),
    ]

    event = build_step_evidence_event(step_id="step-1", run_id="run-1", records=records)

    assert event is not None
    serialized = event.model_dump_json()
    assert "https://secret.example.com" not in serialized
    assert "abcdefghi" not in serialized
    assert "hunter2" not in serialized
    assert "raw stdout" not in serialized
    assert "complete shell output" not in serialized
    assert "evidence-url" in serialized
    assert "evidence-token" in serialized
    assert "evidence-raw" in serialized
    assert event.evidence_refs[0].summary == "该 evidence 已持久化，事件摘要包含敏感或原始内容，已省略；请通过 evidence_refs 回查。"


def test_evidence_ledger_service_should_reject_cross_scope_evidence_event_records(caplog) -> None:
    sandbox_fact = _fact(SandboxFactKind.FILE_READ)
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(facts=[sandbox_fact], evidence=evidence_repo))
    saved = asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    base_record = saved[0]

    with caplog.at_level(logging.ERROR):
        results = [
            asyncio.run(service.build_step_evidence_event(
                scope=_scope("step-1"),
                step=Step(id="step-1"),
                records=[base_record.model_copy(update={"step_id": "step-other", "source_step_id": "step-other"})],
            )),
            asyncio.run(service.build_step_evidence_event(
                scope=_scope("step-1"),
                step=Step(id="step-1"),
                records=[base_record.model_copy(update={"run_id": "run-other"})],
            )),
            asyncio.run(service.build_step_evidence_event(
                scope=_scope("step-1"),
                step=Step(id="step-1"),
                records=[base_record.model_copy(update={"session_id": "session-other"})],
            )),
        ]

    assert results == [None, None, None]
    mismatch_logs = [
        record for record in caplog.records
        if record.message == "evidence_event_projection_failed"
        and record.__dict__.get("reason_code") == "evidence_event_scope_mismatch"
    ]
    assert len(mismatch_logs) == 3


def test_agent_task_runner_should_not_persist_evidence_event_for_cross_scope_records(caplog) -> None:
    sandbox_fact = _fact(SandboxFactKind.FILE_READ)
    evidence_repo = _EvidenceRepo()
    service = EvidenceLedgerService(
        uow_factory=lambda: _UoW(facts=[sandbox_fact], evidence=evidence_repo),
        assembler=EvidenceFactAssembler(),
    )
    saved = asyncio.run(service.reconcile_step_evidence(scope=_scope("step-1"), step=Step(id="step-1")))
    workflow_run_repo = _RecordingRunnerWorkflowRunRepo()
    runner = object.__new__(AgentTaskRunner)
    runner._uow_factory = lambda: _RunnerUoW(workflow_run_repo=workflow_run_repo)

    with caplog.at_level(logging.ERROR):
        asyncio.run(runner._persist_evidence_event_before_step_completed(
            reconciler=service,
            scope=_scope("step-1"),
            step=Step(id="step-1"),
            records=[saved[0].model_copy(update={"run_id": "run-other"})],
        ))

    assert workflow_run_repo.events == []
    assert any(
        record.message == "evidence_event_projection_failed"
        and record.__dict__.get("reason_code") == "evidence_event_scope_mismatch"
        for record in caplog.records
    )


def test_only_legacy_facts_learned_should_not_generate_reusable_evidence_or_duplicate_reuse() -> None:
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(facts=[], evidence=evidence_repo))

    saved = asyncio.run(service.reconcile_step_evidence(
        scope=_scope("step-1"),
        step=Step(
            id="step-1",
            description="读取 /workspace/a.txt",
            outcome=StepOutcome(done=True, facts_learned=["已读取 /workspace/a.txt，后续可复用。"]),
        ),
    ))
    digest = asyncio.run(EvidenceDigestProjector(uow_factory=lambda: _UoW(evidence=evidence_repo)).build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
    ))

    assert saved
    assert [record.evidence_kind for record in saved] == [EvidenceKind.EVIDENCE_GAP]
    assert evidence_repo.records[0].payload["reason_code"] == "step_fact_missing"
    assert digest is not None
    assert digest.do_not_repeat == []
    assert digest.result_handles == []
    assert digest.evidence_backed_facts == []
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
                "evidence_reuse_snapshot": None,
                "has_previous_completed_steps": False,
            },
            runtime_tools=[],
        )
    )
    assert decision is None


def test_working_memory_and_step_reuse_should_ignore_legacy_facts_learned_without_projection_refs() -> None:
    legacy_outcome = StepOutcome(
        done=True,
        summary="模型文本",
        facts_learned=["没有 evidence refs 的旧事实"],
    )
    backed_projection = EvidenceBackedFactProjection(
        text="有 refs 的 evidence-backed fact",
        evidence_ids=["evidence-1"],
        fact_ids=["fact-1"],
        artifact_ids=[],
        source_event_ids=["event-1"],
        user_confirmation_event_ids=[],
    )
    backed_outcome = StepOutcome(
        done=True,
        summary="可审计结果",
        evidence_backed_facts=[backed_projection],
        facts_learned=["旧字段不应被复制"],
    )

    memory = _merge_step_outcome_into_working_memory(
        {},
        step=Step(id="step-1", outcome=legacy_outcome),
    )
    reused = _build_reused_step_outcome(
        legacy_outcome,
        reused_from_run_id="run-1",
        reused_from_step_id="step-1",
    )
    backed_memory = _merge_step_outcome_into_working_memory(
        {},
        step=Step(id="step-2", outcome=backed_outcome),
    )
    backed_reused = _build_reused_step_outcome(
        backed_outcome,
        reused_from_run_id="run-1",
        reused_from_step_id="step-2",
    )

    assert memory.get("facts_in_session") in (None, [])
    assert reused.facts_learned == []
    assert backed_memory["facts_in_session"] == ["有 refs 的 evidence-backed fact"]
    assert backed_reused.facts_learned == ["有 refs 的 evidence-backed fact"]


def test_tool_content_and_workspace_environment_summary_should_not_be_evidence_sources() -> None:
    tool_event = ToolEvent(
        id="evt-tool-1",
        tool_call_id="tool-call-1",
        tool_name="workspace",
        tool_content=FileToolContent(content="完整文件正文不应作为 evidence 来源"),
        function_name="read_file",
        function_args={"path": "/workspace/a.txt"},
        status=ToolEventStatus.CALLED,
    )
    workflow_record = WorkflowRunEventRecord(
        run_id="run-1",
        session_id="session-1",
        event_id=tool_event.id,
        event_type=tool_event.type,
        event_payload=tool_event,
    )
    workspace_environment_summary = {
        "latest_shell_result": {"stdout": "raw stdout 不应作为 evidence"},
        "read_page_summaries": [{"url": "https://example.com", "summary": "页面摘要不应作为 evidence"}],
    }
    evidence_repo = _EvidenceRepo()
    service = _ledger_service(uow_factory=lambda: _UoW(evidence=evidence_repo))

    saved = asyncio.run(service.reconcile_step_evidence(
        scope=_scope("step-1"),
        step=Step(id="step-1", description="读取文件"),
    ))

    assert workflow_record.event_payload.tool_content.content == "完整文件正文不应作为 evidence 来源"
    assert workspace_environment_summary["latest_shell_result"]["stdout"] == "raw stdout 不应作为 evidence"
    assert [record.evidence_kind for record in saved] == [EvidenceKind.EVIDENCE_GAP]
    assert evidence_repo.records[0].payload["reason_code"] == "step_fact_missing"
    assert evidence_repo.records[0].result_refs == []


def test_legacy_search_and_web_reading_summaries_should_not_project_to_digest_truth_source() -> None:
    evidence_repo = _EvidenceRepo()
    projector = EvidenceDigestProjector(uow_factory=lambda: _UoW(evidence=evidence_repo))
    legacy_runtime_recent_action = {
        "search_evidence_summaries": [{"query": "物业", "summary": "搜索摘要"}],
        "web_reading_evidence_summaries": [{"url": "https://example.com", "summary": "网页摘要"}],
    }

    digest = asyncio.run(projector.build_digest(
        scope=_scope("step-2"),
        current_step_id="step-2",
        completed_step_ids=["step-1"],
    ))

    assert legacy_runtime_recent_action["search_evidence_summaries"]
    assert digest is not None
    assert digest.completed_actions == []
    assert digest.do_not_repeat == []
    assert digest.evidence_backed_facts == []
    assert digest.result_handles == []


def test_evidence_ledger_service_should_not_instantiate_digest_projector_for_step_projection() -> None:
    source = inspect.getsource(EvidenceLedgerService.build_step_evidence_backed_facts)

    assert "EvidenceDigestProjector(" not in source


def test_evidence_ledger_service_should_use_injected_step_projection() -> None:
    class _Projection:
        def __init__(self) -> None:
            self.called = False

        async def build_step_evidence_backed_facts(self, *, scope, step):
            self.called = True
            return []

    projection = _Projection()
    service = EvidenceLedgerService(
        uow_factory=lambda: _UoW(),
        assembler=EvidenceFactAssembler(),
        step_projection=projection,
    )

    result = asyncio.run(service.build_step_evidence_backed_facts(scope=_scope("step-1"), step=Step(id="step-1")))

    assert result == []
    assert projection.called is True


def test_agent_task_runner_should_persist_single_aggregated_evidence_event_before_completed_step() -> None:
    sandbox_fact = _fact(SandboxFactKind.FILE_READ)
    evidence_repo = _EvidenceRepo()
    service = EvidenceLedgerService(
        uow_factory=lambda: _UoW(facts=[sandbox_fact], evidence=evidence_repo),
        assembler=EvidenceFactAssembler(),
        step_projection=EvidenceDigestProjector(uow_factory=lambda: _UoW(evidence=evidence_repo)),
    )
    workflow_run_repo = _RecordingRunnerWorkflowRunRepo()
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _RunnerUoW(workflow_run_repo=workflow_run_repo)
    runner._evidence_step_reconciler = service
    event = StepEvent(step=Step(id="step-1", outcome=StepOutcome(done=True, summary="done")), status="completed")

    asyncio.run(runner._reconcile_evidence_before_step_completed(event))

    assert len(workflow_run_repo.events) == 1
    persisted_event = workflow_run_repo.events[0]
    assert isinstance(persisted_event, EvidenceEvent)
    assert persisted_event.step_id == "step-1"
    assert len(persisted_event.evidence_refs) == 1
    assert event.step.outcome is not None
    assert event.step.outcome.facts_learned


def test_agent_task_runner_should_not_block_completed_when_evidence_event_projection_fails(caplog) -> None:
    class _Reconciler:
        async def reconcile_step_evidence(self, *, scope, step):
            return [object()]

        async def build_step_evidence_event(self, *, scope, step, records):
            raise RuntimeError("projection failed")

    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "session-1"
    runner._user_id = "user-1"
    runner._uow_factory = lambda: _RunnerUoW()
    runner._evidence_step_reconciler = _Reconciler()
    event = StepEvent(step=Step(id="step-1"), status="completed")

    with caplog.at_level(logging.ERROR):
        asyncio.run(runner._reconcile_evidence_before_step_completed(event))

    assert any(record.message == "evidence_event_projection_failed" for record in caplog.records)


def test_runtime_observation_should_hide_evidence_event_but_keep_persistent_cursor() -> None:
    service = RuntimeObservationService(
        uow_factory=lambda: _ObservationUoW(),
    )
    event = EvidenceEvent(
        id="evt-evidence-1",
        step_id="step-1",
        quality_status_counts={"valid": 1},
        support_level_counts={"strong": 1},
        summary="step evidence 对账完成：1 条记录。",
    )

    envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=event,
        run_id="run-1",
        source_event_id=event.id,
        cursor_event_id=event.id,
        source="snapshot",
    ))

    assert envelope.runtime.durability == "persistent"
    assert envelope.runtime.visibility == "hidden"
    assert envelope.runtime.cursor_event_id == "evt-evidence-1"
    assert envelope.runtime.status_after_event is None
    assert envelope.runtime.current_step_id is None


def test_runtime_observation_should_keep_status_capabilities_and_current_step_when_evidence_event_exists() -> None:
    evidence_event = EvidenceEvent(
        id="evt-evidence-1",
        step_id="step-1",
        quality_status_counts={"valid": 1},
        support_level_counts={"strong": 1},
        summary="step evidence 对账完成：1 条记录。",
    )
    service = RuntimeObservationService(
        uow_factory=lambda: _ObservationUoW(records=[
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                event_id=evidence_event.id,
                event_type=evidence_event.type,
                event_payload=evidence_event,
            )
        ]),
    )

    observation = asyncio.run(service.build_session_observation(user_id="user-1", session_id="session-1"))
    context = asyncio.run(service.build_event_context(user_id="user-1", session_id="session-1"))
    envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=evidence_event,
        run_id=context.run_id,
        source_event_id=evidence_event.id,
        cursor_event_id=evidence_event.id,
        source="snapshot",
        context=context,
    ))
    advanced_context = service.advance_event_context(context, evidence_event)

    assert observation.status == SessionStatus.RUNNING
    assert observation.current_step_id == "step-running"
    assert observation.capabilities.can_cancel is True
    assert observation.capabilities.can_send_message is False
    assert observation.cursor.latest_event_id == "evt-evidence-1"
    assert envelope.runtime.visibility == "hidden"
    assert envelope.runtime.status_after_event is None
    assert envelope.runtime.current_step_id == "step-running"
    assert advanced_context.status == context.status
    assert advanced_context.current_step_id == context.current_step_id


def test_evidence_event_should_round_trip_from_db_jsonb_to_sse_schema() -> None:
    evidence_event = EvidenceEvent(
        id="evt-evidence-1",
        step_id="step-1",
        source_event_ids=["evt-tool-1"],
        quality_status_counts={"valid": 1},
        support_level_counts={"strong": 1},
        gap_count=0,
        summary="step evidence 对账完成：1 条记录。",
    )
    record = WorkflowRunEventRecord(
        id="record-1",
        run_id="run-1",
        session_id="session-1",
        user_id="user-1",
        event_id=evidence_event.id,
        event_type=evidence_event.type,
        event_payload=evidence_event,
    )
    model = WorkflowRunEventModel.from_domain(record)
    restored_record = model.to_domain()
    service = RuntimeObservationService(uow_factory=lambda: _ObservationUoW(records=[restored_record]))
    envelope = asyncio.run(service.build_observable_event(
        session_id="session-1",
        event=restored_record.event_payload,
        run_id=restored_record.run_id,
        source_event_id=restored_record.event_id,
        cursor_event_id=restored_record.event_id,
        source="snapshot",
    ))

    sse_event = EventMapper.observable_event_to_sse_event(envelope)

    assert isinstance(restored_record.event_payload, EvidenceEvent)
    assert sse_event.event == "evidence"
    assert sse_event.data.event_id == "evt-evidence-1"
    assert sse_event.data.step_id == "step-1"
    assert sse_event.data.source_event_ids == ["evt-tool-1"]
    assert sse_event.data.quality_status_counts == {"valid": 1}
    assert sse_event.data.runtime.visibility == "hidden"
    assert sse_event.data.runtime.durability == "persistent"
    serialized = sse_event.model_dump(mode="json")
    assert "payload" not in str(serialized)
    assert "source_event_id" not in serialized["data"]


def test_runtime_replay_should_include_persistent_evidence_event_for_audit_cursor() -> None:
    evidence_event = EvidenceEvent(
        id="evt-evidence-1",
        step_id="step-1",
        quality_status_counts={"valid": 1},
        support_level_counts={"strong": 1},
        summary="step evidence 对账完成：1 条记录。",
    )
    service = RuntimeObservationService(
        uow_factory=lambda: _ObservationUoW(records=[
            WorkflowRunEventRecord(
                run_id="run-1",
                session_id="session-1",
                event_id=evidence_event.id,
                event_type=evidence_event.type,
                event_payload=evidence_event,
            )
        ]),
    )

    replay = asyncio.run(service.list_persistent_events_after_cursor(
        user_id="user-1",
        session_id="session-1",
        cursor_event_id=None,
    ))

    assert [record.event_id for record in replay.records] == ["evt-evidence-1"]
    assert replay.live_attach_after_event_id == "evt-evidence-1"


class _RecordingRunnerWorkflowRunRepo(_RunnerWorkflowRunRepo):
    def __init__(self) -> None:
        self.events = []

    async def add_event_record_if_absent(self, *, session_id, run_id, event):
        self.events.append(event)
        return True


class _RunnerUoW:
    def __init__(self, *, workflow_run_repo=None) -> None:
        self.session = _RunnerSessionRepo()
        self.workspace = _RunnerWorkspaceRepo()
        self.workflow_run = workflow_run_repo or _RunnerWorkflowRunRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ObservationUoW:
    def __init__(self, *, records=None) -> None:
        self.records = list(records or [])
        self.session = self
        self.workflow_run = self
        self.workspace = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_by_id(self, *args, **kwargs):
        object_id = str(args[0] if args else kwargs.get("session_id") or kwargs.get("workspace_id") or "")
        if object_id == "workspace-1":
            return Workspace(
                id="workspace-1",
                user_id="user-1",
                session_id="session-1",
                current_run_id="run-1",
            )
        return Session(
            id="session-1",
            user_id="user-1",
            workspace_id="workspace-1",
            current_run_id="run-1",
            status=SessionStatus.RUNNING,
        )

    async def get_by_id_for_update(self, object_id):
        if str(object_id or "") == "run-1":
            return await self.get_by_id_for_user(run_id=object_id, user_id="user-1")
        return await self.get_by_id(session_id=object_id)

    async def get_by_id_for_user(self, **kwargs):
        return WorkflowRun(
            id="run-1",
            session_id="session-1",
            user_id="user-1",
            status=WorkflowRunStatus.RUNNING,
            current_step_id="step-running",
        )

    async def get_by_session_id_for_user(self, **kwargs):
        return Workspace(
            id="workspace-1",
            user_id="user-1",
            session_id="session-1",
            current_run_id="run-1",
        )

    async def get_by_session_id(self, session_id):
        return Workspace(
            id="workspace-1",
            user_id="user-1",
            session_id=session_id,
            current_run_id="run-1",
        )

    async def update_runtime_state(self, *args, **kwargs):
        return None

    async def update_status(self, *args, **kwargs):
        return None

    async def list_event_records_by_session(self, session_id):
        return [record for record in self.records if record.session_id == session_id]
