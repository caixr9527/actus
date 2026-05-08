import asyncio
import json

import pytest

from app.domain.models import ExecutionStatus, Plan, Step
from app.domain.models.evidence import (
    EvidenceDuplicateDecision,
    EvidenceQualityStatus,
    EvidenceReadStrategy,
    EvidenceResultRef,
    EvidenceResultRefType,
    EvidenceReusePolicy,
    EvidenceReuseSnapshot,
    EvidenceStalenessPolicy,
    EvidenceSupportLevel,
    build_evidence_result_handle,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.graphs.planner_react.constraint_engine.reason_codes import (
    REASON_EVIDENCE_CONTEXT_INVALID_SCHEMA,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.evidence_context_guard import (
    validate_stage_evidence_context_packet,
)


def _runtime_context():
    handle = build_evidence_result_handle(_result_ref())
    snapshot = EvidenceReuseSnapshot(
        run_id="run-1",
        current_step_id="step-2",
        source_step_ids=["step-1"],
        cursor="cursor-1",
        do_not_repeat=[
            {
                "action_key": "file_read:/workspace/a.txt",
                "subject_key": "file:/workspace/a.txt",
                "reason_code": "evidence_reuse_pending_resolution",
                "source_step_id": "step-1",
                "evidence_ids": ["evidence-1"],
                "reuse_policy": EvidenceReusePolicy.REUSE_ALLOWED,
                "staleness_policy": EvidenceStalenessPolicy.RUN_SCOPED,
                "support_level": EvidenceSupportLevel.STRONG,
                "quality_status": EvidenceQualityStatus.VALID,
                "result_status": "successful",
                "duplicate_decision": EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION,
                "reuse_result_ref": _result_ref(),
                "result_handle_id": handle.result_handle_id,
                "reuse_summary": "safe summary",
            }
        ],
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


class _Provider:
    def __init__(self, runtime_context=None) -> None:
        self.completed_step_ids = []
        self._runtime_context = runtime_context

    async def build_context(self, *, completed_step_ids, **kwargs):
        self.completed_step_ids = completed_step_ids
        return self._runtime_context or _runtime_context()


def test_runtime_context_service_should_build_replan_and_summary_evidence_views() -> None:
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

    replan_packet = asyncio.run(service.build_packet_async(stage="replan", state=state, step=step1))
    summary_packet = asyncio.run(service.build_packet_async(stage="summary", state=state))

    assert "evidence_replan_context" in replan_packet
    assert replan_packet["evidence_replan_context"]["do_not_repeat"]
    assert "evidence_context" in replan_packet
    assert "summary_evidence_context" in summary_packet
    assert "evidence_context" not in summary_packet
    assert summary_packet["evidence_context_cursor"] == "cursor-1"
    assert "evidence_reuse_snapshot" not in json.dumps(summary_packet["summary_evidence_context"])
    assert "result_handle_index" not in json.dumps(summary_packet["summary_evidence_context"])
    assert summary_packet["summary_evidence_context"]["cursor"] == "cursor-1"


@pytest.mark.parametrize("missing_mode", ["provider_missing", "provider_returns_none"])
def test_runtime_context_service_should_mark_missing_evidence_context_for_replan_summary(missing_mode) -> None:
    class _NoneProvider:
        async def build_context(self, **kwargs):
            return None

    service = (
        RuntimeContextService(evidence_context_provider=None)
        if missing_mode == "provider_missing"
        else RuntimeContextService(evidence_context_provider=_NoneProvider())
    )
    step1 = Step(id="step-1", status=ExecutionStatus.COMPLETED)
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "plan": Plan(steps=[step1]),
        "step_states": [{"step_id": "step-1", "status": "completed"}],
        "graph_metadata": {},
    }

    packet = asyncio.run(service.build_packet_async(stage="summary", state=state))

    assert packet["evidence_context_error"]["reason_code"] == "evidence_context_missing"
    assert "evidence_context_error" not in packet["prompt_visible_fields"]


def test_runtime_context_service_should_mark_missing_evidence_context_when_scope_missing() -> None:
    service = RuntimeContextService(evidence_context_provider=_Provider(runtime_context=_runtime_context()))
    step1 = Step(id="step-1", status=ExecutionStatus.COMPLETED)
    state = {
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "plan": Plan(steps=[step1]),
        "step_states": [{"step_id": "step-1", "status": "completed"}],
        "graph_metadata": {},
    }

    packet = asyncio.run(service.build_packet_async(stage="replan", state=state, step=step1))

    assert packet["evidence_context_error"]["reason_code"] == "evidence_context_missing"


def test_runtime_context_service_should_reserve_future_review_stage_without_evidence_context() -> None:
    service = RuntimeContextService(evidence_context_provider=_Provider(runtime_context=_runtime_context()))
    packet = service.build_packet(
        stage="future_review",  # type: ignore[arg-type]
        state={
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "plan": Plan(steps=[Step(id="step-1", status=ExecutionStatus.COMPLETED)]),
            "step_states": [{"step_id": "step-1", "status": "completed"}],
            "graph_metadata": {},
        },
    )

    assert packet["stage"] == "future_review"
    assert "evidence_context" not in packet
    assert "summary_evidence_context" not in packet
    assert "raw_stdout" not in json.dumps(packet, ensure_ascii=False)


def test_evidence_context_guard_should_reject_invalid_schema() -> None:
    result = validate_stage_evidence_context_packet(
        stage="summary",
        context_packet={
            "evidence_context_cursor": "cursor-1",
            "summary_evidence_context": "invalid",
        },
    )

    assert result.blocked is True
    assert result.reason_code == REASON_EVIDENCE_CONTEXT_INVALID_SCHEMA
