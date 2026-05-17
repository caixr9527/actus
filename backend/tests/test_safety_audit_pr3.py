from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from app.application.service.runtime_tool_event_persistence_service import (
    RuntimeToolEventPersistenceService,
)
from app.domain.models import Step, ToolEvent, ToolEventStatus, ToolResult
from app.domain.models.sandbox_fact import SandboxFactRecord
from app.domain.models.evidence import (
    EvidenceDoNotRepeatResult,
    EvidenceDuplicateDecision,
    EvidenceQualityStatus,
    EvidenceResultRef,
    EvidenceResultRefType,
    EvidenceReusePolicy,
    EvidenceReuseSnapshot,
    EvidenceStalenessPolicy,
    EvidenceSupportLevel,
    RuntimeEvidenceContextResult,
)
from app.domain.models.safety_audit import (
    SafetyAuditDecision,
    SafetyAuditRecordCommand,
    SafetyAuditRecordResult,
    SafetyAuditRiskLevel,
    SafetyAuditWriteResult,
    SafetyAuditWriteStatus,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_ports import ArtifactRevisionProjectionResult
from app.domain.services.runtime.contracts.evidence_key_normalizer import hash_query
from app.domain.services.runtime.contracts.sandbox_fact_ports import (
    SandboxFactProjectionContext,
)
from app.domain.services.runtime.contracts.sandbox_fact_contract import (
    SandboxFactKind,
    SandboxFactScope,
    SandboxFactSourceRef,
    SandboxFactSourceType,
    SandboxFactSubjectRef,
    build_sandbox_fact_idempotency_key,
    build_sandbox_fact_payload_hash,
)
from app.domain.services.runtime.contracts.langgraph_settings import ASK_USER_FUNCTION_NAME
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.tools import (
    execute_step_with_prompt,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.execute_nodes import (
    _build_safety_audit_scope,
)
from tests.safety_audit_test_helpers import execute_step_with_fake_safety_audit


def _scope(*, step_id: str = "step-1") -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        current_step_id=step_id,
    )


class _AccessControlService:
    def __init__(self, scope: AccessScopeResult) -> None:
        self.scope = scope

    async def assert_session_access(self, **_kwargs: Any) -> AccessScopeResult:
        return self.scope


class _ToolCallLLM:
    def __init__(self, *, function_name: str, function_args: dict[str, Any]) -> None:
        self._function_name = function_name
        self._function_args = dict(function_args)
        self.calls = 0

    async def invoke(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": self._function_name,
                            "arguments": json.dumps(self._function_args, ensure_ascii=False),
                        },
                    }
                ]
            }
        return {"content": json.dumps({"success": True, "summary": "done"}, ensure_ascii=False)}


class _RuntimeTool(BaseTool):
    def __init__(self, *, name: str, function_names: set[str], operation_log: list[str] | None = None) -> None:
        super().__init__()
        self.name = name
        self.function_names = set(function_names)
        self.invocations: list[tuple[str, dict[str, Any]]] = []
        self._operation_log = operation_log

    def get_tools(self) -> list[dict[str, Any]]:
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

    async def invoke(self, function_name: str, **kwargs: Any) -> ToolResult:
        self.invocations.append((function_name, dict(kwargs)))
        if self._operation_log is not None:
            self._operation_log.append("tool")
        return ToolResult(success=True, message="ok", data={"invoked": function_name, **dict(kwargs)})


class _SafetyAuditRecorder:
    def __init__(self, *, operation_log: list[str] | None = None, fail_record: bool = False, fail_attach: bool = False) -> None:
        self.commands: list[SafetyAuditRecordCommand] = []
        self.attach_calls: list[dict[str, Any]] = []
        self._operation_log = operation_log
        self._fail_record = fail_record
        self._fail_attach = fail_attach

    async def record_constraint_decision(self, command: SafetyAuditRecordCommand) -> SafetyAuditWriteResult:
        if self._fail_record:
            raise RuntimeError("record failed")
        self.commands.append(command)
        if self._operation_log is not None:
            self._operation_log.append("audit")
        return _write_result(
            audit_id=f"audit-{len(self.commands)}",
            command=command,
            risk_level=SafetyAuditRiskLevel.MEDIUM,
        )

    async def attach_tool_event_source(
            self,
            audit_id: str,
            tool_event_source_event_id: str,
            *,
            scope: AccessScopeResult,
    ) -> SafetyAuditWriteResult:
        if self._fail_attach:
            raise RuntimeError("attach failed")
        self.attach_calls.append(
            {
                "audit_id": audit_id,
                "tool_event_source_event_id": tool_event_source_event_id,
                "scope": scope,
            }
        )
        return _write_result_for_attach(audit_id=audit_id, scope=scope)


def _write_result(
        *,
        audit_id: str,
        command: SafetyAuditRecordCommand,
        risk_level: SafetyAuditRiskLevel,
) -> SafetyAuditWriteResult:
    record = SafetyAuditRecordResult(
        audit_id=audit_id,
        action_id=f"action-{audit_id}",
        decision=command.decision,
        risk_level=risk_level,
        reason_code=command.reason_code,
        run_id=command.run_id,
        step_id=command.step_id,
        tool_call_id=command.tool_call_id,
    )
    return SafetyAuditWriteResult(
        audit_id=audit_id,
        record=record,
        status=SafetyAuditWriteStatus.CREATED,
        reason_code=command.reason_code,
    )


def _write_result_for_attach(*, audit_id: str, scope: AccessScopeResult) -> SafetyAuditWriteResult:
    record = SafetyAuditRecordResult(
        audit_id=audit_id,
        action_id=f"action-{audit_id}",
        decision=SafetyAuditDecision.ALLOW,
        risk_level=SafetyAuditRiskLevel.MEDIUM,
        reason_code="allow",
        run_id=str(scope.run_id or ""),
        step_id=scope.current_step_id,
        tool_call_id="call-1",
    )
    return SafetyAuditWriteResult(
        audit_id=audit_id,
        record=record,
        status=SafetyAuditWriteStatus.REUSED,
        reason_code="allow",
    )


def test_execute_step_records_audit_before_allow_executor_and_emits_minimal_metadata() -> None:
    operation_log: list[str] = []
    recorder = _SafetyAuditRecorder(operation_log=operation_log)
    tool = _RuntimeTool(name="filesystem", function_names={"read_file"}, operation_log=operation_log)

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=_ToolCallLLM(function_name="read_file", function_args={"path": "/workspace/a.txt", "secret": "s"}),
            step=Step(id="step-1", description="读取 /workspace/a.txt"),
            runtime_tools=[tool],
            max_tool_iterations=2,
            task_mode="general",
            access_scope=_scope(),
            safety_audit_recorder=recorder,
        )
    )

    assert payload["success"] is True
    assert operation_log[:2] == ["audit", "tool"]
    assert recorder.commands[0].requested_args == {"path": "/workspace/a.txt", "secret": "s"}
    assert recorder.commands[0].decision == SafetyAuditDecision.ALLOW
    assert tool.invocations == [("read_file", {"path": "/workspace/a.txt", "secret": "s"})]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    safety_audit = called_events[0].runtime_metadata["safety_audit"]
    assert safety_audit == {
        "audit_id": "audit-1",
        "action_id": "action-audit-1",
        "decision": "allow",
        "risk_level": "medium",
        "reason_code": "allow",
    }
    assert "secret" not in json.dumps(safety_audit, ensure_ascii=False)
    assert "policy_trace" not in safety_audit


def test_execute_step_records_audit_for_block_without_invoking_real_tool() -> None:
    recorder = _SafetyAuditRecorder()
    tool = _RuntimeTool(name="message", function_names={ASK_USER_FUNCTION_NAME})

    _payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=_ToolCallLLM(function_name=ASK_USER_FUNCTION_NAME, function_args={"question": "继续吗"}),
            step=Step(id="step-1", description="直接完成当前步骤"),
            runtime_tools=[tool],
            max_tool_iterations=2,
            task_mode="general",
            access_scope=_scope(),
            safety_audit_recorder=recorder,
        )
    )

    assert tool.invocations == []
    assert recorder.commands[0].decision == SafetyAuditDecision.BLOCK
    assert recorder.commands[0].reason_code == "ask_user_not_allowed"
    called_event = next(event for event in events if event.status == ToolEventStatus.CALLED)
    assert called_event.runtime_metadata["safety_audit"]["decision"] == "block"
    assert called_event.runtime_metadata["safety_audit"]["audit_id"] == "audit-1"


def test_execute_step_rewrite_keeps_requested_and_final_digest_inputs_but_executes_stripped_args() -> None:
    query = "OpenAI pricing"
    query_hash = hash_query(query)
    snapshot = EvidenceReuseSnapshot(
        run_id="run-1",
        current_step_id="step-1",
        source_step_ids=["source-step"],
        cursor="cursor-1",
        do_not_repeat=[
            EvidenceDoNotRepeatResult(
                action_key=f"search:{query_hash}",
                subject_key=f"query:{query_hash}",
                reason_code="requires_verification",
                source_step_id="source-step",
                evidence_ids=["evidence-1"],
                reuse_policy=EvidenceReusePolicy.VERIFY_BEFORE_REUSE,
                staleness_policy=EvidenceStalenessPolicy.STEP_SCOPED,
                support_level=EvidenceSupportLevel.STRONG,
                quality_status=EvidenceQualityStatus.VALID,
                result_status="available",
                duplicate_decision=EvidenceDuplicateDecision.REQUIRE_VERIFICATION,
                reuse_result_ref=EvidenceResultRef(
                    result_ref_type=EvidenceResultRefType.VERIFICATION_REF,
                    ref_id="result-ref-1",
                    subject_key=f"query:{query_hash}",
                    reason_code="verify_current_result",
                    allowed_verification_actions=["search_web"],
                ),
            )
        ],
    )
    recorder = _SafetyAuditRecorder()
    tool = _RuntimeTool(name="search", function_names={"search_web"})

    _payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=_ToolCallLLM(function_name="search_web", function_args={"query": query}),
            step=Step(id="step-1", description="检索 OpenAI pricing"),
            runtime_tools=[tool],
            max_tool_iterations=2,
            task_mode="research",
            runtime_evidence_context=RuntimeEvidenceContextResult(
                run_id="run-1",
                current_step_id="step-1",
                source_step_ids=["source-step"],
                has_previous_completed_steps=True,
                evidence_reuse_snapshot=snapshot,
                cursor="cursor-1",
            ),
            access_scope=_scope(),
            safety_audit_recorder=recorder,
        )
    )

    command = recorder.commands[0]
    assert command.decision == SafetyAuditDecision.REWRITE
    assert command.requested_args == {"query": query}
    assert command.final_args["query"] == query
    assert command.final_args["query_hash"] == query_hash
    assert command.final_args["verification_reason_code"] == "verify_current_result"
    assert tool.invocations == [("search_web", {"query": query})]
    called_event = next(event for event in events if event.status == ToolEventStatus.CALLED)
    assert called_event.runtime_metadata["safety_audit"]["decision"] == "rewrite"


@pytest.mark.parametrize("access_scope", [None, _scope(step_id="other-step")])
def test_execute_step_fails_closed_without_valid_scope(access_scope: AccessScopeResult | None) -> None:
    recorder = _SafetyAuditRecorder()
    tool = _RuntimeTool(name="filesystem", function_names={"read_file"})

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=_ToolCallLLM(function_name="read_file", function_args={"path": "/workspace/a.txt"}),
            step=Step(id="step-1", description="读取 /workspace/a.txt"),
            runtime_tools=[tool],
            max_tool_iterations=1,
            task_mode="general",
            access_scope=access_scope,
            safety_audit_recorder=recorder,
        )
    )

    assert payload["success"] is False
    assert tool.invocations == []
    assert events == []


def test_execute_step_fails_closed_when_audit_record_write_fails() -> None:
    recorder = _SafetyAuditRecorder(fail_record=True)
    tool = _RuntimeTool(name="filesystem", function_names={"read_file"})

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=_ToolCallLLM(function_name="read_file", function_args={"path": "/workspace/a.txt"}),
            step=Step(id="step-1", description="读取 /workspace/a.txt"),
            runtime_tools=[tool],
            max_tool_iterations=1,
            task_mode="general",
            access_scope=_scope(),
            safety_audit_recorder=recorder,
        )
    )

    assert payload["success"] is False
    assert tool.invocations == []
    assert events == []


def test_fake_safety_audit_helper_preserves_underlying_payload_success() -> None:
    async def _raw_execute(**_kwargs: Any):
        return {"success": False, "blockers": ["真实失败"]}, []

    payload, events = asyncio.run(execute_step_with_fake_safety_audit(
        _raw_execute,
        step=Step(id="step-1", description="读取页面"),
    ))

    assert payload == {"success": False, "blockers": ["真实失败"]}
    assert events == []


def test_build_safety_audit_scope_fills_empty_step_from_graph_authority() -> None:
    scope = asyncio.run(_build_safety_audit_scope(
        state={"user_id": "user-1", "session_id": "session-1"},
        step=Step(id="step-1", description="读取 /workspace/a.txt"),
        access_control_service=_AccessControlService(_scope(step_id="")),
    ))

    assert scope is not None
    assert scope.current_step_id == "step-1"


def test_build_safety_audit_scope_fails_closed_on_step_mismatch(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        scope = asyncio.run(_build_safety_audit_scope(
            state={"user_id": "user-1", "session_id": "session-1"},
            step=Step(id="step-1", description="读取 /workspace/a.txt"),
            access_control_service=_AccessControlService(_scope(step_id="other-step")),
        ))

    assert scope is None
    assert "safety_audit_scope_step_mismatch" in caplog.text


class _OutputStream:
    def __init__(self) -> None:
        self.records: list[str] = []
        self.deleted: list[str] = []
        self.sequence = 0

    async def put(self, _message: str) -> str:
        self.sequence += 1
        event_id = f"evt-{self.sequence}"
        self.records.append(event_id)
        return event_id

    async def delete_message(self, message_id: str) -> bool:
        self.deleted.append(message_id)
        return True


class _Task:
    def __init__(self) -> None:
        self.output_stream = _OutputStream()


class _Coordinator:
    def __init__(self) -> None:
        self.persisted_event_ids: list[str] = []

    async def persist_runtime_event(self, *, event, **_kwargs):
        self.persisted_event_ids.append(event.id)
        return type("PersistResult", (), {"event_inserted": True})()


class _FactContextBuilder:
    async def build_for_tool_event(self, *, source_event_id: str, current_step_id: str | None = None):
        return SandboxFactProjectionContext(
            scope=_scope(step_id=current_step_id or "step-1"),
            profile_ref={},
            source_event_id=source_event_id,
            current_step_id=current_step_id,
        )


class _FactRecorder:
    def __init__(self, *, facts: list[SandboxFactRecord] | None = None) -> None:
        self.calls = 0
        self.facts = list(facts or [])

    async def record_from_tool_event(self, *, context, event):
        self.calls += 1
        return list(self.facts)


class _ArtifactRevisionProjector:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def project_from_tool_event_facts(self, *, scope, event, facts):
        self.calls.append({"scope": scope, "event": event, "facts": list(facts or [])})
        return ArtifactRevisionProjectionResult(revision_count=len(list(facts or [])))

    async def project_from_document_facts(self, *, scope, facts):
        return ArtifactRevisionProjectionResult()


class _FailingSafetyAuditEventProjector:
    async def project_tool_event_source(self, **_kwargs):
        raise RuntimeError("projection failed")


def _search_result_fact(*, source_event_id: str = "evt-1", tool_call_id: str = "call-1") -> SandboxFactRecord:
    payload = {
        "query_hash": hash_query("safety audit"),
        "query_excerpt": "safety audit",
        "result_count": 0,
        "top_results": [],
        "is_truncated": False,
        "missing_fields": None,
        "reason_code": None,
    }
    payload_hash = build_sandbox_fact_payload_hash(payload)
    source_ref = SandboxFactSourceRef(
        source_type=SandboxFactSourceType.SANDBOX_API,
        source_event_id=source_event_id,
        source_event_status="available",
        tool_event_id=source_event_id,
        tool_call_id=tool_call_id,
        function_name="search_web",
    )
    subject_ref = SandboxFactSubjectRef(
        subject_type="search",
        subject_key="search:safety audit",
    )
    return SandboxFactRecord(
        id="fact-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        fact_scope=SandboxFactScope.STEP,
        run_id="run-1",
        step_id="step-1",
        fact_kind=SandboxFactKind.SEARCH_RESULT,
        source_ref=source_ref,
        subject_ref=subject_ref,
        summary="search fact",
        payload=payload,
        payload_hash=payload_hash,
        idempotency_key=build_sandbox_fact_idempotency_key(
            user_id="user-1",
            session_id="session-1",
            workspace_id="workspace-1",
            fact_scope=SandboxFactScope.STEP,
            run_id="run-1",
            step_id="step-1",
            fact_kind=SandboxFactKind.SEARCH_RESULT,
            source_event_id=source_event_id,
            tool_call_id=tool_call_id,
            subject_key=subject_ref.subject_key,
            payload_hash=payload_hash,
        ),
    )


def _persistence_service(
        *,
        task: _Task,
        coordinator: _Coordinator,
        recorder: _SafetyAuditRecorder,
        fact_recorder: _FactRecorder | None = None,
        safety_audit_event_projector=None,
        artifact_revision_projector=None,
):
    return RuntimeToolEventPersistenceService(
        session_id="session-1",
        task=task,
        uow_factory=lambda: None,
        runtime_state_coordinator=coordinator,
        sandbox_fact_recorder=fact_recorder or _FactRecorder(),
        sandbox_fact_context_builder=_FactContextBuilder(),
        safety_audit_recorder=recorder,
        safety_audit_event_projector=safety_audit_event_projector,
        artifact_revision_projector=artifact_revision_projector,
    )


def test_runtime_tool_event_persistence_attaches_audit_source_after_event_persist() -> None:
    task = _Task()
    coordinator = _Coordinator()
    recorder = _SafetyAuditRecorder()
    event = ToolEvent(
        step_id="step-1",
        tool_call_id="call-1",
        tool_name="filesystem",
        function_name="read_file",
        function_args={"path": "/workspace/a.txt"},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
        runtime_metadata={"safety_audit": {"audit_id": "audit-1"}},
    )

    result = asyncio.run(_persistence_service(
        task=task,
        coordinator=coordinator,
        recorder=recorder,
    ).persist_tool_event_and_record_facts(
        event=event,
        run_id="run-1",
        session_id="session-1",
        current_step_id="step-1",
    ))

    assert result.source_event_id == "evt-1"
    assert coordinator.persisted_event_ids == ["evt-1"]
    assert recorder.attach_calls[0]["audit_id"] == "audit-1"
    assert recorder.attach_calls[0]["tool_event_source_event_id"] == "evt-1"
    assert recorder.attach_calls[0]["scope"].run_id == "run-1"


def test_runtime_tool_event_persistence_keeps_persisted_event_when_audit_attach_fails(caplog) -> None:
    task = _Task()
    coordinator = _Coordinator()
    recorder = _SafetyAuditRecorder(fail_attach=True)
    event = ToolEvent(
        step_id="step-1",
        tool_call_id="call-1",
        tool_name="filesystem",
        function_name="read_file",
        function_args={"path": "/workspace/a.txt"},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
        runtime_metadata={"safety_audit": {"audit_id": "audit-1"}},
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="attach failed"):
            asyncio.run(_persistence_service(
                task=task,
                coordinator=coordinator,
                recorder=recorder,
            ).persist_tool_event_and_record_facts(
                event=event,
                run_id="run-1",
                session_id="session-1",
                current_step_id="step-1",
            ))

    assert task.output_stream.records == ["evt-1"]
    assert task.output_stream.deleted == []
    assert coordinator.persisted_event_ids == ["evt-1"]
    assert "safety_audit_source_event_attach_failed" in caplog.text
    assert "tool_event_safety_audit_source_attach_failed_after_source_event_persisted" in caplog.text
    assert "tool_event_fact_projection_contract_failed" not in caplog.text


def test_runtime_tool_event_persistence_continues_when_audit_event_projection_fails(caplog) -> None:
    task = _Task()
    coordinator = _Coordinator()
    recorder = _SafetyAuditRecorder()
    fact_recorder = _FactRecorder(facts=[_search_result_fact()])
    artifact_projector = _ArtifactRevisionProjector()
    event = ToolEvent(
        step_id="step-1",
        tool_call_id="call-1",
        tool_name="search",
        function_name="search_web",
        function_args={"query": "safety audit"},
        function_result=ToolResult(success=True, data={}),
        status=ToolEventStatus.CALLED,
        runtime_metadata={"safety_audit": {"audit_id": "audit-1"}},
    )

    with caplog.at_level(logging.ERROR):
        result = asyncio.run(_persistence_service(
            task=task,
            coordinator=coordinator,
            recorder=recorder,
            fact_recorder=fact_recorder,
            safety_audit_event_projector=_FailingSafetyAuditEventProjector(),
            artifact_revision_projector=artifact_projector,
        ).persist_tool_event_and_record_facts(
            event=event,
            run_id="run-1",
            session_id="session-1",
            current_step_id="step-1",
        ))

    assert result.source_event_id == "evt-1"
    assert fact_recorder.calls == 1
    assert result.fact_count == 1
    assert result.artifact_revision_count == 1
    assert event.runtime_fact_projection["fact_count"] == 1
    assert event.runtime_fact_projection["artifact_revision_count"] == 1
    assert len(artifact_projector.calls) == 1
    assert artifact_projector.calls[0]["facts"][0].id == "fact-1"
    assert recorder.attach_calls[0]["tool_event_source_event_id"] == "evt-1"
    assert "safety_audit_event_projection_failed" in caplog.text
    assert "tool_event_fact_projection_contract_failed" not in caplog.text


def test_policy_and_constraint_layers_do_not_import_audit_services() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_files = [
        root / "app/infrastructure/runtime/langgraph/graphs/planner_react/policy_engine/engine.py",
        *(
            root / "app/infrastructure/runtime/langgraph/graphs/planner_react/constraint_engine"
        ).rglob("*.py"),
    ]
    forbidden_tokens = {
        "SafetyAuditRecorderPort",
        "SafetyAuditLedgerService",
        "RuntimeToolEventPersistenceService",
        "SafetyAuditEventProjector",
    }
    for path in checked_files:
        content = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in content, f"{path} must not depend on {token}"
