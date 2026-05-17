from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.domain.models import Step, ToolEvent, ToolEventStatus, ToolResult
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.tools import (
    execute_step_with_prompt,
)
from tests.test_safety_audit_pr3 import (
    _RuntimeTool,
    _SafetyAuditRecorder,
    _ToolCallLLM,
    _Coordinator,
    _Task,
    _persistence_service,
    _scope,
)


class _SafetyAuditEventProjector:
    def __init__(self) -> None:
        self.tool_source_calls: list[dict[str, str]] = []

    async def project_tool_event_source(self, **kwargs):
        self.tool_source_calls.append(dict(kwargs))


def test_tool_event_failure_text_cannot_create_or_link_safety_audit() -> None:
    task = _Task()
    coordinator = _Coordinator()
    recorder = _SafetyAuditRecorder()
    projector = _SafetyAuditEventProjector()
    event = ToolEvent(
        step_id="step-1",
        tool_call_id="call-1",
        tool_name="shell",
        function_name="shell_execute",
        function_args={"command": "rm -rf /*"},
        function_result=ToolResult(
            success=False,
            message="安全策略阻断：禁止执行 rm -rf /*",
            data={"error": "blocked_by_policy"},
        ),
        status=ToolEventStatus.CALLED,
        runtime_metadata={},
    )

    result = asyncio.run(
        _persistence_service(
            task=task,
            coordinator=coordinator,
            recorder=recorder,
            safety_audit_event_projector=projector,
        ).persist_tool_event_and_record_facts(
            event=event,
            run_id="run-1",
            session_id="session-1",
            current_step_id="step-1",
        )
    )

    assert result.source_event_id == "evt-1"
    assert coordinator.persisted_event_ids == ["evt-1"]
    assert recorder.attach_calls == []
    assert projector.tool_source_calls == []


def test_execute_step_fails_closed_when_safety_audit_recorder_is_missing() -> None:
    tool = _RuntimeTool(name="filesystem", function_names={"read_file"})

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=_ToolCallLLM(function_name="read_file", function_args={"path": "/workspace/a.txt"}),
            step=Step(id="step-1", description="读取 /workspace/a.txt"),
            runtime_tools=[tool],
            max_tool_iterations=1,
            task_mode="general",
            access_scope=_scope(),
            safety_audit_recorder=None,
        )
    )

    assert payload["success"] is False
    assert payload["blockers"] == ["Safety Audit 运行时依赖缺失，已停止当前工具调用。"]
    assert tool.invocations == []
    assert events == []


@pytest.mark.parametrize(
    ("tool_family", "function_name", "function_args"),
    [
        ("mcp", "mcp_github_search", {"query": "safety audit"}),
        ("a2a", "call_remote_agent", {"id": "agent-1", "query": "review this"}),
    ],
)
def test_remote_tool_families_enter_unified_safety_audit_before_execution(
        tool_family: str,
        function_name: str,
        function_args: dict[str, str],
) -> None:
    operation_log: list[str] = []
    recorder = _SafetyAuditRecorder(operation_log=operation_log)
    tool = _RuntimeTool(name=tool_family, function_names={function_name}, operation_log=operation_log)

    payload, _events = asyncio.run(
        execute_step_with_prompt(
            llm=_ToolCallLLM(function_name=function_name, function_args=function_args),
            step=Step(id="step-1", description="调用外部能力"),
            runtime_tools=[tool],
            max_tool_iterations=2,
            task_mode="general",
            access_scope=_scope(),
            safety_audit_recorder=recorder,
        )
    )

    assert payload["success"] is True
    assert operation_log[0] == "audit"
    assert recorder.commands[0].tool_family == tool_family
    assert recorder.commands[0].capability_id == f"{tool_family}.{function_name}"
    assert recorder.commands[0].requested_args == function_args


def test_policy_constraint_and_tool_capability_layers_cannot_write_audit_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_files = [
        root / "app/infrastructure/runtime/langgraph/graphs/planner_react/policy_engine/engine.py",
        *(
            root / "app/infrastructure/runtime/langgraph/graphs/planner_react/constraint_engine"
        ).rglob("*.py"),
        *(
            root / "app/domain/services/tools"
        ).rglob("*.py"),
        *(
            root / "app/domain/services/workspace_runtime/capabilities"
        ).rglob("*.py"),
    ]
    forbidden_tokens = {
        "SafetyAuditLedgerService",
        "ProjectingSafetyAuditRecorder",
        "SafetyAuditEventProjector",
        "DBSafetyAuditRepository",
        "SafetyAuditRecordModel",
        "safety_audit_records",
        "record_constraint_decision(",
        "record_non_tool_action(",
        "attach_tool_event_source(",
        "attach_decision_event(",
    }

    for path in checked_files:
        content = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in content, f"{path} must not write Safety Audit directly via {token}"


def test_run_engine_assembly_keeps_remote_tools_behind_shared_audit_recorder() -> None:
    root = Path(__file__).resolve().parents[1]
    run_engine_selector = (root / "app/application/service/run_engine_selector.py").read_text(encoding="utf-8")
    run_engine = (root / "app/infrastructure/runtime/langgraph/engine/run_engine.py").read_text(encoding="utf-8")
    graph = (
        root / "app/infrastructure/runtime/langgraph/graphs/planner_react/graph.py"
    ).read_text(encoding="utf-8")

    assert "build_runtime_tools_with_snapshot" in run_engine_selector
    assert "ProjectingSafetyAuditRecorder" in run_engine_selector
    assert "safety_audit_recorder=safety_audit_recorder" in run_engine_selector
    assert "self._safety_audit_recorder = safety_audit_recorder" in run_engine
    assert '"safety_audit_recorder": safety_audit_recorder' in graph
