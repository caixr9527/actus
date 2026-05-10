import asyncio
from datetime import datetime

import pytest

from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.application.service.sandbox_fact_ledger_service import (
    CommandExecutionFactInput,
    FileMutationFactInput,
    SandboxFactLedgerService,
    SandboxFactProfileMismatchError,
)
from app.application.service.sandbox_fact_projection_context_builder import SandboxFactProjectionContextBuilder
from app.domain.models import ToolEvent, ToolEventStatus, ToolResult
from app.domain.models.sandbox_fact import SandboxFactKind, SandboxFactProfileRef, SandboxFactRecord
from app.domain.services.runtime.contracts.evidence_key_normalizer import build_file_mutation_intent_hash
from app.domain.services.runtime.contracts.sandbox_fact_ports import SandboxFactProjectionContext, SandboxFactRecorderPort
from app.domain.services.workspace_runtime.projectors import SandboxFactToolEventProjector


class _SandboxFactRepo:
    def __init__(self) -> None:
        self.saved: list[SandboxFactRecord] = []

    async def save_once(self, fact: SandboxFactRecord) -> SandboxFactRecord:
        for saved in self.saved:
            if saved.idempotency_key == fact.idempotency_key:
                return saved
        self.saved.append(fact)
        return fact


class _UoW:
    def __init__(self, repo: _SandboxFactRepo) -> None:
        self.sandbox_fact = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Workspace:
    def __init__(self, *, sandbox_id: str | None = "sandbox-1") -> None:
        self.sandbox_id = sandbox_id


class _AccessControl:
    def __init__(self, *, scope: AccessScopeResult | None = None, error: Exception | None = None) -> None:
        self.scope = scope or AccessScopeResult(
            tenant_id="user-1",
            user_id="user-1",
            session_id="session-1",
            workspace_id="workspace-1",
            run_id="run-1",
            current_step_id="step-1",
        )
        self.error = error
        self.calls: list[dict] = []

    async def assert_session_access(self, **kwargs) -> AccessScopeResult:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.scope


class _WorkspaceRuntime:
    def __init__(self, *, workspace: _Workspace | None = None, profile=None, profile_error: Exception | None = None) -> None:
        self.workspace = workspace
        self.profile = profile
        self.profile_error = profile_error

    async def get_workspace(self):
        return self.workspace

    async def get_sandbox_capability_profile(self):
        if self.profile_error is not None:
            raise self.profile_error
        return self.profile


class _Profile:
    profile_id = "profile-1"
    profile_hash = "sha256:" + "a" * 64
    sandbox_id = "sandbox-1"
    generated_at = datetime(2026, 5, 6, 9, 0, 0)


def _context(*, current_step_id: str | None = "step-1", run_id: str | None = "run-1") -> SandboxFactProjectionContext:
    return SandboxFactProjectionContext(
        scope=AccessScopeResult(
            tenant_id="user-1",
            user_id="user-1",
            session_id="session-1",
            workspace_id="workspace-1",
            run_id=run_id,
            current_step_id=current_step_id,
        ),
        profile_ref=SandboxFactProfileRef(
            profile_id="profile-1",
            profile_hash="sha256:" + "a" * 64,
            sandbox_id="sandbox-1",
            generated_at=datetime(2026, 5, 6, 9, 0, 0),
            status="available",
        ),
        sandbox_id="sandbox-1",
        source_event_id="stream-event-1",
        current_step_id=current_step_id,
    )


def _projector(repo: _SandboxFactRepo) -> SandboxFactToolEventProjector:
    return SandboxFactToolEventProjector(
        ledger_service=SandboxFactLedgerService(uow_factory=lambda: _UoW(repo)),
    )


def test_sandbox_fact_recorder_port_should_only_be_implemented_by_tool_event_projector() -> None:
    repo = _SandboxFactRepo()
    ledger_service = SandboxFactLedgerService(uow_factory=lambda: _UoW(repo))
    projector = SandboxFactToolEventProjector(ledger_service=ledger_service)

    assert not hasattr(ledger_service, "record_from_tool_event")
    assert isinstance(projector, SandboxFactRecorderPort)
    assert SandboxFactToolEventProjector.__mro__[1] is SandboxFactRecorderPort
    assert SandboxFactRecorderPort not in SandboxFactLedgerService.__mro__


def test_projection_context_builder_should_build_available_profile_context() -> None:
    builder = SandboxFactProjectionContextBuilder(
        access_control_service=_AccessControl(),
        workspace_runtime_service=_WorkspaceRuntime(workspace=_Workspace(), profile=_Profile()),
        user_id="user-1",
        session_id="session-1",
    )

    context = asyncio.run(builder.build_for_tool_event(source_event_id="stream-event-1"))

    assert context.source_event_id == "stream-event-1"
    assert context.sandbox_id == "sandbox-1"
    assert context.profile_ref.status == "available"
    assert context.current_step_id == "step-1"


def test_projection_context_builder_should_preserve_missing_profile_ref() -> None:
    builder = SandboxFactProjectionContextBuilder(
        access_control_service=_AccessControl(),
        workspace_runtime_service=_WorkspaceRuntime(workspace=_Workspace(sandbox_id="sandbox-1"), profile=None),
        user_id="user-1",
        session_id="session-1",
    )

    context = asyncio.run(builder.build_for_tool_event(source_event_id="stream-event-1"))

    assert context.sandbox_id == "sandbox-1"
    assert context.profile_ref.status == "missing"
    assert context.profile_ref.sandbox_id is None


def test_file_mutation_fact_input_should_require_mutation_intent_hash() -> None:
    with pytest.raises(Exception):
        FileMutationFactInput(
            fact_kind=SandboxFactKind.FILE_WRITE,
            path="/workspace/a.txt",
            operation="write",
            exists=True,
            content_sha256_kind="read_content_sha256",
            changed=True,
        )


def test_tool_event_projector_should_use_normalized_file_mutation_intent_hash() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-1",
        tool_call_id="call-1",
        tool_name="workspace",
        function_name="write_file",
        function_args={
            "path": "/workspace/dir/../a.txt",
            "content": "new content",
        },
        function_result=ToolResult(
            success=True,
            data={
                "path": "/workspace/dir/../a.txt",
                "after_content_sha256": "sha256:file-v2",
                "content_sha256_kind": "read_content_sha256",
                "size_after": 11,
                "changed": True,
            },
        ),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert facts[0].payload["mutation_intent_hash"] == build_file_mutation_intent_hash(
        path="/workspace/a.txt",
        operation="write",
        content="new content",
        old_str="",
        new_str="",
        append=False,
        leading_newline=False,
        trailing_newline=False,
    )


def test_projection_context_builder_should_mark_profile_ref_invalid_when_profile_load_fails() -> None:
    builder = SandboxFactProjectionContextBuilder(
        access_control_service=_AccessControl(),
        workspace_runtime_service=_WorkspaceRuntime(
            workspace=_Workspace(sandbox_id="sandbox-1"),
            profile_error=RuntimeError("profile invalid"),
        ),
        user_id="user-1",
        session_id="session-1",
    )

    context = asyncio.run(builder.build_for_tool_event(source_event_id="stream-event-1"))

    assert context.sandbox_id == "sandbox-1"
    assert context.profile_ref.status == "invalid"


def test_projection_context_builder_should_not_hide_profile_workspace_sandbox_mismatch() -> None:
    repo = _SandboxFactRepo()
    builder = SandboxFactProjectionContextBuilder(
        access_control_service=_AccessControl(),
        workspace_runtime_service=_WorkspaceRuntime(workspace=_Workspace(sandbox_id="sandbox-a"), profile=_Profile()),
        user_id="user-1",
        session_id="session-1",
    )
    context = asyncio.run(builder.build_for_tool_event(source_event_id="stream-event-1"))
    service = SandboxFactLedgerService(uow_factory=lambda: _UoW(repo))

    with pytest.raises(SandboxFactProfileMismatchError):
        asyncio.run(
            service.record_fact(
                context=context,
                fact_input=CommandExecutionFactInput(
                    command="echo ok",
                    cwd="/workspace",
                    exit_code=0,
                    duration_ms=1,
                ),
            )
        )

    assert context.sandbox_id == "sandbox-a"
    assert context.profile_ref.sandbox_id == "sandbox-1"
    assert repo.saved == []


def test_projection_context_builder_should_fail_when_access_scope_fails() -> None:
    builder = SandboxFactProjectionContextBuilder(
        access_control_service=_AccessControl(error=RuntimeError("scope denied")),
        workspace_runtime_service=_WorkspaceRuntime(workspace=_Workspace(), profile=_Profile()),
        user_id="user-1",
        session_id="session-1",
    )

    with pytest.raises(RuntimeError, match="scope denied"):
        asyncio.run(builder.build_for_tool_event(source_event_id="stream-event-1"))


def test_projection_context_builder_should_preserve_missing_current_step() -> None:
    builder = SandboxFactProjectionContextBuilder(
        access_control_service=_AccessControl(
            scope=AccessScopeResult(
                tenant_id="user-1",
                user_id="user-1",
                session_id="session-1",
                workspace_id="workspace-1",
                run_id="run-1",
                current_step_id=None,
            )
        ),
        workspace_runtime_service=_WorkspaceRuntime(workspace=_Workspace(), profile=_Profile()),
        user_id="user-1",
        session_id="session-1",
    )

    context = asyncio.run(builder.build_for_tool_event(source_event_id="stream-event-1"))

    assert context.current_step_id is None
    assert context.scope.current_step_id is None


def test_shell_tool_event_should_record_command_fact() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-1",
        tool_call_id="call-1",
        tool_name="shell",
        function_name="exec_command",
        function_args={"command": "pytest -q", "cwd": "/workspace"},
        function_result=ToolResult(
            success=True,
            data={"stdout": "51 passed", "stderr": "", "exit_code": 0, "duration_ms": 10},
        ),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert len(facts) == 1
    assert facts[0].fact_kind == SandboxFactKind.COMMAND_EXECUTION
    assert facts[0].source_ref.source_event_id == "stream-event-1"
    assert facts[0].source_ref.tool_event_id == "tool-event-1"
    assert facts[0].source_ref.tool_call_id == "call-1"
    assert facts[0].step_id == "step-1"


def test_shell_tool_event_should_preserve_long_stdout_and_source_truncation_flag() -> None:
    repo = _SandboxFactRepo()
    long_stdout = "x" * 6000
    event = ToolEvent(
        id="tool-event-long-output",
        tool_call_id="call-long-output",
        tool_name="shell",
        function_name="exec_command",
        function_args={"command": "pytest -q", "cwd": "/workspace"},
        function_result=ToolResult(
            success=True,
            data={
                "stdout": long_stdout,
                "stdout_truncated": True,
                "stderr": "",
                "exit_code": 0,
                "duration_ms": 10,
            },
        ),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert facts[0].payload["stdout_excerpt"] == long_stdout
    assert facts[0].payload["stdout_truncated"] is True


def test_failed_tool_event_should_record_tool_failure_fact() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-2",
        tool_call_id="call-2",
        tool_name="file",
        function_name="read_file",
        function_args={"filepath": "/workspace/missing.txt"},
        function_result=ToolResult(success=False, message="file not found", data={"reason_code": "file_not_found"}),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert [fact.fact_kind for fact in facts] == [SandboxFactKind.TOOL_FAILURE]
    assert facts[0].payload["reason_code"] == "file_not_found"


def test_tool_event_fact_projection_should_not_depend_on_tool_content() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-3",
        tool_call_id="call-3",
        tool_name="search",
        function_name="search_web",
        function_args={"query": "runtime contracts"},
        function_result=ToolResult(
            success=True,
            data={"query": "runtime contracts", "results": [{"title": "Doc", "url": "https://example.com", "content": "snippet"}]},
        ),
        tool_content=None,
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert facts[0].fact_kind == SandboxFactKind.SEARCH_RESULT
    assert facts[0].payload["result_count"] == 1


def test_tool_event_without_current_step_should_downgrade_to_run_scope() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-4",
        tool_call_id="call-4",
        tool_name="file",
        function_name="read_file",
        function_args={"filepath": "/workspace/a.txt"},
        function_result=ToolResult(success=True, data={"content": "hello"}),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(
        _projector(repo).record_from_tool_event(
            context=_context(current_step_id=None),
            event=event,
        )
    )

    assert facts[0].fact_scope.value == "run"
    assert facts[0].run_id == "run-1"
    assert facts[0].step_id is None
    assert facts[0].payload["reason_code"] == "current_step_missing"


def test_file_list_tool_event_should_record_file_list_fact() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-5",
        tool_call_id="call-5",
        tool_name="file",
        function_name="list_files",
        function_args={"dir_path": "/workspace"},
        function_result=ToolResult(success=True, data={"dir_path": "/workspace", "files": ["/workspace/a.py"]}),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert facts[0].fact_kind == SandboxFactKind.FILE_LIST
    assert facts[0].payload["dir_path"] == "/workspace"
    assert facts[0].payload["entry_count"] == 1


def test_check_file_exists_tool_event_should_record_structured_fact() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-6",
        tool_call_id="call-6",
        tool_name="file",
        function_name="check_file_exists",
        function_args={"filepath": "/workspace/a.py"},
        function_result=ToolResult(success=True, data={"exists": True}),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert facts[0].fact_kind == SandboxFactKind.FILE_LIST
    assert facts[0].payload["entry_count"] == 1
    assert facts[0].payload["entries"][0]["name"] == "/workspace/a.py"


def test_pr3_should_skip_document_context_fact_projection() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-7",
        tool_call_id="call-7",
        tool_name="document",
        function_name="document_context",
        function_args={},
        function_result=ToolResult(success=True, data={"file_id": "file-1"}),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert facts == []
    assert repo.saved == []


def test_pr3_should_not_create_browser_screenshot_artifact_fact() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-8",
        tool_call_id="call-8",
        tool_name="browser",
        function_name="browser_view",
        function_args={},
        function_result=ToolResult(success=True, data={"url": "https://example.com", "title": "Example"}),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert facts[0].fact_kind == SandboxFactKind.BROWSER_SNAPSHOT
    assert facts[0].source_ref.artifact_id is None
    assert facts[0].payload["screenshot_artifact_id"] is None


def test_browser_screenshot_fact_should_reference_workspace_artifact() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-9",
        tool_call_id="call-9",
        tool_name="browser",
        function_name="browser_view",
        function_args={},
        function_result=ToolResult(
            success=True,
            data={
                "url": "https://example.com",
                "title": "Example",
                "screenshot_artifact": {
                    "artifact_id": "artifact-1",
                    "artifact_path": "/.workspace/browser-screenshots/shot.png",
                },
            },
        ),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert facts[0].fact_kind == SandboxFactKind.BROWSER_SNAPSHOT
    assert facts[0].payload["screenshot_artifact_id"] == "artifact-1"
    assert facts[0].payload["screenshot_artifact_path"] == "/.workspace/browser-screenshots/shot.png"


def test_browser_screenshot_fact_should_not_read_tool_content_screenshot_as_artifact() -> None:
    repo = _SandboxFactRepo()
    event = ToolEvent(
        id="tool-event-10",
        tool_call_id="call-10",
        tool_name="browser",
        function_name="browser_view",
        function_args={},
        function_result=ToolResult(
            success=True,
            data={"url": "https://example.com", "title": "Example"},
        ),
        status=ToolEventStatus.CALLED,
    )
    event.tool_content = {"screenshot": "https://cdn.example.com/shot.png"}

    facts = asyncio.run(_projector(repo).record_from_tool_event(context=_context(), event=event))

    assert facts[0].payload["screenshot_artifact_id"] is None
    assert facts[0].payload["screenshot_artifact_path"] is None
    assert facts[0].payload["reason_code"] == "screenshot_artifact_missing"
