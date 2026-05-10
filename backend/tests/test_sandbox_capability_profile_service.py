import asyncio
import logging
from datetime import datetime, timedelta

import pytest

from app.application.errors import NotFoundError, error_keys
from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.application.service.sandbox_capability_profile_service import SandboxCapabilityProfileService
from app.domain.models import Session, ToolResult, WorkflowRun, Workspace
from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY,
    RuntimeToolCapabilitySnapshot,
    RuntimeToolCapabilitySnapshotItem,
    SandboxCapabilityItem,
    SandboxCapabilityKind,
    SandboxCapabilityProbePayload,
    SandboxCapabilityPromptSummary,
    SandboxCapabilityStatus,
    SandboxProfileRefreshReason,
    SandboxResourceLimits,
    build_sandbox_capability_profile_from_draft,
)


class _SessionRepo:
    def __init__(self, session: Session | None) -> None:
        self.session = session

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        if self.session is None or self.session.id != session_id:
            return None
        if user_id is not None and self.session.user_id != user_id:
            return None
        return self.session.model_copy(deep=True)


class _WorkspaceRepo:
    def __init__(self, workspace: Workspace | None) -> None:
        self.workspace = workspace
        self.saved_workspaces: list[Workspace] = []

    async def save(self, workspace: Workspace) -> None:
        cloned = workspace.model_copy(deep=True)
        self.workspace = cloned
        self.saved_workspaces.append(cloned)

    async def get_by_id_for_user(self, workspace_id: str, user_id: str):
        if self.workspace is None:
            return None
        if self.workspace.id == workspace_id and self.workspace.user_id == user_id:
            return self.workspace.model_copy(deep=True)
        return None

    async def get_by_session_id_for_user(self, session_id: str, user_id: str):
        if self.workspace is None:
            return None
        if self.workspace.session_id == session_id and self.workspace.user_id == user_id:
            return self.workspace.model_copy(deep=True)
        return None

    async def list_by_session_id(self, session_id: str):
        if self.workspace is not None and self.workspace.session_id == session_id:
            return [self.workspace.model_copy(deep=True)]
        return []


class _WorkflowRunRepo:
    def __init__(self, run: WorkflowRun | None) -> None:
        self.run = run

    async def get_by_id_for_user(self, run_id: str, user_id: str):
        if self.run is not None and self.run.id == run_id and self.run.user_id == user_id:
            return self.run.model_copy(deep=True)
        return None


class _WorkspaceArtifactRepo:
    async def list_by_user_workspace_id(self, *, user_id: str, workspace_id: str):
        return []


class _UoW:
    def __init__(
            self,
            *,
            session_repo: _SessionRepo,
            workspace_repo: _WorkspaceRepo,
            run_repo: _WorkflowRunRepo,
    ) -> None:
        self.session = session_repo
        self.workspace = workspace_repo
        self.workflow_run = run_repo
        self.workspace_artifact = _WorkspaceArtifactRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSandbox:
    probe_calls = 0
    result = ToolResult[SandboxCapabilityProbePayload](
        success=True,
        data=SandboxCapabilityProbePayload(raw_profile={}),
    )
    instances: dict[str, "_FakeSandbox"] = {}

    def __init__(self, sandbox_id: str) -> None:
        self.id = sandbox_id

    @classmethod
    def reset(cls) -> None:
        cls.probe_calls = 0
        cls.result = ToolResult[SandboxCapabilityProbePayload](
            success=True,
            data=SandboxCapabilityProbePayload(raw_profile=_raw_profile()),
        )
        cls.instances = {"sandbox-1": _FakeSandbox("sandbox-1")}

    @classmethod
    async def get(cls, id: str):
        return cls.instances.get(id)

    async def probe_capabilities(self):
        type(self).probe_calls += 1
        return type(self).result


def _raw_profile(*, python_version: str = "3.12.1", node_version: str = "20.18.0") -> dict:
    return {
        "health_status": "available",
        "cwd": "/workspace",
        "capabilities": [
            {
                "kind": "python",
                "name": "python3",
                "status": "available",
                "version": python_version,
                "path": "/usr/bin/python3",
                "details": {},
            },
            {
                "kind": "node",
                "name": "node",
                "status": "available",
                "version": node_version,
                "path": "/usr/bin/node",
                "details": {},
            },
        ],
        "resource_limits": {
            "max_file_read_bytes": 10000,
            "max_command_seconds": 600,
            "writable_dirs": ["/workspace", "/tmp"],
            "readable_dirs": ["/workspace", "/tmp"],
            "network_policy": "restricted",
        },
        "disabled_capabilities": [],
        "confirmation_required_capabilities": [],
    }


def _existing_profile_payload(*, expires_at: datetime | None = None) -> dict:
    now = datetime.now()
    profile = build_sandbox_capability_profile_from_draft({
        "profile_id": "profile-existing",
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "sandbox_id": "sandbox-1",
        "generated_at": now,
        "expires_at": expires_at or now + timedelta(minutes=30),
        "refresh_reason": SandboxProfileRefreshReason.SANDBOX_CREATED,
        "health_status": SandboxCapabilityStatus.AVAILABLE,
        "cwd": "/workspace",
        "capabilities": [
            SandboxCapabilityItem(
                kind=SandboxCapabilityKind.PYTHON,
                name="python3",
                status=SandboxCapabilityStatus.AVAILABLE,
                version="3.11.0",
            ),
            SandboxCapabilityItem(
                kind=SandboxCapabilityKind.NODE,
                name="node",
                status=SandboxCapabilityStatus.AVAILABLE,
                version="18.0.0",
            ),
        ],
        "resource_limits": SandboxResourceLimits(
            max_file_read_bytes=111,
            max_command_seconds=222,
            writable_dirs=["/old"],
            readable_dirs=["/old"],
            network_policy="unknown",
        ),
        "runtime_tool_capabilities": RuntimeToolCapabilitySnapshot(
            items=[
                RuntimeToolCapabilitySnapshotItem(
                    capability_id="local_shell",
                    tool_family="shell",
                    source="local",
                ),
            ]
        ),
        "prompt_summary": SandboxCapabilityPromptSummary(
            health_status=SandboxCapabilityStatus.AVAILABLE,
            cwd="/workspace",
            available_runtime={"python": "3.11.0", "node": "18.0.0"},
            available_tools=["shell"],
            resource_limits={"writable_dirs": ["/old"], "max_command_seconds": 222},
            generated_at=now,
        ),
    })
    return profile.model_dump(mode="json")


def _build_service(
        *,
        workspace: Workspace | None = None,
        session: Session | None = None,
        run: WorkflowRun | None = None,
) -> tuple[SandboxCapabilityProfileService, _WorkspaceRepo]:
    _FakeSandbox.reset()
    session = session or Session(
        id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
        current_run_id="run-1",
    )
    workspace = workspace or Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
    )
    run = run or WorkflowRun(id="run-1", session_id="session-1", user_id="user-1")
    workspace_repo = _WorkspaceRepo(workspace=workspace)
    uow = _UoW(
        session_repo=_SessionRepo(session=session),
        workspace_repo=workspace_repo,
        run_repo=_WorkflowRunRepo(run=run),
    )
    service = SandboxCapabilityProfileService(
        uow_factory=lambda: uow,
        sandbox_cls=_FakeSandbox,
        access_control_service=RuntimeAccessControlService(uow_factory=lambda: uow),
    )
    return service, workspace_repo


def _stored_profile(workspace_repo: _WorkspaceRepo) -> dict:
    assert workspace_repo.workspace is not None
    return workspace_repo.workspace.environment_summary[SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY]


def test_refresh_profile_should_probe_and_write_workspace_profile() -> None:
    service, workspace_repo = _build_service()

    profile = asyncio.run(service.refresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.SANDBOX_CREATED,
    ))

    assert _FakeSandbox.probe_calls == 1
    assert profile.workspace_id == "workspace-1"
    assert profile.run_id == "run-1"
    assert profile.sandbox_id == "sandbox-1"
    assert profile.health_status == SandboxCapabilityStatus.AVAILABLE
    assert _stored_profile(workspace_repo)["profile_hash"] == profile.profile_hash


def test_refresh_should_reject_cross_user_before_sandbox_lookup() -> None:
    service, workspace_repo = _build_service()

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.refresh_profile(
            user_id="user-other",
            session_id="session-1",
            reason=SandboxProfileRefreshReason.PERIODIC,
        ))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
    assert _FakeSandbox.probe_calls == 0
    assert workspace_repo.saved_workspaces == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"workspace_id": "workspace-other"},
        {"run_id": "run-other"},
        {"sandbox_id": "sandbox-other"},
        {"task_id": "task-other"},
    ],
)
def test_refresh_after_sandbox_bound_should_fail_closed_when_snapshot_mismatch(
        overrides: dict,
        caplog,
) -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: _existing_profile_payload(),
        },
    )
    service, workspace_repo = _build_service(workspace=workspace)
    before = _stored_profile(workspace_repo)

    with caplog.at_level(logging.WARNING, logger="app.application.service.sandbox_capability_profile_service"):
        payload = {
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "sandbox_id": "sandbox-1",
            "task_id": "task-1",
            "reason": SandboxProfileRefreshReason.SANDBOX_CREATED,
        }
        payload.update(overrides)
        asyncio.run(service.refresh_after_sandbox_bound(
            **payload,
        ))

    assert _FakeSandbox.probe_calls == 0
    stored = _stored_profile(workspace_repo)
    assert stored["profile_hash"] == before["profile_hash"]
    assert stored["last_refresh_error"]["reason_code"] == "sandbox_profile_refresh_failed"
    assert any(record.message == "sandbox_profile_refresh_failed" for record in caplog.records)


def test_refresh_after_sandbox_bound_should_write_unknown_when_sandbox_missing_without_existing_profile() -> None:
    service, workspace_repo = _build_service()
    _FakeSandbox.instances = {}

    asyncio.run(service.refresh_after_sandbox_bound(
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        reason=SandboxProfileRefreshReason.SANDBOX_CREATED,
    ))

    stored = _stored_profile(workspace_repo)
    assert _FakeSandbox.probe_calls == 0
    assert stored["health_status"] == "unknown"
    assert stored["last_refresh_error"]["reason_code"] == "sandbox_profile_refresh_failed"
    assert stored["prompt_summary"]["sandbox_profile_stale"] is True


def test_ensure_fresh_profile_should_skip_unexpired_profile() -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: _existing_profile_payload(
                expires_at=datetime.now() + timedelta(minutes=5),
            ),
        },
    )
    service, workspace_repo = _build_service(workspace=workspace)

    profile = asyncio.run(service.ensure_fresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.PERIODIC,
    ))

    assert _FakeSandbox.probe_calls == 0
    assert profile.profile_hash == _stored_profile(workspace_repo)["profile_hash"]


def test_ensure_fresh_profile_should_refresh_expired_profile() -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: _existing_profile_payload(
                expires_at=datetime.now() - timedelta(minutes=1),
            ),
        },
    )
    service, workspace_repo = _build_service(workspace=workspace)

    profile = asyncio.run(service.ensure_fresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.PERIODIC,
    ))

    assert _FakeSandbox.probe_calls == 1
    assert profile.prompt_summary.available_runtime["python"] == "3.12.1"
    assert _stored_profile(workspace_repo)["profile_hash"] == profile.profile_hash


def test_section_refresh_should_only_merge_selected_capability_kind() -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: _existing_profile_payload(
                expires_at=datetime.now() - timedelta(minutes=1),
            ),
        },
    )
    service, _workspace_repo = _build_service(workspace=workspace)
    _FakeSandbox.result = ToolResult[SandboxCapabilityProbePayload](
        success=True,
        data=SandboxCapabilityProbePayload(raw_profile=_raw_profile(python_version="3.13.0", node_version="22.0.0")),
    )

    profile = asyncio.run(service.refresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.TOOL_ENV_ERROR,
        section_kinds=[SandboxCapabilityKind.PYTHON],
    ))

    versions = {item.kind: item.version for item in profile.capabilities}
    assert versions[SandboxCapabilityKind.PYTHON] == "3.13.0"
    assert versions[SandboxCapabilityKind.NODE] == "18.0.0"
    assert profile.resource_limits.max_command_seconds == 222
    assert profile.last_refresh_error is None


def test_section_refresh_failure_should_preserve_sections_and_record_error() -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: _existing_profile_payload(),
        },
    )
    service, _workspace_repo = _build_service(workspace=workspace)
    _FakeSandbox.result = ToolResult[SandboxCapabilityProbePayload](
        success=False,
        message="probe unavailable",
        data=SandboxCapabilityProbePayload(
            reason_code="sandbox_profile_probe_unavailable",
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        ),
    )

    profile = asyncio.run(service.refresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.TOOL_ENV_ERROR,
        section_kinds=[SandboxCapabilityKind.PYTHON],
    ))

    versions = {item.kind: item.version for item in profile.capabilities}
    assert versions[SandboxCapabilityKind.PYTHON] == "3.11.0"
    assert versions[SandboxCapabilityKind.NODE] == "18.0.0"
    assert profile.resource_limits.max_command_seconds == 222
    assert profile.last_refresh_error is not None
    assert profile.last_refresh_error.reason_code == "sandbox_profile_probe_unavailable"


def test_probe_failure_without_existing_profile_should_write_unknown_profile() -> None:
    service, workspace_repo = _build_service()
    _FakeSandbox.result = ToolResult[SandboxCapabilityProbePayload](
        success=False,
        message="missing api",
        data=SandboxCapabilityProbePayload(
            reason_code="sandbox_profile_probe_unavailable",
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        ),
    )

    profile = asyncio.run(service.refresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.SANDBOX_CREATED,
    ))

    assert profile.health_status == SandboxCapabilityStatus.UNKNOWN
    assert profile.capabilities == []
    assert profile.prompt_summary.sandbox_profile_stale is True
    assert profile.last_refresh_error is not None
    assert _stored_profile(workspace_repo)["profile_hash"] == profile.profile_hash


def test_probe_failure_should_not_persist_sensitive_message() -> None:
    service, workspace_repo = _build_service()
    _FakeSandbox.result = ToolResult[SandboxCapabilityProbePayload](
        success=False,
        message="password=hunter2 token=abc cookie=session http://user:pass@proxy.local/path",
        data=SandboxCapabilityProbePayload(
            reason_code="sandbox_profile_probe_unavailable",
            probe_status=SandboxCapabilityStatus.UNKNOWN,
        ),
    )

    asyncio.run(service.refresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.SANDBOX_CREATED,
    ))

    message = _stored_profile(workspace_repo)["last_refresh_error"]["message"]
    assert "hunter2" not in message
    assert "abc" not in message
    assert "session" not in message
    assert "http://user:pass@proxy.local" not in message
    assert message == "sandbox capability probe failed"


@pytest.mark.parametrize(
    "raw_patch",
    [
        {"profile_hash": "sha256:" + "0" * 64},
        {"prompt_summary": {}},
        {"generated_at": "2026-05-05T10:00:00"},
        {"resource_limits": {"max_tool_iterations": 20}},
        {"capabilities": [{"kind": "python", "name": "python3", "status": "available", "extra": "bad"}]},
        {"capabilities": [{"kind": "gpu", "name": "gpu", "status": "available"}]},
    ],
)
def test_full_refresh_invalid_raw_payload_should_write_unknown_with_normalize_error(raw_patch: dict) -> None:
    service, workspace_repo = _build_service()
    raw_profile = _raw_profile()
    if raw_patch.get("resource_limits"):
        raw_profile["resource_limits"] = {
            **raw_profile["resource_limits"],
            **raw_patch["resource_limits"],
        }
    else:
        raw_profile.update(raw_patch)
    _FakeSandbox.result = ToolResult[SandboxCapabilityProbePayload](
        success=True,
        data=SandboxCapabilityProbePayload(raw_profile=raw_profile),
    )

    profile = asyncio.run(service.refresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.SANDBOX_CREATED,
    ))

    assert profile.health_status == SandboxCapabilityStatus.UNKNOWN
    assert profile.capabilities == []
    assert profile.last_refresh_error is not None
    assert profile.last_refresh_error.reason_code == "sandbox_profile_normalize_failed"
    assert _stored_profile(workspace_repo)["last_refresh_error"]["reason_code"] == "sandbox_profile_normalize_failed"


def test_section_refresh_invalid_raw_payload_should_preserve_old_sections_and_record_normalize_error() -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: _existing_profile_payload(),
        },
    )
    service, _workspace_repo = _build_service(workspace=workspace)
    raw_profile = _raw_profile(python_version="3.13.0")
    raw_profile["capabilities"][0]["extra"] = "bad"
    _FakeSandbox.result = ToolResult[SandboxCapabilityProbePayload](
        success=True,
        data=SandboxCapabilityProbePayload(raw_profile=raw_profile),
    )

    profile = asyncio.run(service.refresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.TOOL_ENV_ERROR,
        section_kinds=[SandboxCapabilityKind.PYTHON],
    ))

    versions = {item.kind: item.version for item in profile.capabilities}
    assert versions[SandboxCapabilityKind.PYTHON] == "3.11.0"
    assert versions[SandboxCapabilityKind.NODE] == "18.0.0"
    assert profile.last_refresh_error is not None
    assert profile.last_refresh_error.reason_code == "sandbox_profile_normalize_failed"


def test_ensure_fresh_profile_should_persist_minimal_unknown_when_lookup_fails_without_existing_profile() -> None:
    service, workspace_repo = _build_service()
    _FakeSandbox.instances = {}

    profile = asyncio.run(service.ensure_fresh_profile(
        user_id="user-1",
        session_id="session-1",
        reason=SandboxProfileRefreshReason.PERIODIC,
    ))

    assert profile.health_status == SandboxCapabilityStatus.UNKNOWN
    assert profile.resource_limits.max_command_seconds is None
    assert profile.prompt_summary.sandbox_profile_stale is True
    assert _stored_profile(workspace_repo)["last_refresh_error"]["reason_code"] == "sandbox_profile_prompt_refresh_failed"


def test_record_runtime_tool_snapshot_should_write_snapshot_and_rehash(caplog) -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        sandbox_id="sandbox-1",
        task_id="task-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: _existing_profile_payload(),
        },
    )
    service, workspace_repo = _build_service(workspace=workspace)
    old_hash = _stored_profile(workspace_repo)["profile_hash"]
    snapshot = RuntimeToolCapabilitySnapshot(
        items=[
            RuntimeToolCapabilitySnapshotItem(
                capability_id="search",
                tool_family="search",
                source="local",
            ),
        ]
    )

    with caplog.at_level(logging.INFO, logger="app.application.service.sandbox_capability_profile_service"):
        profile = asyncio.run(service.record_runtime_tool_snapshot(
            user_id="user-1",
            session_id="session-1",
            snapshot=snapshot,
        ))

    assert profile.profile_hash != old_hash
    assert profile.runtime_tool_capabilities.items[0].capability_id == "search"
    assert "search" in profile.prompt_summary.available_tools
    assert _stored_profile(workspace_repo)["runtime_tool_capabilities"]["items"][0]["capability_id"] == "search"
    assert "search" in _stored_profile(workspace_repo)["prompt_summary"]["available_tools"]
    assert any(record.message == "sandbox_profile_runtime_tool_snapshot_recorded" for record in caplog.records)


def test_record_runtime_tool_snapshot_should_reject_when_existing_profile_missing(caplog) -> None:
    service, workspace_repo = _build_service()
    snapshot = RuntimeToolCapabilitySnapshot(
        items=[
            RuntimeToolCapabilitySnapshotItem(
                capability_id="search",
                tool_family="search",
                source="local",
            ),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="app.application.service.sandbox_capability_profile_service"):
        with pytest.raises(Exception):
            asyncio.run(service.record_runtime_tool_snapshot(
                user_id="user-1",
                session_id="session-1",
                snapshot=snapshot,
            ))

    assert workspace_repo.workspace is not None
    assert SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY not in workspace_repo.workspace.environment_summary
    assert any(
        record.message == "sandbox_profile_runtime_tool_snapshot_rejected"
        and getattr(record, "reason_code", "") == "sandbox_profile_runtime_tool_snapshot_rejected"
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "workspace",
    [
        Workspace(id="workspace-1", session_id="session-1", user_id="user-1", current_run_id=None, sandbox_id="sandbox-1"),
        Workspace(id="workspace-1", session_id="session-1", user_id="user-1", current_run_id="run-1", sandbox_id=None),
    ],
)
def test_record_runtime_tool_snapshot_should_reject_missing_run_or_sandbox(workspace: Workspace) -> None:
    session = Session(id="session-1", user_id="user-1", workspace_id="workspace-1", current_run_id="run-1")
    service, workspace_repo = _build_service(workspace=workspace, session=session)
    snapshot = RuntimeToolCapabilitySnapshot()

    with pytest.raises(Exception):
        asyncio.run(service.record_runtime_tool_snapshot(
            user_id="user-1",
            session_id="session-1",
            snapshot=snapshot,
        ))

    assert workspace_repo.saved_workspaces == []
