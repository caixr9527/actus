import asyncio
import logging
from datetime import datetime, timedelta

import pytest

from app.domain.models import Workspace
from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY,
    RuntimeToolCapabilitySnapshot,
    RuntimeToolCapabilitySnapshotItem,
    SandboxCapabilityItem,
    SandboxCapabilityKind,
    SandboxCapabilityProfile,
    SandboxCapabilityPromptSummary,
    SandboxCapabilityStatus,
    SandboxProfileRefreshReason,
    build_sandbox_capability_profile_from_draft,
)
from app.domain.services.workspace_runtime import WorkspaceRuntimeService


class _WorkspaceArtifactRepo:
    async def list_by_user_workspace_id(self, *, user_id: str, workspace_id: str):
        return []


class _WorkspaceRepo:
    def __init__(self, workspace: Workspace | None = None) -> None:
        self.workspace = workspace
        self.saved_workspaces: list[Workspace] = []

    async def save(self, workspace: Workspace) -> None:
        cloned = workspace.model_copy(deep=True)
        self.workspace = cloned
        self.saved_workspaces.append(cloned)

    async def get_by_session_id_for_user(self, *, session_id: str, user_id: str):
        if self.workspace is None:
            return None
        if self.workspace.session_id != session_id or self.workspace.user_id != user_id:
            return None
        return self.workspace.model_copy(deep=True)


class _UoW:
    def __init__(self, workspace_repo: _WorkspaceRepo) -> None:
        self.workspace = workspace_repo
        self.workspace_artifact = _WorkspaceArtifactRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _build_service(workspace_repo: _WorkspaceRepo) -> WorkspaceRuntimeService:
    return WorkspaceRuntimeService(
        session_id="session-1",
        user_id="user-1",
        uow_factory=lambda: _UoW(workspace_repo=workspace_repo),
    )


def _build_profile() -> SandboxCapabilityProfile:
    now = datetime(2026, 5, 5, 10, 0, 0)
    draft = {
        "schema_version": "sandbox_capability_profile.v1",
        "profile_id": "profile-1",
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "sandbox_id": "sandbox-1",
        "generated_at": now,
        "expires_at": now + timedelta(minutes=30),
        "refresh_reason": SandboxProfileRefreshReason.SANDBOX_CREATED,
        "health_status": SandboxCapabilityStatus.AVAILABLE,
        "cwd": "/workspace",
        "capabilities": [
            SandboxCapabilityItem(
                kind=SandboxCapabilityKind.PYTHON,
                name="python3",
                status=SandboxCapabilityStatus.AVAILABLE,
                version="3.12.1",
                path="/usr/bin/python3",
            ),
        ],
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
            available_runtime={"python": "3.12.1"},
            available_tools=["shell"],
            generated_at=now,
        ),
    }
    return build_sandbox_capability_profile_from_draft(draft)


def test_workspace_runtime_should_record_profile_under_fixed_key_only() -> None:
    workspace_repo = _WorkspaceRepo(
        Workspace(
            id="workspace-1",
            session_id="session-1",
            user_id="user-1",
            environment_summary={"existing": {"value": 1}},
        )
    )
    service = _build_service(workspace_repo)
    profile = _build_profile()

    workspace = asyncio.run(service.record_sandbox_capability_profile(profile=profile))

    stored_profile = workspace.environment_summary[SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY]
    assert workspace.environment_summary["existing"] == {"value": 1}
    assert set(workspace.environment_summary.keys()) == {
        "existing",
        SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY,
    }
    assert stored_profile["schema_version"] == "sandbox_capability_profile.v1"
    assert isinstance(stored_profile["runtime_tool_capabilities"], dict)
    assert stored_profile["runtime_tool_capabilities"]["items"][0]["capability_id"] == "local_shell"
    assert not isinstance(stored_profile["runtime_tool_capabilities"], list)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("user_id", "user-other"),
        ("session_id", "session-other"),
        ("workspace_id", "workspace-other"),
    ],
)
def test_workspace_runtime_should_reject_profile_scope_mismatch(field_name: str, field_value: str) -> None:
    existing_profile = _build_profile().model_dump(mode="json")
    workspace_repo = _WorkspaceRepo(
        Workspace(
            id="workspace-1",
            session_id="session-1",
            user_id="user-1",
            environment_summary={
                SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: existing_profile,
            },
        )
    )
    service = _build_service(workspace_repo)
    mismatched_profile = _build_profile().model_copy(update={field_name: field_value})

    with pytest.raises(ValueError, match="sandbox capability profile scope 与当前 workspace 不一致"):
        asyncio.run(service.record_sandbox_capability_profile(profile=mismatched_profile))

    assert workspace_repo.workspace is not None
    assert workspace_repo.saved_workspaces == []
    assert workspace_repo.workspace.environment_summary[SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY] == existing_profile


def test_workspace_runtime_should_read_valid_profile() -> None:
    profile = _build_profile()
    workspace_repo = _WorkspaceRepo(
        Workspace(
            id="workspace-1",
            session_id="session-1",
            user_id="user-1",
            environment_summary={
                SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: profile.model_dump(mode="json"),
            },
        )
    )
    service = _build_service(workspace_repo)

    restored = asyncio.run(service.get_sandbox_capability_profile())

    assert restored is not None
    assert restored.profile_hash == profile.profile_hash
    assert isinstance(restored.runtime_tool_capabilities, RuntimeToolCapabilitySnapshot)


def test_workspace_runtime_should_reject_invalid_profile_payload(caplog) -> None:
    profile_payload = _build_profile().model_dump(mode="json")
    profile_payload["legacy_extra"] = "not allowed"
    workspace_repo = _WorkspaceRepo(
        Workspace(
            id="workspace-1",
            session_id="session-1",
            user_id="user-1",
            environment_summary={
                SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: profile_payload,
            },
        )
    )
    service = _build_service(workspace_repo)

    with caplog.at_level(logging.WARNING, logger="app.domain.services.workspace_runtime.service"):
        restored = asyncio.run(service.get_sandbox_capability_profile())

    assert restored is None
    assert any(
        record.message == "sandbox_profile_invalid_payload"
        and getattr(record, "reason_code", "") == "sandbox_profile_payload_invalid"
        for record in caplog.records
    )
