from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    RuntimeToolCapabilitySnapshot,
    RuntimeToolCapabilitySnapshotItem,
    SandboxCapabilityItem,
    SandboxCapabilityKind,
    SandboxCapabilityProfile,
    SandboxCapabilityPromptSummary,
    SandboxCapabilityStatus,
    SandboxProfileRefreshReason,
    SandboxResourceLimits,
    build_profile_hash_from_payload,
    build_sandbox_capability_profile_from_draft,
)


def _profile_draft(*, generated_at: datetime | None = None, python_version: str = "3.12.1") -> dict:
    now = generated_at or datetime(2026, 5, 5, 10, 0, 0)
    capabilities = [
        SandboxCapabilityItem(
            kind=SandboxCapabilityKind.PYTHON,
            name="python3",
            status=SandboxCapabilityStatus.AVAILABLE,
            version=python_version,
            path="/usr/bin/python3",
            details={},
        ),
        SandboxCapabilityItem(
            kind=SandboxCapabilityKind.NODE,
            name="node",
            status=SandboxCapabilityStatus.UNAVAILABLE,
            reason_code="node_not_installed",
        ),
    ]
    runtime_tool_capabilities = RuntimeToolCapabilitySnapshot(
        items=[
            RuntimeToolCapabilitySnapshotItem(
                capability_id="local_shell",
                tool_family="shell",
                source="local",
            ),
            RuntimeToolCapabilitySnapshotItem(
                capability_id="sandbox_file",
                tool_family="file",
                source="local",
            ),
        ]
    )
    return {
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
        "capabilities": capabilities,
        "resource_limits": SandboxResourceLimits(
            max_file_read_bytes=10485760,
            max_command_seconds=600,
            max_tool_iterations=20,
            writable_dirs=["/workspace", "/tmp"],
            readable_dirs=["/workspace", "/tmp"],
            network_policy="restricted",
        ),
        "disabled_capabilities": ["node", "node"],
        "confirmation_required_capabilities": [],
        "runtime_tool_capabilities": runtime_tool_capabilities,
        "prompt_summary": SandboxCapabilityPromptSummary(
            health_status=SandboxCapabilityStatus.AVAILABLE,
            cwd="/workspace",
            available_runtime={"python": python_version},
            available_tools=["shell", "file"],
            unavailable_capabilities=["node"],
            requires_confirmation=[],
            resource_limits={
                "writable_dirs": ["/workspace", "/tmp"],
                "max_command_seconds": 600,
            },
            generated_at=now,
        ),
        "last_refresh_error": None,
    }


def _build_profile(**overrides) -> SandboxCapabilityProfile:
    draft = _profile_draft()
    draft.update(overrides)
    return build_sandbox_capability_profile_from_draft(draft)


def test_profile_contract_should_build_valid_profile_from_draft_hash_order() -> None:
    profile = _build_profile()

    assert profile.schema_version == "sandbox_capability_profile.v1"
    assert profile.profile_hash.startswith("sha256:")
    assert profile.runtime_tool_capabilities.items[0].capability_id == "local_shell"
    assert isinstance(profile.prompt_summary, SandboxCapabilityPromptSummary)


def test_profile_contract_should_reject_draft_with_existing_hash() -> None:
    draft = _profile_draft()
    draft["profile_hash"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError):
        build_sandbox_capability_profile_from_draft(draft)


def test_profile_contract_should_reject_invalid_status_and_scope() -> None:
    draft = _profile_draft()
    draft["profile_hash"] = build_profile_hash_from_payload(draft)
    draft["health_status"] = "ready"

    with pytest.raises(ValidationError):
        SandboxCapabilityProfile.model_validate(draft)

    with pytest.raises(ValidationError):
        _build_profile(sandbox_id="")


def test_profile_contract_should_forbid_unknown_fields() -> None:
    profile_payload = _build_profile().model_dump(mode="json")
    profile_payload["legacy_environment"] = {"python": "3.12"}

    with pytest.raises(ValidationError):
        SandboxCapabilityProfile.model_validate(profile_payload)

    prompt_payload = profile_payload["prompt_summary"]
    prompt_payload["secret_extra"] = "should-not-enter-prompt"
    with pytest.raises(ValidationError):
        SandboxCapabilityPromptSummary.model_validate(prompt_payload)


def test_profile_contract_should_reject_runtime_tool_snapshot_string_array() -> None:
    profile_payload = _build_profile().model_dump(mode="json")
    profile_payload["runtime_tool_capabilities"] = ["local_shell", "sandbox_file"]

    with pytest.raises(ValidationError):
        SandboxCapabilityProfile.model_validate(profile_payload)


def test_profile_contract_should_reject_hash_mismatch() -> None:
    profile_payload = _build_profile().model_dump(mode="json")
    profile_payload["capabilities"][0]["version"] = "3.13.0"

    with pytest.raises(ValidationError):
        SandboxCapabilityProfile.model_validate(profile_payload)


def test_profile_hash_should_be_stable_for_order_and_time_changes() -> None:
    draft = _profile_draft()
    first_hash = build_profile_hash_from_payload(draft)
    reordered_draft = _profile_draft(generated_at=datetime(2026, 5, 5, 11, 0, 0))
    reordered_draft["capabilities"] = list(reversed(reordered_draft["capabilities"]))
    reordered_draft["disabled_capabilities"] = ["node", "node"]
    reordered_draft["runtime_tool_capabilities"] = RuntimeToolCapabilitySnapshot(
        items=list(reversed(reordered_draft["runtime_tool_capabilities"].items))
    )
    reordered_draft["last_refresh_error"] = {
        "reason_code": "probe_failed",
        "stage": "probe",
        "message": "redacted error changed",
        "occurred_at": "2026-05-05T11:00:00",
        "retryable": True,
        "probe_status": "unknown",
    }

    assert build_profile_hash_from_payload(reordered_draft) == first_hash


def test_profile_hash_should_be_stable_for_duplicate_sort_keys() -> None:
    draft = _profile_draft()
    draft["capabilities"] = [
        SandboxCapabilityItem(
            kind=SandboxCapabilityKind.PYTHON,
            name="python",
            status=SandboxCapabilityStatus.AVAILABLE,
            version="3.12.1",
            path="/usr/bin/python",
            reason_code="",
        ),
        SandboxCapabilityItem(
            kind=SandboxCapabilityKind.PYTHON,
            name="python",
            status=SandboxCapabilityStatus.DEGRADED,
            version="3.12.1",
            path="/usr/bin/python",
            reason_code="version_probe_timeout",
        ),
    ]
    first_hash = build_profile_hash_from_payload(draft)
    draft["capabilities"] = list(reversed(draft["capabilities"]))

    assert build_profile_hash_from_payload(draft) == first_hash


def test_profile_hash_should_change_for_capability_and_runtime_tool_changes() -> None:
    first_hash = build_profile_hash_from_payload(_profile_draft())

    version_changed = _profile_draft(python_version="3.13.0")
    assert build_profile_hash_from_payload(version_changed) != first_hash

    tool_changed = _profile_draft()
    tool_changed["runtime_tool_capabilities"] = RuntimeToolCapabilitySnapshot(
        items=[
            *tool_changed["runtime_tool_capabilities"].items,
            RuntimeToolCapabilitySnapshotItem(
                capability_id="search",
                tool_family="search",
                source="local",
            ),
        ]
    )
    assert build_profile_hash_from_payload(tool_changed) != first_hash


def test_profile_contract_should_accept_fixed_storage_shape_example_equivalent() -> None:
    profile = _build_profile()
    dumped = profile.model_dump(mode="json")

    restored = SandboxCapabilityProfile.model_validate(dumped)

    assert restored.profile_hash == profile.profile_hash
    assert isinstance(restored.runtime_tool_capabilities, RuntimeToolCapabilitySnapshot)
    assert restored.runtime_tool_capabilities.items[0].tool_family == "shell"
