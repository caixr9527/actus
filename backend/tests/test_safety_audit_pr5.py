from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from app.application.service.safety_audit_event_projector import SafetyAuditEventProjector
from app.application.service.safety_audit_ledger_service import SafetyAuditLedgerService
from app.domain.models import MessageEvent, SafetyAuditEvent, WorkflowRunEventRecord
from app.domain.models.safety_audit import (
    NonToolSafetyAuditCommand,
    SafetyAuditDecision,
    SafetyAuditExternalCapabilityGovernanceDigest,
    SafetyAuditNonToolActionKind,
    SafetyAuditRecordCommand,
    SafetyAuditRiskClassificationInput,
    SafetyAuditRiskLevel,
    classify_safety_audit_risk,
)
from app.infrastructure.models.safety_audit import SafetyAuditRecordModel

from tests.test_safety_audit_pr2 import _FakeSafetyAuditRepository, _FakeUow, _event, _non_tool_command, _record_command, _scope
from tests.test_safety_audit_pr4 import _Coordinator, _Recorder, _Uow


def test_external_capability_governance_digest_is_strict_and_normalized() -> None:
    digest = SafetyAuditExternalCapabilityGovernanceDigest.model_validate(
        {
            "external_provider": "  mcp:github  ",
            "manifest_ref": "  manifest:v1:github  ",
            "permission_claims": [" external_send ", "credential", "credential", ""],
            "network_required": True,
            "filesystem_access": "read_write",
        }
    )

    assert digest.external_provider == "mcp:github"
    assert digest.manifest_ref == "manifest:v1:github"
    assert digest.permission_claims == ["credential", "external_send"]
    with pytest.raises(ValidationError):
        SafetyAuditExternalCapabilityGovernanceDigest.model_validate(
            {
                "external_provider": "mcp:github",
                "manifest_ref": "manifest:v1:github",
                "permission_claims": [],
                "network_required": False,
                "filesystem_access": "read",
                "raw_args": {"secret": "token"},
            }
        )


@pytest.mark.parametrize(
    ("input_data", "expected", "matched_rule"),
    [
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="remote_search",
                tool_family="mcp",
                capability_id="mcp.github.search",
                action_kind="remote_search",
                reason_code="allow",
            ),
            SafetyAuditRiskLevel.HIGH,
            "external_capability_high_risk",
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="invoke_agent",
                tool_family="a2a",
                capability_id="a2a.agent.invoke",
                action_kind="invoke_agent",
                reason_code="allow",
            ),
            SafetyAuditRiskLevel.HIGH,
            "external_capability_high_risk",
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="govern_capability",
                tool_family="non_tool",
                capability_id="external.capability",
                action_kind="external_capability_governance",
                reason_code="allow",
                external_capability=SafetyAuditExternalCapabilityGovernanceDigest(
                    network_required=True,
                    filesystem_access="none",
                ),
            ),
            SafetyAuditRiskLevel.HIGH,
            "external_capability_high_risk",
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="govern_capability",
                tool_family="non_tool",
                capability_id="external.capability",
                action_kind="external_capability_governance",
                reason_code="allow",
                external_capability=SafetyAuditExternalCapabilityGovernanceDigest(
                    permission_claims=["user_data_export"],
                    network_required=True,
                    filesystem_access="read",
                ),
            ),
            SafetyAuditRiskLevel.CRITICAL,
            "sensitive_external_permission_claim",
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                reason_code="allow",
                external_capability=SafetyAuditExternalCapabilityGovernanceDigest(
                    permission_claims=["secret"],
                ),
            ),
            SafetyAuditRiskLevel.CRITICAL,
            "sensitive_external_permission_claim",
        ),
        (
            SafetyAuditRiskClassificationInput(
                decision=SafetyAuditDecision.ALLOW,
                normalized_function_name="inspect_local_manifest",
                tool_family="local",
                capability_id="local.manifest.read",
                action_kind="manifest_read",
                reason_code="allow",
                external_capability=SafetyAuditExternalCapabilityGovernanceDigest(
                    network_required=False,
                    filesystem_access="read",
                ),
            ),
            SafetyAuditRiskLevel.LOW,
            "low_reason_code",
        ),
    ],
)
def test_external_capability_risk_classifier_rules(
        input_data: SafetyAuditRiskClassificationInput,
        expected: SafetyAuditRiskLevel,
        matched_rule: str,
) -> None:
    result = classify_safety_audit_risk(input_data)
    assert result.risk_level == expected
    assert result.matched_rule == matched_rule


def test_confirmation_records_and_linkage_preserve_original_decision_payload() -> None:
    repo = _FakeSafetyAuditRepository()
    service = SafetyAuditLedgerService(uow_factory=lambda: _FakeUow(repo))
    require_command = _record_command(
        decision=SafetyAuditDecision.REQUIRE_CONFIRMATION,
        reason_code="shell_requires_confirmation",
        function_name="run_shell",
        normalized_function_name="run_shell",
        capability_id="shell.execute",
        tool_family="shell",
    )
    require_record = asyncio.run(service.record_constraint_decision(require_command))
    original = repo.records[0]
    approved_command = _record_command(
        decision=SafetyAuditDecision.CONFIRMATION_APPROVED,
        reason_code="user_confirmation_approved",
        function_name="run_shell",
        normalized_function_name="run_shell",
        capability_id="shell.execute",
        tool_family="shell",
    ).model_copy(update={"confirmation_id": "confirmation-1"})
    rejected_command = _record_command(
        decision=SafetyAuditDecision.CONFIRMATION_REJECTED,
        reason_code="user_confirmation_rejected",
        function_name="run_shell",
        normalized_function_name="run_shell",
        capability_id="shell.execute",
        tool_family="shell",
    ).model_copy(update={"tool_call_id": "call-2", "confirmation_id": "confirmation-2"})
    repo.events["msg-user-1"] = _event("msg-user-1", "message", role="user")
    repo.events["msg-assistant-1"] = _event("msg-assistant-1", "message", role="assistant")
    repo.events["wait-1"] = _event("wait-1", "wait")
    repo.events["resume-1"] = _event("resume-1", "resume")

    approved = asyncio.run(service.record_confirmation_decision(approved_command))
    rejected = asyncio.run(service.record_confirmation_decision(rejected_command))
    attached = asyncio.run(service.attach_confirmation_event(require_record.audit_id, "msg-user-1", scope=_scope()))

    updated_original = repo.records[0]
    assert require_record.record.decision == SafetyAuditDecision.REQUIRE_CONFIRMATION
    assert approved.record.decision == SafetyAuditDecision.CONFIRMATION_APPROVED
    assert rejected.record.decision == SafetyAuditDecision.CONFIRMATION_REJECTED
    assert attached.audit_id == require_record.audit_id
    assert updated_original.confirmation_event_id == "msg-user-1"
    assert updated_original.decision == original.decision
    assert updated_original.risk_level == original.risk_level
    assert updated_original.requested_args_digest == original.requested_args_digest
    assert updated_original.final_args_digest == original.final_args_digest
    assert updated_original.policy_trace == original.policy_trace
    assert updated_original.reason_code == original.reason_code
    assert updated_original.tool_call_fingerprint == original.tool_call_fingerprint
    for event_id in ["msg-assistant-1", "wait-1", "resume-1"]:
        with pytest.raises(Exception):
            asyncio.run(service.attach_confirmation_event(approved.audit_id, event_id, scope=_scope()))


def _external_governance_command() -> NonToolSafetyAuditCommand:
    scope = _scope()
    return NonToolSafetyAuditCommand(
        scope=scope,
        action_kind=SafetyAuditNonToolActionKind.EXTERNAL_CAPABILITY_GOVERNANCE,
        user_id=scope.user_id,
        session_id=str(scope.session_id),
        workspace_id=str(scope.workspace_id),
        run_id=str(scope.run_id),
        step_id=scope.current_step_id,
        action_id_hint="mcp-github-enable",
        capability_id="mcp.github",
        tool_family="mcp",
        function_name="external_capability_governance",
        requested_args={"server_name": "github"},
        final_args={"server_name": "github"},
        decision=SafetyAuditDecision.ALLOW,
        reason_code="allow",
        external_provider="mcp:github",
        external_capability=SafetyAuditExternalCapabilityGovernanceDigest(
            external_provider="mcp:github",
            manifest_ref="manifest:v1:github",
            permission_claims=["external_send"],
            network_required=True,
            filesystem_access="read",
        ),
    )


def test_non_tool_external_governance_forms_projectable_audit_record() -> None:
    repo = _FakeSafetyAuditRepository()
    ledger = SafetyAuditLedgerService(uow_factory=lambda: _FakeUow(repo))
    command = _external_governance_command()

    result = asyncio.run(ledger.record_external_capability_governance_decision(command))

    record = repo.records[0]
    assert result.audit_id == record.id
    assert result.record.risk_level == SafetyAuditRiskLevel.CRITICAL
    assert record.risk_classification_digest is not None
    assert record.risk_classification_digest.matched_rule == "sensitive_external_permission_claim"
    serialized = record.model_dump(mode="json")
    assert serialized["external_capability_governance"] == {
        "external_provider": "mcp:github",
        "manifest_ref": "manifest:v1:github",
        "permission_claims": ["external_send"],
        "network_required": True,
        "filesystem_access": "read",
    }
    assert serialized["requested_args_digest"]["fields"]["server_name"]["hash"].startswith("sha256:")
    assert "permission_claims" not in serialized["requested_args_digest"]["fields"]

    round_tripped = SafetyAuditRecordModel.from_domain(record).to_domain()
    assert round_tripped.external_capability_governance == record.external_capability_governance


def test_pr5_non_tool_records_are_projectable_without_business_route_integration() -> None:
    repo = _FakeSafetyAuditRepository()
    ledger = SafetyAuditLedgerService(uow_factory=lambda: _FakeUow(repo))
    download = asyncio.run(ledger.record_artifact_download_decision(_non_tool_command()))
    preview = asyncio.run(ledger.record_artifact_preview_decision(
        _non_tool_command(action_kind=SafetyAuditNonToolActionKind.ARTIFACT_PREVIEW)
    ))
    document = asyncio.run(ledger.record_document_preflight_decision(
        _non_tool_command(action_kind=SafetyAuditNonToolActionKind.DOCUMENT_PREFLIGHT)
    ))
    governance = asyncio.run(ledger.record_external_capability_governance_decision(_external_governance_command()))
    coordinator = _Coordinator(repo)
    projector = SafetyAuditEventProjector(
        uow_factory=lambda: _Uow(repo),
        runtime_state_coordinator=coordinator,
        recorder=_Recorder(repo),
    )

    for audit_id in [download.audit_id, preview.audit_id, document.audit_id, governance.audit_id]:
        projection = asyncio.run(projector.project_single_audit(scope=_scope(), audit_id=audit_id))
        assert projection.projected is True
        event = repo.events[str(projection.event_id)].event_payload
        assert isinstance(event, SafetyAuditEvent)
        assert event.payload.audit_refs[0].audit_id == audit_id

    assert len([event for event in repo.events.values() if event.event_type == "safety_audit"]) == 4
