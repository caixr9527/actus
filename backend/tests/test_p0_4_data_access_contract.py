import pytest

from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    DefaultDataClassificationPolicy,
    PrivacyLevel,
    RetentionPolicyKind,
    is_cross_scope_access_allowed,
    normalize_tenant_id,
)


def test_p0_4_contract_should_reject_empty_tenant_id() -> None:
    with pytest.raises(ValueError):
        normalize_tenant_id("")


def test_p0_4_contract_should_not_allow_missing_or_cross_user_scope() -> None:
    assert is_cross_scope_access_allowed(source_user_id="user-1", target_user_id="user-1") is True
    assert is_cross_scope_access_allowed(source_user_id="user-1", target_user_id="user-2") is False
    assert is_cross_scope_access_allowed(source_user_id=None, target_user_id="user-1") is False
    assert is_cross_scope_access_allowed(source_user_id="user-1", target_user_id=None) is False


def test_p0_4_default_classification_policy_should_keep_memory_private_to_user_scope() -> None:
    policy = DefaultDataClassificationPolicy()

    result = policy.classify_data(
        tenant_id="user-1",
        origin=DataOrigin.LONG_TERM_MEMORY,
    )

    assert result.tenant_id == "user-1"
    assert result.origin == DataOrigin.LONG_TERM_MEMORY
    assert result.trust_level == DataTrustLevel.SYSTEM_GENERATED
    assert result.privacy_level == PrivacyLevel.SENSITIVE
    assert result.retention_policy == RetentionPolicyKind.USER_MEMORY
