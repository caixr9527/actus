from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.interfaces.schemas import auth as auth_schema
from app.interfaces.schemas.auth import (
    SendRegisterCodeRequest,
    RegisterRequest,
    LoginRequest,
    AccessTokenResponse,
    LogoutResponse,
    CurrentUserResponse,
    UpdatePasswordRequest,
)


def test_send_register_code_request_invalid_email_should_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        SendRegisterCodeRequest(email="invalid-email")


def test_send_register_code_request_valid_email_should_pass_validation() -> None:
    payload = SendRegisterCodeRequest(email="tester@example.com")
    assert payload.email == "tester@example.com"


def test_register_request_invalid_email_should_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(
            email="invalid-email",
            password="Password123",
            confirm_password="Password123",
        )


def test_register_request_weak_password_should_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(
            email="tester@example.com",
            password="Password",
            confirm_password="Password",
        )


def test_register_request_password_with_illegal_symbol_should_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(
            email="tester@example.com",
            password="Password123~",
            confirm_password="Password123~",
        )


def test_register_request_password_with_common_symbol_should_pass_validation() -> None:
    payload = RegisterRequest(
        email="tester@example.com",
        password="Password123!",
        confirm_password="Password123!",
    )
    assert payload.password == "Password123!"


def test_register_request_verification_code_with_non_digits_should_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(
            email="tester@example.com",
            password="Password123!",
            verification_code="12ab56",
        )


def test_register_request_verification_code_with_valid_digits_should_pass_validation() -> None:
    payload = RegisterRequest(
        email="tester@example.com",
        password="Password123!",
        confirm_password="Password123!",
        verification_code="123456",
    )
    assert payload.verification_code == "123456"


def test_register_request_verification_code_should_follow_configured_length(monkeypatch) -> None:
    monkeypatch.setattr(
        auth_schema,
        "get_settings",
        lambda: SimpleNamespace(auth_register_code_length=4),
    )
    payload = RegisterRequest(
        email="tester@example.com",
        password="Password123!",
        confirm_password="Password123!",
        verification_code="1234",
    )
    assert payload.verification_code == "1234"


def test_register_request_verification_code_with_wrong_configured_length_should_raise_validation_error(
        monkeypatch,
) -> None:
    monkeypatch.setattr(
        auth_schema,
        "get_settings",
        lambda: SimpleNamespace(auth_register_code_length=4),
    )
    with pytest.raises(ValidationError):
        RegisterRequest(
            email="tester@example.com",
            password="Password123!",
            confirm_password="Password123!",
            verification_code="123456",
        )


def test_login_request_valid_email_should_pass_validation() -> None:
    payload = LoginRequest(email="tester@example.com", password="Password123")
    assert payload.email == "tester@example.com"


def test_access_token_response_should_use_expected_defaults() -> None:
    payload = AccessTokenResponse(
        access_token="access-token",
    )
    assert payload.token_type == "Bearer"
    assert payload.access_token_expires_in == 1800


def test_current_user_response_should_allow_optional_login_fields() -> None:
    payload = CurrentUserResponse(
        user_id="u-1",
        email="tester@example.com",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    assert payload.last_login_at is None
    assert payload.last_login_ip is None


def test_logout_response_should_default_success_true() -> None:
    payload = LogoutResponse()
    assert payload.success is True


def test_update_password_request_should_validate_min_length() -> None:
    with pytest.raises(ValidationError):
        UpdatePasswordRequest(
            old_password="short",
            new_password="Password123",
            confirm_password="Password123",
        )


def test_update_password_request_should_validate_old_password_strength() -> None:
    with pytest.raises(ValidationError):
        UpdatePasswordRequest(
            old_password="12345678",
            new_password="Password123",
            confirm_password="Password123",
        )
