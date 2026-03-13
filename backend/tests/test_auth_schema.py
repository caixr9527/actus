from datetime import datetime

import pytest
from pydantic import ValidationError

from app.interfaces.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    TokenPairResponse,
    CurrentUserResponse,
    UpdatePasswordRequest,
)


def test_register_request_invalid_email_should_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="invalid-email", password="Password123")


def test_register_request_weak_password_should_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="tester@example.com", password="Password")


def test_register_request_password_with_illegal_symbol_should_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="tester@example.com", password="Password123~")


def test_register_request_password_with_common_symbol_should_pass_validation() -> None:
    payload = RegisterRequest(email="tester@example.com", password="Password123!")
    assert payload.password == "Password123!"


def test_login_request_valid_email_should_pass_validation() -> None:
    payload = LoginRequest(email="tester@example.com", password="Password123")
    assert payload.email == "tester@example.com"


def test_token_pair_response_should_use_expected_defaults() -> None:
    payload = TokenPairResponse(
        access_token="access-token",
        refresh_token="refresh-token",
    )
    assert payload.token_type == "Bearer"
    assert payload.access_token_expires_in == 1800
    assert payload.refresh_token_expires_in == 604800


def test_current_user_response_should_allow_optional_login_fields() -> None:
    payload = CurrentUserResponse(
        user_id="u-1",
        email="tester@example.com",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    assert payload.last_login_at is None
    assert payload.last_login_ip is None


def test_update_password_request_should_validate_min_length() -> None:
    with pytest.raises(ValidationError):
        UpdatePasswordRequest(old_password="short", new_password="Password123")


def test_update_password_request_should_validate_old_password_strength() -> None:
    with pytest.raises(ValidationError):
        UpdatePasswordRequest(old_password="12345678", new_password="Password123")
