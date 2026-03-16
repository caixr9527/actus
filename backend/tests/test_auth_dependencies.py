import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Optional

import jwt
import pytest
from starlette.requests import HTTPConnection

from app.application.errors import UnauthorizedError
from app.domain.models import User, UserStatus
from app.interfaces.dependencies import auth as auth_dependencies


class _FakeUserRepository:
    def __init__(self, user: Optional[User]) -> None:
        self._user = user

    async def get_by_id(self, user_id: str) -> Optional[User]:
        if self._user is None:
            return None
        return self._user if self._user.id == user_id else None


class _FakeUoW:
    def __init__(self, user: Optional[User]) -> None:
        self.user = _FakeUserRepository(user)

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeAccessTokenBlacklistStore:
    def __init__(self, blacklisted_tokens: Optional[set[str]] = None) -> None:
        self.blacklisted_tokens = blacklisted_tokens or set()

    async def is_access_token_blacklisted(self, access_token: str) -> bool:
        return access_token in self.blacklisted_tokens


def _build_access_token(
        *,
        user_id: str,
        email: str = "tester@example.com",
        secret_key: str = "unit-test-secret",
        algorithm: str = "HS256",
        expires_delta: timedelta = timedelta(minutes=30),
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "email": email,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def _build_http_connection(authorization: Optional[str], scope_type: str = "http") -> HTTPConnection:
    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("utf-8")))
    return HTTPConnection(
        scope={
            "type": scope_type,
            "method": "GET",
            "path": "/api/sessions",
            "headers": headers,
        }
    )


def _patch_auth_dependency_runtime(monkeypatch, *, user: Optional[User], blacklisted_tokens: Optional[set[str]] = None):
    monkeypatch.setattr(
        auth_dependencies,
        "get_settings",
        lambda: SimpleNamespace(
            auth_jwt_secret="unit-test-secret",
            auth_jwt_algorithm="HS256",
        ),
    )
    monkeypatch.setattr(auth_dependencies, "get_uow", lambda: _FakeUoW(user))
    monkeypatch.setattr(
        auth_dependencies,
        "get_access_token_blacklist_store",
        lambda: _FakeAccessTokenBlacklistStore(blacklisted_tokens=blacklisted_tokens),
    )


def test_get_current_auth_context_should_parse_user_and_write_state(monkeypatch) -> None:
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )
    _patch_auth_dependency_runtime(monkeypatch, user=user)
    access_token = _build_access_token(user_id=user.id)
    connection = _build_http_connection(f"Bearer {access_token}")

    context = asyncio.run(auth_dependencies.get_current_auth_context(connection))

    assert context.user.id == user.id
    assert context.access_token == access_token
    assert connection.state.current_user.id == user.id
    assert connection.state.current_access_token == access_token
    assert connection.state.current_access_token_payload["user_id"] == user.id


def test_get_current_auth_context_should_reject_when_authorization_header_missing(monkeypatch) -> None:
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )
    _patch_auth_dependency_runtime(monkeypatch, user=user)
    connection = _build_http_connection(None)

    with pytest.raises(UnauthorizedError) as exc:
        asyncio.run(auth_dependencies.get_current_auth_context(connection))
    assert exc.value.code == 401


def test_get_current_auth_context_should_reject_blacklisted_access_token(monkeypatch) -> None:
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )
    access_token = _build_access_token(user_id=user.id)
    _patch_auth_dependency_runtime(monkeypatch, user=user, blacklisted_tokens={access_token})
    connection = _build_http_connection(f"Bearer {access_token}")

    with pytest.raises(UnauthorizedError) as exc:
        asyncio.run(auth_dependencies.get_current_auth_context(connection))
    assert exc.value.msg == "登录状态已失效，请重新登录"


def test_get_current_auth_context_should_reject_invalid_signature_token(monkeypatch) -> None:
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )
    _patch_auth_dependency_runtime(monkeypatch, user=user)
    token_with_other_secret = _build_access_token(user_id=user.id, secret_key="another-secret")
    connection = _build_http_connection(f"Bearer {token_with_other_secret}")

    with pytest.raises(UnauthorizedError) as exc:
        asyncio.run(auth_dependencies.get_current_auth_context(connection))
    assert exc.value.msg == "Access Token 无效，请重新登录"


def test_get_current_auth_context_should_reject_disabled_user(monkeypatch) -> None:
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
        status=UserStatus.DISABLED,
    )
    _patch_auth_dependency_runtime(monkeypatch, user=user)
    access_token = _build_access_token(user_id=user.id)
    connection = _build_http_connection(f"Bearer {access_token}")

    with pytest.raises(UnauthorizedError) as exc:
        asyncio.run(auth_dependencies.get_current_auth_context(connection))
    assert exc.value.msg == "账号状态异常，暂不可访问"


def test_get_current_auth_context_should_reject_expired_token(monkeypatch) -> None:
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )
    _patch_auth_dependency_runtime(monkeypatch, user=user)
    expired_access_token = _build_access_token(
        user_id=user.id,
        expires_delta=timedelta(seconds=-5),
    )
    connection = _build_http_connection(f"Bearer {expired_access_token}")

    with pytest.raises(UnauthorizedError) as exc:
        asyncio.run(auth_dependencies.get_current_auth_context(connection))
    assert exc.value.msg == "登录已过期，请重新登录"


def test_get_current_auth_context_should_support_websocket_connection(monkeypatch) -> None:
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )
    _patch_auth_dependency_runtime(monkeypatch, user=user)
    access_token = _build_access_token(user_id=user.id)
    websocket_connection = _build_http_connection(f"Bearer {access_token}", scope_type="websocket")

    context = asyncio.run(auth_dependencies.get_current_auth_context(websocket_connection))

    assert context.user.id == user.id
