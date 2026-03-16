from datetime import datetime
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.errors import BadRequestError
from app.domain.models import User, UserProfile
from app.interfaces.dependencies.auth import AuthContext, get_current_auth_context, get_current_user
from app.interfaces.dependencies.services import (
    get_access_token_blacklist_store,
    get_refresh_token_store,
    get_user_service,
)
from app.interfaces.endpoints.users_routes import router as users_router
from app.interfaces.errors.exception_handlers import register_exception_handlers


def _assert_auth_security_headers(response) -> None:
    assert response.headers.get("cache-control") == "no-store"
    assert response.headers.get("pragma") == "no-cache"
    assert response.headers.get("x-frame-options") == "SAMEORIGIN"
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


def _build_current_user() -> User:
    return User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


class _FakeUserService:
    def __init__(self) -> None:
        self.last_password_update_kwargs: dict | None = None

    async def get_current_user_profile(self, user_id: str):
        user = _build_current_user()
        profile = UserProfile(
            user_id=user_id,
            nickname="tester",
            avatar_url="https://example.com/avatar.png",
            timezone="Asia/Shanghai",
            locale="zh-CN",
        )
        return user, profile

    async def update_current_user_profile(self, user_id: str, updates: dict[str, str | None]):
        user = _build_current_user()
        profile = UserProfile(
            user_id=user_id,
            nickname=updates.get("nickname", "tester"),
            avatar_url=updates.get("avatar_url", "https://example.com/avatar.png"),
            timezone=updates.get("timezone", "Asia/Shanghai"),
            locale=updates.get("locale", "zh-CN"),
        )
        return user, profile

    async def update_current_user_password(
            self,
            user_id: str,
            old_password: str,
            new_password: str,
            confirm_password: str,
            **kwargs,
    ) -> None:
        self.last_password_update_kwargs = {
            "user_id": user_id,
            "old_password": old_password,
            "new_password": new_password,
            "confirm_password": confirm_password,
            **kwargs,
        }
        return None


class _ErrorUserService:
    async def get_current_user_profile(self, user_id: str):
        raise BadRequestError("用户不存在，请重新登录")

    async def update_current_user_profile(self, user_id: str, updates: dict[str, str | None]):
        raise BadRequestError("至少需要更新一个字段")

    async def update_current_user_password(
            self,
            user_id: str,
            old_password: str,
            new_password: str,
            confirm_password: str,
            **kwargs,
    ) -> None:
        raise BadRequestError("旧密码错误")


class _FakeRefreshTokenStore:
    def __init__(self) -> None:
        self.revoked_user_ids: list[str] = []

    async def revoke_user_refresh_tokens(self, user_id: str) -> None:
        self.revoked_user_ids.append(user_id)


class _FakeAccessTokenBlacklistStore:
    def __init__(self) -> None:
        self.blacklisted_tokens: list[tuple[str, int]] = []

    async def add_access_token_to_blacklist(
            self,
            access_token: str,
            expires_in_seconds: int,
    ) -> None:
        self.blacklisted_tokens.append((access_token, expires_in_seconds))


def _build_auth_context() -> AuthContext:
    return AuthContext(
        user=_build_current_user(),
        access_token="access-token-1",
        token_payload={"exp": int(datetime.now().timestamp()) + 300},
    )


def test_get_current_user_profile_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(users_router, prefix="/api")
    app.dependency_overrides[get_user_service] = lambda: _FakeUserService()
    app.dependency_overrides[get_current_user] = _build_current_user

    with TestClient(app) as client:
        response = client.get("/api/users/me")

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "获取成功"
    assert payload["data"]["email"] == "tester@example.com"
    assert payload["data"]["nickname"] == "tester"


def test_update_current_user_profile_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(users_router, prefix="/api")
    app.dependency_overrides[get_user_service] = lambda: _FakeUserService()
    app.dependency_overrides[get_current_user] = _build_current_user

    with TestClient(app) as client:
        response = client.patch(
            "/api/users/me",
            json={"nickname": "neo"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "更新成功"
    assert payload["data"]["user"]["nickname"] == "neo"


def test_update_current_user_password_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(users_router, prefix="/api")
    fake_user_service = _FakeUserService()
    refresh_token_store = _FakeRefreshTokenStore()
    access_token_blacklist_store = _FakeAccessTokenBlacklistStore()
    app.dependency_overrides[get_user_service] = lambda: fake_user_service
    app.dependency_overrides[get_current_auth_context] = _build_auth_context
    app.dependency_overrides[get_refresh_token_store] = lambda: refresh_token_store
    app.dependency_overrides[get_access_token_blacklist_store] = lambda: access_token_blacklist_store

    with TestClient(app) as client:
        response = client.patch(
            "/api/users/me/password",
            json={
                "old_password": "Password123!",
                "new_password": "Password456!",
                "confirm_password": "Password456!",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "密码更新成功"
    assert payload["data"]["success"] is True
    assert fake_user_service.last_password_update_kwargs is not None
    assert fake_user_service.last_password_update_kwargs["user_id"] == "user-1"
    assert fake_user_service.last_password_update_kwargs["refresh_token_store"] is refresh_token_store
    assert (
        fake_user_service.last_password_update_kwargs["access_token_blacklist_store"]
        is access_token_blacklist_store
    )
    assert fake_user_service.last_password_update_kwargs["current_access_token"] == "access-token-1"
    assert fake_user_service.last_password_update_kwargs["access_token_expires_in_seconds"] > 0
    _assert_auth_security_headers(response)


def test_update_current_user_password_route_should_reject_http_when_auth_require_https_enabled(
        monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.interfaces.dependencies.request_security.get_settings",
        lambda: SimpleNamespace(auth_require_https=True),
    )
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(users_router, prefix="/api")
    app.dependency_overrides[get_user_service] = lambda: _FakeUserService()
    app.dependency_overrides[get_current_auth_context] = _build_auth_context
    app.dependency_overrides[get_refresh_token_store] = lambda: _FakeRefreshTokenStore()
    app.dependency_overrides[get_access_token_blacklist_store] = lambda: _FakeAccessTokenBlacklistStore()

    with TestClient(app) as client:
        response = client.patch(
            "/api/users/me/password",
            json={
                "old_password": "Password123!",
                "new_password": "Password456!",
                "confirm_password": "Password456!",
            },
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "当前环境仅允许通过 HTTPS 访问该接口"


def test_update_current_user_password_route_should_allow_x_forwarded_proto_https_when_required(
        monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.interfaces.dependencies.request_security.get_settings",
        lambda: SimpleNamespace(auth_require_https=True),
    )
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(users_router, prefix="/api")
    app.dependency_overrides[get_user_service] = lambda: _FakeUserService()
    app.dependency_overrides[get_current_auth_context] = _build_auth_context
    app.dependency_overrides[get_refresh_token_store] = lambda: _FakeRefreshTokenStore()
    app.dependency_overrides[get_access_token_blacklist_store] = lambda: _FakeAccessTokenBlacklistStore()

    with TestClient(app) as client:
        response = client.patch(
            "/api/users/me/password",
            headers={"x-forwarded-proto": "https"},
            json={
                "old_password": "Password123!",
                "new_password": "Password456!",
                "confirm_password": "Password456!",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "密码更新成功"
    _assert_auth_security_headers(response)
    assert response.headers.get("strict-transport-security") == "max-age=31536000; includeSubDomains"


def test_update_current_user_password_route_should_map_bad_request_error() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(users_router, prefix="/api")
    app.dependency_overrides[get_user_service] = lambda: _ErrorUserService()
    app.dependency_overrides[get_current_auth_context] = _build_auth_context
    app.dependency_overrides[get_refresh_token_store] = lambda: _FakeRefreshTokenStore()
    app.dependency_overrides[get_access_token_blacklist_store] = lambda: _FakeAccessTokenBlacklistStore()

    with TestClient(app) as client:
        response = client.patch(
            "/api/users/me/password",
            json={
                "old_password": "Password123!",
                "new_password": "Password456!",
                "confirm_password": "Password456!",
            },
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "旧密码错误"
