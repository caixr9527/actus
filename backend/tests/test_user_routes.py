from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.errors import BadRequestError
from app.domain.models import User, UserProfile
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.services import get_user_service
from app.interfaces.endpoints.users_routes import router as users_router
from app.interfaces.errors.exception_handlers import register_exception_handlers


def _build_current_user() -> User:
    return User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
        password_salt="salt",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


class _FakeUserService:
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
    ) -> None:
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
    ) -> None:
        raise BadRequestError("旧密码错误")


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
    app.dependency_overrides[get_user_service] = lambda: _FakeUserService()
    app.dependency_overrides[get_current_user] = _build_current_user

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


def test_update_current_user_password_route_should_map_bad_request_error() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(users_router, prefix="/api")
    app.dependency_overrides[get_user_service] = lambda: _ErrorUserService()
    app.dependency_overrides[get_current_user] = _build_current_user

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
