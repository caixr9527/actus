from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.errors import BadRequestError
from app.domain.models import User, UserProfile
from app.interfaces.dependencies.auth import AuthContext, get_current_auth_context
from app.interfaces.endpoints.auth_routes import router as auth_router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from app.interfaces.schemas.auth import RegisterVerificationCodeResult, LoginResult, RefreshResult
from app.interfaces.dependencies.services import get_auth_service


class _FakeAuthService:
    async def send_register_verification_code(self, email: str):
        return RegisterVerificationCodeResult(
            verification_required=True,
            expires_in_seconds=300,
        )

    async def register(self, *, email: str, password: str, verification_code=None):
        return User(
            email=email.lower(),
            password="hashed-password",
            password_salt="salt",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    async def login(self, *, email: str, password: str, client_ip=None):
        user = User(
            email=email.lower(),
            password="hashed-password",
            password_salt="salt",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_login_at=datetime.now(),
            last_login_ip=client_ip,
        )
        profile = UserProfile(user_id=user.id, nickname="tester")
        return LoginResult(
            user=user,
            profile=profile,
            access_token="access-token",
            refresh_token="refresh-token",
            access_token_expires_in=1800,
            refresh_token_expires_in=604800,
        )

    async def refresh_tokens(self, *, refresh_token: str):
        return RefreshResult(
            access_token="next-access-token",
            refresh_token="next-refresh-token",
            access_token_expires_in=1800,
            refresh_token_expires_in=604800,
        )

    async def logout(
            self,
            *,
            refresh_token: str,
            access_token: str,
            access_token_expires_in_seconds: int,
    ):
        return None


class _ErrorAuthService:
    async def send_register_verification_code(self, email: str):
        raise BadRequestError("该邮箱已注册，请直接登录")

    async def register(self, *, email: str, password: str, verification_code=None):
        raise BadRequestError("该邮箱已注册，请直接登录")

    async def login(self, *, email: str, password: str, client_ip=None):
        raise BadRequestError("邮箱或密码错误")

    async def refresh_tokens(self, *, refresh_token: str):
        raise BadRequestError("Refresh Token 无效或已过期")

    async def logout(
            self,
            *,
            refresh_token: str,
            access_token: str,
            access_token_expires_in_seconds: int,
    ):
        raise BadRequestError("Refresh Token 无效或已过期")


def _build_auth_context() -> AuthContext:
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
        password_salt="salt",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    return AuthContext(
        user=user,
        access_token="access-token",
        token_payload={"exp": 4_102_444_800},
    )


def test_send_register_verification_code_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/register/send-code",
            json={"email": "tester@example.com"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "验证码发送成功"
    assert payload["data"]["verification_required"] is True
    assert payload["data"]["expires_in_seconds"] == 300


def test_send_register_verification_code_route_should_map_bad_request_error() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _ErrorAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/register/send-code",
            json={"email": "tester@example.com"},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "该邮箱已注册，请直接登录"


def test_register_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/register",
            json={"email": "tester@example.com", "password": "Password123!"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "注册成功"
    assert payload["data"]["email"] == "tester@example.com"


def test_register_route_should_map_bad_request_error() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _ErrorAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/register",
            json={"email": "tester@example.com", "password": "Password123!"},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "该邮箱已注册，请直接登录"


def test_login_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/login",
            json={"email": "tester@example.com", "password": "Password123!"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "登录成功"
    assert payload["data"]["tokens"]["access_token"] == "access-token"
    assert payload["data"]["tokens"]["refresh_token"] == "refresh-token"
    assert payload["data"]["user"]["email"] == "tester@example.com"


def test_login_route_should_map_bad_request_error() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _ErrorAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/login",
            json={"email": "tester@example.com", "password": "Password123!"},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "邮箱或密码错误"


def test_refresh_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": "rt-1"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "刷新成功"
    assert payload["data"]["tokens"]["access_token"] == "next-access-token"
    assert payload["data"]["tokens"]["refresh_token"] == "next-refresh-token"


def test_refresh_route_should_map_bad_request_error() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _ErrorAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": "rt-invalid"},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "Refresh Token 无效或已过期"


def test_logout_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()
    app.dependency_overrides[get_current_auth_context] = _build_auth_context

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/logout",
            json={"refresh_token": "rt-1"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "退出成功"
    assert payload["data"]["success"] is True


def test_logout_route_should_map_bad_request_error() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _ErrorAuthService()
    app.dependency_overrides[get_current_auth_context] = _build_auth_context

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/logout",
            json={"refresh_token": "rt-invalid"},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "Refresh Token 无效或已过期"
