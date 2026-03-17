from datetime import datetime
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.errors import BadRequestError
from app.application.errors import error_keys
from app.domain.models import User, UserProfile
from app.interfaces.dependencies.auth import AuthContext, get_current_auth_context
from app.interfaces.endpoints.auth_routes import router as auth_router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from app.interfaces.schemas.auth import RegisterVerificationCodeResult, LoginResult, RefreshResult
from app.interfaces.dependencies.services import get_auth_service


def _assert_auth_security_headers(response) -> None:
    assert response.headers.get("cache-control") == "no-store"
    assert response.headers.get("pragma") == "no-cache"
    assert response.headers.get("x-frame-options") == "SAMEORIGIN"
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


class _FakeAuthService:
    async def send_register_verification_code(self, email: str, client_ip=None):
        return RegisterVerificationCodeResult(
            verification_required=True,
            expires_in_seconds=300,
        )

    async def register(self, *, email: str, password: str, confirm_password: str, verification_code=None):
        return User(
            email=email.lower(),
            password="hashed-password",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    async def login(self, *, email: str, password: str, client_ip=None):
        user = User(
            email=email.lower(),
            password="hashed-password",
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
    async def send_register_verification_code(self, email: str, client_ip=None):
        raise BadRequestError(
            "该邮箱已注册，请直接登录",
            error_key=error_keys.AUTH_EMAIL_ALREADY_REGISTERED,
        )

    async def register(self, *, email: str, password: str, confirm_password: str, verification_code=None):
        raise BadRequestError(
            "该邮箱已注册，请直接登录",
            error_key=error_keys.AUTH_EMAIL_ALREADY_REGISTERED,
        )

    async def login(self, *, email: str, password: str, client_ip=None):
        raise BadRequestError(
            "邮箱或密码错误",
            error_key=error_keys.AUTH_LOGIN_INVALID_CREDENTIALS,
        )

    async def refresh_tokens(self, *, refresh_token: str):
        raise BadRequestError(
            "Refresh Token 无效或已过期",
            error_key=error_keys.AUTH_REFRESH_TOKEN_INVALID,
        )

    async def logout(
            self,
            *,
            refresh_token: str,
            access_token: str,
            access_token_expires_in_seconds: int,
    ):
        raise BadRequestError(
            "Refresh Token 无效或已过期",
            error_key=error_keys.AUTH_REFRESH_TOKEN_INVALID,
        )


def _build_auth_context() -> AuthContext:
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
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
    _assert_auth_security_headers(response)


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
    assert payload["error_key"] == error_keys.AUTH_EMAIL_ALREADY_REGISTERED


def test_register_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/register",
            json={
                "email": "tester@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            },
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
            json={
                "email": "tester@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            },
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "该邮箱已注册，请直接登录"
    assert payload["error_key"] == error_keys.AUTH_EMAIL_ALREADY_REGISTERED


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
    assert "refresh_token" not in payload["data"]["tokens"]
    assert payload["data"]["user"]["email"] == "tester@example.com"
    assert "actus_refresh_token=refresh-token" in response.headers.get("set-cookie", "")
    assert "HttpOnly" in response.headers.get("set-cookie", "")
    _assert_auth_security_headers(response)


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
    assert payload["error_key"] == error_keys.AUTH_LOGIN_INVALID_CREDENTIALS


def test_login_route_should_reject_http_when_auth_require_https_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.interfaces.dependencies.request_security.get_settings",
        lambda: SimpleNamespace(auth_require_https=True),
    )
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/login",
            json={"email": "tester@example.com", "password": "Password123!"},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "当前环境仅允许通过 HTTPS 访问该接口"
    assert payload["error_key"] == error_keys.AUTH_HTTPS_REQUIRED


def test_login_route_should_allow_x_forwarded_proto_https_when_required(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.interfaces.dependencies.request_security.get_settings",
        lambda: SimpleNamespace(auth_require_https=True),
    )
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        response = client.post(
            "/api/auth/login",
            headers={"x-forwarded-proto": "https"},
            json={"email": "tester@example.com", "password": "Password123!"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "登录成功"
    _assert_auth_security_headers(response)
    assert response.headers.get("strict-transport-security") == "max-age=31536000; includeSubDomains"


def test_refresh_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        client.cookies.set("actus_refresh_token", "rt-1")
        response = client.post(
            "/api/auth/refresh",
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "刷新成功"
    assert payload["data"]["tokens"]["access_token"] == "next-access-token"
    assert "refresh_token" not in payload["data"]["tokens"]
    assert "actus_refresh_token=next-refresh-token" in response.headers.get("set-cookie", "")
    _assert_auth_security_headers(response)


def test_refresh_route_should_map_bad_request_error() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _ErrorAuthService()

    with TestClient(app) as client:
        client.cookies.set("actus_refresh_token", "rt-invalid")
        response = client.post(
            "/api/auth/refresh",
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "Refresh Token 无效或已过期"
    assert payload["error_key"] == error_keys.AUTH_REFRESH_TOKEN_INVALID


def test_refresh_route_should_return_bad_request_when_refresh_cookie_missing() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()

    with TestClient(app) as client:
        response = client.post("/api/auth/refresh")

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "登录状态缺失，请重新登录"
    assert payload["error_key"] == error_keys.AUTH_REFRESH_SESSION_MISSING


def test_logout_route_should_return_success_response() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()
    app.dependency_overrides[get_current_auth_context] = _build_auth_context

    with TestClient(app) as client:
        client.cookies.set("actus_refresh_token", "rt-1")
        response = client.post(
            "/api/auth/logout",
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 200
    assert payload["msg"] == "退出成功"
    assert payload["data"]["success"] is True
    assert "actus_refresh_token=" in response.headers.get("set-cookie", "")
    assert "Max-Age=0" in response.headers.get("set-cookie", "")
    _assert_auth_security_headers(response)


def test_logout_route_should_map_bad_request_error() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _ErrorAuthService()
    app.dependency_overrides[get_current_auth_context] = _build_auth_context

    with TestClient(app) as client:
        client.cookies.set("actus_refresh_token", "rt-invalid")
        response = client.post(
            "/api/auth/logout",
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "Refresh Token 无效或已过期"
    assert payload["error_key"] == error_keys.AUTH_REFRESH_TOKEN_INVALID


def test_logout_route_should_return_bad_request_when_refresh_cookie_missing() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router, prefix="/api")
    app.dependency_overrides[get_auth_service] = lambda: _FakeAuthService()
    app.dependency_overrides[get_current_auth_context] = _build_auth_context

    with TestClient(app) as client:
        response = client.post("/api/auth/logout")

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == 400
    assert payload["msg"] == "登录状态缺失，请重新登录"
    assert payload["error_key"] == error_keys.AUTH_REFRESH_SESSION_MISSING
