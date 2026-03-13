import asyncio
from typing import Optional

import pytest

from app.application.errors import BadRequestError
from app.application.service.auth_service import AuthService
from app.application.utils import PasswordHasher
from app.domain.models import User, UserProfile, UserStatus


class _FakeUserRepository:
    def __init__(self) -> None:
        self.users_by_id: dict[str, User] = {}
        self.users_by_email: dict[str, User] = {}
        self.profiles_by_user_id: dict[str, UserProfile] = {}

    async def save(self, user: User) -> None:
        self.users_by_id[user.id] = user
        self.users_by_email[user.email] = user

    async def get_by_id(self, user_id: str) -> Optional[User]:
        return self.users_by_id.get(user_id)

    async def get_by_email(self, email: str) -> Optional[User]:
        return self.users_by_email.get(email)

    async def save_profile(self, profile: UserProfile) -> None:
        self.profiles_by_user_id[profile.user_id] = profile

    async def get_profile_by_user_id(self, user_id: str) -> Optional[UserProfile]:
        return self.profiles_by_user_id.get(user_id)


class _FakeRefreshTokenStore:
    def __init__(self) -> None:
        self.records: list[dict[str, str | int]] = []

    async def save_refresh_token(
            self,
            *,
            refresh_token: str,
            user_id: str,
            email: str,
            expires_in_seconds: int,
    ) -> None:
        self.records.append(
            {
                "refresh_token": refresh_token,
                "user_id": user_id,
                "email": email,
                "expires_in_seconds": expires_in_seconds,
            }
        )


class _FakeRegisterVerificationCodeStore:
    def __init__(self) -> None:
        self.codes: dict[str, str] = {}

    async def save_verification_code(
            self,
            *,
            email: str,
            verification_code: str,
            expires_in_seconds: int,
    ) -> None:
        self.codes[email] = verification_code

    async def verify_and_consume_verification_code(
            self,
            *,
            email: str,
            verification_code: str,
    ) -> bool:
        current = self.codes.get(email)
        if current is None or current != verification_code:
            return False
        self.codes.pop(email, None)
        return True


class _FakeEmailSender:
    def __init__(self) -> None:
        self.records: list[dict[str, str | int]] = []

    async def send_register_verification_code(
            self,
            *,
            to_email: str,
            verification_code: str,
            expires_in_seconds: int,
    ) -> None:
        self.records.append(
            {
                "to_email": to_email,
                "verification_code": verification_code,
                "expires_in_seconds": expires_in_seconds,
            }
        )


class _FakeUoW:
    def __init__(self, user_repo: _FakeUserRepository) -> None:
        self.user = user_repo

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _build_auth_service(
        user_repo: _FakeUserRepository,
        refresh_token_store: Optional[_FakeRefreshTokenStore] = None,
        register_verification_code_store: Optional[_FakeRegisterVerificationCodeStore] = None,
        email_sender: Optional[_FakeEmailSender] = None,
        email_verification_enabled: bool = False,
) -> AuthService:
    return AuthService(
        uow_factory=lambda: _FakeUoW(user_repo),
        refresh_token_store=refresh_token_store or _FakeRefreshTokenStore(),
        jwt_secret="unit-test-secret",
        jwt_algorithm="HS256",
        access_token_expires_in=1800,
        refresh_token_expires_in=604800,
        email_verification_enabled=email_verification_enabled,
        register_code_expires_in=300,
        register_verification_code_store=register_verification_code_store,
        email_sender=email_sender,
        register_verification_code_length=6,
    )


def test_register_should_create_user_and_profile() -> None:
    user_repo = _FakeUserRepository()
    service = _build_auth_service(user_repo)

    user = asyncio.run(
        service.register(
            email="Tester@Example.com",
            password="Password123!",
        )
    )

    assert user.email == "tester@example.com"
    assert user.password != "Password123!"
    assert len(user.password_salt) > 0
    assert user_repo.profiles_by_user_id[user.id].user_id == user.id


def test_register_should_raise_bad_request_when_email_exists() -> None:
    user_repo = _FakeUserRepository()
    existing = User(
        email="tester@example.com",
        password="hashed-password",
        password_salt="salt",
    )
    user_repo.users_by_id[existing.id] = existing
    user_repo.users_by_email[existing.email] = existing

    service = _build_auth_service(user_repo)

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            service.register(
                email="tester@example.com",
                password="Password123!",
            )
        )
    assert "该邮箱已注册" in exc.value.msg


def test_send_register_verification_code_should_store_and_send_when_enabled() -> None:
    user_repo = _FakeUserRepository()
    code_store = _FakeRegisterVerificationCodeStore()
    email_sender = _FakeEmailSender()
    service = _build_auth_service(
        user_repo=user_repo,
        register_verification_code_store=code_store,
        email_sender=email_sender,
        email_verification_enabled=True,
    )

    result = asyncio.run(service.send_register_verification_code("Tester@Example.com"))

    assert result.verification_required is True
    assert result.expires_in_seconds == 300
    assert "tester@example.com" in code_store.codes
    assert len(email_sender.records) == 1
    assert email_sender.records[0]["to_email"] == "tester@example.com"


def test_register_should_require_verification_code_when_enabled() -> None:
    user_repo = _FakeUserRepository()
    code_store = _FakeRegisterVerificationCodeStore()
    email_sender = _FakeEmailSender()
    service = _build_auth_service(
        user_repo=user_repo,
        register_verification_code_store=code_store,
        email_sender=email_sender,
        email_verification_enabled=True,
    )

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            service.register(
                email="tester@example.com",
                password="Password123!",
            )
        )
    assert "请输入邮箱验证码" in exc.value.msg


def test_register_should_raise_bad_request_when_verification_code_invalid() -> None:
    user_repo = _FakeUserRepository()
    code_store = _FakeRegisterVerificationCodeStore()
    email_sender = _FakeEmailSender()
    service = _build_auth_service(
        user_repo=user_repo,
        register_verification_code_store=code_store,
        email_sender=email_sender,
        email_verification_enabled=True,
    )
    code_store.codes["tester@example.com"] = "123456"

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            service.register(
                email="tester@example.com",
                password="Password123!",
                verification_code="654321",
            )
        )
    assert "邮箱验证码错误或已过期" in exc.value.msg


def test_register_should_pass_with_verification_code_when_enabled() -> None:
    user_repo = _FakeUserRepository()
    code_store = _FakeRegisterVerificationCodeStore()
    email_sender = _FakeEmailSender()
    service = _build_auth_service(
        user_repo=user_repo,
        register_verification_code_store=code_store,
        email_sender=email_sender,
        email_verification_enabled=True,
    )
    code_store.codes["tester@example.com"] = "123456"

    user = asyncio.run(
        service.register(
            email="tester@example.com",
            password="Password123!",
            verification_code="123456",
        )
    )

    assert user.email == "tester@example.com"
    assert "tester@example.com" not in code_store.codes


def test_login_should_issue_tokens_and_update_last_login_fields() -> None:
    user_repo = _FakeUserRepository()
    refresh_token_store = _FakeRefreshTokenStore()
    service = _build_auth_service(user_repo, refresh_token_store)

    salt = "test-salt"
    hashed_password = PasswordHasher.hash_password_with_salt("Password123!", salt)
    existing = User(
        email="tester@example.com",
        password=hashed_password,
        password_salt=salt,
    )
    profile = UserProfile(user_id=existing.id, nickname="tester")
    user_repo.users_by_id[existing.id] = existing
    user_repo.users_by_email[existing.email] = existing
    user_repo.profiles_by_user_id[existing.id] = profile

    result = asyncio.run(
        service.login(
            email="tester@example.com",
            password="Password123!",
            client_ip="127.0.0.1",
        )
    )

    assert result.user.email == "tester@example.com"
    assert result.user.last_login_ip == "127.0.0.1"
    assert result.user.last_login_at is not None
    assert result.access_token
    assert result.refresh_token
    assert result.access_token_expires_in == 1800
    assert result.refresh_token_expires_in == 604800
    assert len(refresh_token_store.records) == 1
    assert refresh_token_store.records[0]["user_id"] == existing.id


def test_login_should_raise_bad_request_when_password_is_wrong() -> None:
    user_repo = _FakeUserRepository()
    service = _build_auth_service(user_repo)

    salt = "test-salt"
    hashed_password = PasswordHasher.hash_password_with_salt("Password123!", salt)
    existing = User(
        email="tester@example.com",
        password=hashed_password,
        password_salt=salt,
    )
    user_repo.users_by_id[existing.id] = existing
    user_repo.users_by_email[existing.email] = existing

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            service.login(
                email="tester@example.com",
                password="WrongPassword123!",
            )
        )
    assert exc.value.msg == "邮箱或密码错误"


def test_login_should_raise_bad_request_when_user_status_is_disabled() -> None:
    user_repo = _FakeUserRepository()
    service = _build_auth_service(user_repo)

    salt = "test-salt"
    hashed_password = PasswordHasher.hash_password_with_salt("Password123!", salt)
    existing = User(
        email="tester@example.com",
        password=hashed_password,
        password_salt=salt,
        status=UserStatus.DISABLED,
    )
    user_repo.users_by_id[existing.id] = existing
    user_repo.users_by_email[existing.email] = existing

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            service.login(
                email="tester@example.com",
                password="Password123!",
            )
        )
    assert exc.value.msg == "账号状态异常，暂不可登录"
