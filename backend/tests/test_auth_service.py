import asyncio
from types import SimpleNamespace
from typing import Optional

import pytest

from app.application.errors import BadRequestError, TooManyRequestsError
from app.application.service.auth_service import AuthService
from app.application.utils import PasswordHasher
from app.domain.external import RefreshTokenConsumeResult, RefreshTokenConsumeStatus
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
        self.consume_results: dict[str, RefreshTokenConsumeResult] = {}
        self.consumed_tokens: list[str] = []
        self.revoked_user_ids: list[str] = []
        self.deleted_tokens: list[str] = []

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

    async def consume_refresh_token(self, refresh_token: str) -> RefreshTokenConsumeResult:
        self.consumed_tokens.append(refresh_token)
        return self.consume_results.get(
            refresh_token,
            RefreshTokenConsumeResult(status=RefreshTokenConsumeStatus.NOT_FOUND),
        )

    async def revoke_user_refresh_tokens(self, user_id: str) -> None:
        self.revoked_user_ids.append(user_id)

    async def delete_refresh_token(self, refresh_token: str) -> None:
        self.deleted_tokens.append(refresh_token)


class _FakeAccessTokenBlacklistStore:
    def __init__(self) -> None:
        self.records: list[dict[str, str | int]] = []

    async def add_access_token_to_blacklist(
            self,
            access_token: str,
            expires_in_seconds: int,
    ) -> None:
        self.records.append(
            {
                "access_token": access_token,
                "expires_in_seconds": expires_in_seconds,
            }
        )

    async def is_access_token_blacklisted(self, access_token: str) -> bool:
        return any(record["access_token"] == access_token for record in self.records)


class _FakeAuthRateLimitStore:
    def __init__(self) -> None:
        self.login_attempts_by_ip: dict[str, int] = {}
        self.login_attempts_by_email: dict[str, int] = {}
        self.send_code_attempts_by_ip: dict[str, int] = {}

    async def get_login_attempt_count_by_ip(self, ip: str) -> int:
        return self.login_attempts_by_ip.get(ip, 0)

    async def get_login_attempt_count_by_email(self, email: str) -> int:
        return self.login_attempts_by_email.get(email.strip().lower(), 0)

    async def increase_login_attempt_count(
            self,
            ip: str | None,
            email: str,
            expires_in_seconds: int,
    ) -> None:
        if ip is not None and ip.strip():
            self.login_attempts_by_ip[ip] = self.login_attempts_by_ip.get(ip, 0) + 1
        normalized_email = email.strip().lower()
        self.login_attempts_by_email[normalized_email] = self.login_attempts_by_email.get(normalized_email, 0) + 1

    async def clear_login_attempt_count(self, ip: str | None, email: str) -> None:
        if ip is not None and ip.strip():
            self.login_attempts_by_ip.pop(ip, None)
        self.login_attempts_by_email.pop(email.strip().lower(), None)

    async def increase_register_send_code_attempt_count_by_ip(
            self,
            ip: str,
            expires_in_seconds: int,
    ) -> int:
        self.send_code_attempts_by_ip[ip] = self.send_code_attempts_by_ip.get(ip, 0) + 1
        return self.send_code_attempts_by_ip[ip]


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
        access_token_blacklist_store: Optional[_FakeAccessTokenBlacklistStore] = None,
        auth_rate_limit_store: Optional[_FakeAuthRateLimitStore] = None,
        register_verification_code_store: Optional[_FakeRegisterVerificationCodeStore] = None,
        email_sender: Optional[_FakeEmailSender] = None,
        email_verification_enabled: bool = False,
) -> AuthService:
    service = AuthService(
        uow_factory=lambda: _FakeUoW(user_repo),
        refresh_token_store=refresh_token_store or _FakeRefreshTokenStore(),
        access_token_blacklist_store=access_token_blacklist_store or _FakeAccessTokenBlacklistStore(),
        auth_rate_limit_store=auth_rate_limit_store,
        register_verification_code_store=register_verification_code_store,
        email_sender=email_sender,
    )
    service._setting = SimpleNamespace(
        auth_jwt_secret="unit-test-secret",
        auth_jwt_algorithm="HS256",
        auth_access_token_expires_in=1800,
        auth_refresh_token_expires_in=604800,
        auth_register_code_expires_in=300,
        auth_register_code_length=6,
        auth_login_rate_limit_max_attempts=10,
        auth_login_rate_limit_window_seconds=300,
        auth_send_code_rate_limit_max_attempts=10,
        auth_send_code_rate_limit_window_seconds=300,
        auth_register_verification_enabled=email_verification_enabled,
    )
    return service


def test_register_should_create_user_and_profile() -> None:
    user_repo = _FakeUserRepository()
    service = _build_auth_service(user_repo)

    user = asyncio.run(
        service.register(
            email="Tester@Example.com",
            password="Password123!",
            confirm_password="Password123!",
        )
    )

    assert user.email == "tester@example.com"
    assert user.password != "Password123!"
    assert PasswordHasher.verify_password("Password123!", user.password)
    assert user_repo.profiles_by_user_id[user.id].user_id == user.id


def test_register_should_raise_bad_request_when_email_exists() -> None:
    user_repo = _FakeUserRepository()
    existing = User(
        email="tester@example.com",
        password="hashed-password",
    )
    user_repo.users_by_id[existing.id] = existing
    user_repo.users_by_email[existing.email] = existing

    service = _build_auth_service(user_repo)

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            service.register(
                email="tester@example.com",
                password="Password123!",
                confirm_password="Password123!",
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
                confirm_password="Password123!",
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
                confirm_password="Password123!",
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
            confirm_password="Password123!",
            verification_code="123456",
        )
    )

    assert user.email == "tester@example.com"
    assert "tester@example.com" not in code_store.codes


def test_login_should_issue_tokens_and_update_last_login_fields() -> None:
    user_repo = _FakeUserRepository()
    refresh_token_store = _FakeRefreshTokenStore()
    service = _build_auth_service(user_repo, refresh_token_store)

    hashed_password = PasswordHasher.hash_password("Password123!")
    existing = User(
        email="tester@example.com",
        password=hashed_password,
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

    hashed_password = PasswordHasher.hash_password("Password123!")
    existing = User(
        email="tester@example.com",
        password=hashed_password,
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


def test_login_should_raise_too_many_requests_when_ip_or_email_is_rate_limited() -> None:
    user_repo = _FakeUserRepository()
    rate_limit_store = _FakeAuthRateLimitStore()
    service = _build_auth_service(user_repo, auth_rate_limit_store=rate_limit_store)
    rate_limit_store.login_attempts_by_ip["127.0.0.1"] = 10

    with pytest.raises(TooManyRequestsError) as exc:
        asyncio.run(
            service.login(
                email="tester@example.com",
                password="Password123!",
                client_ip="127.0.0.1",
            )
        )

    assert exc.value.msg == "登录尝试过于频繁，请稍后重试"


def test_login_should_record_failed_attempt_count_when_password_is_wrong() -> None:
    user_repo = _FakeUserRepository()
    rate_limit_store = _FakeAuthRateLimitStore()
    service = _build_auth_service(user_repo, auth_rate_limit_store=rate_limit_store)

    existing = User(
        email="tester@example.com",
        password=PasswordHasher.hash_password("Password123!"),
    )
    user_repo.users_by_id[existing.id] = existing
    user_repo.users_by_email[existing.email] = existing

    with pytest.raises(BadRequestError):
        asyncio.run(
            service.login(
                email="tester@example.com",
                password="WrongPassword123!",
                client_ip="127.0.0.1",
            )
        )

    assert rate_limit_store.login_attempts_by_ip["127.0.0.1"] == 1
    assert rate_limit_store.login_attempts_by_email["tester@example.com"] == 1


def test_login_should_clear_failed_attempt_count_when_login_success() -> None:
    user_repo = _FakeUserRepository()
    rate_limit_store = _FakeAuthRateLimitStore()
    refresh_token_store = _FakeRefreshTokenStore()
    service = _build_auth_service(
        user_repo,
        refresh_token_store=refresh_token_store,
        auth_rate_limit_store=rate_limit_store,
    )

    existing = User(
        email="tester@example.com",
        password=PasswordHasher.hash_password("Password123!"),
    )
    user_repo.users_by_id[existing.id] = existing
    user_repo.users_by_email[existing.email] = existing
    rate_limit_store.login_attempts_by_ip["127.0.0.1"] = 2
    rate_limit_store.login_attempts_by_email["tester@example.com"] = 2

    asyncio.run(
        service.login(
            email="tester@example.com",
            password="Password123!",
            client_ip="127.0.0.1",
        )
    )

    assert "127.0.0.1" not in rate_limit_store.login_attempts_by_ip
    assert "tester@example.com" not in rate_limit_store.login_attempts_by_email


def test_login_should_raise_bad_request_when_user_status_is_disabled() -> None:
    user_repo = _FakeUserRepository()
    service = _build_auth_service(user_repo)

    hashed_password = PasswordHasher.hash_password("Password123!")
    existing = User(
        email="tester@example.com",
        password=hashed_password,
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


def test_send_register_verification_code_should_raise_too_many_requests_when_ip_limited() -> None:
    user_repo = _FakeUserRepository()
    rate_limit_store = _FakeAuthRateLimitStore()
    service = _build_auth_service(user_repo=user_repo, auth_rate_limit_store=rate_limit_store)
    rate_limit_store.send_code_attempts_by_ip["127.0.0.1"] = 10

    with pytest.raises(TooManyRequestsError) as exc:
        asyncio.run(
            service.send_register_verification_code(
                email="tester@example.com",
                client_ip="127.0.0.1",
            )
        )

    assert exc.value.msg == "验证码发送过于频繁，请稍后重试"


def test_refresh_tokens_should_issue_new_token_pair_when_refresh_token_valid() -> None:
    user_repo = _FakeUserRepository()
    refresh_token_store = _FakeRefreshTokenStore()
    service = _build_auth_service(user_repo, refresh_token_store)

    existing = User(
        email="tester@example.com",
        password="hashed-password",
    )
    user_repo.users_by_id[existing.id] = existing
    user_repo.users_by_email[existing.email] = existing
    refresh_token_store.consume_results["rt-valid"] = RefreshTokenConsumeResult(
        status=RefreshTokenConsumeStatus.CONSUMED,
        user_id=existing.id,
        email=existing.email,
    )

    result = asyncio.run(service.refresh_tokens("rt-valid"))

    assert result.access_token
    assert result.refresh_token
    assert result.refresh_token != "rt-valid"
    assert result.access_token_expires_in == 1800
    assert result.refresh_token_expires_in == 604800
    assert refresh_token_store.consumed_tokens == ["rt-valid"]
    assert len(refresh_token_store.records) == 1
    assert refresh_token_store.records[0]["user_id"] == existing.id


def test_refresh_tokens_should_raise_bad_request_when_refresh_token_missing() -> None:
    user_repo = _FakeUserRepository()
    refresh_token_store = _FakeRefreshTokenStore()
    service = _build_auth_service(user_repo, refresh_token_store)

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(service.refresh_tokens("rt-not-found"))

    assert exc.value.msg == "Refresh Token 无效或已过期"


def test_refresh_tokens_should_revoke_user_tokens_when_refresh_token_replayed() -> None:
    user_repo = _FakeUserRepository()
    refresh_token_store = _FakeRefreshTokenStore()
    service = _build_auth_service(user_repo, refresh_token_store)
    compromised_user_id = "user-1"
    refresh_token_store.consume_results["rt-replayed"] = RefreshTokenConsumeResult(
        status=RefreshTokenConsumeStatus.REPLAYED,
        user_id=compromised_user_id,
        email="tester@example.com",
    )

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(service.refresh_tokens("rt-replayed"))

    assert exc.value.msg == "检测到登录状态异常，请重新登录"
    assert refresh_token_store.revoked_user_ids == [compromised_user_id]


def test_refresh_tokens_should_revoke_user_tokens_when_user_status_is_disabled() -> None:
    user_repo = _FakeUserRepository()
    refresh_token_store = _FakeRefreshTokenStore()
    service = _build_auth_service(user_repo, refresh_token_store)

    existing = User(
        email="tester@example.com",
        password="hashed-password",
        status=UserStatus.DISABLED,
    )
    user_repo.users_by_id[existing.id] = existing
    user_repo.users_by_email[existing.email] = existing
    refresh_token_store.consume_results["rt-valid"] = RefreshTokenConsumeResult(
        status=RefreshTokenConsumeStatus.CONSUMED,
        user_id=existing.id,
        email=existing.email,
    )

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(service.refresh_tokens("rt-valid"))

    assert exc.value.msg == "账号状态异常，暂不可登录"
    assert refresh_token_store.revoked_user_ids == [existing.id]


def test_logout_should_delete_refresh_token() -> None:
    user_repo = _FakeUserRepository()
    refresh_token_store = _FakeRefreshTokenStore()
    access_token_blacklist_store = _FakeAccessTokenBlacklistStore()
    service = _build_auth_service(
        user_repo,
        refresh_token_store=refresh_token_store,
        access_token_blacklist_store=access_token_blacklist_store,
    )

    asyncio.run(
        service.logout(
            refresh_token="  rt-logout  ",
            access_token="  access-token  ",
            access_token_expires_in_seconds=60,
        )
    )

    assert refresh_token_store.deleted_tokens == ["rt-logout"]
    assert access_token_blacklist_store.records == [
        {"access_token": "access-token", "expires_in_seconds": 60}
    ]
