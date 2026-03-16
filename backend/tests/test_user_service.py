import asyncio
from datetime import datetime
from typing import Optional

import pytest

from app.application.errors import BadRequestError
from app.application.service.user_service import UserService
from app.application.utils import PasswordHasher
from app.domain.models import User, UserProfile


class _FakeUserRepository:
    def __init__(self) -> None:
        self.users_by_id: dict[str, User] = {}
        self.profiles_by_user_id: dict[str, UserProfile] = {}

    async def save(self, user: User) -> None:
        self.users_by_id[user.id] = user

    async def get_by_id(self, user_id: str) -> Optional[User]:
        return self.users_by_id.get(user_id)

    async def get_by_email(self, email: str) -> Optional[User]:
        for user in self.users_by_id.values():
            if user.email == email:
                return user
        return None

    async def save_profile(self, profile: UserProfile) -> None:
        self.profiles_by_user_id[profile.user_id] = profile

    async def get_profile_by_user_id(self, user_id: str) -> Optional[UserProfile]:
        return self.profiles_by_user_id.get(user_id)


class _FakeUoW:
    def __init__(self, user_repo: _FakeUserRepository) -> None:
        self.user = user_repo

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _build_service(user_repo: _FakeUserRepository) -> UserService:
    return UserService(uow_factory=lambda: _FakeUoW(user_repo))


def _build_user(*, email: str = "tester@example.com", raw_password: str = "Password123!") -> User:
    salt = "test-salt"
    hashed_password = PasswordHasher.hash_password_with_salt(raw_password, salt)
    return User(
        email=email,
        password=hashed_password,
        password_salt=salt,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


def test_get_current_user_profile_should_create_default_profile_when_missing() -> None:
    user_repo = _FakeUserRepository()
    user = _build_user()
    user_repo.users_by_id[user.id] = user
    service = _build_service(user_repo)

    result_user, profile = asyncio.run(service.get_current_user_profile(user.id))

    assert result_user.id == user.id
    assert profile.user_id == user.id
    assert user_repo.profiles_by_user_id[user.id].user_id == user.id


def test_update_current_user_profile_should_patch_fields() -> None:
    user_repo = _FakeUserRepository()
    user = _build_user()
    profile = UserProfile(
        user_id=user.id,
        nickname="old-name",
        avatar_url="https://old.example.com/avatar.png",
        timezone="Asia/Shanghai",
        locale="zh-CN",
    )
    user_repo.users_by_id[user.id] = user
    user_repo.profiles_by_user_id[user.id] = profile
    service = _build_service(user_repo)

    updated_user, updated_profile = asyncio.run(
        service.update_current_user_profile(
            user_id=user.id,
            updates={"nickname": "new-name", "locale": "en-US"},
        )
    )

    assert updated_profile.nickname == "new-name"
    assert updated_profile.locale == "en-US"
    assert updated_profile.timezone == "Asia/Shanghai"
    assert updated_user.updated_at >= user.updated_at


def test_update_current_user_profile_should_raise_when_updates_empty() -> None:
    user_repo = _FakeUserRepository()
    user = _build_user()
    user_repo.users_by_id[user.id] = user
    service = _build_service(user_repo)

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(service.update_current_user_profile(user_id=user.id, updates={}))

    assert exc.value.msg == "至少需要更新一个字段"


def test_update_current_user_password_should_raise_when_confirm_password_not_match() -> None:
    user_repo = _FakeUserRepository()
    user = _build_user()
    user_repo.users_by_id[user.id] = user
    service = _build_service(user_repo)

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            service.update_current_user_password(
                user_id=user.id,
                old_password="Password123!",
                new_password="Password456!",
                confirm_password="Password789!",
            )
        )

    assert exc.value.msg == "两次输入的新密码不一致"


def test_update_current_user_password_should_raise_when_old_password_is_wrong() -> None:
    user_repo = _FakeUserRepository()
    user = _build_user()
    user_repo.users_by_id[user.id] = user
    service = _build_service(user_repo)

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            service.update_current_user_password(
                user_id=user.id,
                old_password="WrongPassword123!",
                new_password="Password456!",
                confirm_password="Password456!",
            )
        )

    assert exc.value.msg == "旧密码错误"


def test_update_current_user_password_should_update_hash_and_salt() -> None:
    user_repo = _FakeUserRepository()
    user = _build_user()
    user_repo.users_by_id[user.id] = user
    service = _build_service(user_repo)

    before_hash = user.password
    before_salt = user.password_salt

    asyncio.run(
        service.update_current_user_password(
            user_id=user.id,
            old_password="Password123!",
            new_password="Password456!",
            confirm_password="Password456!",
        )
    )

    assert user.password != before_hash
    assert user.password_salt != before_salt
    assert (
        user.password
        == PasswordHasher.hash_password_with_salt("Password456!", user.password_salt)
    )
