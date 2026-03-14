from datetime import datetime, timedelta

from app.domain.models import User, UserStatus
from app.infrastructure.models import UserModel


def test_user_model_from_domain_should_keep_datetime_fields_as_datetime() -> None:
    now = datetime.now()
    user = User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
        password_salt="salt",
        status=UserStatus.ACTIVE,
        created_at=now,
        updated_at=now + timedelta(seconds=1),
        last_login_at=now + timedelta(seconds=2),
        last_login_ip="127.0.0.1",
    )

    record = UserModel.from_domain(user)

    assert isinstance(record.created_at, datetime)
    assert isinstance(record.updated_at, datetime)
    assert isinstance(record.last_login_at, datetime)
    assert record.status == UserStatus.ACTIVE.value


def test_user_model_update_from_domain_should_keep_datetime_fields_as_datetime() -> None:
    base_time = datetime.now()
    original = User(
        id="user-2",
        email="tester2@example.com",
        password="hashed-password",
        password_salt="salt",
        status=UserStatus.ACTIVE,
        created_at=base_time,
        updated_at=base_time,
        last_login_at=None,
        last_login_ip=None,
    )
    record = UserModel.from_domain(original)

    updated = User(
        id=original.id,
        email=original.email,
        password=original.password,
        password_salt=original.password_salt,
        status=UserStatus.DISABLED,
        created_at=base_time,
        updated_at=base_time + timedelta(minutes=1),
        last_login_at=base_time + timedelta(minutes=2),
        last_login_ip="10.0.0.1",
    )

    record.update_from_domain(updated)

    assert isinstance(record.created_at, datetime)
    assert isinstance(record.updated_at, datetime)
    assert isinstance(record.last_login_at, datetime)
    assert record.status == UserStatus.DISABLED.value
