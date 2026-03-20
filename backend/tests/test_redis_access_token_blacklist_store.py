import asyncio
import hashlib

from app.infrastructure.external.token_store import RedisAccessTokenBlacklistStore


class _FakeRedis:
    def __init__(self) -> None:
        self.set_calls: list[tuple[str, str, int]] = []
        self.exists_map: dict[str, int] = {}

    async def set(self, key: str, value: str, ex: int):
        self.set_calls.append((key, value, ex))
        return True

    async def exists(self, key: str):
        return self.exists_map.get(key, 0)


class _FakeRedisClient:
    def __init__(self) -> None:
        self.client = _FakeRedis()


def test_add_access_token_to_blacklist_should_write_hashed_key_with_ttl() -> None:
    redis_client = _FakeRedisClient()
    store = RedisAccessTokenBlacklistStore(redis_client=redis_client)

    asyncio.run(store.add_access_token_to_blacklist("access-token", 120))

    expected_hash = hashlib.sha256("access-token".encode("utf-8")).hexdigest()
    assert redis_client.client.set_calls == [
        (f"auth:access_token_blacklist:{expected_hash}", "1", 120)
    ]


def test_is_access_token_blacklisted_should_return_true_when_key_exists() -> None:
    redis_client = _FakeRedisClient()
    store = RedisAccessTokenBlacklistStore(redis_client=redis_client)
    expected_hash = hashlib.sha256("access-token".encode("utf-8")).hexdigest()
    redis_client.client.exists_map[f"auth:access_token_blacklist:{expected_hash}"] = 1

    result = asyncio.run(store.is_access_token_blacklisted("access-token"))

    assert result is True


def test_add_access_token_to_blacklist_should_normalize_min_ttl_to_one_second() -> None:
    redis_client = _FakeRedisClient()
    store = RedisAccessTokenBlacklistStore(redis_client=redis_client)

    asyncio.run(store.add_access_token_to_blacklist("access-token", 0))

    assert redis_client.client.set_calls[0][2] == 1
