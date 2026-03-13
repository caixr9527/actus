import asyncio
import json

from app.infrastructure.external.token_store.redis_refresh_token_store import RedisRefreshTokenStore


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set(self, key: str, payload: str, ex: int) -> "_FakePipeline":
        self.calls.append(("set", key, payload, ex))
        return self

    def sadd(self, key: str, value: str) -> "_FakePipeline":
        self.calls.append(("sadd", key, value))
        return self

    def expire(self, key: str, seconds: int) -> "_FakePipeline":
        self.calls.append(("expire", key, seconds))
        return self

    async def execute(self) -> list[bool | int]:
        self.calls.append(("execute",))
        return [True, 1, True]


class _FakeRedis:
    def __init__(self) -> None:
        self.pipeline_instance = _FakePipeline()
        self.pipeline_transaction: bool | None = None

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        self.pipeline_transaction = transaction
        return self.pipeline_instance


class _FakeRedisClient:
    def __init__(self) -> None:
        self.client = _FakeRedis()


def test_save_refresh_token_should_write_token_and_user_index_by_pipeline() -> None:
    redis_client = _FakeRedisClient()
    store = RedisRefreshTokenStore(redis_client=redis_client)

    asyncio.run(
        store.save_refresh_token(
            refresh_token="rt-1",
            user_id="u-1",
            email="tester@example.com",
            expires_in_seconds=600,
        )
    )

    pipeline = redis_client.client.pipeline_instance
    assert redis_client.client.pipeline_transaction is True

    assert [call[0] for call in pipeline.calls] == ["set", "sadd", "expire", "execute"]

    set_call = pipeline.calls[0]
    assert set_call[1] == "auth:refresh_token:rt-1"
    assert set_call[3] == 600
    payload = json.loads(set_call[2])
    assert payload["token"] == "rt-1"
    assert payload["user_id"] == "u-1"
    assert payload["email"] == "tester@example.com"
    assert payload["is_used"] is False
    assert payload["issued_at"]

    assert pipeline.calls[1] == ("sadd", "auth:user_refresh_tokens:u-1", "rt-1")
    assert pipeline.calls[2] == ("expire", "auth:user_refresh_tokens:u-1", 600)
