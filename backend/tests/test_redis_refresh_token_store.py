import asyncio
import json

from app.domain.external import RefreshTokenConsumeStatus
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

    def delete(self, key: str) -> "_FakePipeline":
        self.calls.append(("delete", key))
        return self

    def srem(self, key: str, value: str) -> "_FakePipeline":
        self.calls.append(("srem", key, value))
        return self

    async def execute(self) -> list[bool | int]:
        self.calls.append(("execute",))
        return [True]


class _FakeRedis:
    def __init__(self) -> None:
        self.pipeline_transactions: list[bool] = []
        self.pipeline_instances: list[_FakePipeline] = []
        self.eval_results: list[list[int | str] | int] = []
        self.eval_calls: list[tuple] = []
        self.get_values: dict[str, str] = {}
        self.smembers_values: dict[str, set[str]] = {}

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        self.pipeline_transactions.append(transaction)
        pipeline = _FakePipeline()
        self.pipeline_instances.append(pipeline)
        return pipeline

    async def eval(self, script: str, numkeys: int, *keys_and_args):
        self.eval_calls.append((script, numkeys, *keys_and_args))
        if self.eval_results:
            return self.eval_results.pop(0)
        return [0]

    async def get(self, key: str):
        return self.get_values.get(key)

    async def smembers(self, key: str):
        return self.smembers_values.get(key, set())


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

    pipeline = redis_client.client.pipeline_instances[0]
    assert redis_client.client.pipeline_transactions == [True]
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


def test_consume_refresh_token_should_return_consumed_status() -> None:
    redis_client = _FakeRedisClient()
    redis_client.client.eval_results.append([1, "u-1", "tester@example.com"])
    store = RedisRefreshTokenStore(redis_client=redis_client)

    result = asyncio.run(store.consume_refresh_token("rt-1"))

    assert result.status == RefreshTokenConsumeStatus.CONSUMED
    assert result.user_id == "u-1"
    assert result.email == "tester@example.com"
    eval_call = redis_client.client.eval_calls[0]
    assert eval_call[1] == 1
    assert eval_call[2] == "auth:refresh_token:rt-1"


def test_consume_refresh_token_should_return_replayed_status() -> None:
    redis_client = _FakeRedisClient()
    redis_client.client.eval_results.append([2, "u-1", "tester@example.com"])
    store = RedisRefreshTokenStore(redis_client=redis_client)

    result = asyncio.run(store.consume_refresh_token("rt-1"))

    assert result.status == RefreshTokenConsumeStatus.REPLAYED
    assert result.user_id == "u-1"


def test_consume_refresh_token_should_return_not_found_status() -> None:
    redis_client = _FakeRedisClient()
    redis_client.client.eval_results.append([0])
    store = RedisRefreshTokenStore(redis_client=redis_client)

    result = asyncio.run(store.consume_refresh_token("rt-missing"))

    assert result.status == RefreshTokenConsumeStatus.NOT_FOUND
    assert result.user_id is None
    assert result.email is None


def test_revoke_user_refresh_tokens_should_delete_user_token_keys() -> None:
    redis_client = _FakeRedisClient()
    redis_client.client.smembers_values["auth:user_refresh_tokens:u-1"] = {"rt-1", "rt-2"}
    store = RedisRefreshTokenStore(redis_client=redis_client)

    asyncio.run(store.revoke_user_refresh_tokens("u-1"))

    pipeline = redis_client.client.pipeline_instances[0]
    delete_calls = [call for call in pipeline.calls if call[0] == "delete"]
    deleted_keys = {call[1] for call in delete_calls}
    assert "auth:refresh_token:rt-1" in deleted_keys
    assert "auth:refresh_token:rt-2" in deleted_keys
    assert "auth:user_refresh_tokens:u-1" in deleted_keys
    assert pipeline.calls[-1] == ("execute",)


def test_delete_refresh_token_should_remove_token_and_user_index() -> None:
    redis_client = _FakeRedisClient()
    redis_client.client.get_values["auth:refresh_token:rt-1"] = json.dumps(
        {"token": "rt-1", "user_id": "u-1", "email": "tester@example.com"}
    )
    store = RedisRefreshTokenStore(redis_client=redis_client)

    asyncio.run(store.delete_refresh_token("rt-1"))

    pipeline = redis_client.client.pipeline_instances[0]
    assert pipeline.calls[0] == ("delete", "auth:refresh_token:rt-1")
    assert pipeline.calls[1] == ("srem", "auth:user_refresh_tokens:u-1", "rt-1")
    assert pipeline.calls[2] == ("execute",)
