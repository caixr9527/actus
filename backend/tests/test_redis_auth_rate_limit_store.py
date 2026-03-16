import asyncio
import hashlib

from app.infrastructure.external.rate_limit_store import RedisAuthRateLimitStore


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def delete(self, key: str) -> "_FakePipeline":
        self.calls.append(("delete", key))
        return self

    async def execute(self) -> list[bool]:
        self.calls.append(("execute",))
        return [True]


class _FakeRedis:
    def __init__(self) -> None:
        self.get_values: dict[str, str] = {}
        self.eval_results: list[int | str] = []
        self.eval_calls: list[tuple] = []
        self.pipeline_instances: list[_FakePipeline] = []
        self.pipeline_transactions: list[bool] = []

    async def get(self, key: str):
        return self.get_values.get(key)

    async def eval(self, script: str, numkeys: int, *keys_and_args):
        self.eval_calls.append((script, numkeys, *keys_and_args))
        if self.eval_results:
            return self.eval_results.pop(0)
        return 0

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        self.pipeline_transactions.append(transaction)
        pipeline = _FakePipeline()
        self.pipeline_instances.append(pipeline)
        return pipeline


class _FakeRedisClient:
    def __init__(self) -> None:
        self.client = _FakeRedis()


def test_get_login_attempt_count_by_ip_should_parse_existing_counter() -> None:
    redis_client = _FakeRedisClient()
    redis_client.client.get_values["auth:rate_limit:login:ip:127.0.0.1"] = "3"
    store = RedisAuthRateLimitStore(redis_client=redis_client)

    count = asyncio.run(store.get_login_attempt_count_by_ip("127.0.0.1"))

    assert count == 3


def test_get_login_attempt_count_by_email_should_use_hashed_email_key() -> None:
    redis_client = _FakeRedisClient()
    email_hash = hashlib.sha256("tester@example.com".encode("utf-8")).hexdigest()
    redis_client.client.get_values[f"auth:rate_limit:login:email:{email_hash}"] = "5"
    store = RedisAuthRateLimitStore(redis_client=redis_client)

    count = asyncio.run(store.get_login_attempt_count_by_email("Tester@Example.com"))

    assert count == 5


def test_increase_login_attempt_count_should_increase_ip_and_email_counters() -> None:
    redis_client = _FakeRedisClient()
    redis_client.client.eval_results = [1, 2]
    store = RedisAuthRateLimitStore(redis_client=redis_client)

    asyncio.run(
        store.increase_login_attempt_count(
            ip="127.0.0.1",
            email="tester@example.com",
            expires_in_seconds=300,
        )
    )

    email_hash = hashlib.sha256("tester@example.com".encode("utf-8")).hexdigest()
    assert redis_client.client.eval_calls[0][1] == 1
    assert redis_client.client.eval_calls[0][2] == "auth:rate_limit:login:ip:127.0.0.1"
    assert redis_client.client.eval_calls[0][3] == 300
    assert redis_client.client.eval_calls[1][2] == f"auth:rate_limit:login:email:{email_hash}"
    assert redis_client.client.eval_calls[1][3] == 300


def test_clear_login_attempt_count_should_delete_ip_and_email_keys() -> None:
    redis_client = _FakeRedisClient()
    store = RedisAuthRateLimitStore(redis_client=redis_client)

    asyncio.run(
        store.clear_login_attempt_count(
            ip="127.0.0.1",
            email="tester@example.com",
        )
    )

    email_hash = hashlib.sha256("tester@example.com".encode("utf-8")).hexdigest()
    pipeline = redis_client.client.pipeline_instances[0]
    assert redis_client.client.pipeline_transactions == [True]
    assert pipeline.calls == [
        ("delete", "auth:rate_limit:login:ip:127.0.0.1"),
        ("delete", f"auth:rate_limit:login:email:{email_hash}"),
        ("execute",),
    ]


def test_increase_register_send_code_attempt_count_by_ip_should_return_count() -> None:
    redis_client = _FakeRedisClient()
    redis_client.client.eval_results = [4]
    store = RedisAuthRateLimitStore(redis_client=redis_client)

    current = asyncio.run(
        store.increase_register_send_code_attempt_count_by_ip(
            ip="127.0.0.1",
            expires_in_seconds=300,
        )
    )

    assert current == 4
    assert redis_client.client.eval_calls[0][2] == "auth:rate_limit:register_send_code:ip:127.0.0.1"
    assert redis_client.client.eval_calls[0][3] == 300
