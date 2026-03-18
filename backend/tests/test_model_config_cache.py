import asyncio

from app.domain.models import LLMModelConfig
from app.infrastructure.external.cache import ModelConfigCache


class _FakePipeline:
    def __init__(self, storage: dict[str, str]) -> None:
        self._storage = storage
        self._operations: list[tuple[str, str, int | None]] = []

    def set(self, key: str, value: str, ex: int | None = None):
        self._operations.append((key, value, ex))
        return self

    async def execute(self):
        for key, value, _ in self._operations:
            self._storage[key] = value
        return True


class _FakeRedis:
    def __init__(self) -> None:
        self.storage: dict[str, str] = {}

    async def mget(self, keys: list[str]):
        return [self.storage.get(key) for key in keys]

    def pipeline(self):
        return _FakePipeline(self.storage)


class _FakeRedisClient:
    def __init__(self, redis: _FakeRedis) -> None:
        self.client = redis


def _build_model(model_id: str, is_default: bool = False) -> LLMModelConfig:
    return LLMModelConfig(
        id=model_id,
        provider="openai",
        display_name=model_id.upper(),
        base_url="https://api.example.com/v1",
        api_key="secret",
        model_name=model_id,
        is_default=is_default,
        sort_order=1 if is_default else 2,
        config={"temperature": 0.7},
    )


def test_model_config_cache_should_save_models_to_redis() -> None:
    redis = _FakeRedis()
    default_model = _build_model("gpt-5.4", is_default=True)
    cache = ModelConfigCache(redis_client=_FakeRedisClient(redis))

    asyncio.run(
        cache.save_models(
            models=[default_model, _build_model("deepseek")],
            default_model_id=default_model.id,
            expires_in_seconds=60,
        )
    )

    assert ModelConfigCache.MODELS_CACHE_KEY in redis.storage
    assert redis.storage[ModelConfigCache.DEFAULT_MODEL_CACHE_KEY] == "gpt-5.4"


def test_model_config_cache_should_use_redis_when_cache_hit() -> None:
    redis = _FakeRedis()
    default_model = _build_model("gpt-5.4", is_default=True)
    warmup_cache = ModelConfigCache(redis_client=_FakeRedisClient(redis))
    asyncio.run(
        warmup_cache.save_models(
            models=[default_model, _build_model("deepseek")],
            default_model_id=default_model.id,
            expires_in_seconds=60,
        )
    )
    cache = ModelConfigCache(redis_client=_FakeRedisClient(redis))

    cached = asyncio.run(cache.get_models())

    assert cached is not None
    models, default_model_id = cached
    assert [model.id for model in models] == ["gpt-5.4", "deepseek"]
    assert default_model_id == "gpt-5.4"
