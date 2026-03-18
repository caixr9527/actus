import asyncio

import pytest

from app.application.errors import ServerError, error_keys
from app.application.service import ModelConfigService
from app.domain.models import LLMModelConfig


class _FakeModelConfigCache:
    def __init__(self, models: list[LLMModelConfig] | None = None, default_model_id: str | None = None) -> None:
        self._models = models
        self._default_model_id = default_model_id
        self.saved_payload: tuple[list[LLMModelConfig], str | None, int] | None = None

    async def get_models(self):
        if self._models is None:
            return None
        return self._models, self._default_model_id

    async def save_models(
            self,
            models: list[LLMModelConfig],
            default_model_id: str | None,
            expires_in_seconds: int,
    ) -> None:
        self.saved_payload = (models, default_model_id, expires_in_seconds)


class _FakeLLMModelConfigRepo:
    def __init__(self, models: list[LLMModelConfig], default_model: LLMModelConfig | None) -> None:
        self._models = models
        self._default_model = default_model
        self.list_enabled_calls = 0
        self.get_default_calls = 0

    async def list_enabled(self):
        self.list_enabled_calls += 1
        return self._models

    async def get_default(self):
        self.get_default_calls += 1
        return self._default_model


class _FakeUoW:
    def __init__(self, repo: _FakeLLMModelConfigRepo) -> None:
        self.llm_model_config = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _build_model(model_id: str) -> LLMModelConfig:
    return LLMModelConfig(
        id=model_id,
        provider="openai",
        display_name=model_id.upper(),
        base_url="https://api.example.com/v1",
        api_key="secret",
        model_name=model_id,
    )


def test_model_config_service_should_raise_when_default_model_missing() -> None:
    repo = _FakeLLMModelConfigRepo(models=[_build_model("gpt-5.4")], default_model=None)
    service = ModelConfigService(
        uow_factory=lambda: _FakeUoW(repo),
        model_config_cache_store=_FakeModelConfigCache(models=[_build_model("gpt-5.4")], default_model_id=None),
    )

    with pytest.raises(ServerError) as exc:
        asyncio.run(service.get_public_models())

    assert exc.value.error_key == error_keys.APP_CONFIG_DEFAULT_MODEL_UNAVAILABLE


def test_model_config_service_should_return_models_and_default_model_id() -> None:
    repo = _FakeLLMModelConfigRepo(models=[], default_model=None)
    service = ModelConfigService(
        uow_factory=lambda: _FakeUoW(repo),
        model_config_cache_store=_FakeModelConfigCache(
            models=[_build_model("gpt-5.4"), _build_model("deepseek")],
            default_model_id="gpt-5.4",
        )
    )

    default_model_id, models = asyncio.run(service.get_public_models())

    assert default_model_id == "gpt-5.4"
    assert [model.id for model in models] == ["gpt-5.4", "deepseek"]


def test_model_config_service_should_load_from_db_and_write_cache_when_cache_miss() -> None:
    default_model = _build_model("gpt-5.4")
    cache = _FakeModelConfigCache(models=None, default_model_id=None)
    repo = _FakeLLMModelConfigRepo(
        models=[default_model, _build_model("deepseek")],
        default_model=default_model,
    )
    service = ModelConfigService(
        uow_factory=lambda: _FakeUoW(repo),
        model_config_cache_store=cache,
    )

    default_model_id, models = asyncio.run(service.get_public_models())

    assert default_model_id == "gpt-5.4"
    assert [model.id for model in models] == ["gpt-5.4", "deepseek"]
    assert repo.list_enabled_calls == 1
    assert repo.get_default_calls == 1
    assert cache.saved_payload is not None
    _, saved_default_model_id, saved_ttl = cache.saved_payload
    assert saved_default_model_id == "gpt-5.4"
    assert saved_ttl == 60
