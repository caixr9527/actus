import asyncio

import pytest

from app.application.errors import ServerError, error_keys
from app.application.service import ModelRuntimeResolver
from app.domain.models import LLMModelConfig, Session


class _FakeModelConfigService:
    def __init__(self, models: list[LLMModelConfig], default_model_id: str | None) -> None:
        self._models = models
        self._default_model_id = default_model_id

    async def get_enabled_models(self):
        return self._models, self._default_model_id


def _build_model(
        model_id: str,
        *,
        base_url: str = "https://api.example.com/v1",
        api_key: str = "secret",
        model_name: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        capabilities: dict | None = None,
) -> LLMModelConfig:
    config = {"temperature": temperature, "max_tokens": max_tokens}
    if capabilities is not None:
        config["capabilities"] = capabilities
    return LLMModelConfig(
        id=model_id,
        provider="openai",
        display_name=model_id.upper(),
        base_url=base_url,
        api_key=api_key,
        model_name=model_name or model_id,
        config=config,
    )


def test_model_runtime_resolver_should_use_default_model_for_auto() -> None:
    resolver = ModelRuntimeResolver(
        model_config_service=_FakeModelConfigService(
            models=[_build_model("gpt-5.4"), _build_model("deepseek")],
            default_model_id="gpt-5.4",
        )
    )
    session = Session(id="session-a", user_id="user-a", current_model_id="auto")

    resolved_model_id, llm_config = asyncio.run(resolver.resolve(session))

    assert resolved_model_id == "gpt-5.4"
    assert llm_config.model_name == "gpt-5.4"


def test_model_runtime_resolver_should_fallback_to_default_when_selected_model_missing() -> None:
    resolver = ModelRuntimeResolver(
        model_config_service=_FakeModelConfigService(
            models=[_build_model("gpt-5.4"), _build_model("deepseek")],
            default_model_id="gpt-5.4",
        )
    )
    session = Session(id="session-a", user_id="user-a", current_model_id="kimi")

    resolved_model_id, llm_config = asyncio.run(resolver.resolve(session))

    assert resolved_model_id == "gpt-5.4"
    assert llm_config.model_name == "gpt-5.4"


def test_model_runtime_resolver_should_raise_when_default_model_unavailable() -> None:
    resolver = ModelRuntimeResolver(
        model_config_service=_FakeModelConfigService(
            models=[_build_model("deepseek")],
            default_model_id="gpt-5.4",
        )
    )
    session = Session(id="session-a", user_id="user-a", current_model_id=None)

    with pytest.raises(ServerError) as exc:
        asyncio.run(resolver.resolve(session))

    assert exc.value.error_key == error_keys.APP_CONFIG_DEFAULT_MODEL_UNAVAILABLE


def test_model_runtime_resolver_should_parse_multimodal_capabilities() -> None:
    resolver = ModelRuntimeResolver(
        model_config_service=_FakeModelConfigService(
            models=[
                _build_model(
                    "gpt-5.4",
                    capabilities={"multimodal": True, "supported": ["image", "audio", "text"]},
                )
            ],
            default_model_id="gpt-5.4",
        )
    )
    session = Session(id="session-a", user_id="user-a", current_model_id="auto")

    _, llm_config = asyncio.run(resolver.resolve(session))

    assert llm_config.multimodal is True
    assert llm_config.supported == ["image", "audio", "text"]


def test_model_runtime_resolver_should_fallback_supported_to_text() -> None:
    resolver = ModelRuntimeResolver(
        model_config_service=_FakeModelConfigService(
            models=[
                _build_model(
                    "deepseek",
                    capabilities={"multimodal": False, "supported": ["unknown_type"]},
                )
            ],
            default_model_id="deepseek",
        )
    )
    session = Session(id="session-a", user_id="user-a", current_model_id="auto")

    _, llm_config = asyncio.run(resolver.resolve(session))

    assert llm_config.multimodal is False
    assert llm_config.supported == ["text"]
