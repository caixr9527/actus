from datetime import datetime, timedelta

from app.domain.models import LLMModelConfig
from app.infrastructure.models import LLMModelConfigModel


def test_llm_model_config_model_from_domain_should_keep_datetime_and_config_types() -> None:
    now = datetime.now()
    model_config = LLMModelConfig(
        id="gpt-5.4",
        provider="openai",
        display_name="GPT-5.4",
        base_url="https://api.openai.com/v1",
        api_key="secret-key",
        model_name="gpt-5.4",
        enabled=True,
        sort_order=10,
        is_default=True,
        config={"temperature": 0.7, "max_tokens": 8192, "badge": "Reasoning"},
        created_at=now,
        updated_at=now + timedelta(seconds=1),
    )

    record = LLMModelConfigModel.from_domain(model_config)

    assert isinstance(record.created_at, datetime)
    assert isinstance(record.updated_at, datetime)
    assert isinstance(record.config, dict)
    assert record.config["badge"] == "Reasoning"


def test_llm_model_config_model_update_from_domain_should_apply_latest_values() -> None:
    base_time = datetime.now()
    original = LLMModelConfig(
        id="deepseek",
        provider="deepseek",
        display_name="DeepSeek",
        base_url="https://api.deepseek.com",
        api_key="legacy-key",
        model_name="deepseek-reasoner",
        config={"temperature": 0.8},
        created_at=base_time,
        updated_at=base_time,
    )
    record = LLMModelConfigModel.from_domain(original)

    updated = LLMModelConfig(
        id="deepseek",
        provider="deepseek",
        display_name="DeepSeek R1",
        base_url="https://api.deepseek.com/v1",
        api_key="new-key",
        model_name="deepseek-r1",
        enabled=False,
        sort_order=3,
        is_default=False,
        config={"temperature": 0.5, "description": "复杂推理"},
        created_at=base_time,
        updated_at=base_time + timedelta(minutes=1),
    )

    record.update_from_domain(updated)

    assert record.display_name == "DeepSeek R1"
    assert record.base_url == "https://api.deepseek.com/v1"
    assert record.enabled is False
    assert record.sort_order == 3
    assert record.config["description"] == "复杂推理"
    assert isinstance(record.updated_at, datetime)
