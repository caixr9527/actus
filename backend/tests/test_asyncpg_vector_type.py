from app.infrastructure.models.vector_types import AsyncpgVector


def test_asyncpg_vector_should_keep_raw_value_for_bind_processor() -> None:
    vector_type = AsyncpgVector(3)
    processor = vector_type.bind_processor(None)

    raw_value = [0.1, 0.2, 0.3]

    assert processor(raw_value) == raw_value
