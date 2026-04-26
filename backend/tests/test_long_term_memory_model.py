from app.domain.models import LongTermMemory
from app.infrastructure.models.long_term_memory import LongTermMemoryModel


def test_long_term_memory_model_should_derive_content_text_from_summary_content_and_tags() -> None:
    memory = LongTermMemory(
        namespace="user/user-1/profile",
        memory_type="profile",
        summary="用户偏好",
        content={"language": "zh", "response_style": "concise"},
        tags=["language", "style"],
    )

    record = LongTermMemoryModel.from_domain(memory)

    assert "用户偏好" in record.content_text
    assert "language: zh" in record.content_text
    assert "response_style: concise" in record.content_text
    assert "language" in record.content_text
