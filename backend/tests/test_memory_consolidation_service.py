import asyncio

from app.domain.services.memory_consolidation import (
    MemoryConsolidationInput,
    MemoryConsolidationResult,
    MemoryConsolidationService,
)


def test_memory_consolidation_service_should_build_rule_based_result() -> None:
    payload = MemoryConsolidationInput(
        user_message="请总结当前任务",
        assistant_message="任务已经完成",
        previous_conversation_summary="历史摘要",
        goal="验证沉淀服务",
        completed_step_count=2,
        total_step_count=3,
        message_window=[
            {"role": "user", "message": "请总结当前任务", "attachment_paths": []},
        ],
        selected_artifacts=["/home/ubuntu/report.md"],
        facts_in_session=["当前会话只关注 backend"],
        user_preferences={"language": "zh"},
        pending_memory_writes=[
            {
                "namespace": "session/session-1/fact",
                "memory_type": "fact",
                "summary": "当前会话只关注 backend",
                "content": {"text": "当前会话只关注 backend"},
                "tags": ["backend"],
                "confidence": 0.8,
            }
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert result.degraded is False
    assert result.message_window[-1]["role"] == "assistant"
    assert result.message_window[-1]["message"] == "任务已经完成"
    assert result.message_window[-1]["attachment_paths"] == ["/home/ubuntu/report.md"]
    assert "历史摘要" in result.conversation_summary
    assert "目标:验证沉淀服务" in result.conversation_summary
    assert "进度:2/3" in result.conversation_summary
    assert result.facts_in_session == ["当前会话只关注 backend"]
    assert result.user_preferences == {"language": "zh"}
    assert len(result.memory_candidates) == 1
    assert result.stats.kept_candidate_count == 1


def test_memory_consolidation_service_should_trim_message_window_and_merge_profile_candidates() -> None:
    payload = MemoryConsolidationInput(
        assistant_message="最终总结",
        message_window=[
            {"role": "user", "message": f"历史消息 {index}", "attachment_paths": []}
            for index in range(105)
        ],
        pending_memory_writes=[
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "用户偏好语言",
                "content": {"language": "zh"},
                "tags": ["language"],
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "用户偏好风格",
                "content": {"response_style": "concise"},
                "tags": ["response_style"],
                "confidence": 0.7,
            },
            {
                "namespace": "session/session-1/fact",
                "memory_type": "fact",
                "summary": "低置信度事实",
                "content": {"text": "低置信度事实"},
                "tags": ["fact"],
                "confidence": 0.1,
            },
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert len(result.message_window) == 100
    assert result.stats.trimmed_message_count == 6
    assert len(result.memory_candidates) == 1
    assert result.memory_candidates[0]["memory_type"] == "profile"
    assert result.memory_candidates[0]["content"] == {
        "language": "zh",
        "response_style": "concise",
    }
    assert result.stats.merged_profile_count == 1
    assert result.stats.dropped_low_confidence_count == 1


def test_memory_consolidation_service_should_degrade_when_provider_fails() -> None:
    class _FailingProvider:
        async def consolidate(self, payload: MemoryConsolidationInput) -> MemoryConsolidationResult:
            raise TimeoutError("provider timeout")

    payload = MemoryConsolidationInput(
        assistant_message="最终总结",
        pending_memory_writes=[
            {
                "namespace": "session/session-1/fact",
                "memory_type": "fact",
                "summary": "可保留事实",
                "content": {"text": "可保留事实"},
                "confidence": 0.8,
            }
        ],
    )

    result = asyncio.run(MemoryConsolidationService(provider=_FailingProvider()).consolidate(payload))

    assert result.degraded is True
    assert result.degrade_reason == "provider_error:TimeoutError"
    assert len(result.memory_candidates) == 1
    assert result.conversation_summary == "结果:最终总结"


def test_memory_consolidation_service_should_reject_secret_memory_candidates() -> None:
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "用户 token",
                "content": {"text": "access_token=abcdefghi123456789"},
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "用户密码",
                "content": {"text": "password=secret123"},
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "provider payload secret",
                "content": {"api_key": "abcdefghi123456789"},
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "payload key secret",
                "content": {"api_key=abcdefghi123456789": "偏好中文"},
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "payload list secret",
                "content": {"items": ["token=abcdefghi123456789"]},
                "confidence": 0.9,
            },
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert result.memory_candidates == []
    assert result.stats.dropped_sensitive_count == 5
    assert result.stats.dropped_invalid_count == 0


def test_memory_consolidation_service_should_redact_pii_before_write() -> None:
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "邮箱 test@example.com 手机 13812345678",
                "content": {
                    "email": "test@example.com",
                    "phone": "13812345678",
                    "id_card": "11010519491231002X",
                    "test@example.com": "偏好中文",
                    "numeric_phone": 13812345678,
                    "items": ["11010519491231002X"],
                },
                "tags": ["pii"],
                "confidence": 0.9,
            }
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert len(result.memory_candidates) == 1
    candidate = result.memory_candidates[0]
    assert "test@example.com" not in candidate["summary"]
    assert "13812345678" not in candidate["summary"]
    assert candidate["content"] == {
        "email": "[REDACTED_EMAIL]",
        "phone": "[REDACTED_PHONE]",
        "id_card": "[REDACTED_ID_CARD]",
        "[REDACTED_EMAIL]": "偏好中文",
        "numeric_phone": "[REDACTED_PHONE]",
        "items": ["[REDACTED_ID_CARD]"],
    }


def test_memory_consolidation_service_should_deterministically_merge_redacted_duplicate_keys() -> None:
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "重复脱敏 key",
                "content": {
                    "first@example.com": "旧值",
                    "second@example.com": "新值",
                },
                "confidence": 0.9,
            }
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert len(result.memory_candidates) == 1
    assert result.memory_candidates[0]["content"] == {"[REDACTED_EMAIL]": "新值"}


def test_memory_consolidation_service_should_govern_tags_source_and_dedupe_key() -> None:
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "联系方式",
                "content": {"text": "联系方式"},
                "tags": ["foo@example.com", "13812345678", "11010519491231002X"],
                "source": {
                    "kind": "summary",
                    "stage": "memory",
                    "run_id": "run-1",
                    "note": "foo@example.com",
                },
                "dedupe_key": "contact foo@example.com",
                "confidence": 0.9,
            }
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert len(result.memory_candidates) == 1
    candidate = result.memory_candidates[0]
    assert candidate["tags"] == [
        "[REDACTED_EMAIL]",
        "[REDACTED_PHONE]",
        "[REDACTED_ID_CARD]",
    ]
    assert candidate["source"] == {
        "kind": "summary",
        "stage": "memory",
        "run_id": "run-1",
    }
    assert candidate["dedupe_key"] == "contact [REDACTED_EMAIL]"


def test_memory_consolidation_service_should_reject_secret_in_tags_source_or_dedupe_key() -> None:
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "tag secret",
                "content": {"text": "tag secret"},
                "tags": ["api_key=abcdefghi123456789"],
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "source secret",
                "content": {"text": "source secret"},
                "source": {"kind": "summary", "stage": "cookie=sessionid123456789"},
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "dedupe secret",
                "content": {"text": "dedupe secret"},
                "dedupe_key": "password=secret123",
                "confidence": 0.9,
            },
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert result.memory_candidates == []
    assert result.stats.dropped_sensitive_count == 3


def test_memory_consolidation_service_should_reject_unserializable_source_without_breaking_consolidation() -> None:
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "source unserializable",
                "content": {"text": "source unserializable"},
                "source": {"kind": "summary", "raw": object()},
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": "valid candidate",
                "content": {"text": "valid candidate"},
                "source": {"kind": "summary"},
                "confidence": 0.9,
            },
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert len(result.memory_candidates) == 1
    assert result.memory_candidates[0]["summary"] == "valid candidate"
    assert result.memory_candidates[0]["source"] == {"kind": "summary"}
    assert result.stats.dropped_sensitive_count == 1
    assert result.stats.dropped_invalid_count == 0


def test_memory_consolidation_service_should_keep_profile_merge_governance() -> None:
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "用户偏好一",
                "content": {"language": "zh"},
                "tags": ["foo@example.com"],
                "source": {"kind": "summary", "stage": "profile", "extra": "13812345678"},
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "用户偏好二",
                "content": {"style": "concise"},
                "tags": ["13812345678"],
                "source": {"kind": "provider", "run_id": "run-1", "note": "11010519491231002X"},
                "confidence": 0.9,
            },
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert len(result.memory_candidates) == 1
    candidate = result.memory_candidates[0]
    assert candidate["tags"] == ["[REDACTED_EMAIL]", "[REDACTED_PHONE]"]
    assert candidate["source"] == {"kind": "provider", "stage": "profile", "run_id": "run-1"}


def test_memory_consolidation_service_should_truncate_candidate_summary() -> None:
    long_summary = "长期记忆候选摘要" * 30
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "summary": long_summary,
                "content": {"text": "候选正文"},
                "confidence": 0.9,
            }
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert len(result.memory_candidates) == 1
    assert result.memory_candidates[0]["summary"] == long_summary[:120]
    assert result.memory_candidates[0]["summary"] != long_summary


def test_memory_consolidation_service_should_truncate_summary_derived_from_content_text() -> None:
    long_content_text = "正文派生摘要" * 40
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/fact",
                "memory_type": "fact",
                "content": {"text": long_content_text},
                "confidence": 0.9,
            }
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert len(result.memory_candidates) == 1
    assert result.memory_candidates[0]["summary"] == long_content_text[:120]


def test_memory_consolidation_service_should_truncate_profile_merge_summary() -> None:
    long_summary = "合并后的偏好摘要" * 30
    payload = MemoryConsolidationInput(
        pending_memory_writes=[
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": "旧摘要",
                "content": {"language": "zh"},
                "confidence": 0.9,
            },
            {
                "namespace": "user/user-1/profile",
                "memory_type": "profile",
                "summary": long_summary,
                "content": {"style": "concise"},
                "confidence": 0.9,
            },
        ],
    )

    result = asyncio.run(MemoryConsolidationService().consolidate(payload))

    assert len(result.memory_candidates) == 1
    assert result.memory_candidates[0]["summary"] == long_summary[:120]
