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
