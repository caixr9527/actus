import asyncio

import pytest

from app.domain.services.memory_consolidation import (
    MemoryConsolidationInput,
    MemoryConsolidationResult,
    MemoryConsolidationService,
)
from app.infrastructure.external.llm.ollama_memory_consolidation_provider import (
    OllamaMemoryConsolidationProvider,
)


class _FakeOllamaLLM:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or MemoryConsolidationResult(
            conversation_summary="主题:测试记忆沉淀",
            facts_in_session=["当前任务只关注 backend"],
            user_preferences={"language": "zh"},
            memory_candidates=[
                {
                    "namespace": "session/session-1/fact",
                    "memory_type": "fact",
                    "summary": "当前任务只关注 backend",
                    "content": {"text": "当前任务只关注 backend"},
                    "confidence": 0.8,
                }
            ],
        )
        self.error = error
        self.calls = []

    async def generate_structured(self, *, prompt, output_model):
        self.calls.append({"prompt": prompt, "output_model": output_model})
        if self.error is not None:
            raise self.error
        return self.result


def test_ollama_memory_consolidation_provider_should_delegate_to_ollama_llm_structured_output() -> None:
    llm = _FakeOllamaLLM()
    provider = OllamaMemoryConsolidationProvider(llm=llm)

    result = asyncio.run(
        provider.consolidate(
            MemoryConsolidationInput(
                session_id="session-1",
                user_message="请记住只关注 backend",
                assistant_message="已完成",
            )
        )
    )

    assert result.conversation_summary == "主题:测试记忆沉淀"
    assert result.user_preferences == {"language": "zh"}
    assert len(result.memory_candidates) == 1
    assert llm.calls[0]["output_model"] is MemoryConsolidationResult
    assert '"session_id": "session-1"' in llm.calls[0]["prompt"]
    assert "禁止把完整最终回答当成 conversation_summary" in llm.calls[0]["prompt"]


def test_memory_consolidation_service_should_degrade_when_ollama_llm_fails() -> None:
    provider = OllamaMemoryConsolidationProvider(
        llm=_FakeOllamaLLM(error=ValueError("invalid structured output"))
    )

    result = asyncio.run(
        MemoryConsolidationService(provider=provider).consolidate(
            MemoryConsolidationInput(assistant_message="最终总结")
        )
    )

    assert result.degraded is True
    assert result.degrade_reason == "provider_error:ValueError"
    assert result.conversation_summary == "结果:最终总结"


def test_ollama_memory_consolidation_provider_should_require_llm() -> None:
    with pytest.raises(ValueError, match="OllamaLLM"):
        OllamaMemoryConsolidationProvider(llm=None)
