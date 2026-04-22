import asyncio
import json

import pytest
from pydantic import BaseModel

from app.application.errors.exceptions import ServerError
from app.infrastructure.external.llm.ollama_llm import OllamaLLM


class _StructuredOutputForTest(BaseModel):
    answer: str = ""
    confidence: float = 0.0


class _FakeOllamaClient:
    def __init__(self, response_content: str = "") -> None:
        self.response_content = response_content
        self.requests = []

    async def chat(self, **kwargs):
        self.requests.append(kwargs)
        return {"message": {"role": "assistant", "content": self.response_content}}


def test_ollama_llm_should_invoke_chat_with_json_schema_format() -> None:
    client = _FakeOllamaClient(
        json.dumps(
            {
                "answer": "结构化成功",
                "confidence": 0.9,
            },
            ensure_ascii=False,
        )
    )
    llm = OllamaLLM(
        base_url="http://ollama.test",
        model="qwen2.5:3b",
        timeout_seconds=3,
        client=client,
    )

    result = asyncio.run(
        llm.generate_structured(
            prompt="请返回结构化结果",
            output_model=_StructuredOutputForTest,
        )
    )

    assert result.answer == "结构化成功"
    request = client.requests[0]
    assert request["model"] == "qwen2.5:3b"
    assert request["stream"] is False
    assert request["format"]["title"] == "_StructuredOutputForTest"
    assert request["messages"] == [{"role": "user", "content": "请返回结构化结果"}]


def test_ollama_llm_should_map_json_object_response_format() -> None:
    client = _FakeOllamaClient('{"ok": true}')
    llm = OllamaLLM(
        base_url="http://ollama.test",
        model="qwen2.5:3b",
        timeout_seconds=3,
        client=client,
    )

    message = asyncio.run(
        llm.invoke(
            messages=[{"role": "user", "content": "返回 JSON"}],
            response_format={"type": "json_object"},
        )
    )

    assert message["content"] == '{"ok": true}'
    assert client.requests[0]["format"] == "json"


def test_ollama_llm_should_reject_tools() -> None:
    llm = OllamaLLM(
        base_url="http://ollama.test",
        model="qwen2.5:3b",
        timeout_seconds=3,
        client=_FakeOllamaClient(),
    )

    with pytest.raises(ServerError, match="不支持工具调用"):
        asyncio.run(
            llm.invoke(
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"type": "function"}],
            )
        )


def test_ollama_llm_should_allow_tool_choice_none() -> None:
    client = _FakeOllamaClient("plain text")
    llm = OllamaLLM(
        base_url="http://ollama.test",
        model="qwen2.5:3b",
        timeout_seconds=3,
        client=client,
    )

    message = asyncio.run(
        llm.invoke(
            messages=[{"role": "user", "content": "hello"}],
            tool_choice="none",
        )
    )

    assert message["content"] == "plain text"
    assert client.requests[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_ollama_llm_should_reject_non_text_multiplexed_input() -> None:
    llm = OllamaLLM(
        base_url="http://ollama.test",
        model="qwen2.5:3b",
        timeout_seconds=3,
        client=_FakeOllamaClient(),
    )

    with pytest.raises(ServerError, match="不支持 image 输入"):
        asyncio.run(
            llm.format_multiplexed_message(
                [{"type": "image", "file_url": "https://example.com/a.png"}]
            )
        )
