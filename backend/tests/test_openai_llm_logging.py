import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from app.domain.models import RuntimeLLMConfig
from app.infrastructure.external.llm.openai_llm import OpenAILLM


class _FakeMessage:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._payload)


class _FakeResponse:
    def __init__(self, response_payload: Dict[str, Any], message_payload: Dict[str, Any]) -> None:
        self._response_payload = response_payload
        self.choices = [SimpleNamespace(message=_FakeMessage(message_payload))]

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._response_payload)


class _FakeCompletions:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_kwargs: Optional[Dict[str, Any]] = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def _build_llm(response_payload: Dict[str, Any], message_payload: Dict[str, Any]) -> OpenAILLM:
    llm = OpenAILLM(
        llm_config=RuntimeLLMConfig(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model_name="qwen3.5-plus",
            temperature=0.7,
            max_tokens=8192,
            multimodal=True,
            supported=["text", "image"],
        )
    )
    llm._client = _FakeClient(_FakeResponse(response_payload=response_payload, message_payload=message_payload))
    return llm


def _collect_log_text(caplog) -> str:
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "app.infrastructure.external.llm.openai_llm"
    )


def test_openai_llm_invoke_should_log_summary_without_reasoning_content(caplog) -> None:
    message_payload = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "write_file", "arguments": "{\"filepath\":\"/tmp/a.md\"}"},
            }
        ],
        "reasoning_content": "这是内部思考，不应该出现在日志中",
    }
    response_payload = {
        "model": "qwen3.5-plus",
        "choices": [{"finish_reason": "tool_calls", "message": dict(message_payload)}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    llm = _build_llm(response_payload=response_payload, message_payload=message_payload)
    caplog.set_level(logging.INFO, logger="app.infrastructure.external.llm.openai_llm")

    result = asyncio.run(
        llm.invoke(
            messages=[{"role": "user", "content": "读取附件"}],
            tools=[
                {"type": "function", "function": {"name": "write_file", "parameters": {}}},
                {"type": "function", "function": {"name": "read_file", "parameters": {}}},
            ],
        )
    )

    log_text = _collect_log_text(caplog)
    assert result == message_payload
    assert "调用模型: model=qwen3.5-plus tools_enabled=True request_tools=['write_file', 'read_file']" in log_text
    assert "大模型返回摘要:" in log_text
    assert "tool_calls=['write_file']" in log_text
    assert "reasoning_content" not in log_text
    assert "这是内部思考，不应该出现在日志中" not in log_text


def test_openai_llm_invoke_should_log_summary_without_tools(caplog) -> None:
    message_payload = {
        "role": "assistant",
        "content": "第一行\n第二行",
    }
    response_payload = {
        "model": "qwen3.5-plus",
        "choices": [{"finish_reason": "stop", "message": dict(message_payload)}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
    }
    llm = _build_llm(response_payload=response_payload, message_payload=message_payload)
    caplog.set_level(logging.INFO, logger="app.infrastructure.external.llm.openai_llm")

    result = asyncio.run(llm.invoke(messages=[{"role": "user", "content": "你好"}], tools=[]))

    log_text = _collect_log_text(caplog)
    assert result == message_payload
    assert "调用模型: model=qwen3.5-plus tools_enabled=False request_tools=[]" in log_text
    assert "大模型返回摘要:" in log_text
    assert "tool_calls=[]" in log_text
    assert "content_preview=第一行\\n第二行" in log_text
