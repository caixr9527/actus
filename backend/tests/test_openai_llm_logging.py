import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional

from app.domain.models import RuntimeLLMConfig
from app.infrastructure.external.llm.openai_llm import OpenAILLM


class _FakeMessage:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._payload)


class _FakeChatResponse:
    def __init__(self, message_payload: Dict[str, Any]) -> None:
        self.choices = [SimpleNamespace(message=_FakeMessage(message_payload))]


class _FakeChatCompletions:
    def __init__(self, message_payload: Dict[str, Any]) -> None:
        self._response = _FakeChatResponse(message_payload=message_payload)
        self.last_kwargs: Optional[Dict[str, Any]] = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeResponses:
    def __init__(self, response_payload: Dict[str, Any]) -> None:
        self._response_payload = response_payload
        self.last_kwargs: Optional[Dict[str, Any]] = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(model_dump=lambda: dict(self._response_payload))


class _FakeClient:
    def __init__(
            self,
            *,
            chat_message_payload: Dict[str, Any] | None = None,
            responses_payload: Dict[str, Any] | None = None,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(chat_message_payload or {"role": "assistant", "content": ""})
        )
        self.responses = _FakeResponses(responses_payload or {"id": "resp-empty", "output": []})


def _build_llm(
        *,
        api_style: str,
        chat_message_payload: Dict[str, Any] | None = None,
        responses_payload: Dict[str, Any] | None = None,
) -> OpenAILLM:
    llm = OpenAILLM(
        llm_config=RuntimeLLMConfig(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model_name="gpt-5.4-mini" if api_style == "responses" else "qwen3.5-plus",
            temperature=0.7,
            max_tokens=8192,
            api_style=api_style,
            multimodal=True,
            supported=["text", "image"],
        )
    )
    llm._client = _FakeClient(
        chat_message_payload=chat_message_payload,
        responses_payload=responses_payload,
    )
    llm._transport = llm._build_transport()
    return llm


def _collect_log_text(caplog) -> str:
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "app.infrastructure.external.llm.openai_llm"
    )


def test_openai_llm_invoke_should_log_summary_without_reasoning_content_for_chat_transport(caplog) -> None:
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
    llm = _build_llm(api_style="chat_completions", chat_message_payload=message_payload)
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
    assert "调用模型: model=qwen3.5-plus api_style=chat_completions tools_enabled=True" in log_text
    assert "大模型返回摘要: model=qwen3.5-plus api_style=chat_completions" in log_text
    assert "tool_calls=['write_file']" in log_text
    assert "reasoning_content" not in log_text
    assert "这是内部思考，不应该出现在日志中" not in log_text


def test_openai_llm_invoke_should_log_summary_without_tools_for_chat_transport(caplog) -> None:
    message_payload = {
        "role": "assistant",
        "content": "第一行\n第二行",
    }
    llm = _build_llm(api_style="chat_completions", chat_message_payload=message_payload)
    caplog.set_level(logging.INFO, logger="app.infrastructure.external.llm.openai_llm")

    result = asyncio.run(llm.invoke(messages=[{"role": "user", "content": "你好"}], tools=[]))

    log_text = _collect_log_text(caplog)
    assert result == message_payload
    assert "调用模型: model=qwen3.5-plus api_style=chat_completions tools_enabled=False" in log_text
    assert "tool_calls=[]" in log_text
    assert "content_preview=第一行\\n第二行" in log_text


def test_openai_llm_invoke_should_route_to_responses_api_and_normalize_output(caplog) -> None:
    responses_payload = {
        "id": "resp-123",
        "output": [
            {
                "id": "rs-1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "先检索再总结"}],
            },
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "我先搜索课程。"}],
            },
            {
                "id": "fc-1",
                "type": "function_call",
                "call_id": "call-search-1",
                "name": "search_web",
                "arguments": "{\"query\":\"imooc ai agent\"}",
            },
        ],
    }
    llm = _build_llm(api_style="responses", responses_payload=responses_payload)
    caplog.set_level(logging.INFO, logger="app.infrastructure.external.llm.openai_llm")

    result = asyncio.run(
        llm.invoke(
            messages=[{"role": "user", "content": "搜索课程"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "search web",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    },
                }
            ],
            response_format={"type": "json_object"},
            tool_choice="auto",
        )
    )

    request_kwargs = llm._client.responses.last_kwargs
    log_text = _collect_log_text(caplog)
    assert result == {
        "role": "assistant",
        "content": "我先搜索课程。",
        "tool_calls": [
            {
                "id": "fc-1",
                "call_id": "call-search-1",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": "{\"query\":\"imooc ai agent\"}",
                },
            }
        ],
        "reasoning_summary": "先检索再总结",
        "response_id": "resp-123",
    }
    assert request_kwargs is not None
    assert request_kwargs["input"] == [{"type": "message", "role": "user", "content": "搜索课程"}]
    assert request_kwargs["tools"] == [
        {
            "type": "function",
            "name": "search_web",
            "description": "search web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            "strict": True,
        }
    ]
    assert request_kwargs["text"] == {"format": {"type": "json_object"}}
    assert request_kwargs["tool_choice"] == "auto"
    assert "调用模型: model=gpt-5.4-mini api_style=responses tools_enabled=True" in log_text
    assert "tool_calls=['search_web']" in log_text
