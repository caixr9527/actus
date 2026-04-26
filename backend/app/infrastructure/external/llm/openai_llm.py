#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenAI SDK 统一 LLM 门面。

设计约束：
1. OpenAI 官方模型走 Responses API。
2. 第三方 OpenAI-compatible 模型走 Chat Completions 公共子集。
3. 上层仍复用现有 chat-style message 契约，避免一次性扩散改造范围。
"""
import logging
from typing import Any, Dict, List, Literal, Protocol

from openai import AsyncOpenAI

from app.application.errors.exceptions import ServerError
from app.domain.external import LLM
from app.domain.models import RuntimeLLMConfig

logger = logging.getLogger(__name__)

ChatMessage = Dict[str, Any]


class _OpenAITransport(Protocol):
    """OpenAI SDK 传输协议，约束不同 API 风格的最小调用面。"""

    async def invoke(
            self,
            messages: list[ChatMessage],
            tools: list[Dict[str, Any]] | None,
            response_format: Dict[str, Any] | None,
            tool_choice: str | Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        ...


def _extract_tool_call_names(message_payload: Dict[str, Any]) -> List[str]:
    tool_call_names: List[str] = []
    raw_tool_calls = message_payload.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return tool_call_names
    for raw_call in raw_tool_calls:
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        function_name = str(function.get("name") or "").strip()
        if function_name:
            tool_call_names.append(function_name)
    return tool_call_names


def _build_content_preview(content: Any, max_len: int = 160) -> str:
    if content is None:
        return ""
    preview = str(content).replace("\n", "\\n")
    return preview[:max_len] + ("..." if len(preview) > max_len else "")


def _normalize_api_style(value: Any) -> Literal["chat_completions", "responses"]:
    normalized = str(value or "").strip().lower()
    if normalized == "responses":
        return "responses"
    return "chat_completions"


def _normalize_chat_tools(tools: list[Dict[str, Any]] | None) -> list[Dict[str, Any]] | None:
    normalized_tools: list[Dict[str, Any]] = []
    for item in list(tools or []):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        normalized_tools.append(
            {
                "type": "function",
                "name": name,
                "description": function.get("description"),
                "parameters": function.get("parameters") or {},
                # strict=True 是 OpenAI Responses 的默认语义，这里显式保持收敛。
                "strict": True,
            }
        )
    return normalized_tools or None


def _normalize_responses_tool_choice(tool_choice: str | Dict[str, Any] | None) -> str | Dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip().lower()
        if normalized in {"auto", "none", "required"}:
            return normalized
        return tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice
    function = tool_choice.get("function")
    if tool_choice.get("type") == "function" and isinstance(function, dict):
        name = str(function.get("name") or "").strip()
        if name:
            return {"type": "function", "name": name}
    return tool_choice


def _normalize_responses_text_config(response_format: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(response_format, dict) or not response_format:
        return None
    response_type = str(response_format.get("type") or "").strip()
    if response_type == "json_object":
        return {"format": {"type": "json_object"}}
    if response_type != "json_schema":
        return {"format": {"type": response_type}} if response_type else None

    raw_schema = response_format.get("json_schema")
    if not isinstance(raw_schema, dict):
        return {"format": {"type": "json_schema", "name": "response", "schema": {}}}
    schema_name = str(raw_schema.get("name") or "response").strip() or "response"
    schema_body = raw_schema.get("schema")
    if not isinstance(schema_body, dict):
        schema_body = {}
    format_payload: Dict[str, Any] = {
        "type": "json_schema",
        "name": schema_name,
        "schema": schema_body,
    }
    if "description" in raw_schema:
        format_payload["description"] = raw_schema.get("description")
    if "strict" in raw_schema:
        format_payload["strict"] = bool(raw_schema.get("strict"))
    return {"format": format_payload}


def _normalize_responses_content_part(part: Dict[str, Any]) -> Dict[str, Any]:
    part_type = str(part.get("type") or "").strip()
    if part_type == "text":
        return {"type": "input_text", "text": str(part.get("text") or "")}
    if part_type == "input_text":
        return {"type": "input_text", "text": str(part.get("text") or "")}
    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        return {
            "type": "input_image",
            "image_url": str(image_url or ""),
            "detail": "auto",
        }
    if part_type == "input_image":
        return {
            "type": "input_image",
            "image_url": str(part.get("image_url") or ""),
            "detail": str(part.get("detail") or "auto"),
        }
    if part_type == "input_file":
        normalized_part = {"type": "input_file"}
        for key in ("file_id", "file_url", "file_data", "filename"):
            if key in part:
                normalized_part[key] = part.get(key)
        return normalized_part
    raise ServerError(msg=f"Responses API 暂不支持的消息片段类型: {part_type or 'unknown'}")


def _normalize_responses_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    normalized_parts: list[Dict[str, Any]] = []
    for raw_part in content:
        if not isinstance(raw_part, dict):
            normalized_parts.append({"type": "input_text", "text": str(raw_part)})
            continue
        normalized_parts.append(_normalize_responses_content_part(raw_part))
    return normalized_parts


def _normalize_responses_input(messages: list[ChatMessage]) -> list[Dict[str, Any]]:
    normalized_input: list[Dict[str, Any]] = []
    for message in list(messages or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role in {"system", "developer", "user", "assistant"}:
            content = message.get("content")
            if content not in (None, "", []):
                normalized_input.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": _normalize_responses_message_content(content),
                    }
                )
            raw_tool_calls = message.get("tool_calls")
            if role == "assistant" and isinstance(raw_tool_calls, list):
                for raw_call in raw_tool_calls:
                    if not isinstance(raw_call, dict):
                        continue
                    function = raw_call.get("function")
                    if not isinstance(function, dict):
                        continue
                    function_name = str(function.get("name") or "").strip()
                    if not function_name:
                        continue
                    call_id = str(raw_call.get("call_id") or raw_call.get("id") or "").strip()
                    if not call_id:
                        raise ServerError(msg="Responses API 工具回放缺少 call_id")
                    normalized_input.append(
                        {
                            "type": "function_call",
                            "id": str(raw_call.get("id") or call_id),
                            "call_id": call_id,
                            "name": function_name,
                            "arguments": str(function.get("arguments") or "{}"),
                        }
                    )
            continue

        if role == "tool":
            call_id = str(message.get("call_id") or message.get("tool_call_id") or "").strip()
            if not call_id:
                raise ServerError(msg="Responses API 工具结果回放缺少 call_id")
            normalized_input.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(message.get("content") or ""),
                }
            )
    return normalized_input


def _normalize_responses_message(response_dump: Dict[str, Any]) -> Dict[str, Any]:
    output_items = response_dump.get("output")
    if not isinstance(output_items, list):
        output_items = []

    content_parts: list[str] = []
    tool_calls: list[Dict[str, Any]] = []
    reasoning_summary_parts: list[str] = []

    for item in output_items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "message":
            for content_item in list(item.get("content") or []):
                if not isinstance(content_item, dict):
                    continue
                content_type = str(content_item.get("type") or "").strip()
                if content_type in {"output_text", "refusal"}:
                    content_parts.append(str(content_item.get("text") or ""))
            continue
        if item_type == "function_call":
            function_name = str(item.get("name") or "").strip()
            if not function_name:
                continue
            call_id = str(item.get("call_id") or item.get("id") or "").strip()
            tool_calls.append(
                {
                    "id": str(item.get("id") or call_id or function_name),
                    "call_id": call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": str(item.get("arguments") or "{}"),
                    },
                }
            )
            continue
        if item_type == "reasoning":
            for summary_item in list(item.get("summary") or []):
                if not isinstance(summary_item, dict):
                    continue
                text = str(summary_item.get("text") or "").strip()
                if text:
                    reasoning_summary_parts.append(text)

    normalized_message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if tool_calls:
        normalized_message["tool_calls"] = tool_calls
    if reasoning_summary_parts:
        normalized_message["reasoning_summary"] = "\n".join(reasoning_summary_parts)
    if response_dump.get("id"):
        normalized_message["response_id"] = response_dump.get("id")
    return normalized_message


class _OpenAICompatibleChatTransport(_OpenAITransport):
    """第三方兼容网关使用的最小公共 Chat Completions 通道。"""

    def __init__(
            self,
            *,
            client: AsyncOpenAI,
            model_name: str,
            temperature: float,
            max_tokens: int,
            timeout: int,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    async def invoke(
            self,
            messages: list[ChatMessage],
            tools: list[Dict[str, Any]] | None,
            response_format: Dict[str, Any] | None,
            tool_choice: str | Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        request_payload: Dict[str, Any] = {
            "model": self._model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": messages,
            "parallel_tool_calls": False,
            "timeout": self._timeout,
        }
        if response_format:
            request_payload["response_format"] = response_format
        if tools:
            request_payload["tools"] = tools
            request_payload["tool_choice"] = tool_choice
        response = await self._client.chat.completions.create(**request_payload)
        return response.choices[0].message.model_dump()


class _OpenAIResponsesTransport(_OpenAITransport):
    """OpenAI 官方模型使用的 Responses API 通道。"""

    def __init__(
            self,
            *,
            client: AsyncOpenAI,
            model_name: str,
            temperature: float,
            max_tokens: int,
            timeout: int,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    async def invoke(
            self,
            messages: list[ChatMessage],
            tools: list[Dict[str, Any]] | None,
            response_format: Dict[str, Any] | None,
            tool_choice: str | Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        request_payload: Dict[str, Any] = {
            "model": self._model_name,
            "input": _normalize_responses_input(messages),
            "temperature": self._temperature,
            "max_output_tokens": self._max_tokens,
            "parallel_tool_calls": False,
            "timeout": self._timeout,
        }
        text_config = _normalize_responses_text_config(response_format)
        if text_config:
            request_payload["text"] = text_config
        normalized_tools = _normalize_chat_tools(tools)
        if normalized_tools:
            request_payload["tools"] = normalized_tools
            request_payload["tool_choice"] = _normalize_responses_tool_choice(tool_choice)
        response = await self._client.responses.create(**request_payload)
        return _normalize_responses_message(response.model_dump())


class OpenAILLM(LLM):
    """基于 OpenAI SDK 的统一 LLM 门面。"""

    def __init__(self, llm_config: RuntimeLLMConfig, **kwargs) -> None:
        self._llm_config = llm_config.model_copy(deep=True)
        self._client = AsyncOpenAI(
            base_url=str(llm_config.base_url),
            api_key=llm_config.api_key,
            **kwargs,
        )
        self._model_name = llm_config.model_name
        self._temperature = llm_config.temperature
        self._max_tokens = llm_config.max_tokens
        self._api_style = _normalize_api_style(llm_config.api_style)
        self._multimodal = bool(llm_config.multimodal)
        self._supported = [str(item) for item in list(llm_config.supported or []) if str(item).strip()]
        self._timeout = 180
        self._transport = self._build_transport()

    def _build_transport(self) -> _OpenAITransport:
        if self._api_style == "responses":
            return _OpenAIResponsesTransport(
                client=self._client,
                model_name=self._model_name,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
            )
        return _OpenAICompatibleChatTransport(
            client=self._client,
            model_name=self._model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )

    @property
    def llm_config(self) -> RuntimeLLMConfig:
        return self._llm_config.model_copy(deep=True)

    def clone_with_overrides(
            self,
            *,
            model_name: str | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
    ) -> "OpenAILLM":
        next_config = self._llm_config.model_copy(deep=True)
        if model_name is not None and str(model_name).strip():
            next_config.model_name = str(model_name).strip()
        if temperature is not None:
            next_config.temperature = float(temperature)
        if max_tokens is not None:
            next_config.max_tokens = int(max_tokens)
        return OpenAILLM(llm_config=next_config)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def multimodal(self) -> bool:
        return self._multimodal

    @property
    def supported(self) -> list[str]:
        return list(self._supported)

    async def _image_message_format(self, input_part: Dict[str, Any]) -> Dict[str, Any]:
        if self.model_name.startswith("Qwen"):
            return {
                "type": "image_url",
                "image_url": {
                    "url": input_part.get("file_url")
                }
            }
        if self.model_name.startswith("Kimi"):
            image_url = f"data:{input_part.get('mime_type')};base64,{input_part.get('base64_payload')}"
            return {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        image_url = f"data:{input_part.get('mime_type')};base64,{input_part.get('base64_payload')}"
        return {
            "type": "image_url",
            "image_url": {
                "url": image_url,
            },
        }

    async def _video_message_format(self, input_part: Dict[str, Any]) -> Dict[str, Any]:
        if self.model_name.startswith("Qwen"):
            return {
                "type": "video",
                "video": [
                    input_part.get("file_url")
                ]
            }
        if self.model_name.startswith("Kimi"):
            video_url = f"data:{input_part.get('mime_type')};base64,{input_part.get('base64_payload')}"
            return {
                "type": "video_url",
                "video_url": {
                    "url": video_url,
                },
            }
        return {}

    async def _audio_message_format(self, input_part: Dict[str, Any]) -> Dict[str, Any]:
        if self.model_name.startswith("Qwen"):
            return {}
        if self.model_name.startswith("Kimi"):
            return {}
        return {}

    async def format_multiplexed_message(self, input_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        attachments_message: List[Dict[str, Any]] = []
        for input_part in input_parts:
            if input_part.get("type") == "image":
                attachments_message.append(await self._image_message_format(input_part))
            if input_part.get("type") == "audio":
                attachments_message.append(await self._audio_message_format(input_part))
            if input_part.get("type") == "video":
                attachments_message.append(await self._video_message_format(input_part))

        return attachments_message

    async def invoke(
            self,
            messages: list[Dict[str, Any]],
            tools: list[Dict[str, Any]] = None,
            response_format: Dict[str, Any] = None,
            tool_choice: str | Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        try:
            logger.info(
                "调用模型: model=%s api_style=%s tools_enabled=%s",
                self._model_name,
                self._api_style,
                bool(tools),
            )
            message_dump = await self._transport.invoke(
                messages=messages,
                tools=tools,
                response_format=response_format,
                tool_choice=tool_choice,
            )
            logger.info(
                "大模型返回摘要: model=%s api_style=%s tool_calls=%s content_preview=%s",
                self._model_name,
                self._api_style,
                _extract_tool_call_names(message_dump),
                _build_content_preview(message_dump.get("content"), 2000),
            )
            return message_dump
        except Exception as e:
            logger.error(f"调用模型失败: {e}")
            raise ServerError("调用模型出错") from e
