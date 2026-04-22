#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用 Ollama LLM 客户端。

本模块只封装 Ollama 本地模型调用能力，实现 domain.external.LLM 协议，
不依赖任何 memory consolidation 业务合同。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from app.application.errors.exceptions import ServerError
from app.domain.external import LLM

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


def _normalize_text_content(content: Any) -> str:
    """把当前 runtime 的消息 content 收敛为 Ollama 文本输入。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
                continue
            if item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _normalize_messages(messages: list[Dict[str, Any]]) -> list[Dict[str, str]]:
    """转换为 ollama.chat 需要的最小消息结构。"""
    normalized_messages: list[Dict[str, str]] = []
    for item in list(messages or []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip() or "user"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = _normalize_text_content(item.get("content"))
        if not content:
            continue
        normalized_messages.append({"role": role, "content": content})
    return normalized_messages


def _build_ollama_format(response_format: Dict[str, Any] | None) -> Any:
    """把现有 LLM response_format 映射为 Ollama format 参数。"""
    if not isinstance(response_format, dict) or not response_format:
        return None
    response_type = str(response_format.get("type") or "").strip()
    if response_type == "json_object":
        return "json"
    if response_type != "json_schema":
        return None
    raw_schema = response_format.get("json_schema")
    if not isinstance(raw_schema, dict):
        return None
    schema_body = raw_schema.get("schema")
    return schema_body if isinstance(schema_body, dict) else None


class OllamaLLM(LLM):
    """基于官方 ollama Python 客户端的通用 LLM 实现。"""

    def __init__(
            self,
            *,
            base_url: str,
            model: str,
            timeout_seconds: float,
            temperature: float = 0.0,
            max_tokens: int = 1024,
            client: Any | None = None,
    ) -> None:
        self._base_url = str(base_url or "").strip().rstrip("/")
        self._model_name = str(model or "").strip()
        self._timeout_seconds = max(float(timeout_seconds or 0), 0.1)
        self._temperature = float(temperature or 0.0)
        self._max_tokens = int(max_tokens or 0) if int(max_tokens or 0) > 0 else 1024
        if not self._base_url:
            raise ValueError("Ollama base_url 不能为空")
        if not self._model_name:
            raise ValueError("Ollama model 不能为空")
        self._client = client or ollama.AsyncClient(
            host=self._base_url,
            timeout=self._timeout_seconds,
        )

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
        return False

    @property
    def supported(self) -> list[str]:
        return ["text"]

    async def format_multiplexed_message(self, input_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ollama 当前作为文本 LLM 使用，非文本输入直接拒绝。"""
        formatted_parts: List[Dict[str, Any]] = []
        for input_part in list(input_parts or []):
            input_type = str(input_part.get("type") or "").strip()
            if input_type == "text":
                formatted_parts.append({"type": "text", "text": str(input_part.get("text") or "")})
                continue
            if input_type:
                raise ServerError(f"OllamaLLM 当前不支持 {input_type} 输入")
        return formatted_parts

    async def invoke(
            self,
            messages: list[Dict[str, Any]],
            tools: list[Dict[str, Any]] = None,
            response_format: Dict[str, Any] = None,
            tool_choice: str | Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """调用 Ollama chat；当前不声明工具调用能力。"""
        if tools:
            raise ServerError("OllamaLLM 当前不支持工具调用")
        normalized_tool_choice = (
            tool_choice.strip().lower()
            if isinstance(tool_choice, str)
            else tool_choice
        )
        if normalized_tool_choice not in {None, "", "none"}:
            raise ServerError("OllamaLLM 当前不支持 tool_choice")
        normalized_messages = _normalize_messages(messages)
        if not normalized_messages:
            raise ServerError("OllamaLLM 调用缺少有效消息")

        request_payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": normalized_messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        ollama_format = _build_ollama_format(response_format)
        if ollama_format is not None:
            request_payload["format"] = ollama_format

        try:
            response = await self._client.chat(**request_payload)
        except Exception as exc:
            raise ServerError("调用 Ollama 模型出错") from exc

        message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        if isinstance(message, dict):
            content = str(message.get("content") or "")
        else:
            content = str(getattr(message, "content", "") or "")
        return {"role": "assistant", "content": content}

    async def generate_structured(
            self,
            *,
            prompt: str,
            output_model: type[StructuredModel],
    ) -> StructuredModel:
        """使用 Ollama structured outputs 生成并校验 Pydantic 结构化结果。"""
        schema = output_model.model_json_schema()
        llm_message = await self.invoke(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        raw_content = str(llm_message.get("content") or "").strip()
        try:
            parsed = json.loads(raw_content)
        except Exception as exc:
            raise ValueError("Ollama structured output 返回非 JSON") from exc
        try:
            return output_model.model_validate(parsed)
        except ValidationError as exc:
            raise ValueError("Ollama structured output 不符合 Pydantic 合同") from exc
