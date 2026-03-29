#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/17 14:45
@Author : caixiaorong01@outlook.com
@File   : openai_llm.py
"""
import logging
from typing import Dict, Any, List

from openai import AsyncOpenAI

from app.application.errors.exceptions import ServerError
from app.domain.external import LLM
from app.domain.models import RuntimeLLMConfig

logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    """OpenAI语言模型"""

    def __init__(self, llm_config: RuntimeLLMConfig, **kwargs) -> None:
        self._client = AsyncOpenAI(
            base_url=str(llm_config.base_url),
            api_key=llm_config.api_key,
            **kwargs,
        )
        self._model_name = llm_config.model_name
        self._temperature = llm_config.temperature
        self._max_tokens = llm_config.max_tokens
        self._multimodal = bool(llm_config.multimodal)
        self._supported = [str(item) for item in list(llm_config.supported or []) if str(item).strip()]
        self._timeout = 3600

    @property
    def model_name(self) -> str:
        """模型名称"""
        return self._model_name

    @property
    def temperature(self) -> float:
        """温度"""
        return self._temperature

    @property
    def max_tokens(self) -> int:
        """最大生成长度"""
        return self._max_tokens

    @property
    def multimodal(self) -> bool:
        """是否支持原生多模态输入"""
        return self._multimodal

    @property
    def supported(self) -> list[str]:
        """支持的输入类型集合"""
        return list(self._supported)

    async def _image_message_format(self, input_part: Dict[str, Any]) -> Dict[str, Any]:
        if self.model_name.startswith("Qwen"):
            return {
                "type": "image_url",
                "image_url":
                    {
                        "url": input_part.get("file_url")
                    }
            }
        if self.model_name.startswith("Kimi"):
            image_url = f"data:{input_part.get("mime_type")};base64,{input_part.get("base64_payload")}"
            return {
                "type": "image_url",  # <-- 使用 image_url 类型来上传图片，内容为使用 base64 编码过的图片内容
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
            video_url = f"data:{input_part.get("mime_type")};base64,{input_part.get("base64_payload")}"
            return {
                "type": "video_url",  # <-- 使用 video_url 类型来上传视频，内容为使用 base64 编码过的视频内容
                "video_url": {
                    "url": video_url,
                },
            }

    async def _audio_message_format(self, input_part: Dict[str, Any]) -> Dict[str, Any]:
        if self.model_name.startswith("Qwen"):
            return {}
        if self.model_name.startswith("Kimi"):
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

    @staticmethod
    def _extract_requested_tool_names(tools: list[Dict[str, Any]] | None) -> List[str]:
        tool_names: List[str] = []
        for item in list(tools or []):
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if name:
                tool_names.append(name)
        return tool_names

    @staticmethod
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

    @staticmethod
    def _build_content_preview(content: Any, max_len: int = 160) -> str:
        if content is None:
            return ""
        preview = str(content).replace("\n", "\\n")
        return preview[:max_len] + ("..." if len(preview) > max_len else "")

    async def invoke(self,
                     messages: list[Dict[str, Any]],
                     tools: list[Dict[str, Any]] = None,
                     response_format: Dict[str, Any] = None,
                     tool_choice: str = None,
                     ) -> Dict[str, Any]:
        """调用模型"""
        try:
            requested_tool_names = []
            if tools:
                logger.info(
                    "调用模型: model=%s tools_enabled=%s request_tools=%s",
                    self._model_name,
                    True,
                    requested_tool_names,
                )
                response = await self._client.chat.completions.create(
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=messages,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,
                    timeout=self._timeout,
                )
            else:
                logger.info(
                    "调用模型: model=%s tools_enabled=%s request_tools=%s",
                    self._model_name,
                    False,
                    requested_tool_names,
                )
                response = await self._client.chat.completions.create(
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=messages,
                    response_format=response_format,
                    parallel_tool_calls=False,
                    timeout=self._timeout,
                )

            response_dump = response.model_dump()
            message_dump = response.choices[0].message.model_dump()
            first_choice = (response_dump.get("choices") or [{}])[0]
            if not isinstance(first_choice, dict):
                first_choice = {}
            usage = response_dump.get("usage")
            usage_summary = usage if isinstance(usage, dict) else {}
            logger.info(
                "大模型返回摘要: model=%s finish_reason=%s tool_calls=%s content_preview=%s usage=%s",
                self._model_name,
                str(first_choice.get("finish_reason") or ""),
                self._extract_tool_call_names(message_dump),
                self._build_content_preview(message_dump.get("content"), 2000),
                {
                    "prompt_tokens": usage_summary.get("prompt_tokens"),
                    "completion_tokens": usage_summary.get("completion_tokens"),
                    "total_tokens": usage_summary.get("total_tokens"),
                },
            )
            return message_dump
        except Exception as e:
            logger.error(f"调用模型失败: {e}")
            raise ServerError(f"调用模型出错")


if __name__ == "__main__":
    async def main():
        llm = OpenAILLM(llm_config=RuntimeLLMConfig(
            base_url="https://api.deepseek.com",
            api_key="",
            model_name="deepseek-reasoner",
            temperature=0.7,
            max_tokens=8192,
        ))
        response = await llm.invoke(messages=[{"role": "user", "content": "Hi"}], tools=[])
        print(response)


    import asyncio

    asyncio.run(main())
