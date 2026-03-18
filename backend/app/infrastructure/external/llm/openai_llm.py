#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/17 14:45
@Author : caixiaorong01@outlook.com
@File   : openai_llm.py
"""
import logging
from typing import Dict, Any

from app.application.errors.exceptions import ServerError
from app.domain.external import LLM
from app.domain.models import RuntimeLLMConfig
from openai import AsyncOpenAI

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

    async def invoke(self,
                     messages: list[Dict[str, Any]],
                     tools: list[Dict[str, Any]] = None,
                     response_format: Dict[str, Any] = None,
                     tool_choice: str = None,
                     ) -> Dict[str, Any]:
        """调用模型"""
        try:
            if tools:
                logger.info(f"调用模型携带工具信息: {self._model_name}")
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
                logger.info(f"调用模型未携带工具信息: {self._model_name}")
                response = await self._client.chat.completions.create(
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=messages,
                    response_format=response_format,
                    parallel_tool_calls=False,
                    timeout=self._timeout,
                )

            logger.info(f"大模型返回结果: {response.model_dump()}")
            return response.choices[0].message.model_dump()
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
