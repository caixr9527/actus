#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/17 14:40
@Author : caixiaorong01@outlook.com
@File   : llm.py
"""
from typing import Protocol, Any, Dict, List


class LLM(Protocol):
    """用于Agent应用与LLM进行交互的接口协议"""

    async def invoke(self,
                     messages: list[Dict[str, Any]],
                     tools: list[Dict[str, Any]] = None,
                     response_format: Dict[str, Any] = None,
                     tool_choice: str = None,
                     ) -> Dict[str, Any]:
        """调用LLM接口"""
        ...

    @property
    def model_name(self) -> str:
        """获取LLM模型名称"""
        ...

    @property
    def temperature(self) -> float:
        """获取LLM模型温度"""
        ...

    @property
    def max_tokens(self) -> int:
        """获取LLM模型最大token数"""
        ...

    @property
    def multimodal(self) -> bool:
        """是否支持原生多模态输入"""
        ...

    @property
    def supported(self) -> list[str]:
        """支持的输入类型集合"""
        ...

    async def format_multiplexed_message(self, input_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化多模态消息"""
        ...
