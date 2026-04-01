#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""语义向量嵌入服务协议。"""
from typing import List, Protocol


class EmbeddingService(Protocol):
    """文本向量嵌入服务协议。"""

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对文本批量生成向量。"""
        ...
