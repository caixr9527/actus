#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenAI Compatible 向量嵌入服务实现。"""
from typing import List

from openai import AsyncOpenAI

from app.domain.external import EmbeddingService


class OpenAIEmbeddingService(EmbeddingService):
    """基于 OpenAI Compatible embeddings API 的向量服务。"""

    def __init__(
            self,
            *,
            base_url: str,
            api_key: str,
            embedding_model: str,
            dimensions: int | None = None,
            **kwargs,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
        self._embedding_model = embedding_model
        self._dimensions = dimensions

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        normalized_texts = [str(text or "").strip() for text in texts]
        if len(normalized_texts) == 0:
            return []

        request_kwargs = {
            "model": self._embedding_model,
            "input": normalized_texts,
        }
        if self._dimensions is not None:
            request_kwargs["dimensions"] = int(self._dimensions)

        response = await self._client.embeddings.create(**request_kwargs)
        return [list(item.embedding or []) for item in list(response.data or [])]
