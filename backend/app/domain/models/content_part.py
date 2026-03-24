#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/24
@Author : caixiaorong01@outlook.com
@File   : content_part.py
"""
from typing import Annotated, Literal, Optional, Union, List

from pydantic import BaseModel, Field


SUPPORTED_CONTENT_PART_TYPES: tuple[str, ...] = (
    "text",
    "image",
    "audio",
    "pdf",
    "file_ref",
)


class TextContentPart(BaseModel):
    """文本内容片段。"""

    type: Literal["text"] = "text"
    text: str = ""


class FileRefContentPart(BaseModel):
    """文件引用片段。"""

    type: Literal["file_ref"] = "file_ref"
    file_id: Optional[str] = None
    filepath: str = ""
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    extension: Optional[str] = None
    size: Optional[int] = None
    # 文本文件的内联内容（可选）。用于“模型支持多模态时优先走模型输入”。
    text_content: Optional[str] = None


class ImageContentPart(BaseModel):
    """图片内容片段。"""

    type: Literal["image"] = "image"
    file_id: Optional[str] = None
    filepath: str = ""
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    extension: Optional[str] = None
    size: Optional[int] = None


class AudioContentPart(BaseModel):
    """音频内容片段。"""

    type: Literal["audio"] = "audio"
    file_id: Optional[str] = None
    filepath: str = ""
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    extension: Optional[str] = None
    size: Optional[int] = None


class PdfContentPart(BaseModel):
    """PDF 内容片段。"""

    type: Literal["pdf"] = "pdf"
    file_id: Optional[str] = None
    filepath: str = ""
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    extension: Optional[str] = None
    size: Optional[int] = None


ContentPart = Annotated[
    Union[
        TextContentPart,
        ImageContentPart,
        AudioContentPart,
        PdfContentPart,
        FileRefContentPart,
    ],
    Field(discriminator="type"),
]


class MessageInputEnvelope(BaseModel):
    """统一输入内容封装。"""

    source: Literal["user", "tool", "system"] = "user"
    text: str = ""
    parts: List[ContentPart] = Field(default_factory=list)
