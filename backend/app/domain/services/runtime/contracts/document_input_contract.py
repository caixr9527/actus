#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档输入领域契约。

该模块只承载纯领域枚举、模型和纯函数，不依赖数据库、接口层、
运行引擎、文件存储或 LLM 适配器。
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentInputKind(str, Enum):
    """Runtime 支持识别的文档格式族。"""

    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    WORD = "word"
    SPREADSHEET = "spreadsheet"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    CODE = "code"
    ARCHIVE = "archive"
    UNKNOWN_DOCUMENT = "unknown_document"


DOCUMENT_INPUT_TYPE = "document"
LEGACY_NATIVE_INPUT_TYPES = frozenset({"image", "audio", "video", "file", "file_ref"})
ALLOWED_DOCUMENT_INPUT_TYPES = frozenset({DOCUMENT_INPUT_TYPE})


class DocumentParseStatus(str, Enum):
    """文档解析生命周期状态。"""

    ACCEPTED = "accepted"
    PARSED = "parsed"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    TOO_LARGE = "too_large"
    ENCRYPTED = "encrypted"
    EMPTY = "empty"


class DocumentInputSourceRef(BaseModel):
    """文档输入的可追溯来源引用。"""

    file_id: str
    user_id: str
    session_id: str
    workspace_id: str | None = None
    run_id: str | None = None
    sandbox_filepath: str | None = None
    filename: str
    mime_type: str
    extension: str
    size: int
    sha256: str | None = None


class DocumentInputPart(BaseModel):
    """进入 Runtime graph state 的唯一附件输入类型。"""

    type: Literal["document"] = "document"
    kind: DocumentInputKind
    source: DocumentInputSourceRef
    parse_status: DocumentParseStatus
    text_excerpt: str = ""
    summary: str = ""
    structured_preview: dict[str, Any] = Field(default_factory=dict)
    reason_code: str | None = None
    is_truncated: bool = False


def is_document_input_part(raw_part: object) -> bool:
    """判断裸输入是否符合 P0-5 文档 input part 的类型边界。"""
    return isinstance(raw_part, dict) and str(raw_part.get("type") or "").strip().lower() == DOCUMENT_INPUT_TYPE


def normalize_document_supported_input_types(raw_supported: object) -> list[str]:
    """将模型声明的输入能力收口为 Runtime 唯一文档输入类型。"""
    if isinstance(raw_supported, str):
        candidates = [raw_supported]
    elif isinstance(raw_supported, list):
        candidates = raw_supported
    else:
        return []

    normalized_supported: list[str] = []
    for item in candidates:
        input_type = str(item or "").strip().lower()
        if not input_type or input_type not in ALLOWED_DOCUMENT_INPUT_TYPES:
            continue
        if input_type not in normalized_supported:
            normalized_supported.append(input_type)
    return normalized_supported
