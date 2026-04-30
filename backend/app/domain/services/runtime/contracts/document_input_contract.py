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

DOCUMENT_INPUT_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
DOCUMENT_INPUT_MAX_ATTACHMENTS = 10
DOCUMENT_INPUT_EXCERPT_CHAR_LIMIT = 8_000
DOCUMENT_INPUT_CONTEXT_CHAR_LIMIT = 24_000

DOCUMENT_REASON_UNSUPPORTED_DOCUMENT_FORMAT = "unsupported_document_format"
DOCUMENT_REASON_UNSUPPORTED_MEDIA_IMAGE = "unsupported_media_image"
DOCUMENT_REASON_UNSUPPORTED_MEDIA_AUDIO = "unsupported_media_audio"
DOCUMENT_REASON_UNSUPPORTED_MEDIA_VIDEO = "unsupported_media_video"
DOCUMENT_REASON_UNSUPPORTED_BINARY = "unsupported_binary"
DOCUMENT_REASON_TOO_LARGE = "too_large"
DOCUMENT_REASON_TOO_MANY_ATTACHMENTS = "too_many_attachments"
DOCUMENT_REASON_EMPTY = "empty"
DOCUMENT_REASON_DECODE_FAILED = "decode_failed"
DOCUMENT_REASON_PARSE_FAILED = "parse_failed"

DOCUMENT_KIND_EXTENSIONS: dict[DocumentInputKind, frozenset[str]] = {
    DocumentInputKind.TEXT: frozenset({".txt", ".log"}),
    DocumentInputKind.MARKDOWN: frozenset({".md", ".markdown"}),
    DocumentInputKind.CSV: frozenset({".csv", ".tsv"}),
    DocumentInputKind.JSON: frozenset({".json", ".jsonl"}),
    DocumentInputKind.HTML: frozenset({".html", ".htm"}),
    DocumentInputKind.PDF: frozenset({".pdf"}),
    DocumentInputKind.WORD: frozenset({".doc", ".docx"}),
    DocumentInputKind.SPREADSHEET: frozenset({".xlsx", ".xls"}),
    DocumentInputKind.ARCHIVE: frozenset({".zip"}),
}

CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".cs", ".php", ".rb", ".swift", ".kt", ".kts", ".scala", ".sh",
    ".bash", ".zsh", ".fish", ".sql", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".xml", ".css", ".scss", ".less", ".vue", ".svelte",
})

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"})
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
BINARY_EXTENSIONS = frozenset({".exe", ".bin", ".dmg"})


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


def normalize_document_extension(extension: object, filename: object = "") -> str:
    """归一化文件扩展名，保证带点、小写。"""
    raw_extension = str(extension or "").strip().lower()
    if raw_extension:
        return raw_extension if raw_extension.startswith(".") else f".{raw_extension}"
    raw_filename = str(filename or "").strip().lower()
    if "." not in raw_filename:
        return ""
    suffix = raw_filename.rsplit(".", 1)[-1].strip()
    return f".{suffix}" if suffix else ""


def resolve_document_kind(*, extension: object, mime_type: object = "", filename: object = "") -> DocumentInputKind:
    """按 P0-5 白名单识别文档格式族，不表达业务任务类型。"""
    normalized_extension = normalize_document_extension(extension, filename)
    for kind, extensions in DOCUMENT_KIND_EXTENSIONS.items():
        if normalized_extension in extensions:
            return kind
    if normalized_extension in CODE_EXTENSIONS:
        return DocumentInputKind.CODE
    normalized_mime = str(mime_type or "").strip().lower()
    if normalized_mime.startswith("text/"):
        return DocumentInputKind.TEXT
    return DocumentInputKind.UNKNOWN_DOCUMENT


def resolve_unsupported_media_reason(*, extension: object, mime_type: object = "", filename: object = "") -> str | None:
    """识别必须在请求前拒绝的非文档媒体和禁止二进制。"""
    normalized_extension = normalize_document_extension(extension, filename)
    normalized_mime = str(mime_type or "").strip().lower()
    if normalized_extension in IMAGE_EXTENSIONS or normalized_mime.startswith("image/"):
        return DOCUMENT_REASON_UNSUPPORTED_MEDIA_IMAGE
    if normalized_extension in AUDIO_EXTENSIONS or normalized_mime.startswith("audio/"):
        return DOCUMENT_REASON_UNSUPPORTED_MEDIA_AUDIO
    if normalized_extension in VIDEO_EXTENSIONS or normalized_mime.startswith("video/"):
        return DOCUMENT_REASON_UNSUPPORTED_MEDIA_VIDEO
    if normalized_extension in BINARY_EXTENSIONS:
        return DOCUMENT_REASON_UNSUPPORTED_BINARY
    return None


def is_supported_document_metadata(*, extension: object, mime_type: object = "", filename: object = "") -> bool:
    """判断附件元数据是否属于 P0-5 文档白名单。"""
    return resolve_document_kind(
        extension=extension,
        mime_type=mime_type,
        filename=filename,
    ) != DocumentInputKind.UNKNOWN_DOCUMENT
