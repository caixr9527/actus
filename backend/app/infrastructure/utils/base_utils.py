#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/25 14:50
@Author : caixiaorong01@outlook.com
@File   : base_utils.py
"""
from pathlib import Path
from typing import Any


class BaseUtils:
    TEXT_ATTACHMENT_READ_CHUNK_BYTES: int = 64 * 1024
    MAX_INLINE_BINARY_ATTACHMENT_BYTES: int = 50 * 1024 * 1024
    IMAGE_EXTENSIONS: set[str] = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"}
    AUDIO_EXTENSIONS: set[str] = {"mp3", "wav", "m4a", "aac", "flac", "ogg"}
    VIDEO_EXTENSIONS: set[str] = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}

    @classmethod
    def read_limited_bytes(cls, stream: Any, max_bytes: int = MAX_INLINE_BINARY_ATTACHMENT_BYTES) -> tuple[bytes, bool]:
        """按字节上限分块读取，返回(raw_bytes, is_truncated)。"""
        chunks: list[bytes] = []
        total = 0

        while total < max_bytes:
            remaining = max_bytes - total
            chunk_size = min(cls.TEXT_ATTACHMENT_READ_CHUNK_BYTES, remaining)
            chunk = stream.read(chunk_size)
            if not chunk:
                break

            if isinstance(chunk, str):
                data = chunk.encode("utf-8", errors="ignore")
            elif isinstance(chunk, (bytes, bytearray)):
                data = bytes(chunk)
            else:
                data = str(chunk).encode("utf-8", errors="ignore")

            if not data:
                break
            chunks.append(data)
            total += len(data)

        truncated = False
        if total >= max_bytes:
            probe = stream.read(1)
            if probe:
                truncated = True

        return b"".join(chunks), truncated

    @classmethod
    def resolve_part_type_by_filepath(cls, filepath: str) -> str:
        """按文件路径后缀判定类型；无后缀时默认 file。"""
        extension = Path(filepath).suffix.strip().lower().lstrip(".")
        if extension in cls.IMAGE_EXTENSIONS:
            return "image"
        if extension in cls.AUDIO_EXTENSIONS:
            return "audio"
        if extension in cls.VIDEO_EXTENSIONS:
            return "video"
        return "file"
