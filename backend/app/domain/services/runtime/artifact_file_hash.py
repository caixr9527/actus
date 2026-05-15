#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Artifact file storage 内容 hash 校验工具。"""

from __future__ import annotations

import hashlib
from tempfile import SpooledTemporaryFile
from typing import BinaryIO

from app.domain.models import File

CHUNK_SIZE = 1024 * 1024
HASH_PREFIX = "sha256:"


class ArtifactFileHashMismatchError(RuntimeError):
    """file storage 实际内容与 revision 锁定 hash 不一致。"""


class VerifiedArtifactFile:
    """校验后的可复用文件流。"""

    def __init__(self, *, stream: BinaryIO, file: File, content_hash: str) -> None:
        self.stream = stream
        self.file = file
        self.content_hash = content_hash


def is_valid_sha256_content_hash(value: str | None) -> bool:
    normalized = str(value or "").strip()
    if not normalized.startswith(HASH_PREFIX):
        return False
    digest = normalized[len(HASH_PREFIX):]
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest.lower())


def normalize_sha256_content_hash(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if not is_valid_sha256_content_hash(normalized):
        return ""
    return normalized


def calculate_sha256_stream(stream: BinaryIO, *, spool_max_size: int = 8 * 1024 * 1024) -> tuple[str, BinaryIO]:
    """流式计算 sha256，并返回已回到起点的可重读流。"""
    digest = hashlib.sha256()
    buffered = SpooledTemporaryFile(max_size=spool_max_size, mode="w+b")
    while True:
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        buffered.write(chunk)
    buffered.seek(0)
    return f"{HASH_PREFIX}{digest.hexdigest()}", buffered


def verify_file_storage_stream(
        *,
        stream: BinaryIO,
        file: File,
        expected_content_hash: str,
        trusted_storage_hash: str | None = None,
) -> VerifiedArtifactFile:
    """统一校验 file storage 内容 hash，供下载、预览和最终附件复用。"""
    expected = normalize_sha256_content_hash(expected_content_hash)
    if not expected:
        raise ValueError("artifact_content_hash_invalid")
    trusted = normalize_sha256_content_hash(trusted_storage_hash)
    if trusted:
        if trusted != expected:
            raise ArtifactFileHashMismatchError("artifact_hash_changed")
        return VerifiedArtifactFile(stream=stream, file=file, content_hash=trusted)

    actual_hash, verified_stream = calculate_sha256_stream(stream)
    if actual_hash != expected:
        raise ArtifactFileHashMismatchError("artifact_hash_changed")
    return VerifiedArtifactFile(stream=verified_stream, file=file, content_hash=actual_hash)

