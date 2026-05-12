#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox 路径访问策略。"""

from __future__ import annotations

import posixpath
import re


_URL_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")


def normalize_sandbox_path(path: str) -> str:
    """归一化 sandbox 内部路径，拒绝越界或外部资源路径。"""
    normalized = str(path or "").strip().replace("\\", "/")
    if not normalized:
        raise ValueError("sandbox path 不能为空")
    if _URL_SCHEME_PATTERN.match(normalized):
        raise ValueError("sandbox path 不能包含 URL scheme")
    if _WINDOWS_DRIVE_PATTERN.match(normalized):
        raise ValueError("sandbox path 不能包含 Windows drive 前缀")

    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if ".." in parts:
        raise ValueError("sandbox path 不能包含路径穿越段")

    if normalized.startswith("/"):
        normalized_path = posixpath.normpath("/" + "/".join(parts))
        return normalized_path if normalized_path != "." else "/"
    normalized_path = posixpath.normpath("/".join(parts))
    if normalized_path == ".":
        return ""
    return posixpath.normpath(f"/workspace/{normalized_path}")


def is_allowed_sandbox_read_path(path: str) -> bool:
    """判断路径是否允许作为 sandbox 文件读取目标。"""
    try:
        normalized = normalize_sandbox_path(path)
    except ValueError:
        return False
    if not normalized:
        return False
    if normalized == "/workspace" or normalized.startswith("/workspace/"):
        return True
    if normalized == "/tmp" or normalized.startswith("/tmp/"):
        return True
    return False
