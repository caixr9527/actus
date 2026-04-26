#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace projector 共享辅助函数。"""

from __future__ import annotations

import io
from typing import BinaryIO


def get_stream_size(stream: BinaryIO | io.BytesIO) -> int:
    """读取当前文件流大小，并在结束后恢复指针位置。"""
    current_pos = stream.tell()
    stream.seek(0, io.SEEK_END)
    size = stream.tell()
    stream.seek(current_pos)
    return size
