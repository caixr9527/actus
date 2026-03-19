#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/28 12:13
@Author : caixiaorong01@outlook.com
@File   : file_storage.py
"""
from dataclasses import dataclass
from typing import Protocol, Tuple, BinaryIO

from app.domain.models import File


@dataclass(slots=True)
class FileUploadPayload:
    """文件上传载荷，避免 domain 层依赖 Web 框架对象"""
    filename: str
    file: BinaryIO
    content_type: str = ""
    size: int = 0


class FileStorage(Protocol):
    """文件存储桶协议"""

    async def upload_file(self, upload_file: FileUploadPayload, user_id: str | None = None) -> File:
        """根据传递的文件源上传文件后返回文件信息"""
        ...

    async def download_file(self, file_id: str, user_id: str | None = None) -> Tuple[BinaryIO, File]:
        """根据传递的文件id下载文件，并返回文件源+文件信息"""
        ...

    def get_file_url(self, file: File) -> str:
        """根据文件信息生成可访问URL"""
        ...
