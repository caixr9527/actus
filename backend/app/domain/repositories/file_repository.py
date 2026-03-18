#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/28 12:16
@Author : caixiaorong01@outlook.com
@File   : file_repository.py
"""
from typing import Protocol, Optional

from app.domain.models.file import File


class FileRepository(Protocol):
    """文件模型数据仓库"""

    async def save(self, file: File, user_id: str | None = None) -> None:
        """新增或更新文件信息"""
        ...

    async def get_by_id(self, file_id: str) -> Optional[File]:
        """根据传递的文件id获取文件信息"""
        ...

    async def get_by_id_and_user_id(self, file_id: str, user_id: str) -> Optional[File]:
        """根据文件id和用户id获取文件信息"""
        ...
