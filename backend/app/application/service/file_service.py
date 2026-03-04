#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/28 12:38
@Author : caixiaorong01@outlook.com
@File   : file_service.py
"""
from typing import Tuple, BinaryIO, Callable

from fastapi import UploadFile

from app.application.errors import NotFoundError
from app.domain.external import FileStorage
from app.domain.models import File
from app.domain.repositories import IUnitOfWork


class FileService:
    """Actus文件系统服务"""

    def __init__(
            self,
            uow_factory: Callable[[], IUnitOfWork],
            file_storage: FileStorage,
    ) -> None:
        """构造函数，完成文件服务的初始化"""
        self.file_storage = file_storage
        self._uow_factory = uow_factory
        self._uow = uow_factory()

    async def upload_file(self, upload_file: UploadFile) -> File:
        """将传递的文件上传到腾讯云cos并记录上传数据"""
        return await self.file_storage.upload_file(upload_file=upload_file)

    async def get_file_info(self, file_id: str) -> File:
        """根据传递的文件id获取文件信息"""
        async with self._uow:
            file = await self._uow.file.get_by_id(file_id)
        if not file:
            raise NotFoundError(f"该文件[{file_id}]不存在")
        return file

    async def download_file(self, file_id: str) -> Tuple[BinaryIO, File]:
        """根据传递的文件id下载文件"""
        return await self.file_storage.download_file(file_id)
