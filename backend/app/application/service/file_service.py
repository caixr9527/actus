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
from app.application.errors import error_keys
from app.domain.external import FileStorage, FileUploadPayload
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

    async def upload_file(self, user_id: str, upload_file: UploadFile) -> File:
        """将传递的文件上传到腾讯云cos并记录上传数据"""
        return await self.file_storage.upload_file(
            upload_file=FileUploadPayload(
                filename=upload_file.filename or "",
                file=upload_file.file,
                content_type=upload_file.content_type or "",
                size=upload_file.size or 0,
            ),
            user_id=user_id,
        )

    async def get_file_info(self, user_id: str, file_id: str) -> File:
        """根据传递的文件id获取文件信息"""
        # 查询使用短生命周期UoW，避免服务级别缓存事务对象。
        async with self._uow_factory() as uow:
            file = await uow.file.get_by_id_and_user_id(file_id=file_id, user_id=user_id)
        if not file:
            raise NotFoundError(
                msg=f"该文件[{file_id}]不存在",
                error_key=error_keys.FILE_NOT_FOUND,
                error_params={"file_id": file_id},
            )
        return file

    async def download_file(self, user_id: str, file_id: str) -> Tuple[BinaryIO, File]:
        """根据传递的文件id下载文件"""
        return await self.file_storage.download_file(file_id=file_id, user_id=user_id)
