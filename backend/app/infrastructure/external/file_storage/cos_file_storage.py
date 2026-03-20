#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/28 12:15
@Author : caixiaorong01@outlook.com
@File   : cos_file_storage.py.py
"""
import logging
import os.path
import uuid
from datetime import datetime
from typing import Tuple, BinaryIO, Callable

from starlette.concurrency import run_in_threadpool

from app.domain.external import FileStorage, FileUploadPayload
from app.domain.models import File
from app.domain.repositories import IUnitOfWork
from app.infrastructure.storage import Cos

logger = logging.getLogger(__name__)


class CosFileStorage(FileStorage):
    """基于COS的文件存储扩展"""

    def __init__(
            self,
            bucket: str,
            public_base_url: str,
            cos: Cos,
            uow_factory: Callable[[], IUnitOfWork],
    ) -> None:
        """构造函数，完成cos文件存储桶扩展初始化"""
        self.bucket = bucket
        self.public_base_url = public_base_url.rstrip("/")
        self.cos = cos
        self._uow_factory = uow_factory

    async def upload_file(self, upload_file: FileUploadPayload, user_id: str | None = None) -> File:
        """根据传递的文件源将文件上传到腾讯云cos"""
        try:
            # 生成唯一文件ID
            file_id = str(uuid.uuid4())

            # 获取文件扩展名
            _, file_extension = os.path.splitext(upload_file.filename)
            if not file_extension:
                file_extension = ""

            # 创建按日期组织的路径
            date_path = datetime.now().strftime("%Y/%m/%d")
            cos_key = f"{date_path}/{file_id}{file_extension}"

            # 将文件上传到腾讯云COS
            await run_in_threadpool(
                self.cos.client.put_object,
                Bucket=self.bucket,
                Body=upload_file.file,
                Key=cos_key,
                EnableMD5=False,
            )
            logger.info(f"文件上传成功: {upload_file.filename} (ID: {file_id})")

            # 创建文件对象并保存到数据库
            file = File(
                id=file_id,
                filename=upload_file.filename,
                key=cos_key,
                extension=file_extension,
                mime_type=upload_file.content_type or "",
                size=upload_file.size,
            )
            # 文件元数据持久化使用独立UoW，确保每次调用都有独立事务边界。
            async with self._uow_factory() as uow:
                await uow.file.save(file, user_id=user_id)

            return file
        except Exception as e:
            logger.error(f"上传文件[{upload_file.filename}]失败: {str(e)}")
            raise

    async def download_file(self, file_id: str, user_id: str | None = None) -> Tuple[BinaryIO, File]:
        """根据文件id查询数据并下载文件"""
        try:
            # 根据文件ID从数据库获取文件记录
            async with self._uow_factory() as uow:
                if user_id is None:
                    file = await uow.file.get_by_id(file_id)
                else:
                    file = await uow.file.get_by_id_and_user_id(file_id=file_id, user_id=user_id)
            if not file:
                raise ValueError(f"该文件不存在, 文件id: {file_id}")

            # 从腾讯云COS下载文件内容
            response = await run_in_threadpool(
                self.cos.client.get_object,
                Bucket=self.bucket,
                Key=file.key,
                KeySimplifyCheck=True
            )

            # 返回文件流和文件对象
            return response["Body"], file
        except Exception as e:
            logger.error(f"下载文件[{file_id}]失败: {str(e)}")
            raise

    def get_file_url(self, file: File) -> str:
        """构造文件外链URL，屏蔽上层对COS域名规则的感知"""
        return f"{self.public_base_url}/{file.key.lstrip('/')}"
