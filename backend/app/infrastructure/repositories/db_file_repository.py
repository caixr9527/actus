#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/28 12:19
@Author : caixiaorong01@outlook.com
@File   : db_file_repository.py
"""
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import File
from app.domain.repositories import FileRepository
from app.infrastructure.models import FileModel


class DBFileRepository(FileRepository):
    """基于数据库的文件数据仓库"""

    def __init__(self, db_session: AsyncSession) -> None:
        """构造函数，完成数据仓库初始化"""
        self.db_session = db_session

    async def save(self, file: File, user_id: str | None = None) -> None:
        """根据传递的文件模型存储or更新数据"""
        # 查询数据库中是否存在指定ID的文件记录
        stmt = select(FileModel).where(FileModel.id == file.id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        # 如果记录不存在，则创建新的文件模型并添加到数据库
        if not record:
            record = FileModel.from_domain(file)
            if user_id is not None:
                record.user_id = user_id
            self.db_session.add(record)
            return

        # 如果记录存在，则使用传入的文件对象更新现有记录
        record.update_from_domain(file)

    async def get_by_id(self, file_id: str) -> Optional[File]:
        """根据传递的文件id获取文件信息"""
        # 构建查询语句，根据文件ID查找对应的文件记录
        stmt = select(FileModel).where(FileModel.id == file_id)
        # 执行查询操作
        result = await self.db_session.execute(stmt)
        # 获取查询结果，如果不存在则返回None
        record = result.scalar_one_or_none()

        # 如果找到了记录，则将其转换为领域模型并返回；否则返回None
        return record.to_domain() if record is not None else None

    async def get_by_id_and_user_id(self, file_id: str, user_id: str) -> Optional[File]:
        """根据文件id和所属用户查询文件信息"""
        stmt = select(FileModel).where(
            FileModel.id == file_id,
            FileModel.user_id == user_id,
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None
