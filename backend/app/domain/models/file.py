#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/19 20:22
@Author : caixiaorong01@outlook.com
@File   : file.py
"""
import uuid
from typing import Optional

from pydantic import BaseModel, Field


class File(BaseModel):
    """文件模型,记录上传的文件"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # 文件id
    user_id: Optional[str] = None  # 所属用户id
    filename: str = ""  # 文件名字
    filepath: str = ""  # 文件路径
    key: str = ""  # 腾讯云cos中的路径
    extension: str = ""  # 扩展名
    mime_type: str = ""  # mime-type类型
    size: int = 0  # 文件大小，单位为字节
