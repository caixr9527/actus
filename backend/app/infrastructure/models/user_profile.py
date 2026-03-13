#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 17:03
@Author : caixiaorong01@outlook.com
@File   : user_profile.py
"""
from typing import Optional

from sqlalchemy import (
    String,
    PrimaryKeyConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import UserProfile
from .base import Base


class UserProfileModel(Base):
    """用户资料ORM模型"""
    __tablename__ = "user_profiles"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", name="pk_user_profiles_user_id"),
    )

    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
    )  # 用户id（逻辑关联users.id，不使用数据库外键）
    nickname: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )  # 昵称
    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )  # 头像URL
    timezone: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text("'Asia/Shanghai'::character varying"),
    )  # 时区
    locale: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'zh-CN'::character varying"),
    )  # 语言地区

    @classmethod
    def from_domain(cls, profile: UserProfile) -> "UserProfileModel":
        """从领域模型创建ORM模型"""
        return cls(**profile.model_dump(mode="json"))

    def to_domain(self) -> UserProfile:
        """将ORM模型转换为领域模型"""
        return UserProfile.model_validate(self, from_attributes=True)

    def update_from_domain(self, profile: UserProfile) -> None:
        """根据领域模型更新ORM模型"""
        profile_data = profile.model_dump(mode="json")
        for field, value in profile_data.items():
            setattr(self, field, value)
