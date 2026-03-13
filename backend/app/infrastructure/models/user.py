#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 17:03
@Author : caixiaorong01@outlook.com
@File   : user.py
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String,
    DateTime,
    text,
    PrimaryKeyConstraint,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import User, UserStatus
from .base import Base


class UserModel(Base):
    """用户ORM模型"""
    __tablename__ = "users"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_users_id"),
        UniqueConstraint("email", name="uq_users_email"),
    )

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )  # 用户id（UUID字符串）
    email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )  # 邮箱（唯一）
    password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )  # 密码哈希
    password_salt: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )  # 加盐值
    auth_provider: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        server_default=text("'email'::character varying"),
    )  # 认证方式，预留SSO扩展
    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )  # SSO外部ID
    status: Mapped[UserStatus] = mapped_column(
        String(255),
        nullable=False,
        server_default=text("'active'::character varying"),
    )  # 用户状态
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )  # 创建时间
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )  # 更新时间
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
    )  # 最近登录时间
    last_login_ip: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )  # 最近登录IP

    @classmethod
    def from_domain(cls, user: User) -> "UserModel":
        """从领域模型创建ORM模型"""
        return cls(**user.model_dump(mode="json"))

    def to_domain(self) -> User:
        """将ORM模型转换为领域模型"""
        return User.model_validate(self, from_attributes=True)

    def update_from_domain(self, user: User) -> None:
        """根据领域模型更新ORM模型"""
        user_data = user.model_dump(mode="json")
        for field, value in user_data.items():
            setattr(self, field, value)
