#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/18 14:32
@Author : caixiaorong01@outlook.com
@File   : llm_model_config.py
"""
from datetime import datetime
from typing import Any

from sqlalchemy import (
    String,
    Integer,
    Boolean,
    DateTime,
    Index,
    PrimaryKeyConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import LLMModelConfig
from .base import Base


class LLMModelConfigModel(Base):
    """LLM 模型配置 ORM 模型"""

    __tablename__ = "llm_model_configs"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_llm_model_configs_id"),
        Index("ix_llm_model_configs_enabled", "enabled"),
        Index("ix_llm_model_configs_sort_order", "sort_order"),
    )

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
    )
    provider: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    base_url: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )
    api_key: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    sort_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @staticmethod
    def _build_orm_payload(model_config: LLMModelConfig) -> dict[str, Any]:
        """将领域模型转换为 ORM 可写入的数据结构。"""
        return model_config.model_dump(mode="python")

    @classmethod
    def from_domain(cls, model_config: LLMModelConfig) -> "LLMModelConfigModel":
        """从领域模型创建 ORM 模型。"""
        return cls(**cls._build_orm_payload(model_config))

    def to_domain(self) -> LLMModelConfig:
        """将 ORM 模型转换为领域模型。"""
        return LLMModelConfig.model_validate(self, from_attributes=True)

    def update_from_domain(self, model_config: LLMModelConfig) -> None:
        """根据领域模型更新 ORM 模型。"""
        for field, value in self._build_orm_payload(model_config).items():
            setattr(self, field, value)
