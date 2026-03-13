#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/12 16:53
@Author : caixiaorong01@outlook.com
@File   : config.py
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    env: str = "development"
    log_level: str = "INFO"
    # 日志输出配置（单配置项）：
    # - all：输出全部日志（项目 + 系统/框架）
    # - 其他值：按逗号分隔解析为 logger 前缀白名单，仅输出命中的日志
    #   示例：app,core,__main__
    #   示例：app,sqlalchemy.engine.Engine
    log_output_mode: str = "all"
    app_config_filepath: str = "config.yaml"
    auto_run_db_migrations: bool = True
    db_migration_log_interval_seconds: float = 3.0
    cors_allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    cors_allow_credentials: bool = False

    sqlalchemy_database_uri: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/actus"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    auth_jwt_secret: str = "change-this-to-a-secure-random-secret"
    auth_jwt_algorithm: str = "HS256"
    auth_access_token_expires_in: int = 30 * 60
    auth_refresh_token_expires_in: int = 7 * 24 * 60 * 60
    auth_register_code_expires_in: int = 5 * 60
    auth_register_code_length: int = 6

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_password: str = ""
    smtp_from_email: str = ""
    smtp_use_tls: bool = True

    cos_region: str = "ap-guangzhou"
    cos_secret_id: str = ""
    cos_secret_key: str = ""
    cos_scheme: str = "https"
    cos_bucket: str = ""
    cos_domain: str = ""

    sandbox_address: Optional[str] = None
    sandbox_image: Optional[str] = None
    sandbox_name_prefix: Optional[str] = None
    sandbox_ttl_minutes: Optional[int] = 60
    sandbox_network: Optional[str] = None
    sandbox_chrome_args: Optional[str] = ""
    sandbox_https_proxy: Optional[str] = None
    sandbox_http_proxy: Optional[str] = None
    sandbox_no_proxy: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_log_output_all(self) -> bool:
        """判断是否为 all 模式"""
        return self.log_output_mode.strip().lower() == "all"

    @property
    def is_production_env(self) -> bool:
        """判断是否为生产环境"""
        return self.env.strip().lower() in {"production", "prod"}

    @property
    def auth_register_verification_enabled(self) -> bool:
        """注册是否强制邮箱验证码（仅生产环境开启）"""
        return self.is_production_env

    @property
    def log_output_allowed_logger_prefixes(self) -> tuple[str, ...]:
        """解析白名单模式下可输出日志的 logger 前缀列表"""
        if self.is_log_output_all:
            return ()
        prefixes = tuple(
            item.strip()
            for item in self.log_output_mode.split(",")
            if item.strip()
        )
        # 若配置为空串，回退到常用项目前缀，避免误配置导致日志完全静默。
        return prefixes or ("app", "core", "__main__")

    def is_logger_allowed_by_output_mode(self, logger_name: str) -> bool:
        """判断指定 logger 在当前输出配置下是否允许输出"""
        if self.is_log_output_all:
            return True
        for prefix in self.log_output_allowed_logger_prefixes:
            if logger_name == prefix or logger_name.startswith(f"{prefix}."):
                return True
        return False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
