#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/17 10:47
@Author : caixiaorong01@outlook.com
@File   : file_app_config_repository.py
"""
import logging
from pathlib import Path
from typing import Optional

import yaml
from filelock import FileLock

from app.application.errors.exceptions import ServerError
from app.application.errors import error_keys
from app.domain.models import AppConfig, LLMConfig, AgentConfig, MCPConfig
from app.domain.models.app_config import A2AConfig
from app.domain.repositories import AppConfigRepository

logger = logging.getLogger(__name__)


class FileAppConfigRepository(AppConfigRepository):
    """文件应用配置仓库"""

    def __init__(self, config_path: str) -> None:
        # 获取当前目录
        root_dir = Path.cwd()
        # 拼接路径
        self._config_path = root_dir.joinpath(root_dir, config_path)
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self._config_path.with_suffix(".lock")

    def _create_default_app_config_if_not_exists(self) -> None:
        """创建文件"""
        if not self._config_path.exists():
            default_app_config = AppConfig(
                llm_config=LLMConfig(),
                agent_config=AgentConfig(),
                mcp_config=MCPConfig(),
                a2a_config=A2AConfig(),
            )
            self.save(default_app_config)

    def load(self) -> Optional[AppConfig]:
        """加载应用配置"""
        self._create_default_app_config_if_not_exists()

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return AppConfig.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"读取应用配置文件失败: {e}")
            raise ServerError(
                "读取应用配置文件失败，请稍后尝试",
                error_key=error_keys.APP_CONFIG_LOAD_FAILED,
                error_params={"config_path": str(self._config_path)},
            )

    def save(self, app_config: AppConfig) -> None:
        """保存应用配置"""
        lock = FileLock(self._lock_file, timeout=5)
        try:
            with lock:
                data_to_dum = app_config.model_dump(mode="json")

                with open(self._config_path, "w", encoding="utf-8") as f:
                    yaml.dump(data_to_dum, f, allow_unicode=True, sort_keys=False)
        except TimeoutError:
            logger.error("获取应用配置文件锁失败")
            raise ServerError(
                "保存应用配置文件失败，请稍后尝试",
                error_key=error_keys.APP_CONFIG_SAVE_FAILED,
                error_params={"config_path": str(self._config_path)},
            )
