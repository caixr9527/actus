#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/13 14:31
@Author : caixiaorong01@outlook.com
@File   : logging.py
"""
import logging
import sys

from core.config import get_settings


class ProjectLoggerOnlyFilter(logging.Filter):
    """仅放行指定 logger 前缀的日志记录。"""

    def __init__(self, allowed_prefixes: tuple[str, ...]):
        super().__init__()
        self._allowed_prefixes = allowed_prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        logger_name = record.name
        for prefix in self._allowed_prefixes:
            if logger_name == prefix or logger_name.startswith(f"{prefix}."):
                return True
        return False


def setup_logging():
    settings = get_settings()

    root_logger = logging.getLogger()

    root_logger.handlers.clear()
    log_level = getattr(logging, settings.log_level)
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    if not settings.is_log_output_all:
        # 非 all 模式时，按 LOG_OUTPUT_MODE 配置的前缀白名单过滤日志输出。
        console_handler.addFilter(ProjectLoggerOnlyFilter(settings.log_output_allowed_logger_prefixes))

    root_logger.addHandler(console_handler)

    root_logger.info("Logging setup complete.")
