#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/13 14:31
@Author : caixiaorong01@outlook.com
@File   : logging.py
"""
import logging
import re
import sys
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.config import get_settings

BACKEND_ROOT_DIR = Path(__file__).resolve().parents[3]

_SENSITIVE_KEY_VALUE_PATTERN = re.compile(
    r'(?i)("?(?:password|old_password|new_password|confirm_password|access_token|refresh_token|token|cookie|set-cookie|authorization)"?\s*[:=]\s*)("[^"]*"|\'[^\']*\'|[^,\s;]+)'
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b")


def _mask_sensitive_text(text: str) -> str:
    masked = _BEARER_TOKEN_PATTERN.sub("Bearer ***", text)
    masked = _SENSITIVE_KEY_VALUE_PATTERN.sub(r"\1***", masked)
    return masked


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


class SensitiveDataMaskingFilter(logging.Filter):
    """统一脱敏日志中的敏感字段值。"""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        masked_message = _mask_sensitive_text(message)
        if masked_message != message:
            record.msg = masked_message
            record.args = ()
        return True


class SizeAndAgeRotatingFileHandler(RotatingFileHandler):
    """按大小切分，并在启动与滚动时清理超过保留天数的旧日志。"""

    def __init__(
            self,
            filename: str | Path,
            *,
            max_bytes: int,
            retention_days: int,
            backup_count: int = 1024,
            encoding: str = "utf-8",
    ) -> None:
        self._retention_days = max(int(retention_days), 0)
        super().__init__(
            filename=filename,
            maxBytes=max(int(max_bytes), 1),
            backupCount=max(int(backup_count), 1),
            encoding=encoding,
        )

    def cleanup_expired_files(self) -> int:
        log_file_path = Path(self.baseFilename)
        expire_before = datetime.now() - timedelta(days=self._retention_days)
        cleaned_count = 0
        pattern = f"{log_file_path.name}*"
        for candidate in log_file_path.parent.glob(pattern):
            if not candidate.is_file():
                continue
            try:
                modified_at = datetime.fromtimestamp(candidate.stat().st_mtime)
            except FileNotFoundError:
                continue
            if modified_at >= expire_before:
                continue
            candidate.unlink(missing_ok=True)
            cleaned_count += 1
        return cleaned_count

    def doRollover(self) -> None:
        super().doRollover()
        self.cleanup_expired_files()


def _build_log_formatter() -> logging.Formatter:
    return logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def _resolve_log_file_path() -> Path:
    settings = get_settings()
    log_dir = Path(settings.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = BACKEND_ROOT_DIR / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / settings.log_filename


def _build_console_handler(log_level: int) -> logging.Handler:
    settings = get_settings()
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(_build_log_formatter())
    console_handler.setLevel(log_level)
    console_handler.addFilter(SensitiveDataMaskingFilter())
    if not settings.is_log_output_all:
        console_handler.addFilter(ProjectLoggerOnlyFilter(settings.log_output_allowed_logger_prefixes))
    return console_handler


def _build_file_handler(log_level: int) -> SizeAndAgeRotatingFileHandler:
    settings = get_settings()
    log_file_path = _resolve_log_file_path()
    max_bytes = max(int(settings.log_file_max_mb), 1) * 1024 * 1024
    file_handler = SizeAndAgeRotatingFileHandler(
        filename=log_file_path,
        max_bytes=max_bytes,
        retention_days=settings.log_retention_days,
        backup_count=1024,
        encoding="utf-8",
    )
    file_handler.setFormatter(_build_log_formatter())
    file_handler.setLevel(log_level)
    file_handler.addFilter(SensitiveDataMaskingFilter())
    if not settings.is_log_output_all:
        file_handler.addFilter(ProjectLoggerOnlyFilter(settings.log_output_allowed_logger_prefixes))
    return file_handler


def setup_logging():
    settings = get_settings()

    root_logger = logging.getLogger()

    root_logger.handlers.clear()
    log_level = getattr(logging, settings.log_level)
    root_logger.setLevel(log_level)
    console_handler = _build_console_handler(log_level)
    file_handler = _build_file_handler(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    cleaned_count = file_handler.cleanup_expired_files()
    root_logger.info(
        "日志初始化完成。级别=%s 文件=%s 单文件上限=%sMB 保留天数=%s 清理旧文件数=%s",
        settings.log_level,
        str(Path(file_handler.baseFilename)),
        settings.log_file_max_mb,
        settings.log_retention_days,
        cleaned_count,
    )
