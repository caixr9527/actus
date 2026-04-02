from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from app.infrastructure.logging.logging import (
    SizeAndAgeRotatingFileHandler,
    setup_logging,
)


def test_size_and_age_rotating_file_handler_should_cleanup_expired_files(tmp_path: Path) -> None:
    log_file = tmp_path / "backend.log"
    log_file.write_text("current", encoding="utf-8")
    expired_backup = tmp_path / "backend.log.1"
    expired_backup.write_text("expired", encoding="utf-8")
    fresh_backup = tmp_path / "backend.log.2"
    fresh_backup.write_text("fresh", encoding="utf-8")

    expired_timestamp = (datetime.now() - timedelta(days=5)).timestamp()
    fresh_timestamp = (datetime.now() - timedelta(days=1)).timestamp()
    expired_backup.touch()
    fresh_backup.touch()
    log_file.touch()
    Path(expired_backup).chmod(0o644)
    Path(fresh_backup).chmod(0o644)
    import os
    os.utime(expired_backup, (expired_timestamp, expired_timestamp))
    os.utime(fresh_backup, (fresh_timestamp, fresh_timestamp))

    handler = SizeAndAgeRotatingFileHandler(
        filename=log_file,
        max_bytes=1024,
        retention_days=3,
    )

    cleaned_count = handler.cleanup_expired_files()
    handler.close()

    assert cleaned_count == 1
    assert expired_backup.exists() is False
    assert fresh_backup.exists() is True
    assert log_file.exists() is True


def test_setup_logging_should_attach_file_handler(tmp_path: Path, monkeypatch) -> None:
    fake_settings = SimpleNamespace(
        log_level="INFO",
        log_dir=str(tmp_path),
        log_filename="backend.log",
        log_file_max_mb=100,
        log_retention_days=3,
        is_log_output_all=True,
        log_output_allowed_logger_prefixes=(),
    )
    monkeypatch.setattr(
        "app.infrastructure.logging.logging.get_settings",
        lambda: fake_settings,
    )

    setup_logging()

    import logging

    root_logger = logging.getLogger()
    file_handlers = [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, SizeAndAgeRotatingFileHandler)
    ]

    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename) == tmp_path / "backend.log"
    assert file_handlers[0].maxBytes == 100 * 1024 * 1024
