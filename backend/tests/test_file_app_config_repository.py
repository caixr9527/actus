from pathlib import Path

import pytest

from app.application.errors import ServerError, error_keys
from app.infrastructure.repositories import file_app_config_repository
from app.infrastructure.repositories.file_app_config_repository import FileAppConfigRepository


def test_load_should_raise_server_error_with_error_key_when_open_failed(monkeypatch, tmp_path: Path) -> None:
    repo = FileAppConfigRepository("tmp-config.yaml")
    repo._config_path = tmp_path / "app-config.yaml"
    repo._lock_file = repo._config_path.with_suffix(".lock")
    monkeypatch.setattr(repo, "_create_default_app_config_if_not_exists", lambda: None)

    def _raise_open_error(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(file_app_config_repository, "open", _raise_open_error, raising=False)

    with pytest.raises(ServerError) as exc:
        repo.load()

    assert exc.value.error_key == error_keys.APP_CONFIG_LOAD_FAILED
    assert exc.value.error_params == {"config_path": str(repo._config_path)}


def test_save_should_raise_server_error_with_error_key_when_lock_timeout(monkeypatch, tmp_path: Path) -> None:
    repo = FileAppConfigRepository("tmp-config.yaml")
    repo._config_path = tmp_path / "app-config.yaml"
    repo._lock_file = repo._config_path.with_suffix(".lock")

    class _BrokenLock:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            raise TimeoutError

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(file_app_config_repository, "FileLock", _BrokenLock)

    with pytest.raises(ServerError) as exc:
        repo.save(file_app_config_repository.AppConfig(
            llm_config=file_app_config_repository.LLMConfig(),
            agent_config=file_app_config_repository.AgentConfig(),
            mcp_config=file_app_config_repository.MCPConfig(),
            a2a_config=file_app_config_repository.A2AConfig(),
        ))

    assert exc.value.error_key == error_keys.APP_CONFIG_SAVE_FAILED
    assert exc.value.error_params == {"config_path": str(repo._config_path)}
