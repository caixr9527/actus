import asyncio

import pytest

from app.application.errors import NotFoundError, ServerError, error_keys
from app.application.service.app_config_service import AppConfigService
from app.domain.models import AppConfig, AgentConfig, MCPConfig
from app.domain.models.app_config import A2AConfig


class _FakeAppConfigRepository:
    def __init__(self, app_config: AppConfig) -> None:
        self._app_config = app_config

    def load(self) -> AppConfig:
        return self._app_config

    def save(self, app_config: AppConfig) -> None:
        self._app_config = app_config


def _build_service() -> AppConfigService:
    return AppConfigService(
        app_config_repository=_FakeAppConfigRepository(
            AppConfig(
                agent_config=AgentConfig(),
                mcp_config=MCPConfig(),
                a2a_config=A2AConfig(),
            )
        )
    )


def test_delete_mcp_server_should_raise_not_found_with_error_key() -> None:
    service = _build_service()

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.delete_mcp_server("missing-server"))

    assert exc.value.error_key == error_keys.APP_CONFIG_MCP_SERVER_NOT_FOUND
    assert exc.value.error_params == {"server_name": "missing-server"}


def test_delete_a2a_server_should_raise_not_found_with_error_key() -> None:
    service = _build_service()

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.delete_a2a_server("missing-a2a"))

    assert exc.value.error_key == error_keys.APP_CONFIG_A2A_SERVER_NOT_FOUND
    assert exc.value.error_params == {"a2a_id": "missing-a2a"}


def test_get_mcp_servers_should_map_domain_error_to_app_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _build_service()

    async def _raise_init_error(self) -> None:
        raise RuntimeError("mcp init broken")

    monkeypatch.setattr(
        "app.application.service.app_config_service.MCPClientManager.initialize",
        _raise_init_error,
    )

    with pytest.raises(ServerError) as exc:
        asyncio.run(service.get_mcp_servers())

    assert exc.value.error_key == error_keys.APP_CONFIG_MCP_SERVERS_LOAD_FAILED
    assert exc.value.error_params == {"reason": "mcp init broken"}


def test_get_a2a_servers_should_map_domain_error_to_app_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _build_service()

    async def _raise_init_error(self) -> None:
        raise RuntimeError("a2a init broken")

    monkeypatch.setattr(
        "app.application.service.app_config_service.A2AClientManager.initialize",
        _raise_init_error,
    )

    with pytest.raises(ServerError) as exc:
        asyncio.run(service.get_a2a_servers())

    assert exc.value.error_key == error_keys.APP_CONFIG_A2A_SERVERS_LOAD_FAILED
    assert exc.value.error_params == {"reason": "a2a init broken"}
