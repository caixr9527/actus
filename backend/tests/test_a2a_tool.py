import asyncio

from app.domain.models.app_config import A2AConfig, A2AServerConfig
from app.domain.services.tools.a2a import A2AClientManager


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeHttpClient:
    async def get(self, url: str):
        if "ok.example" in url:
            return _FakeResponse({"url": "https://ok.example/rpc", "name": "ok-agent"})
        raise RuntimeError("service unavailable")

    async def post(self, url: str, json):
        return _FakeResponse({"ok": True, "url": url, "jsonrpc": json.get("jsonrpc")})


def test_a2a_get_agent_cards_keeps_available_servers() -> None:
    manager = A2AClientManager(
        A2AConfig(
            a2a_servers=[
                A2AServerConfig(id="s1", base_url="https://ok.example", enabled=True),
                A2AServerConfig(id="s2", base_url="https://bad.example", enabled=True),
            ]
        )
    )
    manager._httpx_client = _FakeHttpClient()

    asyncio.run(manager._get_a2a_agent_cards())

    assert "s1" in manager.agent_cards
    assert "s2" not in manager.agent_cards
    assert manager.agent_cards["s1"]["enabled"] is True


def test_a2a_invoke_uses_initialized_httpx_client() -> None:
    manager = A2AClientManager(A2AConfig(a2a_servers=[]))
    manager._httpx_client = _FakeHttpClient()
    manager._agent_cards = {
        "s1": {
            "url": "https://ok.example/rpc",
        }
    }

    result = asyncio.run(manager.invoke(agent_id="s1", query="hello"))

    assert result.success is True
    assert result.data is not None
