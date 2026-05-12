import asyncio

import httpx

from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    SandboxCapabilityProbePayload,
    SandboxCapabilityStatus,
)
from app.infrastructure.sandbox import docker_sandbox


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "http://127.0.0.1"),
                response=httpx.Response(self.status_code),
            )


class _FakeAsyncClient:
    def __init__(self, *, response: _FakeResponse | None = None, raises: Exception | None = None) -> None:
        self.response = response
        self.raises = raises
        self.last_get_url: str | None = None
        self.exec_command_called = False

    async def get(self, url: str, **kwargs) -> _FakeResponse:
        self.last_get_url = url
        if self.raises is not None:
            raise self.raises
        assert self.response is not None
        return self.response

    async def post(self, *args, **kwargs):
        self.exec_command_called = True
        raise AssertionError("probe_capabilities 不应调用 post 或 shell fallback")


def test_probe_capabilities_should_call_sandbox_profile_api_and_return_typed_payload() -> None:
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(
            status_code=200,
            payload={
                "code": 200,
                "msg": "ok",
                "data": {
                    "raw_profile": {
                        "health_status": "available",
                        "cwd": "/sandbox",
                        "capabilities": [],
                        "resource_limits": {
                            "max_file_read_bytes": 10000,
                            "max_command_seconds": 600,
                            "network_policy": "restricted",
                        },
                        "disabled_capabilities": [],
                        "confirmation_required_capabilities": [],
                    },
                    "reason_code": "",
                    "probe_status": "available",
                },
            },
        )
    )
    sandbox = docker_sandbox.DockerSandbox(ip="127.0.0.1", container_name="sandbox-1")
    sandbox.client = fake_client

    result = asyncio.run(sandbox.probe_capabilities())

    assert fake_client.last_get_url == "http://127.0.0.1:8081/api/capabilities/profile"
    assert result.success is True
    assert isinstance(result.data, SandboxCapabilityProbePayload)
    assert result.data.raw_profile["cwd"] == "/sandbox"
    assert fake_client.exec_command_called is False


def test_probe_capabilities_should_return_typed_unavailable_payload_for_missing_api() -> None:
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(
            status_code=404,
            payload={"code": 404, "msg": "not found", "data": {}},
        )
    )
    sandbox = docker_sandbox.DockerSandbox(ip="127.0.0.1", container_name="sandbox-1")
    sandbox.client = fake_client

    result = asyncio.run(sandbox.probe_capabilities())

    assert result.success is False
    assert isinstance(result.data, SandboxCapabilityProbePayload)
    assert result.data.reason_code == "sandbox_profile_probe_unavailable"
    assert result.data.probe_status == SandboxCapabilityStatus.UNKNOWN
    assert fake_client.exec_command_called is False


def test_probe_capabilities_should_return_typed_unavailable_payload_for_connection_error() -> None:
    fake_client = _FakeAsyncClient(raises=httpx.ConnectError("connection failed"))
    sandbox = docker_sandbox.DockerSandbox(ip="127.0.0.1", container_name="sandbox-1")
    sandbox.client = fake_client

    result = asyncio.run(sandbox.probe_capabilities())

    assert result.success is False
    assert isinstance(result.data, SandboxCapabilityProbePayload)
    assert result.data.reason_code == "sandbox_profile_probe_unavailable"
    assert fake_client.exec_command_called is False
