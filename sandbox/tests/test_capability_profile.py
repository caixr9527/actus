import os
import unittest
from contextlib import AbstractAsyncContextManager
from unittest.mock import patch

from app.interfaces.endpoints.capabilities import get_profile
from app.models import ProcessInfo, SandboxCapabilityProbePayload, SearXNGStatusResult
from app.models.capabilities import SandboxCapabilityItem
from app.services.capabilities import SandboxCapabilityProbeService


def _process(name: str, statename: str = "RUNNING") -> ProcessInfo:
    return ProcessInfo(
        name=name,
        group="services",
        description="pid 1, uptime 0:00:01",
        start=1,
        stop=0,
        now=2,
        state=20 if statename == "RUNNING" else 10,
        statename=statename,
        spawnerr="",
        exitstatus=0,
        logfile="/dev/stdout",
        stdout_logfile="/dev/stdout",
        stderr_logfile="/dev/stderr",
        pid=1,
    )


class _FakeSupervisorService:
    async def get_all_processes(self):
        return [
            _process("chrome"),
            _process("socat"),
            _process("x11vnc"),
            _process("websockify"),
        ]


class _FakeSearXNGService:
    async def get_status(self):
        return SearXNGStatusResult(
            base_url="http://127.0.0.1:8082",
            available=True,
            status_code=200,
            content_type="text/html; charset=utf-8",
        )


def _available_network_egress() -> tuple:
    return (
        SandboxCapabilityItem(
            kind="network",
            name="network_egress",
            status="available",
            details={
                "dns_resolved": True,
                "https_reachable": True,
                "probe_targets": ["connectivity_check"],
                "status_class": "2xx",
                "duration_ms": 1,
            },
        ),
        "",
    )


class CapabilityProfileTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_probe_profile_should_return_sandbox_raw_profile_without_backend_fields(self) -> None:
        service = SandboxCapabilityProbeService(
            supervisor_service=_FakeSupervisorService(),
            searxng_service=_FakeSearXNGService(),
        )

        with (
            patch("app.services.capabilities.shutil.which", return_value="/usr/bin/tool"),
            patch.object(SandboxCapabilityProbeService, "_probe_network_egress_capability",
                         return_value=_available_network_egress()),
        ):
            payload = await service.probe_profile()

        raw_profile = payload.raw_profile
        self.assertIn(raw_profile["health_status"], {"available", "degraded"})
        self.assertTrue(raw_profile["cwd"])
        self.assertEqual(
            set(raw_profile.keys()),
            {
                "health_status",
                "cwd",
                "capabilities",
                "resource_limits",
                "disabled_capabilities",
                "confirmation_required_capabilities",
            },
        )
        forbidden_keys = {
            "user_id",
            "session_id",
            "workspace_id",
            "run_id",
            "profile_id",
            "profile_hash",
            "prompt_summary",
            "generated_at",
            "expires_at",
            "refresh_reason",
        }
        self.assertTrue(forbidden_keys.isdisjoint(raw_profile.keys()))
        capability = raw_profile["capabilities"][0]
        self.assertEqual(
            set(capability.keys()),
            {
                "kind",
                "name",
                "status",
                "version",
                "path",
                "details",
                "reason_code",
                "requires_confirmation",
                "disabled",
            },
        )
        capability_names = {item["name"] for item in raw_profile["capabilities"]}
        self.assertTrue({"python3", "pip", "node", "npm", "git", "curl", "wget"}.issubset(capability_names))
        self.assertIn("chromium", capability_names)
        self.assertIn("x11vnc", capability_names)
        self.assertIn("searxng", capability_names)
        self.assertIn("network_policy", capability_names)
        self.assertIn("proxy_configuration", capability_names)
        self.assertIn("network_egress", capability_names)

    async def test_network_capabilities_should_not_mark_missing_proxy_as_unavailable(self) -> None:
        service = SandboxCapabilityProbeService(
            supervisor_service=_FakeSupervisorService(),
            searxng_service=_FakeSearXNGService(),
        )
        env_patch = {
            "HTTPS_PROXY": "",
            "HTTP_PROXY": "",
            "ALL_PROXY": "",
            "https_proxy": "",
            "http_proxy": "",
            "all_proxy": "",
        }

        with patch.dict(os.environ, env_patch, clear=False):
            resource_limits = service._build_resource_limits()
            capabilities = SandboxCapabilityProbeService._build_network_capabilities(resource_limits)

        capabilities = {item.name: item for item in capabilities}
        self.assertEqual(resource_limits.network_policy, "restricted")
        self.assertEqual(capabilities["proxy_configuration"].status, "available")
        self.assertFalse(capabilities["proxy_configuration"].details["configured"])
        self.assertEqual(capabilities["proxy_configuration"].reason_code, "")
        self.assertEqual(
            SandboxCapabilityProbeService._resolve_health_status(list(capabilities.values())),
            "available",
        )

    async def test_network_egress_probe_should_return_degraded_when_dns_fails(self) -> None:
        service = SandboxCapabilityProbeService(
            supervisor_service=_FakeSupervisorService(),
            searxng_service=_FakeSearXNGService(),
        )

        with patch.object(SandboxCapabilityProbeService, "_probe_dns_resolution", return_value=False):
            item, reason_code = await service._probe_network_egress_capability()

        self.assertEqual(item.name, "network_egress")
        self.assertEqual(item.status, "degraded")
        self.assertEqual(reason_code, "network_dns_probe_failed")
        self.assertFalse(item.details["dns_resolved"])

    async def test_network_egress_probe_should_not_expose_probe_url_or_response_body(self) -> None:
        service = SandboxCapabilityProbeService(
            supervisor_service=_FakeSupervisorService(),
            searxng_service=_FakeSearXNGService(),
        )

        class _FakeResponse:
            status_code = 204

        class _FakeAsyncClient(AbstractAsyncContextManager):
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def head(self, url: str):
                return _FakeResponse()

        with (
            patch.object(SandboxCapabilityProbeService, "_probe_dns_resolution", return_value=True),
            patch("app.services.capabilities.httpx.AsyncClient", _FakeAsyncClient),
        ):
            item, reason_code = await service._probe_network_egress_capability()

        serialized = str(item.model_dump(mode="json"))
        self.assertEqual(item.status, "available")
        self.assertEqual(reason_code, "")
        self.assertIn("connectivity_check", item.details["probe_targets"])
        self.assertNotIn("https://example.com", serialized)
        self.assertNotIn("secret body", serialized)

    async def test_probe_profile_should_return_resource_limits_and_sanitized_proxy(self) -> None:
        service = SandboxCapabilityProbeService(
            supervisor_service=_FakeSupervisorService(),
            searxng_service=_FakeSearXNGService(),
        )
        env_patch = {
            "HTTPS_PROXY": "http://user:password@proxy.internal:8080/token-secret",
            "HTTP_PROXY": "",
            "ALL_PROXY": "",
            "NO_PROXY": "",
        }

        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch.object(SandboxCapabilityProbeService, "_probe_network_egress_capability",
                         return_value=_available_network_egress()),
        ):
            payload = await service.probe_profile()

        resource_limits = payload.raw_profile["resource_limits"]
        self.assertEqual(resource_limits["max_file_read_bytes"], 10000)
        self.assertEqual(resource_limits["max_command_seconds"], 600)
        self.assertNotIn("max_tool_iterations", resource_limits)
        self.assertIn(resource_limits["network_policy"], {"proxy_configured", "restricted", "unknown"})
        self.assertNotIn("open", resource_limits["network_policy"])
        self.assertNotIn("unrestricted", resource_limits["network_policy"])
        serialized = str(resource_limits)
        self.assertNotIn("password", serialized)
        self.assertNotIn("token-secret", serialized)
        self.assertNotIn("proxy.internal:8080", serialized)
        for directory in resource_limits["readable_dirs"]:
            self.assertIn(directory, {"/workspace", "/sandbox", "/tmp", "/home/ubuntu"})
            self.assertTrue(os.path.isdir(directory))
            self.assertTrue(os.access(directory, os.R_OK))
        for directory in resource_limits["writable_dirs"]:
            self.assertIn(directory, {"/workspace", "/sandbox", "/tmp", "/home/ubuntu"})
            self.assertTrue(os.path.isdir(directory))
            self.assertTrue(os.access(directory, os.W_OK))

    async def test_capabilities_endpoint_should_return_probe_payload(self) -> None:
        fake_service = unittest.mock.AsyncMock()
        fake_service.probe_profile.return_value = SandboxCapabilityProbePayload(
            raw_profile={
                "health_status": "available",
                "cwd": "/sandbox",
                "capabilities": [],
                "resource_limits": {},
                "disabled_capabilities": [],
                "confirmation_required_capabilities": [],
            },
            reason_code="",
            probe_status="available",
        )

        response = await get_profile(capability_probe_service=fake_service)

        self.assertEqual(response.code, 200)
        self.assertEqual(response.data.raw_profile["cwd"], "/sandbox")
        fake_service.probe_profile.assert_awaited_once()
