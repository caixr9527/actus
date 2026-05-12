#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox 受控能力探测服务。"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import socket
import time
from urllib.parse import urlparse

import httpx

from app.models import ProcessInfo
from app.models.capabilities import (
    SandboxCapabilityItem,
    SandboxCapabilityProbePayload,
    SandboxResourceLimits,
)
from app.services import SearXNGService, SupervisorService

COMMAND_PROBE_TIMEOUT_SECONDS = 2
SUPERVISOR_PROBE_TIMEOUT_SECONDS = 3
SEARCH_PROBE_TIMEOUT_SECONDS = 3
NETWORK_EGRESS_PROBE_TIMEOUT_SECONDS = 3
DIRECTORY_CANDIDATES = ("/tmp", "/home/ubuntu")
NETWORK_EGRESS_PROBE_TARGETS = (
    ("connectivity_check", "https://example.com/"),
)
NETWORK_DNS_PROBE_HOSTS = ("example.com",)
COMMAND_PROBES: dict[str, tuple[str, ...]] = {
    "python3": ("python3", "--version"),
    "pip": ("pip", "--version"),
    "node": ("node", "--version"),
    "npm": ("npm", "--version"),
    "git": ("git", "--version"),
    "curl": ("curl", "--version"),
    "wget": ("wget", "--version"),
}
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
FORBIDDEN_RAW_PROFILE_KEYS = {
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


class SandboxCapabilityProbeService:
    """执行 sandbox 内固定白名单能力探测。"""

    def __init__(
            self,
            *,
            supervisor_service: SupervisorService,
            searxng_service: SearXNGService,
    ) -> None:
        self._supervisor_service = supervisor_service
        self._searxng_service = searxng_service

    async def probe_profile(self) -> SandboxCapabilityProbePayload:
        capabilities: list[SandboxCapabilityItem] = []
        reason_codes: list[str] = []

        capabilities.extend(self._probe_os_capabilities())
        capabilities.extend(self._probe_directory_capabilities())
        command_items, command_reason_codes = await self._probe_command_capabilities()
        capabilities.extend(command_items)
        reason_codes.extend(command_reason_codes)

        supervisor_items, supervisor_reason_codes = await self._probe_supervisor_capabilities()
        capabilities.extend(supervisor_items)
        reason_codes.extend(supervisor_reason_codes)

        search_item, search_reason_code = await self._probe_search_capability()
        capabilities.append(search_item)
        if search_reason_code:
            reason_codes.append(search_reason_code)

        resource_limits = self._build_resource_limits()
        capabilities.extend(self._build_network_capabilities(resource_limits))
        network_egress_item, network_egress_reason_code = await self._probe_network_egress_capability()
        capabilities.append(network_egress_item)
        if network_egress_reason_code:
            reason_codes.append(network_egress_reason_code)
        raw_profile = {
            "health_status": self._resolve_health_status(capabilities),
            "cwd": self._safe_getcwd(),
            "capabilities": [item.model_dump(mode="json") for item in capabilities],
            "resource_limits": resource_limits.model_dump(mode="json"),
            "disabled_capabilities": [],
            "confirmation_required_capabilities": [],
        }
        for key in FORBIDDEN_RAW_PROFILE_KEYS:
            raw_profile.pop(key, None)

        return SandboxCapabilityProbePayload(
            raw_profile=raw_profile,
            reason_code=";".join(sorted(set(reason_codes))),
            probe_status=raw_profile["health_status"],
        )

    @staticmethod
    def _probe_os_capabilities() -> list[SandboxCapabilityItem]:
        return [
            SandboxCapabilityItem(
                kind="os",
                name="platform",
                status="available",
                version=platform.platform(),
                details={
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "python_runtime": platform.python_version(),
                },
            ),
            SandboxCapabilityItem(
                kind="shell",
                name="default_shell",
                status="available" if os.path.exists("/bin/bash") else "unknown",
                path="/bin/bash" if os.path.exists("/bin/bash") else "",
                reason_code="" if os.path.exists("/bin/bash") else "default_shell_unknown",
            ),
        ]

    @staticmethod
    def _probe_directory_capabilities() -> list[SandboxCapabilityItem]:
        capabilities: list[SandboxCapabilityItem] = []
        for directory in DIRECTORY_CANDIDATES:
            exists = os.path.isdir(directory)
            readable = exists and os.access(directory, os.R_OK)
            writable = exists and os.access(directory, os.W_OK)
            status = "available" if exists and (readable or writable) else "unavailable"
            capabilities.append(
                SandboxCapabilityItem(
                    kind="filesystem",
                    name=directory,
                    status=status,
                    path=directory,
                    details={
                        "readable": readable,
                        "writable": writable,
                    },
                    reason_code="" if status == "available" else "directory_unavailable",
                )
            )
        return capabilities

    async def _probe_command_capabilities(self) -> tuple[list[SandboxCapabilityItem], list[str]]:
        tasks = [
            self._probe_command(name=name, command=command)
            for name, command in COMMAND_PROBES.items()
        ]
        results = await asyncio.gather(*tasks)
        capabilities = [item for item, _ in results]
        reason_codes = [reason_code for _, reason_code in results if reason_code]
        return capabilities, reason_codes

    async def _probe_command(self, *, name: str, command: tuple[str, ...]) -> tuple[SandboxCapabilityItem, str]:
        executable_path = shutil.which(command[0]) or ""
        kind = self._resolve_command_kind(name)
        if not executable_path:
            reason_code = f"{name}_not_found"
            return (
                SandboxCapabilityItem(
                    kind=kind,
                    name=name,
                    status="unavailable",
                    reason_code=reason_code,
                ),
                reason_code,
            )

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=COMMAND_PROBE_TIMEOUT_SECONDS,
            )
            output = stdout.decode("utf-8", errors="replace").strip()
            version = self._extract_version(output)
            if process.returncode == 0:
                return (
                    SandboxCapabilityItem(
                        kind=kind,
                        name=name,
                        status="available",
                        version=version,
                        path=executable_path,
                    ),
                    "",
                )
            reason_code = f"{name}_version_probe_failed"
            return (
                SandboxCapabilityItem(
                    kind=kind,
                    name=name,
                    status="degraded",
                    version=version,
                    path=executable_path,
                    reason_code=reason_code,
                ),
                reason_code,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            reason_code = f"{name}_version_probe_timeout"
            return (
                SandboxCapabilityItem(
                    kind=kind,
                    name=name,
                    status="degraded",
                    path=executable_path,
                    reason_code=reason_code,
                ),
                reason_code,
            )
        except Exception:
            reason_code = f"{name}_version_probe_error"
            return (
                SandboxCapabilityItem(
                    kind=kind,
                    name=name,
                    status="degraded",
                    path=executable_path,
                    reason_code=reason_code,
                ),
                reason_code,
            )

    async def _probe_supervisor_capabilities(self) -> tuple[list[SandboxCapabilityItem], list[str]]:
        try:
            processes = await asyncio.wait_for(
                self._supervisor_service.get_all_processes(),
                timeout=SUPERVISOR_PROBE_TIMEOUT_SECONDS,
            )
        except Exception:
            reason_code = "supervisor_status_probe_failed"
            return (
                [
                    SandboxCapabilityItem(
                        kind="browser",
                        name="browser",
                        status="unknown",
                        reason_code=reason_code,
                    ),
                    SandboxCapabilityItem(
                        kind="vnc",
                        name="vnc",
                        status="unknown",
                        reason_code=reason_code,
                    ),
                ],
                [reason_code],
            )

        process_by_name = {str(process.name): process for process in processes}
        capabilities = [
            self._build_process_capability(
                kind="browser",
                name="chromium",
                process=process_by_name.get("chrome"),
            ),
            self._build_process_capability(
                kind="browser",
                name="cdp",
                process=process_by_name.get("socat"),
            ),
            self._build_process_capability(
                kind="vnc",
                name="x11vnc",
                process=process_by_name.get("x11vnc"),
            ),
            self._build_process_capability(
                kind="vnc",
                name="websockify",
                process=process_by_name.get("websockify"),
            ),
        ]
        return capabilities, [item.reason_code for item in capabilities if item.reason_code]

    async def _probe_search_capability(self) -> tuple[SandboxCapabilityItem, str]:
        try:
            status = await asyncio.wait_for(
                self._searxng_service.get_status(),
                timeout=SEARCH_PROBE_TIMEOUT_SECONDS,
            )
        except Exception:
            reason_code = "search_status_probe_failed"
            return (
                SandboxCapabilityItem(
                    kind="search",
                    name="searxng",
                    status="unknown",
                    reason_code=reason_code,
                ),
                reason_code,
            )

        if status.available:
            return (
                SandboxCapabilityItem(
                    kind="search",
                    name="searxng",
                    status="available",
                    details={
                        "status_code": status.status_code,
                        "content_type": status.content_type or "",
                    },
                ),
                "",
            )
        reason_code = "search_unavailable"
        return (
            SandboxCapabilityItem(
                kind="search",
                name="searxng",
                status="unavailable",
                details={
                    "status_code": status.status_code,
                    "content_type": status.content_type or "",
                },
                reason_code=reason_code,
            ),
            reason_code,
        )

    def _build_resource_limits(self) -> SandboxResourceLimits:
        readable_dirs = [
            directory
            for directory in DIRECTORY_CANDIDATES
            if os.path.isdir(directory) and os.access(directory, os.R_OK)
        ]
        writable_dirs = [
            directory
            for directory in DIRECTORY_CANDIDATES
            if os.path.isdir(directory) and os.access(directory, os.W_OK)
        ]
        network_policy, proxy_host_categories = self._resolve_network_policy()
        return SandboxResourceLimits(
            max_file_read_bytes=10000,
            max_command_seconds=600,
            readable_dirs=readable_dirs,
            writable_dirs=writable_dirs,
            network_policy=network_policy,
            proxy_host_categories=proxy_host_categories,
        )

    @staticmethod
    def _build_network_capabilities(resource_limits: SandboxResourceLimits) -> list[SandboxCapabilityItem]:
        network_policy = resource_limits.network_policy
        proxy_categories = list(resource_limits.proxy_host_categories)
        proxy_configured = network_policy == "proxy_configured"
        network_status = "unknown" if network_policy == "unknown" else "available"
        return [
            SandboxCapabilityItem(
                kind="network",
                name="network_policy",
                status=network_status,
                details={
                    "network_policy": network_policy,
                },
                reason_code="network_policy_unknown" if network_policy == "unknown" else "",
            ),
            SandboxCapabilityItem(
                kind="proxy",
                name="proxy_configuration",
                status="available",
                details={
                    "configured": proxy_configured,
                    "host_categories": proxy_categories,
                },
                reason_code="",
            ),
        ]

    async def _probe_network_egress_capability(self) -> tuple[SandboxCapabilityItem, str]:
        started_at = time.monotonic()
        try:
            dns_resolved = await self._probe_dns_resolution()
            if not dns_resolved:
                reason_code = "network_dns_probe_failed"
                return (
                    SandboxCapabilityItem(
                        kind="network",
                        name="network_egress",
                        status="degraded",
                        details={
                            "dns_resolved": False,
                            "https_reachable": False,
                            "probe_targets": [target for target, _ in NETWORK_EGRESS_PROBE_TARGETS],
                            "duration_ms": self._elapsed_ms(started_at),
                        },
                        reason_code=reason_code,
                    ),
                    reason_code,
                )

            timeout = httpx.Timeout(NETWORK_EGRESS_PROBE_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=timeout,
                    trust_env=True,
            ) as client:
                for target_name, target_url in NETWORK_EGRESS_PROBE_TARGETS:
                    try:
                        response = await client.head(target_url)
                    except httpx.HTTPError:
                        continue
                    if response.status_code < 500:
                        return (
                            SandboxCapabilityItem(
                                kind="network",
                                name="network_egress",
                                status="available",
                                details={
                                    "dns_resolved": True,
                                    "https_reachable": True,
                                    "probe_targets": [target_name],
                                    "status_class": f"{response.status_code // 100}xx",
                                    "duration_ms": self._elapsed_ms(started_at),
                                },
                            ),
                            "",
                        )

            reason_code = "network_https_egress_unavailable"
            return (
                SandboxCapabilityItem(
                    kind="network",
                    name="network_egress",
                    status="degraded",
                    details={
                        "dns_resolved": True,
                        "https_reachable": False,
                        "probe_targets": [target for target, _ in NETWORK_EGRESS_PROBE_TARGETS],
                        "duration_ms": self._elapsed_ms(started_at),
                    },
                    reason_code=reason_code,
                ),
                reason_code,
            )
        except Exception:
            reason_code = "network_egress_probe_error"
            return (
                SandboxCapabilityItem(
                    kind="network",
                    name="network_egress",
                    status="unknown",
                    details={
                        "dns_resolved": False,
                        "https_reachable": False,
                        "probe_targets": [target for target, _ in NETWORK_EGRESS_PROBE_TARGETS],
                        "duration_ms": self._elapsed_ms(started_at),
                    },
                    reason_code=reason_code,
                ),
                reason_code,
            )

    @staticmethod
    async def _probe_dns_resolution() -> bool:
        for host in NETWORK_DNS_PROBE_HOSTS:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(socket.getaddrinfo, host, 443, type=socket.SOCK_STREAM),
                    timeout=NETWORK_EGRESS_PROBE_TIMEOUT_SECONDS,
                )
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, int((time.monotonic() - started_at) * 1000))

    @staticmethod
    def _build_process_capability(
            *,
            kind: str,
            name: str,
            process: ProcessInfo | None,
    ) -> SandboxCapabilityItem:
        if process is None:
            return SandboxCapabilityItem(
                kind=kind,
                name=name,
                status="unavailable",
                reason_code=f"{name}_process_missing",
            )
        is_running = process.statename == "RUNNING"
        return SandboxCapabilityItem(
            kind=kind,
            name=name,
            status="available" if is_running else "degraded",
            details={
                "process_state": process.statename,
            },
            reason_code="" if is_running else f"{name}_process_not_running",
        )

    @staticmethod
    def _resolve_command_kind(name: str) -> str:
        if name in {"python", "python3", "pip"}:
            return "python"
        if name in {"node", "npm"}:
            return "node"
        return "shell"

    @staticmethod
    def _extract_version(output: str) -> str:
        normalized_output = str(output or "").splitlines()[0].strip()
        if not normalized_output:
            return ""
        match = re.search(r"(\d+(?:\.\d+){1,3})", normalized_output)
        if match:
            return match.group(1)
        return normalized_output[:80]

    @staticmethod
    def _safe_getcwd() -> str:
        try:
            return os.getcwd()
        except Exception:
            return ""

    @staticmethod
    def _resolve_health_status(capabilities: list[SandboxCapabilityItem]) -> str:
        if not capabilities:
            return "unknown"
        unavailable_count = len([item for item in capabilities if item.status == "unavailable"])
        degraded_count = len([item for item in capabilities if item.status in {"degraded", "unknown"}])
        if unavailable_count == 0 and degraded_count == 0:
            return "available"
        if unavailable_count == len(capabilities):
            return "unavailable"
        return "degraded"

    @staticmethod
    def _resolve_network_policy() -> tuple[str, list[str]]:
        try:
            proxy_values = [
                os.getenv(key) or os.getenv(key.lower()) or ""
                for key in PROXY_ENV_KEYS
            ]
            proxy_categories = sorted({
                SandboxCapabilityProbeService._classify_proxy_host(value)
                for value in proxy_values
                if str(value or "").strip()
            })
            proxy_categories = [category for category in proxy_categories if category]
            if proxy_categories:
                return "proxy_configured", proxy_categories
            return "restricted", []
        except Exception:
            return "unknown", []

    @staticmethod
    def _classify_proxy_host(proxy_value: str) -> str:
        normalized_value = str(proxy_value or "").strip()
        if not normalized_value:
            return ""
        parsed = urlparse(normalized_value if "://" in normalized_value else f"http://{normalized_value}")
        hostname = str(parsed.hostname or "").lower()
        if not hostname:
            return "unknown_host"
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            return "loopback"
        if hostname.endswith(".local") or hostname.endswith(".internal"):
            return "internal_host"
        if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[0-1])\.)", hostname):
            return "private_network"
        return "external_host"
