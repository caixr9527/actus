#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具事件投影。"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from typing import Optional

from app.domain.external import Browser, FileStorage, FileUploadPayload, Sandbox
from app.domain.models import ToolEvent
from app.domain.services.tools import ToolRuntimeAdapter, ToolRuntimeEventHooks
from .browser_screenshot_artifact_service import BrowserScreenshotArtifactService
from .helpers import get_stream_size
from ..service import WorkspaceRuntimeService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SyncedStorageFile:
    """受控 sandbox 文件同步结果，hash 来自最终文件 bytes。"""

    id: str
    filename: str
    filepath: str
    key: str
    mime_type: str
    size: int
    content_hash: str
    storage_hash: str


class ToolEventProjector:
    """负责 tool event 的展示型副作用与富化。"""

    def __init__(
            self,
            *,
            adapter: ToolRuntimeAdapter,
            browser: Browser,
            file_storage: FileStorage,
            sandbox: Sandbox | None = None,
            workspace_runtime_service: WorkspaceRuntimeService,
            user_id: Optional[str],
    ) -> None:
        self._adapter = adapter
        self._sandbox = sandbox
        self._file_storage = file_storage
        self._user_id = user_id
        self._workspace_runtime_service = workspace_runtime_service
        self._browser_screenshot_service = BrowserScreenshotArtifactService(
            browser=browser,
            file_storage=file_storage,
            user_id=user_id,
        )

    async def project(self, event: ToolEvent) -> None:
        try:
            hooks = ToolRuntimeEventHooks(
                get_browser_screenshot=lambda: self._browser_screenshot_service.capture(
                    source_capability=event.function_name,
                ),
                get_shell_tool_result=self._workspace_runtime_service.get_latest_shell_tool_result,
                sync_file_to_storage=self.sync_file_to_storage,
            )
            await self._adapter.enrich_tool_event(event=event, hooks=hooks)
        except Exception as e:
            logger.exception(f"处理工具事件失败: {e}")

    async def sync_file_to_storage(self, filepath: str) -> SyncedStorageFile | None:
        """同步 sandbox 最终文件到 file storage，并返回最终 bytes 的可信 hash。"""
        if self._sandbox is None:
            return None
        normalized_path = str(filepath or "").strip()
        if not normalized_path:
            return None
        file_data = await self._sandbox.download_file(file_path=normalized_path)
        file_bytes = _read_all_bytes(file_data)
        content_hash = "sha256:" + hashlib.sha256(file_bytes).hexdigest()
        upload_stream = io.BytesIO(file_bytes)
        filename = normalized_path.split("/")[-1] or "artifact"
        uploaded_file = await self._file_storage.upload_file(
            upload_file=FileUploadPayload(
                file=upload_stream,
                filename=filename,
                size=get_stream_size(upload_stream),
            ),
            user_id=self._user_id,
        )
        return SyncedStorageFile(
            id=str(uploaded_file.id or ""),
            filename=str(uploaded_file.filename or filename),
            filepath=normalized_path,
            key=str(uploaded_file.key or ""),
            mime_type=str(uploaded_file.mime_type or ""),
            size=int(uploaded_file.size or len(file_bytes)),
            content_hash=content_hash,
            storage_hash=content_hash,
        )


def _read_all_bytes(file_data) -> bytes:
    if isinstance(file_data, bytes):
        return file_data
    if isinstance(file_data, bytearray):
        return bytes(file_data)
    file_data.seek(0)
    content = file_data.read()
    file_data.seek(0)
    return bytes(content)
