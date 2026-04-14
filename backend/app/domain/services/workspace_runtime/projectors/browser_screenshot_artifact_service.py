#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""浏览器截图 artifact 服务。"""

from __future__ import annotations

import io
import uuid
from typing import Optional

from app.domain.external import Browser, FileStorage, FileUploadPayload
from .helpers import get_stream_size
from ..service import WorkspaceRuntimeService


class BrowserScreenshotArtifactService:
    """负责把浏览器截图落为 workspace artifact。"""

    def __init__(
            self,
            *,
            browser: Browser,
            file_storage: FileStorage,
            workspace_runtime_service: WorkspaceRuntimeService,
            user_id: Optional[str],
    ) -> None:
        self._browser = browser
        self._file_storage = file_storage
        self._workspace_runtime_service = workspace_runtime_service
        self._user_id = user_id

    @staticmethod
    def _build_artifact_path(*, file_id: str, key: str, filepath: str, filename: str) -> str:
        preferred_path = str(key or filepath or filename or "").strip().lstrip("/")
        if not preferred_path:
            preferred_path = f"{str(file_id or '').strip() or str(uuid.uuid4())}.png"
        return f"/.workspace/browser-screenshots/{preferred_path}"

    async def capture(self, *, source_capability: str = "browser_screenshot") -> str:
        screenshot = await self._browser.screenshot()
        screenshot_stream = io.BytesIO(screenshot)
        filename = f"{str(uuid.uuid4())}.png"
        uploaded_file = await self._file_storage.upload_file(
            upload_file=FileUploadPayload(
                file=screenshot_stream,
                filename=filename,
                content_type="image/png",
                size=get_stream_size(screenshot_stream),
            ),
            user_id=self._user_id,
        )
        screenshot_url = self._file_storage.get_file_url(uploaded_file)
        artifact_path = self._build_artifact_path(
            file_id=uploaded_file.id,
            key=uploaded_file.key,
            filepath=uploaded_file.filepath,
            filename=uploaded_file.filename or filename,
        )
        await self._workspace_runtime_service.upsert_artifact(
            path=artifact_path,
            artifact_type="browser_screenshot",
            summary=f"浏览器截图: {artifact_path}",
            source_capability=str(source_capability or "browser_screenshot").strip() or "browser_screenshot",
            metadata={
                "file_id": uploaded_file.id,
                "filename": uploaded_file.filename or filename,
                "filepath": uploaded_file.filepath,
                "key": uploaded_file.key,
                "mime_type": uploaded_file.mime_type or "image/png",
                "size": uploaded_file.size or get_stream_size(screenshot_stream),
                "url": screenshot_url,
            },
            record_as_changed_file=False,
        )
        return screenshot_url
