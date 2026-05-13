#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""浏览器截图临时文件服务。"""

from __future__ import annotations

import io
import uuid
from typing import Optional

from app.domain.external import Browser, FileStorage, FileUploadPayload
from app.domain.services.runtime.contracts.browser_artifact_contract import BrowserScreenshotCaptureResult
from .helpers import get_stream_size


class BrowserScreenshotArtifactService:
    """负责截图并上传为临时文件结果，revision 登记由后续 projector 处理。"""

    def __init__(
            self,
            *,
            browser: Browser,
            file_storage: FileStorage,
            user_id: Optional[str],
    ) -> None:
        self._browser = browser
        self._file_storage = file_storage
        self._user_id = user_id

    async def capture(self, *, source_capability: str = "browser_screenshot") -> BrowserScreenshotCaptureResult:
        screenshot = await self._browser.screenshot()
        screenshot_stream = io.BytesIO(screenshot)
        filename = f"{str(uuid.uuid4())}.png"
        size = get_stream_size(screenshot_stream)
        uploaded_file = await self._file_storage.upload_file(
            upload_file=FileUploadPayload(
                file=screenshot_stream,
                filename=filename,
                content_type="image/png",
                size=size,
            ),
            user_id=self._user_id,
        )
        screenshot_url = self._file_storage.get_file_url(uploaded_file)
        return BrowserScreenshotCaptureResult(
            url=screenshot_url,
            file_id=uploaded_file.id,
            filename=uploaded_file.filename or filename,
            filepath=uploaded_file.filepath,
            key=uploaded_file.key,
            mime_type=uploaded_file.mime_type or "image/png",
            size=uploaded_file.size or size,
        )
