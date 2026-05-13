#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""浏览器临时文件领域契约。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BrowserScreenshotCaptureResult(BaseModel):
    """浏览器截图临时文件结果，不表示 artifact 已登记。"""

    model_config = ConfigDict(extra="forbid")

    url: str = ""
    file_id: str
    filename: str
    filepath: str | None = None
    key: str | None = None
    mime_type: str
    size: int | None = None
