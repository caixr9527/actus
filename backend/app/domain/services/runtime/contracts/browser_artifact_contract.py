#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""浏览器 artifact 领域契约。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BrowserScreenshotArtifactRef(BaseModel):
    """浏览器截图 artifact 的结构化引用。"""

    model_config = ConfigDict(extra="forbid")

    url: str = ""
    artifact_id: str
    artifact_path: str
