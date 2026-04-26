#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具事件投影。"""

from __future__ import annotations

import logging
from typing import Optional

from app.domain.external import Browser, FileStorage
from app.domain.models import ToolEvent
from app.domain.services.tools import ToolRuntimeAdapter, ToolRuntimeEventHooks
from .browser_screenshot_artifact_service import BrowserScreenshotArtifactService
from ..service import WorkspaceRuntimeService

logger = logging.getLogger(__name__)


class ToolEventProjector:
    """负责 tool event 的展示型副作用与富化。"""

    def __init__(
            self,
            *,
            adapter: ToolRuntimeAdapter,
            browser: Browser,
            file_storage: FileStorage,
            workspace_runtime_service: WorkspaceRuntimeService,
            user_id: Optional[str],
    ) -> None:
        self._adapter = adapter
        self._workspace_runtime_service = workspace_runtime_service
        self._browser_screenshot_service = BrowserScreenshotArtifactService(
            browser=browser,
            file_storage=file_storage,
            workspace_runtime_service=workspace_runtime_service,
            user_id=user_id,
        )

    async def project(self, event: ToolEvent) -> None:
        try:
            hooks = ToolRuntimeEventHooks(
                get_browser_screenshot=lambda: self._browser_screenshot_service.capture(
                    source_capability=event.function_name,
                ),
                get_shell_tool_result=self._workspace_runtime_service.get_latest_shell_tool_result,
            )
            await self._adapter.enrich_tool_event(event=event, hooks=hooks)
        except Exception as e:
            logger.exception(f"处理工具事件失败: {e}")
