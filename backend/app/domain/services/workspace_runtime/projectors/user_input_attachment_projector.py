#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用户输入附件投影。"""

from __future__ import annotations

import logging
from typing import Callable

from app.domain.external import FileStorage, Sandbox
from app.domain.models import File, MessageEvent
from app.domain.repositories import IUnitOfWork

logger = logging.getLogger(__name__)


class UserInputAttachmentProjector:
    """处理用户输入附件从存储到沙箱的同步。"""

    def __init__(
            self,
            *,
            session_id: str,
            user_id: str,
            sandbox: Sandbox,
            file_storage: FileStorage,
            uow_factory: Callable[[], IUnitOfWork],
    ) -> None:
        self._session_id = session_id
        self._user_id = str(user_id or "").strip()
        if not self._user_id:
            raise ValueError("UserInputAttachmentProjector 必须提供 user_id")
        self._sandbox = sandbox
        self._file_storage = file_storage
        self._uow_factory = uow_factory

    @staticmethod
    def _build_sandbox_file_path(*, file_id: str, filename: str) -> str:
        normalized_file_id = str(file_id or "").strip()
        normalized_filename = str(filename or "").strip() or normalized_file_id or "attachment"
        return f"/home/ubuntu/upload/{normalized_file_id}/{normalized_filename}"

    async def sync_file_to_sandbox(self, *, file_id: str) -> File:
        try:
            file_data, file = await self._file_storage.download_file(
                file_id=file_id,
                user_id=self._user_id,
            )
            resolved_file_id = str(file.id or file_id).strip() or str(file_id or "").strip()
            file_path = self._build_sandbox_file_path(
                file_id=resolved_file_id,
                filename=file.filename,
            )
            tool_result = await self._sandbox.upload_file(
                file_data=file_data,
                file_path=file_path,
                filename=file.filename,
            )
            if not tool_result.success:
                raise RuntimeError(
                    f"同步文件[{file_id}]到沙箱失败: {tool_result.message or 'upload_file returned unsuccessful result'}"
                )

            file.filepath = file_path
            async with self._uow_factory() as uow:
                await uow.file.save(file=file)
                await uow.session.add_file(session_id=self._session_id, file=file)
            return file
        except Exception as e:
            logger.exception(f"同步文件[{file_id}]到沙箱失败: {e}")
            raise

    async def project(self, event: MessageEvent) -> None:
        if not event.attachments:
            return

        attachments: list[File] = []
        try:
            for attachment in event.attachments:
                file_id = str(attachment.id or "").strip()
                if not file_id:
                    raise RuntimeError("消息附件缺少 file_id，无法同步到沙箱")
                attachments.append(await self.sync_file_to_sandbox(file_id=file_id))
            event.attachments = attachments
        except Exception as e:
            logger.exception(f"同步消息附件到沙箱失败: {e}")
            raise
