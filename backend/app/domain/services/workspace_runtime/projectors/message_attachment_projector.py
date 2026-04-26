#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""消息附件投影。"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from app.domain.external import FileStorage, FileUploadPayload, Sandbox
from app.domain.models import File, MessageEvent
from app.domain.repositories import IUnitOfWork
from .helpers import get_stream_size
from ..service import WorkspaceRuntimeService

logger = logging.getLogger(__name__)


class MessageAttachmentProjector:
    """处理消息附件从沙箱到存储的落账。"""

    def __init__(
            self,
            *,
            session_id: str,
            user_id: Optional[str],
            sandbox: Sandbox,
            file_storage: FileStorage,
            uow_factory: Callable[[], IUnitOfWork],
            workspace_runtime_service: WorkspaceRuntimeService,
    ) -> None:
        self._session_id = session_id
        self._user_id = user_id
        self._sandbox = sandbox
        self._file_storage = file_storage
        self._uow_factory = uow_factory
        self._workspace_runtime_service = workspace_runtime_service

    @staticmethod
    def _parse_sandbox_file_exists_payload(payload: Any) -> Optional[bool]:
        if isinstance(payload, bool):
            return payload
        if isinstance(payload, dict):
            for key in ("exists", "is_exists", "file_exists", "found"):
                if key in payload:
                    return bool(payload[key])
        if payload is None:
            return None
        return bool(payload)

    async def _check_sandbox_file_exists(self, filepath: str) -> bool:
        try:
            result = await self._sandbox.check_file_exists(file_path=filepath)
        except Exception as e:
            logger.warning(f"检查沙箱文件[{filepath}]是否存在失败，跳过附件同步: {e}")
            return False

        if not result.success:
            logger.info(f"沙箱文件[{filepath}]不存在或不可访问，跳过附件同步: {result.message}")
            return False

        parsed = self._parse_sandbox_file_exists_payload(result.data)
        if parsed is None:
            return True
        return parsed

    async def sync_file_to_storage(
            self,
            filepath: str,
            stage: Optional[str] = "intermediate",
    ) -> Optional[File]:
        try:
            normalized_path = str(filepath or "").strip()
            if not normalized_path:
                logger.info("接收到空附件路径，跳过附件同步")
                return None

            async with self._uow_factory() as uow:
                old_file = await uow.session.get_file_by_path(
                    session_id=self._session_id,
                    filepath=normalized_path,
                )
                if old_file is not None and stage != "final":
                    return old_file

            sandbox_file_exists = await self._check_sandbox_file_exists(filepath=normalized_path)
            if not sandbox_file_exists:
                if old_file is not None and stage == "final":
                    async with self._uow_factory() as uow:
                        await uow.session.add_final_files(session_id=self._session_id, file=old_file)
                    return old_file
                return None

            file_data = await self._sandbox.download_file(file_path=normalized_path)
            filename = normalized_path.split("/")[-1]
            upload_file = FileUploadPayload(
                file=file_data,
                filename=filename,
                size=get_stream_size(file_data),
            )
            new_file = await self._file_storage.upload_file(
                upload_file=upload_file,
                user_id=self._user_id,
            )
            new_file.filepath = normalized_path

            async with self._uow_factory() as uow:
                if old_file:
                    await uow.session.remove_file(session_id=self._session_id, file_id=old_file.id)
                await uow.session.add_file(session_id=self._session_id, file=new_file)
                if stage == "final":
                    await uow.session.add_final_files(session_id=self._session_id, file=new_file)
            return new_file
        except Exception as e:
            logger.exception(f"同步文件到存储失败: {e}")
            raise

    async def project(self, event: MessageEvent) -> None:
        attachments: list[File] = []
        delivered_paths: list[str] = []

        try:
            if not event.attachments:
                return

            allowed_final_paths: Optional[set[str]] = None
            if event.stage == "final":
                authoritative_paths = await self._workspace_runtime_service.resolve_authoritative_artifact_paths(
                    paths=[
                        str(attachment.filepath or "").strip()
                        for attachment in event.attachments
                    ],
                )
                allowed_final_paths = set(authoritative_paths)

            for attachment in event.attachments:
                filepath = str(attachment.filepath or "").strip()
                if not filepath:
                    continue
                if event.stage == "final" and allowed_final_paths is not None and filepath not in allowed_final_paths:
                    logger.info("最终附件[%s]未进入 workspace artifact 索引，跳过最终投影", filepath)
                    continue
                file = await self.sync_file_to_storage(filepath, event.stage)
                if file is not None:
                    attachments.append(file)
                    delivered_paths.append(filepath)

            event.attachments = attachments
            if event.stage == "final" and delivered_paths:
                await self._workspace_runtime_service.mark_artifacts_delivery_state(
                    paths=delivered_paths,
                    delivery_state="final_delivered",
                )
        except Exception as e:
            logger.exception(f"同步消息附件到存储失败: {e}")
            raise
