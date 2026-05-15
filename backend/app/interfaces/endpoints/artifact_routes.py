#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Artifact revision 下载与预览路由。"""

import urllib.parse

from fastapi import APIRouter, Depends, Query
from starlette.responses import StreamingResponse

from app.application.service.artifact_delivery_service import ArtifactDeliveryService
from app.domain.models import User
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.services import get_artifact_delivery_service

router = APIRouter(prefix="/sessions", tags=["Artifact 产物模块"])


@router.get(
    path="/{session_id}/artifacts/{artifact_id}/revisions/{revision_id}/download",
    summary="Artifact revision 下载接口",
)
async def download_artifact_revision(
        session_id: str,
        artifact_id: str,
        revision_id: str,
        content_hash: str = Query(...),
        run_id: str | None = Query(default=None),
        source_run_id: str | None = Query(default=None),
        current_user: User = Depends(get_current_user),
        delivery_service: ArtifactDeliveryService = Depends(get_artifact_delivery_service),
) -> StreamingResponse:
    result = await delivery_service.download_revision(
        user_id=current_user.id,
        session_id=session_id,
        artifact_id=artifact_id,
        revision_id=revision_id,
        content_hash=content_hash,
        run_id=run_id,
        source_run_id=source_run_id,
    )
    return _to_streaming_response(result)


@router.get(
    path="/{session_id}/artifacts/{artifact_id}/revisions/{revision_id}/preview",
    summary="Artifact revision 预览接口",
)
async def preview_artifact_revision(
        session_id: str,
        artifact_id: str,
        revision_id: str,
        content_hash: str = Query(...),
        run_id: str | None = Query(default=None),
        source_run_id: str | None = Query(default=None),
        current_user: User = Depends(get_current_user),
        delivery_service: ArtifactDeliveryService = Depends(get_artifact_delivery_service),
) -> StreamingResponse:
    result = await delivery_service.preview_revision(
        user_id=current_user.id,
        session_id=session_id,
        artifact_id=artifact_id,
        revision_id=revision_id,
        content_hash=content_hash,
        run_id=run_id,
        source_run_id=source_run_id,
    )
    return _to_streaming_response(result)


def _to_streaming_response(result) -> StreamingResponse:
    filename = result.file.filename or result.revision.path.rsplit("/", 1)[-1] or result.revision.revision_id
    encoded_filename = urllib.parse.quote(filename)
    return StreamingResponse(
        content=result.stream,
        media_type=result.file.mime_type or result.revision.mime_type or "application/octet-stream",
        headers={
            "Content-Disposition": f"{result.disposition}; filename*=utf-8''{encoded_filename}",
            "Content-Length": str(result.file.size or result.revision.size_bytes or 0),
        },
    )

