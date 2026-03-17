#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/13 14:52
@Author : caixiaorong01@outlook.com
@File   : status_routes.py
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, Response as FastAPIResponse

from app.application.service import StatusService
from app.application.errors import error_keys
from app.domain.models import HealthStatus
from app.interfaces.schemas import Response
from app.interfaces.dependencies.services import get_status_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/status", tags=["状态模块"])


@router.get(
    path="",
    response_model=Response[List[HealthStatus]],
    summary="系统健康监测",
    description="系统健康监测",
)
async def get_status(
        http_response: FastAPIResponse,
        status_service: StatusService = Depends(get_status_service),
) -> Response[List[HealthStatus]]:
    """系统健康监测"""
    status = await status_service.check_all()
    if any(item.status == 'ERROR' or item.status == 'error' for item in status):
        # 健康检查失败时同步返回真实HTTP状态码，避免网关/探针误判为200。
        http_response.status_code = 503
        return Response.fail(
            503,
            "系统服务存在异常",
            status,
            error_key=error_keys.STATUS_UNHEALTHY,
        )
    return Response.success(msg="系统服务正常", data=status)
