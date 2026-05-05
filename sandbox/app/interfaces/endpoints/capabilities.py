#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox capability profile 探测接口。"""

from fastapi import APIRouter, Depends

from app.interfaces.schemas import Response
from app.interfaces.service_dependencies import get_capability_probe_service
from app.models import SandboxCapabilityProbePayload
from app.services import SandboxCapabilityProbeService

router = APIRouter(prefix="/capabilities", tags=["Capabilities模块"])


@router.get(
    path="/profile",
    response_model=Response[SandboxCapabilityProbePayload],
    summary="获取Sandbox能力画像原始探测结果",
    description="执行固定白名单环境探测，返回脱敏后的 sandbox capability raw_profile。",
)
async def get_profile(
        capability_probe_service: SandboxCapabilityProbeService = Depends(get_capability_probe_service),
) -> Response[SandboxCapabilityProbePayload]:
    payload = await capability_probe_service.probe_profile()
    return Response.success(
        msg="获取Sandbox能力画像成功",
        data=payload,
    )
