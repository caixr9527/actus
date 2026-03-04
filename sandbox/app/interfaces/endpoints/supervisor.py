#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/12 12:07
@Author : caixiaorong01@outlook.com
@File   : supervisor.py
"""
from typing import List

from fastapi import APIRouter, Depends

from app.interfaces.schemas import Response, TimeoutRequest
from app.interfaces.service_dependencies import get_supervisor_service
from app.models import ProcessInfo, SupervisorActionResult, SupervisorTimeout
from app.services import SupervisorService

router = APIRouter(prefix="/supervisor", tags=["supervisor模块"])


@router.get(
    path="/status",
    response_model=Response[List[ProcessInfo]],
    summary="获取Supervisor状态",
    description="获取Supervisor状态",
)
async def get_status(
        supervisor_service: SupervisorService = Depends(get_supervisor_service)
) -> Response[List[ProcessInfo]]:
    processes = await supervisor_service.get_all_processes()
    return Response.success(
        msg="获取沙箱服务进程成功",
        data=processes
    )


@router.post(
    path="/stop-all-processes",
    response_model=Response[SupervisorActionResult],
    summary="停止所有进程",
    description="停止所有进程",
)
async def stop_all_processes(
        supervisor_service: SupervisorService = Depends(get_supervisor_service)
) -> Response[SupervisorActionResult]:
    result = await supervisor_service.stop_all_processes()
    return Response.success(
        msg="停止所有进程成功",
        data=result
    )


@router.post(
    path="/shutdown",
    response_model=Response[SupervisorActionResult],
)
async def shutdown(
        supervisor_service: SupervisorService = Depends(get_supervisor_service)
) -> Response[SupervisorActionResult]:
    result = await supervisor_service.shutdown()
    return Response.success(
        msg="关闭Supervisor成功",
        data=result
    )


@router.post(
    path="/restart",
    response_model=Response[SupervisorActionResult],
    summary="重启Supervisor",
    description="重启Supervisor",
)
async def restart(
        supervisor_service: SupervisorService = Depends(get_supervisor_service)
) -> Response[SupervisorActionResult]:
    result = await supervisor_service.restart()
    return Response.success(
        msg="重启Supervisor成功",
        data=result
    )


@router.post(
    path="/activate-timeout",
    response_model=Response[SupervisorTimeout],
)
async def activate_timeout(
        request: TimeoutRequest,
        supervisor_service: SupervisorService = Depends(get_supervisor_service),
) -> Response[SupervisorTimeout]:
    """传递分钟激活超时沙箱销毁设置，并关闭自动保活配置"""
    result = await supervisor_service.activate_timeout(request.minutes)
    supervisor_service.disable_expand()
    return Response.success(
        msg=f"超时销毁已设置, 所有服务与沙箱将在{result.timeout_minutes}分钟后销毁",
        data=result
    )


@router.post(
    path="/extend-timeout",
    response_model=Response[SupervisorTimeout],
)
async def extend_timeout(
        request: TimeoutRequest,
        supervisor_service: SupervisorService = Depends(get_supervisor_service),
) -> Response[SupervisorTimeout]:
    """传递指定的分钟延长超时时间并关闭自动保活"""
    result = await supervisor_service.extend_timeout(request.minutes)
    supervisor_service.disable_expand()
    return Response.success(
        msg=f"超时销毁时间已延长{request.minutes}分钟, 所有服务与沙箱将在{result.timeout_minutes}后销毁",
        data=result,
    )


@router.post(
    path="/cancel-timeout",
    response_model=Response[SupervisorTimeout],
)
async def cancel_timeout(
        supervisor_service: SupervisorService = Depends(get_supervisor_service),
) -> Response[SupervisorTimeout]:
    """取消超时销毁配置"""
    result = await supervisor_service.cancel_timeout()
    return Response.success(
        msg=f"超时销毁已取消" if result.status == "timeout_cancelled" else "超时销毁未激活",
        data=result,
    )


@router.get(
    path="/timeout-status",
    response_model=Response[SupervisorTimeout],
)
async def get_timeout_status(
        supervisor_service: SupervisorService = Depends(get_supervisor_service),
) -> Response[SupervisorTimeout]:
    """获取当前supervisor的超时状态配置"""
    result = await supervisor_service.get_timeout_status()
    msg = "未激活超时销毁" if not result.active else f"剩余超时销毁分钟数: {result.remaining_seconds // 60}"
    return Response.success(
        msg=msg,
        data=result,
    )
