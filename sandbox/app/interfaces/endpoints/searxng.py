#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/4 00:03
@Author : caixiaorong01@outlook.com
@File   : searxng.py
"""
from fastapi import APIRouter, Depends

from app.interfaces.schemas import Response, SearXNGSearchRequest
from app.interfaces.service_dependencies import get_searxng_service
from app.models import SearXNGStatusResult, SearXNGSearchResult
from app.services import SearXNGService

router = APIRouter(prefix="/searxng", tags=["SearXNG模块"])


@router.get(
    path="/status",
    response_model=Response[SearXNGStatusResult],
    summary="获取SearXNG服务状态",
    description="获取容器内SearXNG服务状态",
)
async def get_status(
        searxng_service: SearXNGService = Depends(get_searxng_service)
) -> Response[SearXNGStatusResult]:
    result = await searxng_service.get_status()
    return Response.success(
        msg="SearXNG服务可用" if result.available else "SearXNG服务不可用",
        data=result,
    )


@router.post(
    path="/search",
    response_model=Response[SearXNGSearchResult],
    summary="调用SearXNG执行搜索",
    description="调用容器内SearXNG JSON API执行搜索",
)
async def search(
        request: SearXNGSearchRequest,
        searxng_service: SearXNGService = Depends(get_searxng_service)
) -> Response[SearXNGSearchResult]:
    result = await searxng_service.search(
        query=request.query,
        categories=request.categories,
        engines=request.engines,
        language=request.language,
        page=request.page,
        time_range=request.time_range,
        safesearch=request.safesearch,
    )
    return Response.success(
        msg=f"SearXNG搜索成功, 返回{len(result.results)}条结果",
        data=result,
    )
