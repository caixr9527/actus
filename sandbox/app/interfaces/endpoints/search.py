#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/4 00:03
@Author : caixiaorong01@outlook.com
@File   : searxng.py
"""
from fastapi import APIRouter, Depends

from app.interfaces.schemas import Response, SearXNGFetchPageRequest, SearXNGSearchRequest
from app.interfaces.service_dependencies import get_searxng_service, get_bocha_search_service
from app.models import SearXNGFetchPageResult, SearXNGStatusResult, SearXNGSearchResult
from app.services import SearXNGService, BochaSearchService

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


@router.post(
    path="/ai-search",
    response_model=Response[SearXNGSearchResult],
    summary="调用Bocha执行AI搜索",
    description="调用博查 web-search，并返回与 SearXNG/search 尽量一致的响应结构",
)
async def ai_search(
        request: SearXNGSearchRequest,
        bocha_search_service: BochaSearchService = Depends(get_bocha_search_service)
) -> Response[SearXNGSearchResult]:
    result = await bocha_search_service.search(
        query=request.query,
    )
    return Response.success(
        msg=f"Bocha搜索成功, 返回{len(result.results)}条结果",
        data=result,
    )


@router.post(
    path="/fetch-page",
    response_model=Response[SearXNGFetchPageResult],
    summary="读取单个页面正文",
    description="使用 crawl4ai 读取指定 URL 的页面正文",
)
async def fetch_page(
        request: SearXNGFetchPageRequest,
        searxng_service: SearXNGService = Depends(get_searxng_service)
) -> Response[SearXNGFetchPageResult]:
    result = await searxng_service.fetch_page(
        url=request.url,
        max_chars=request.max_chars,
    )
    return Response.success(
        msg="页面读取成功",
        data=result,
    )
