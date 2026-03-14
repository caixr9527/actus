#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/13 15:00
@Author : caixiaorong01@outlook.com
@File   : routes.py
"""
from fastapi import APIRouter, Depends

from app.interfaces.endpoints import (
    status_routes,
    app_config_routes,
    file_routes,
    session_routes,
    auth_routes,
)
from app.interfaces.dependencies.auth import get_current_user


def create_api_route() -> APIRouter:
    """路由管理"""
    api_router = APIRouter()

    api_router.include_router(status_routes.router)
    api_router.include_router(app_config_routes.router, dependencies=[Depends(get_current_user)])
    api_router.include_router(file_routes.router, dependencies=[Depends(get_current_user)])
    api_router.include_router(session_routes.router, dependencies=[Depends(get_current_user)])
    api_router.include_router(auth_routes.router)
    return api_router


router = create_api_route()
