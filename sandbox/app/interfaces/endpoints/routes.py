#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/12 12:08
@Author : caixiaorong01@outlook.com
@File   : routes.py
"""
from fastapi import APIRouter

from . import file, shell, supervisor, searxng


def create_api_routes() -> APIRouter:
    api_router = APIRouter()
    api_router.include_router(file.router)
    api_router.include_router(shell.router)
    api_router.include_router(supervisor.router)
    api_router.include_router(searxng.router)
    return api_router


router = create_api_routes()
