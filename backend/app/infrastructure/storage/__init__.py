#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/12 17:01
@Author : caixiaorong01@outlook.com
@File   : __init__.py.py
"""
from .cos import get_cos, Cos
from .redis import get_redis_client, RedisClient


def get_postgres():
    from .postgres import get_postgres as _get_postgres
    return _get_postgres()


def get_db_session():
    from .postgres import get_db_session as _get_db_session
    return _get_db_session()


def get_uow():
    from .postgres import get_uow as _get_uow
    return _get_uow()

__all__ = [
    "get_redis_client",
    "get_postgres",
    "get_db_session",
    "get_cos",
    "Cos",
    "RedisClient",
    "get_uow",
]
