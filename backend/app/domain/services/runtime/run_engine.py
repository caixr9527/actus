#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : run_engine.py
"""
from typing import Protocol, AsyncGenerator

from app.domain.models import BaseEvent, Message


class RunEngine(Protocol):
    """统一运行时引擎协议，屏蔽底层具体编排实现。"""

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        """执行一次消息驱动的运行时流程。"""
        ...
