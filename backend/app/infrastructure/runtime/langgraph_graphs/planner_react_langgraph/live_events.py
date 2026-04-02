#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行态实时事件回调绑定与投递。"""

import inspect
import logging
from contextvars import ContextVar, Token
from typing import Awaitable, Callable, Optional

from app.domain.models import BaseEvent
from .runtime_logging import log_runtime

logger = logging.getLogger(__name__)

_LIVE_EVENT_SINK: ContextVar[
    Optional[Callable[[BaseEvent], Optional[Awaitable[None]]]]
] = ContextVar("langgraph_live_event_sink", default=None)


def bind_live_event_sink(
        sink: Optional[Callable[[BaseEvent], Optional[Awaitable[None]]]],
) -> Token:
    """绑定当前运行实例的实时事件输出回调。"""
    return _LIVE_EVENT_SINK.set(sink)


def unbind_live_event_sink(token: Token) -> None:
    """清理当前运行实例的实时事件输出回调。"""
    _LIVE_EVENT_SINK.reset(token)


async def emit_live_events(*events: BaseEvent) -> None:
    """将节点事件实时投递给运行时（best effort，不影响主链路）。"""
    sink = _LIVE_EVENT_SINK.get()
    if sink is None:
        return

    for event in events:
        try:
            emitted = sink(event)
            if inspect.isawaitable(emitted):
                await emitted
        except Exception as e:
            log_runtime(
                logger,
                logging.WARNING,
                "实时事件投递失败",
                event_type=event.type,
                error=str(e),
            )
