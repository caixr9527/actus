#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行态实时事件回调绑定与投递。"""

import inspect
import logging
from dataclasses import dataclass
from contextvars import ContextVar, Token
from typing import Awaitable, Callable, Optional

from app.domain.models import BaseEvent
from app.domain.services.runtime.contracts.runtime_logging import log_runtime

logger = logging.getLogger(__name__)

_LIVE_EVENT_SINK: ContextVar[
    Optional[Callable[[BaseEvent], Optional[Awaitable[None]]]]
] = ContextVar("langgraph_live_event_sink", default=None)
_LIVE_EVENT_ACK_SINK: ContextVar[
    Optional[Callable[[BaseEvent], Optional[Awaitable[BaseEvent | None]]]]
] = ContextVar("langgraph_live_event_ack_sink", default=None)


@dataclass(frozen=True)
class LiveEventBinding:
    sink_token: Token
    ack_sink_token: Token


def bind_live_event_sink(
        sink: Optional[Callable[[BaseEvent], Optional[Awaitable[None]]]],
) -> Token:
    """绑定当前运行实例的实时事件输出回调。"""
    return _LIVE_EVENT_SINK.set(sink)


def bind_live_event_ack_sink(
        sink: Optional[Callable[[BaseEvent], Optional[Awaitable[BaseEvent | None]]]],
) -> Token:
    """绑定实时事件同步回调，graph 会等待该回调完成后继续执行。"""
    return _LIVE_EVENT_ACK_SINK.set(sink)


def bind_live_event_sinks(
        sink: Optional[Callable[[BaseEvent], Optional[Awaitable[None]]]],
        ack_sink: Optional[Callable[[BaseEvent], Optional[Awaitable[BaseEvent | None]]]] = None,
) -> LiveEventBinding:
    return LiveEventBinding(
        sink_token=bind_live_event_sink(sink),
        ack_sink_token=bind_live_event_ack_sink(ack_sink),
    )


def unbind_live_event_sink(token: Token) -> None:
    """清理当前运行实例的实时事件输出回调。"""
    _LIVE_EVENT_SINK.reset(token)


def unbind_live_event_ack_sink(token: Token) -> None:
    """清理当前运行实例的实时事件同步回调。"""
    _LIVE_EVENT_ACK_SINK.reset(token)


def unbind_live_event_sinks(binding: LiveEventBinding | Token) -> None:
    if isinstance(binding, LiveEventBinding):
        unbind_live_event_ack_sink(binding.ack_sink_token)
        unbind_live_event_sink(binding.sink_token)
        return
    unbind_live_event_sink(binding)


async def emit_live_events(*events: BaseEvent) -> None:
    """将节点事件实时投递给运行时（best effort，不影响主链路）。"""
    sink = _LIVE_EVENT_SINK.get()
    ack_sink = _LIVE_EVENT_ACK_SINK.get()
    if sink is None and ack_sink is None:
        return

    for event in events:
        try:
            if ack_sink is not None:
                acknowledged = ack_sink(event)
                acknowledged_event = await acknowledged if inspect.isawaitable(acknowledged) else acknowledged
                if isinstance(acknowledged_event, BaseEvent):
                    event = acknowledged_event
            if sink is not None:
                emitted = sink(event)
                if inspect.isawaitable(emitted):
                    await emitted
        except Exception as e:
            log_runtime(
                logger,
                logging.ERROR,
                "实时事件投递失败",
                event_type=event.type,
                error=str(e),
            )
            if ack_sink is not None:
                raise
