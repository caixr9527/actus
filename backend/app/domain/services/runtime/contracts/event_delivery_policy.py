#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行时事件投递/持久化策略。

统一定义事件是否需要：
1. 进入实时输出流；
2. 落入持久化事件历史；
3. 参与历史回放。
"""

from __future__ import annotations

from enum import Enum

from app.domain.models import BaseEvent


class EventDeliveryPolicy(str, Enum):
    """事件投递策略。"""

    PERSISTENT_AND_LIVE = "persistent_and_live"
    LIVE_ONLY = "live_only"


_LIVE_ONLY_EVENT_TYPES = {
    "text_stream_start",
    "text_stream_delta",
    "text_stream_end",
}


def get_event_delivery_policy(event: BaseEvent) -> EventDeliveryPolicy:
    """根据事件类型解析统一投递策略。"""
    event_type = str(getattr(event, "type", "") or "").strip()
    if event_type in _LIVE_ONLY_EVENT_TYPES:
        return EventDeliveryPolicy.LIVE_ONLY
    return EventDeliveryPolicy.PERSISTENT_AND_LIVE


def should_persist_event(event: BaseEvent) -> bool:
    """判断事件是否允许进入持久化事件历史。"""
    return get_event_delivery_policy(event) == EventDeliveryPolicy.PERSISTENT_AND_LIVE
