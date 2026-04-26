#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : langgraph_events.py
"""
from typing import List

from app.domain.models import BaseEvent


def append_events(current_events: List[BaseEvent] | None, *events: BaseEvent) -> List[BaseEvent]:
    """构建新的事件列表，避免节点内原地修改导致状态语义混乱。"""
    merged = list(current_events or [])
    merged.extend(events)
    return merged
