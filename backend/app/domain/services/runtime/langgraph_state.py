#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : langgraph_state.py
"""
from typing import TypedDict, Optional, List

from app.domain.models import BaseEvent, Plan


class PlannerReActPOCState(TypedDict, total=False):
    """LangGraph POC 使用的最小状态对象。"""

    session_id: str
    user_message: str
    plan: Plan
    final_message: str
    emitted_events: List[BaseEvent]
    error: Optional[str]
