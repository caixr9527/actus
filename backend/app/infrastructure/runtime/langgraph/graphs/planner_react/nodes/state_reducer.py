#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层 state reducer helper。

本模块只负责把新增事件归并回 graph state，避免后续节点读到过期状态。
"""

from typing import Any, Dict, List, Optional

from app.domain.services.runtime.langgraph_events import append_events
from app.domain.services.runtime.langgraph_state import (
    GraphStateContractMapper,
    PlannerReActLangGraphState,
)


def _reduce_state_with_events(
        state: PlannerReActLangGraphState,
        *,
        updates: Dict[str, Any],
        events: Optional[List[Any]] = None,
) -> PlannerReActLangGraphState:
    """把新增事件立即收敛回 graph state，避免图内后续节点读到过期 step 状态。"""
    new_events = list(events or [])
    next_state: PlannerReActLangGraphState = {
        **state,
        **updates,
        "emitted_events": append_events(state.get("emitted_events"), *new_events),
    }
    if len(new_events) == 0:
        return next_state
    return GraphStateContractMapper.apply_emitted_events(state=next_state)
