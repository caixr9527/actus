#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层收尾节点。

本模块只承载 finalize 节点实现，不改收尾事件语义。
"""

import logging
import sys

from app.domain.models import DoneEvent
from app.domain.services.runtime.contracts.runtime_logging import (
    elapsed_ms,
    log_runtime,
    now_perf,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from ..live_events import emit_live_events
from .state_reducer import _reduce_state_with_events

logger = logging.getLogger(__name__)


def _resolve_emit_live_events():
    """统一从 nodes 包级入口解析事件发送函数，保持聚合入口的可替换性。"""
    package_module = sys.modules.get(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes"
    )
    if package_module is not None:
        package_emit_live_events = getattr(package_module, "emit_live_events", None)
        if callable(package_emit_live_events):
            return package_emit_live_events
    return emit_live_events


async def finalize_node(state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
    """结束节点，追加 done 事件。"""
    started_at = now_perf()
    events = list(state.get("emitted_events") or [])
    if events and isinstance(events[-1], DoneEvent):
        log_runtime(
            logger,
            logging.INFO,
            "结束事件已存在，跳过收尾",
            state=state,
            reason="已存在完成事件",
            elapsed_ms=elapsed_ms(started_at),
        )
        return state

    done_event = DoneEvent()
    await _resolve_emit_live_events()(done_event)
    log_runtime(
        logger,
        logging.INFO,
        "流程收尾完成",
        state=state,
        emitted_event_count=len(events) + 1,
        elapsed_ms=elapsed_ms(started_at),
    )
    return _reduce_state_with_events(
        state,
        updates={},
        events=[done_event],
    )
